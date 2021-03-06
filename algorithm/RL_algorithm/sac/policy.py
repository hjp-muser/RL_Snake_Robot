import tensorflow as tf
import numpy as np
from gym.spaces import Box

from algorithm.RL_algorithm.utils.tensorflow1.layer_utils import mlp
from algorithm.RL_algorithm.utils.tensorflow1.policy_utils import BasePolicy, nature_cnn, register_policy

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)


def log_gaussian_likelihood(x, mu, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param x: (tf.Tensor)
    :param mu: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: gaussian_prob (tf.Tensor)
    """
    log_gaussian_prob = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 - log_std - 0.5 * np.log(2 * np.pi))
    return tf.reduce_sum(log_gaussian_prob, axis=1)


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal Gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def clip_but_pass_gradient(x, lower=-1., upper=1.):
    clip_up = tf.cast(x > upper, tf.float32)
    clip_low = tf.cast(x < lower, tf.float32)
    return x + tf.stop_gradient((upper - x) * clip_up + (lower - x) * clip_low)


def apply_squashing_func(mu, x, logp_pi):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu: (tf.Tensor) Mean of the gaussian
    :param x: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu)
    policy = tf.tanh(x)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)  # - tanh'(z) ?

    return deterministic_policy, policy, logp_pi


class SACPolicy(BasePolicy):
    """
    Policy object that implements a SAC-like actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False):
        super(SACPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.policy = None
        self.deterministic_policy = None
        self.act_mu = None
        self.std = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    def make_critics(self, obs=None, action=None, reuse=False,
                     scope="values_fn", create_vf=True, create_qf=True):
        """
        Creates the two Q-Values approximator along with the Value function

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies.py)
        :param mask: ([float]) The last masks (used in recurrent policies.py)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float]) actions
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability params (mean, std) for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies.py)
        :param mask: ([float]) The last masks (used in recurrent policies.py)
        :return: ([float], [float])
        """
        raise NotImplementedError


class FeedForwardPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None, act_fns=None, cnn_extractor=nature_cnn, feature_extraction="cnn",
                 reg_weight=0.0, layer_norm=False, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        self.feature_extraction = feature_extraction
        self.cnn_extractor = cnn_extractor
        self.cnn_kwargs = kwargs
        self.layer_norm = layer_norm
        self.reg_weight = reg_weight
        self.reg_loss = None
        self.entropy = None

        if layers is None:
            self.layers = {'mu': [128, 128, 128, 64], 'std': [128, 64, 64], 'vf': [128, 128, 128, 64]}
        else:
            self.layers = layers
        # if isinstance(self.layers, list):
        #     assert len(self.layers) >= 1, "Error: must have at least one hidden layer for the policy."
        # elif isinstance(self.layers, dict):
        #     assert isinstance(self.layers['pi'], list), "Error: the type of value must be list."
        #     assert isinstance(self.layers['vf'], list), "Error: the type of value must be list."
        #     assert len(self.layers['pi']) >= 1, "Error: pi network must have at least one hidden layer for the policy."
        #     assert len(self.layers['vf']) >= 1, "Error: value network must have at least one hidden layer for the policy."
        # else:
        #     raise TypeError("The type of layers must be list or dict.")

        if layers is None:
            self.activ_fns = {'mu': [tf.nn.sigmoid, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu],
                              'std': [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid],
                              'vf': [tf.nn.sigmoid, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.sigmoid]}
        elif act_fns is None:
            self.activ_fns = tf.nn.relu
        # else:
        #     self.activ_fns = act_fns
        # if callable(self.activ_fns):
        #     tmp_activ_fns = self.activ_fns
        #     if isinstance(self.layers, list):
        #         self.activ_fns = []
        #         for _ in range(len(self.layers)):
        #             self.activ_fns.append(tmp_activ_fns)
        #     elif isinstance(self.layers, dict):
        #         self.activ_fns = {'pi': [], 'vf': []}
        #         for _ in range(len(self.layers['pi'])):
        #             self.activ_fns['pi'].append(tmp_activ_fns)
        #         for _ in range(len(self.layers['vf'])):
        #             self.activ_fns['vf'].append(tmp_activ_fns)
        # else:
        #     assert isinstance(self.activ_fns, type(self.layers)), "Error: if act_fns is not a callable function, " \
        #                                                           "it must have the same type as the variable layers."
        #     if isinstance(self.activ_fns, list):
        #         assert len(self.activ_fns) == len(self.layers), "Error: The number of activation functions must " \
        #                                                         "be equal to that of layers."
        #     elif isinstance(self.activ_fns, dict):
        #         assert isinstance(self.activ_fns['pi'], list), "Error: the type of dict value must be list."
        #         assert isinstance(self.activ_fns['vf'], list), "Error: the type of dict value must be list."
        #         assert len(self.activ_fns['pi']) == len(self.layers['pi']), "Error: The number of activation functions " \
        #                                                                     "must be equal to that of layers."
        #         assert len(self.activ_fns['vf']) == len(self.layers['vf']), "Error: The number of activation functions " \
        #                                                                     "must be equal to that of layers."

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        kernel_regularizer = tf.keras.regularizers.l2(0.3)
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("mu", reuse=reuse):
                if self.feature_extraction == "cnn":
                    pi_h_mu = self.cnn_extractor(obs, **self.cnn_kwargs)
                else:
                    pi_h_mu = tf.layers.flatten(obs)

                if isinstance(self.layers, list):
                    pi_h_mu = mlp(pi_h_mu, self.layers, self.activ_fns, layer_norm=self.layer_norm,
                               kernel_initializer=tf.variance_scaling_initializer(), kernel_regularizer=kernel_regularizer)
                elif isinstance(self.layers, dict):
                    pi_h_mu = mlp(pi_h_mu, self.layers['mu'], self.activ_fns['mu'], layer_norm=self.layer_norm,
                               kernel_initializer=tf.variance_scaling_initializer(), kernel_regularizer=kernel_regularizer)
                self.act_mu = tf.nn.tanh(tf.layers.dense(pi_h_mu, self.ac_space.shape[0],
                                                            kernel_initializer=tf.variance_scaling_initializer(),
                                                            kernel_regularizer=kernel_regularizer))

            kernel_regularizer = tf.keras.regularizers.l2(0.3)
            with tf.variable_scope("std", reuse=reuse):
                if self.feature_extraction == "cnn":
                    pi_h_std = self.cnn_extractor(obs, **self.cnn_kwargs)
                else:
                    pi_h_std = tf.layers.flatten(obs)

                if isinstance(self.layers, list):
                    pi_h_std = mlp(pi_h_std, self.layers, self.activ_fns, layer_norm=self.layer_norm,
                               kernel_initializer=tf.variance_scaling_initializer(), kernel_regularizer=kernel_regularizer)
                elif isinstance(self.layers, dict):
                    pi_h_std = mlp(pi_h_std, self.layers['std'], self.activ_fns['std'], layer_norm=self.layer_norm,
                               kernel_initializer=tf.variance_scaling_initializer(), kernel_regularizer=kernel_regularizer)
                log_std = tf.layers.dense(pi_h_std, self.ac_space.shape[0],
                            kernel_initializer=tf.variance_scaling_initializer(), kernel_regularizer=kernel_regularizer)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        # CAP the standard deviation of the actor
        # LOG_STD_MAX = -10
        # LOG_STD_MIN = -20
        # log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # log_std = tf.log(self.std)
        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        self.policy = self.act_mu + tf.random_normal(tf.shape(self.act_mu)) * std
        logp_pi = log_gaussian_likelihood(self.policy, self.act_mu, log_std)
        self.deterministic_policy = self.act_mu
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        # deterministic_policy, policy, logp_pi = apply_squashing_func(mu, x, logp_pi)

        return self.deterministic_policy, self.policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True, double_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    if isinstance(self.layers, list):
                        vf_h = mlp(critics_h, self.layers, self.activ_fns, layer_norm=self.layer_norm,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                    elif isinstance(self.layers, dict):
                        vf_h = mlp(critics_h, self.layers['vf'], self.activ_fns['vf'], layer_norm=self.layer_norm,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    if isinstance(self.layers, list):
                        qf1_h = mlp(qf_h, self.layers, self.activ_fns, layer_norm=self.layer_norm)
                    elif isinstance(self.layers, dict):
                        qf1_h = mlp(qf_h, self.layers['vf'], self.activ_fns['vf'], layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")
                self.qf1 = qf1
                if double_qf:
                    with tf.variable_scope('qf2', reuse=reuse):
                        if isinstance(self.layers, list):
                            qf2_h = mlp(qf_h, self.layers, self.activ_fns, layer_norm=self.layer_norm)
                        elif isinstance(self.layers, dict):
                            qf2_h = mlp(qf_h, self.layers['vf'], self.activ_fns['vf'], layer_norm=self.layer_norm)
                        qf2 = tf.layers.dense(qf2_h, 1, name="qf2")
                    self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action = self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        else:
            action = self.sess.run(self.policy, {self.obs_ph: obs})
        return action, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class LnCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", layer_norm=True, **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("sac_CnnPolicy", CnnPolicy)
register_policy("sac_LnCnnPolicy", LnCnnPolicy)
register_policy("sac_MlpPolicy", MlpPolicy)
register_policy("sac_LnMlpPolicy", LnMlpPolicy)
