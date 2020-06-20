from abc import abstractmethod
from gym import spaces
import tensorflow as tf
import numpy as np

from gym.spaces import Box
from algorithm.RL_algorithm.utils.tensorflow1.distribution_utils1 import make_proba_dist_type
from algorithm.RL_algorithm.utils.tensorflow1.layer_utils import mlp
from algorithm.RL_algorithm.utils.tensorflow1.policy_utils import register_policy, BasePolicy, nature_cnn


class MultiActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(MultiActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                     reuse=reuse, scale=scale, add_action_ph=True)
        self._pdtype_list = []
        self._pd_list = []
        self._action = []
        self._deterministic_action = []
        self._value_fn = None
        self._value_flat = None

        assert isinstance(ac_space, spaces.Tuple), "Error: The type of ac_space must be gym.spaces.Tuple."
        for ac_sub_space in ac_space.spaces:
            self._pdtype_list.append(make_proba_dist_type(ac_sub_space))

    @property
    def pdtype_list(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype_list

    @property
    def pd_list(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._pd_list

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @abstractmethod
    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: ([[tf.Tensor],[tf.Tensor],[tf.Tensor]]) policy_distribution list, action list
                and determination action list
        """
        raise NotImplementedError

    @abstractmethod
    def make_critics(self, obs=None, reuse=False, scope="critic"):
        """
        Creates the Value function

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :return: ([tf.Tensor]) the output tensor list
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class MultiFeedForwardPolicy(MultiActorCriticPolicy):
    """
    Policy object that implements actor critic, using a multi-branch feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_reward: The number of different scales for reward
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fns: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 layers=None, act_fns=None, cnn_extractor=nature_cnn, feature_extraction="cnn",
                 reg_weight=0.0, layer_norm=False, **kwargs):
        super(MultiFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                     reuse=reuse, scale=(feature_extraction == "cnn"))

        self.feature_extraction = feature_extraction
        self.reuse = reuse
        self.cnn_extractor = cnn_extractor
        self.cnn_kwargs = kwargs
        self.layer_norm = layer_norm
        self.reg_weight = reg_weight
        self.reg_loss = None
        self.entropy = None

        if layers is None:
            self.layers = {'pi': [[256, 128], [64, 64]], 'vf': [256, 128, 64]}
        else:
            self.layers = layers

        if isinstance(self.layers, dict):
            assert isinstance(self.layers['pi'], list), "Error: the type of value must be list"
            assert isinstance(self.layers['vf'], list), "Error: the type of value must be list"
            assert len(self.layers['pi']) >= 1, "Error: pi network must have at least one hidden layer for the policy."
            assert len(self.layers['pi']) == len(ac_space), "Error: The number of branches in pi network must be " \
                                                                "equal to the number of elements in action space tuple."
            for sub_layers in self.layers['pi']:
                assert isinstance(sub_layers, list), "Error: layers must be divided to several sub-layers, " \
                                                     "such as [[64, 128], [256]]"
                assert len(sub_layers) >= 1, "Error: sub-layers of the pi network must have at least one hidden layer " \
                                             "for the policy."
            assert len(self.layers['vf']) >= 1, "Error: value network must have at least one hidden layer."
            # assert len(self.layers['vf']) == self.n_reward, "Error: The number of branches in value network must be " \
            #                                                 "equal to the number of different scale rewards."
            # for sub_layers in self.layers['vf']:
            #     assert isinstance(sub_layers, list), "Error: layers must be divided to several sub-layers, " \
            #                                          "such as [[64, 128], [256]]"
            #     assert len(
            #         sub_layers) >= 1, "Error: sub-layers of the value network must have at least one hidden layer."
        else:
            raise TypeError(
                "The type of layers must be dict, such as {'pi': [[64, 128], [64]], 'vf': [[64, 128], [64]]}")

        if act_fns is None and layers is None:
            self.activ_fns = {'pi': [[tf.nn.relu, tf.nn.tanh], [tf.nn.relu, tf.nn.tanh]],
                              'vf': [tf.nn.relu, tf.nn.tanh, tf.nn.relu]}
        elif act_fns is None:
            self.activ_fns = tf.nn.relu
        else:
            self.activ_fns = act_fns
        if callable(self.activ_fns):
            tmp_activ_fns = self.activ_fns
            if isinstance(self.layers, dict):
                self.activ_fns = {'pi': [], 'vf': []}
                for sub_layers in range(len(self.layers['pi'])):
                    sub_activ_fns = []
                    for _ in range(len(sub_layers)):
                        sub_activ_fns.append(tmp_activ_fns)
                    self.activ_fns['pi'].append(sub_activ_fns)
                # for sub_layers in range(len(self.layers['vf'])):
                #     sub_activ_fns = []
                #     for _ in range(len(sub_layers)):
                #         sub_activ_fns.append(tmp_activ_fns)
                #     self.activ_fns['vf'].append(sub_activ_fns)
                for _ in range(len(self.layers['vf'])):
                    self.activ_fns['vf'].append(tmp_activ_fns)
            else:
                raise TypeError(
                    "The type of layers must be dict, such as {'pi': [[64, 128], [64]], 'vf': [64, 128, 256]}")
        else:
            if isinstance(self.activ_fns, dict):
                assert isinstance(self.activ_fns['pi'], list), "Error: the type of dict value must be list."
                assert isinstance(self.activ_fns['vf'], list), "Error: the type of dict value must be list."
                assert len(self.activ_fns['pi']) == len(self.layers['pi']), "Error: The number of activation functions " \
                                                                            "must be equal to that of layers for policy network."
                for i in range(len(self.layers['pi'])):
                    assert isinstance(self.activ_fns['pi'][i],
                                      list), "Error: acti_fns must be divided to several sub-acti_fns," \
                                             "such as [[relu, tanh], [relu]] for policy network."
                    assert len(self.layers['pi'][i]) == len(self.activ_fns['pi'][i]), \
                        "Error: The number of sub-act_fns must be equal to that of sub-layers for policy network."
                assert len(self.activ_fns['vf']) == len(self.layers['vf']), "Error: The number of activation functions." \
                                                                            "must be equal to that of layers for value network"
                # for i in range(len(self.layers['vf'])):
                #     assert isinstance(self.activ_fns['vf'][i],
                #                       list), "Error: acti_fns must be divided to several sub-acti_fns," \
                #                              "such as [[relu, tanh], [relu]] for value network."
                #     assert len(self.layers['vf'][i]) == len(self.activ_fns['vf'][i]), \
                #         "Error: The number of sub-act_fns must be equal to that of sub-layers for value network."
            else:
                raise TypeError(
                    "The type of activ_fns must be a callable function or dict which has the same structure as layers.")

        self.make_actor(reuse=self.reuse)
        self.make_critics(reuse=self.reuse)

    def make_actor(self, obs=None, reuse=False, scope="action"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=self.reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            assert isinstance(self.layers, dict), "Error: The type of layers must be dict."
            for sub_id in range(len(self.layers['pi'])):
                if sub_id == 0:
                    pi_h = mlp(pi_h, self.layers['pi'][sub_id], self.activ_fns['pi'][sub_id],
                               add_fc_id=0, layer_norm=self.layer_norm)
                else:
                    pi_h = mlp(pi_h, self.layers['pi'][sub_id], self.activ_fns['pi'][sub_id],
                               add_fc_id=len(self.layers['pi'][sub_id-1]), layer_norm=self.layer_norm)
                pd = self.pdtype_list[sub_id].proba_distribution_from_latent(pi_h, name_id=sub_id, init_scale=0.01)
                self.pd_list.append(pd)

            assert len(self.pd_list) >= 1
            for sub_id in range(len(self.ac_space)):
                action = self.pd_list[sub_id].sample()
                if isinstance(self.ac_space[sub_id], Box):
                    action = tf.clip_by_value(action, self.ac_space[sub_id].low, self.ac_space[sub_id].high)
                deterministic_action = self.pd_list[sub_id].mode()
                if isinstance(self.ac_space[sub_id], Box):
                    deterministic_action = tf.clip_by_value(deterministic_action, self.ac_space[sub_id].low,
                                                            self.ac_space[sub_id].high)
                self._action.append(action)
                self._deterministic_action.append(deterministic_action)

    def make_critics(self, obs=None, reuse=False, scope="critic"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            vf_h = mlp(critics_h, self.layers['vf'], self.activ_fns['vf'],
                       layer_norm=self.layer_norm)
            vf_h = tf.layers.dense(vf_h, 1, name="vf")
        self._value_fn = vf_h
        self._value_flat = self.value_fn[:, 0]

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value = self.sess.run([self.deterministic_action, self.value_fn], {self.obs_ph: obs})
        else:
            action, value = self.sess.run([self.action, self.value_fn], {self.obs_ph: obs})
        action = np.hstack(action)
        return action, value, self.initial_state, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.pd_list], {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class MlpPolicy(MultiFeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_reward: The number of different scales for reward
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                        reuse, layer_norm=False, feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(MultiFeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP, with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_reward: The number of different scales for reward
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                          reuse, layer_norm=True, feature_extraction="mlp", **_kwargs)


register_policy("multia2c_MlpPolicy", MlpPolicy)
register_policy("multia2c_LnMlpPolicy", LnMlpPolicy)
