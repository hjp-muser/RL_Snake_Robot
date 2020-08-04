import tensorflow as tf

from algorithm.RL_algorithm.utils.tensorflow1.policy_utils import FeedForwardPolicy, register_policy


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

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        pi_layers = [512, 256, 256, 64]
        vf_layers = [512, 256, 256, 64]
        net_arch = [dict(vf=pi_layers, pi=vf_layers)]
        pi_act_funs = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.tanh]
        vf_act_funs = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.tanh]
        act_funs = [dict(vf=pi_act_funs, pi=vf_act_funs)]
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=net_arch,
                                        act_funs=act_funs, feature_extraction="mlp", **_kwargs)


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

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("ac_MlpPolicy", MlpPolicy)
register_policy("ac_LnMlpPolicy", LnMlpPolicy)
