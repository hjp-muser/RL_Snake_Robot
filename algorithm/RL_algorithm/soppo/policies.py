import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class PolicyModel(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, policy_latent, vf_latent=None, estimate_q=True, q_network=None, sess=None, **extra_tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **extra_tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.__dict__.update(extra_tensors)

        self.policy_latent = tf.layers.flatten(policy_latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi_mean = self.pdtype.pdfromlatent(self.policy_latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if vf_latent is None:
            vf_latent = policy_latent
        self.vf_latent = tf.layers.flatten(vf_latent)
        self.vf = fc(vf_latent, 'vf', 1)
        self.vf = self.vf[:, 0]

        if estimate_q:
            assert callable(q_network)
            self.q_network = q_network
            with tf.variable_scope('qf', reuse=tf.AUTO_REUSE):
                qf_input = tf.stop_gradient(tf.concat(self.policy_latent, self.vf_latent))
                # qf_input = self.policy_latent * vf_latent
                qf_latent = self.q_network(qf_input)
                self.qf = fc(qf_latent, 'qf', 1)
                self.qf = self.qf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt.op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, neglogp, v = self._evaluate([self.action, self.neglogp, self.vf], observation, **extra_feed)
        return a, neglogp, v

    def estimate_q(self, observation, **extra_feed):
        qf_value = self._evaluate(self.qf, observation, **extra_feed)
        return qf_value

    def estimate_v(self, observation, **extra_feed):
        vf_value = self._evaluate(self.vf, observation, **extra_feed)
        return vf_value

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_policy(env, policy_network, estimate_q=True, q_network=None, normalize_observations=False, **network_kwargs):
    if isinstance(policy_network, str):
        policy_network = get_network_builder(policy_network)(**network_kwargs)
    else:
        assert callable(policy_network)

    if estimate_q:
        # The architecture of q_network is the same as policy_network's.
        if q_network is None:
            q_network = get_network_builder("mlp")(**network_kwargs)
        elif isinstance(q_network, str):
            q_network = get_network_builder(q_network)(**network_kwargs)
        else:
            assert callable(q_network)
            q_network = q_network

    def policy_fn(nbatch=None, sess=None, vf_latent=None, observ_placeholder=None):
        # preprocess input
        ob_space = env.observation_space
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        extra_tensors = {}
        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)

        policy = PolicyModel(
            env=env,
            observations=X,
            policy_latent=policy_latent,
            vf_latent=vf_latent,
            estimate_q=estimate_q,
            q_network=q_network,
            sess=sess,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=None):
    if clip_range is None:
        clip_range = [-5.0, 5.0]
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

