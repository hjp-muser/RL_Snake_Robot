import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common import tf_util
# from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd

# from baselines.common.models import get_network_builder
from algorithm.RL_algorithm.soppo.distributions import make_pdtype
from algorithm.RL_algorithm.soppo.network import get_network_builder

# import gym


class StochasticActorModel(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, action_space, observations, actor_latent, sess=None, **extra_tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        sess            tensorflow session to run calculations in (if None, default session is used)

        **extra_tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.obs_ph = observations
        self.__dict__.update(extra_tensors)

        self.actor_latent = tf.layers.flatten(actor_latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(action_space)

        self.pd, self.pd_param = self.pdtype.pdfromlatent(self.actor_latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.obs_ph: adjust_shape(self.obs_ph, observation)}
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

        action = self._evaluate(self.action, observation, **extra_feed)
        return action

    def cal_action_neglogp(self, action):
        return self.pd.neglogp(action)

    def cal_entropy(self):
        return self.pd.entropy()

    def cal_pdparams(self, observation, **extra_feed):
        pdparams = self._evaluate(self.pd_param, observation, **extra_feed)
        return pdparams

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


class DeterminedActorModel(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, action_space, observations, actor_latent, sess=None, **extra_tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        sess            tensorflow session to run calculations in (if None, default session is used)

        **extra_tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.obs_ph = observations
        self.__dict__.update(extra_tensors)

        self.actor_latent = tf.layers.flatten(actor_latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(action_space)

        self.pd, self.pd_param = self.pdtype.pdfromlatent(self.actor_latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.mode()

        self.sess = sess or tf.get_default_session()

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.obs_ph: adjust_shape(self.obs_ph, observation)}
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

        action = self._evaluate(self.action, observation, **extra_feed)
        return action

    def cal_pdparams(self, observation, **extra_feed):
        pdparams = self._evaluate(self.pd_param, observation, **extra_feed)
        return pdparams

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_actor(observation_space, action_space, actor_network, is_determined=False, **network_kwargs):
    if isinstance(actor_network, str):
        actor_network = get_network_builder(actor_network, **network_kwargs)
    else:
        assert callable(actor_network)

    def actor_fn(obs_ph=None, normalize_observations=False, vf_latent=None, sess=None):
        # preprocess input
        ob_space = observation_space
        X = obs_ph if obs_ph is not None else observation_placeholder(ob_space)
        extra_tensors = {}
        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            actor_latent = actor_network(encoded_x)

        if is_determined is False:
            actor_model = StochasticActorModel(
                action_space=action_space,
                observations=X,
                actor_latent=actor_latent,
                vf_latent=vf_latent,
                sess=sess,
                **extra_tensors
            )
        else:
            actor_model = DeterminedActorModel(
                action_space=action_space,
                observations=X,
                actor_latent=actor_latent,
                vf_latent=vf_latent,
                sess=sess,
                **extra_tensors
            )

        return actor_model

    return actor_fn


def _normalize_clip_observation(x, clip_range=None):
    if clip_range is None:
        clip_range = [-5.0, 5.0]
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

