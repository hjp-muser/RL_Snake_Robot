import tensorflow as tf

from baselines.a2c.utils import fc
from baselines.common.models import get_network_builder
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util
from baselines.common.tf_util import adjust_shape

from algorithm.RL_algorithm.utils.tensorflow1.input import observation_placeholder, encode_observation, encode_action, \
    action_placeholder


class ValueModel(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, observations, latent, sess=None, **extra_tensors):
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

        self.obs_ph = observations
        self.__dict__.update(extra_tensors)

        self.vf_latent = tf.layers.flatten(latent)
        self.sess = sess or tf.get_default_session()
        self.vf = fc(self.vf_latent, 'vf_out', 1)
        self.vf = self.vf[:, 0]

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

        vpred = self._evaluate(self.vf, observation, **extra_feed)
        return vpred

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_value(observation_space, value_network, **network_kwargs):
    if isinstance(value_network, str):
        value_network = get_network_builder(value_network)(**network_kwargs)
    else:
        assert callable(value_network)

    def value_fn(obs_ph=None, normalize_observations=False, sess=None):
        ob_space = observation_space
        X = obs_ph if obs_ph is not None else observation_placeholder(ob_space)
        extra_tensors = {}
        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            vf_latent = value_network(encoded_x)

        value = ValueModel(
            observations=X,
            latent=vf_latent,
            sess=sess,
            **extra_tensors
        )
        return value

    return value_fn


class QValueModel(object):
    """
        Encapsulates fields and methods for RL policy and value function estimation with shared parameters
        """

    def __init__(self, vf_model, actor_model, latent, sess=None):
        """
        Parameters:
        ----------

        latent          latent state from which policy distribution parameters should be inferred

        sess            tensorflow session to run calculations in (if None, default session is used)

        **extra_tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.qf_latent = latent
        self.vf_model = vf_model
        self.actor_model = actor_model
        self.qf = fc(self.qf_latent, 'qf_out', 1)[:, 0]
        self.sess = sess or tf.get_default_session()

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.vf_model.obs_ph: adjust_shape(self.vf_model.obs_ph, observation),
                     self.actor_model.obs_ph: adjust_shape(self.actor_model.obs_ph, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt.op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observations, **extra_feed):
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

        qpred = self._evaluate(self.qf, observations, **extra_feed)
        return qpred

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_qvalue(vf_model, actor_model, qvalue_network, **network_kwargs):
    if isinstance(qvalue_network, str):
        qvalue_network = get_network_builder(qvalue_network)(**network_kwargs)
    else:
        assert callable(qvalue_network)

    def qvalue_fn(sess=None):
        qf_input = tf.concat([vf_model.vf_latent, actor_model.actor_latent], axis=-1)
        with tf.variable_scope('qf', reuse=tf.AUTO_REUSE):
            qf_latent = qvalue_network(qf_input)
        qvalue = QValueModel(
            vf_model=vf_model,
            actor_model=actor_model,
            latent=qf_latent,
            sess=sess
        )
        return qvalue

    return qvalue_fn


def _normalize_clip_observation(x, clip_range=None):
    if clip_range is None:
        clip_range = [-5.0, 5.0]
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
