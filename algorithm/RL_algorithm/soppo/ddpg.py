import os
from copy import copy
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common.mpi_running_mean_std import RunningMeanStd

from algorithm.RL_algorithm.soppo.memory import Memory
from algorithm.RL_algorithm.soppo.actors import build_actor
from algorithm.RL_algorithm.soppo.critics import build_value, build_qvalue
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import get_session

try:
    from mpi4py import MPI
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
except ImportError:
    MPI = None


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class Model(object):
    def __init__(self, *, network, obs_space, ac_space, memory=None, tau=0.01, max_grad_norm=0.5,
                 normalize_observations=False, noise_type=None,
                 batch_size=4, actor_reg_coef=0.1, critic_reg_coef=0.2, actor_lr=3e-5, critic_lr=1e-5,
                 **network_kwargs):
        # logger.info('Using agent with the following configuration:')
        # logger.info(str(self.__dict__.items()))
        self.observation_space = obs_space
        self.action_space = ac_space
        self.observation_shape = obs_space.shape
        self.action_shape = ac_space.shape
        self.sess = get_session()

        # Inputs.
        self.adv_ph = tf.placeholder(tf.float32, shape=(None, 1), name='advantages')
        self.target_qf_ph = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        # Parameters.
        self.tau = tau
        self.normalize_observations = normalize_observations
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_reg_coef = actor_reg_coef
        self.critic_reg_coef = critic_reg_coef
        self.stats_sample = None
        self.initial_state = None  # recurrent architectures not supported yet
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/ddpg_tmp/'

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
        else:
            self.obs_rms = None

        # set action_noise and param_noise
        self.action_noise, self.param_noise = self._init_noise(noise_type)

        with tf.variable_scope('shared_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('vf_model'):
                    # create vf model, default: mlp value network
                    value_network = build_value(obs_space, network, **network_kwargs)
                    self.vf_model = value_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('vf_model'):
                    value_network = build_value(obs_space, network, **network_kwargs)
                    self.target_vf_model = value_network(sess=self.sess)
                    # self.target_vf_model.vf = tf.stop_gradient(self.target_vf_model.vf)     # stop gradient

        with tf.variable_scope('ddpg_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('policy'):
                    # create policy model, default: mlp policy network
                    policy_network = build_actor(obs_space, ac_space, network, is_determined=True, **network_kwargs)
                    self.policy_model = policy_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('policy'):
                    policy_network = build_actor(obs_space, ac_space, network, is_determined=True, **network_kwargs)
                    self.target_policy_model = policy_network(sess=self.sess)
                    # self.target_policy_model.action = tf.stop_gradient(self.target_policy_model.action)  # stop gradient
            with tf.variable_scope('current'):
                with tf.variable_scope('qf_model'):
                    qvalue_network = build_qvalue(self.target_vf_model, self.target_policy_model, network,
                                                  **network_kwargs)
                    self.qf_model = qvalue_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('qf_model'):
                    target_qvalue_network = build_qvalue(self.vf_model, self.policy_model, network, **network_kwargs)
                    self.target_qf_model = target_qvalue_network(sess=self.sess)

        self.vf_train_variables = tf.trainable_variables('shared_model/current/vf_model')
        self.policy_train_variables = tf.trainable_variables('ddpg_model/current/policy')
        self.qf_train_variables = tf.trainable_variables('ddpg_model/current/qf_model')
        self.target_vf_train_variables = tf.trainable_variables('shared_model/target/vf_model')
        self.target_policy_train_variables = tf.trainable_variables('ddpg_model/target/policy')
        self.target_qf_train_variables = tf.trainable_variables('ddpg_model/target/qf_model')

        self.target_qf = self.adv_ph + self.target_vf_model.vf

        self.step = self.policy_model.step
        self.value = self.vf_model.step
        # self.action_qvalue = self.target_qf_model
        # self.policy_action = self.policy_model.action
        # self.action_qvalue = self.qf_model(self.obs0_ph, self.ac_ph)
        # self.policy_qvalue = self.qf_model(self.obs0_ph, self.policy_model(self.obs0_ph))
        # target_policy_qvalue_obs1 = self.target_qf_model(self.obs1_ph, self.target_policy_model(self.obs1_ph))
        # self.target_policy_qvalue = self.adv_ph + target_policy_qvalue_obs1
        if memory is None:
            self.memory = Memory(limit=int(1e6), action_shape=ac_space.shape, observation_shape=obs_space.shape)
        else:
            self.memory = memory

        # Set up parts
        self._setup_actor_optimizer()
        self._setup_critic_optimizer()
        self._setup_target_network_updates()

        self._initialize()

    def _init_noise(self, noise_type):
        action_noise = None
        param_noise = None
        if noise_type is not None:
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                         desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(self.action_shape[-1]),
                                                     sigma=float(stddev) * np.ones(self.action_shape[-1]))
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_shape[-1]),
                                                                sigma=float(stddev) * np.ones(self.action_shape[-1]))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
        return action_noise, param_noise

    def _initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    def _setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.target_qf_model.qf)
        self.actor_train_variables = self.policy_train_variables + self.vf_train_variables
        if self.actor_reg_coef > 0.:
            actor_reg_vars = [var for var in self.actor_train_variables if
                              var.name.endswith('/w:0') and 'output' not in var.name]
            for var in actor_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying actor regularization with {}'.format(self.actor_reg_coef))
            actor_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.actor_reg_coef),
                weights_list=actor_reg_vars
            )
            self.actor_loss += actor_reg
        actor_shapes = [var.get_shape().as_list() for var in self.actor_train_variables]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))

        self.actor_trainer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, epsilon=1e-5)
        grads_and_var = self.actor_trainer.compute_gradients(self.actor_loss, self.actor_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.actor_grads = grads
        grads_and_var = list(zip(grads, var))
        self.actor_optimizer = self.actor_trainer.apply_gradients(grads_and_var)

    def _setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        self.critic_train_variables = self.qf_train_variables
        self.critic_loss = tf.reduce_mean(tf.square(self.qf_model.qf - self.target_qf_ph))
        if self.critic_reg_coef > 0.:
            critic_reg_vars = [var for var in self.critic_train_variables if
                               var.name.endswith('/w:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_reg_coef))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_reg_coef),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic_train_variables]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))

        self.critic_trainer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, epsilon=1e-5)
        grads_and_var = self.critic_trainer.compute_gradients(self.critic_loss, self.critic_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.critic_grads = grads
        grads_and_var = list(zip(grads, var))
        self.critic_optimizer = self.critic_trainer.apply_gradients(grads_and_var)

    def _setup_target_network_updates(self):
        policy_init_updates, policy_soft_updates = get_target_updates(self.policy_train_variables,
                                                                      self.target_policy_train_variables,
                                                                      self.tau)
        vf_init_updates, vf_soft_updates = get_target_updates(self.vf_train_variables,
                                                              self.target_vf_train_variables,
                                                              self.tau)
        qf_init_updates, qf_soft_updates = get_target_updates(self.qf_train_variables,
                                                              self.target_qf_train_variables,
                                                              self.tau)
        self.target_init_updates = [policy_init_updates, vf_init_updates, qf_init_updates]
        self.target_soft_updates = [policy_soft_updates, vf_soft_updates, qf_soft_updates]

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)
        target_qf = self.sess.run(self.target_qf, feed_dict={
            self.target_vf_model.obs_ph: batch['observations'],
            self.adv_ph: batch['advantages'],
        })

        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss, self.actor_optimizer,
               self.critic_optimizer]
        actor_grads, actor_loss, critic_grads, critic_loss, _, _ = self.sess.run(ops, feed_dict={
            self.vf_model.obs_ph: batch['observations'],
            self.policy_model.obs_ph: batch['observations'],
            self.target_vf_model.obs_ph: batch['observations'],
            self.target_policy_model.obs_ph: batch['observations'],
            self.target_qf_ph: target_qf,
        })

        self.sess.run(self.target_soft_updates)

        return critic_loss, actor_loss


