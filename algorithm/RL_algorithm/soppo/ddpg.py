import os
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from algorithm.RL_algorithm.soppo.memory import Memory
from algorithm.RL_algorithm.soppo.actors import build_actor
from algorithm.RL_algorithm.soppo.critics import build_value, build_qvalue
from algorithm.RL_algorithm.soppo.utils import get_target_updates, constfn, get_session


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


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
    def __init__(self, *, network, obs_space, ac_space, vf_model=None, target_vf_model=None, memory=None, sess=None,
                 noise_type=None, lr=3e-5, gamma=0.99, lam=0.95, tau=0.01, pg_coef=1.0, ref_coef=0.1, qf_coef=0.25,
                 actor_reg_coef=0.1, critic_reg_coef=0.1, max_grad_norm=0.5, **network_kwargs):
        # logger.info('Using agent with the following configuration:')
        # logger.info(str(self.__dict__.items()))
        # Model Parameters.
        self.observation_space = obs_space
        self.action_space = ac_space
        self.observation_shape = obs_space.shape
        self.action_shape = ac_space.shape
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.pg_coef = pg_coef
        self.qf_coef = qf_coef
        self.ref_coef = ref_coef
        self.actor_reg_coef = actor_reg_coef
        self.critic_reg_coef = critic_reg_coef
        self.max_grad_norm = max_grad_norm

        if isinstance(lr, float):
            self.lr = constfn(lr)
        else:
            assert callable(lr)
        self.initial_state = None  # recurrent architectures not supported yet
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/ddpg_tmp/'

        if sess is None:
            self.sess = get_session()
        else:
            self.sess = sess

        # set action_noise and param_noise
        self.action_noise, self.param_noise = self._init_noise(noise_type)

        if memory is None:
            self.memory = Memory(limit=int(1e6), action_shape=ac_space.shape, observation_shape=obs_space.shape)
        else:
            self.memory = memory

        if vf_model is None:
            self.has_shared_vf_model = False
            with tf.variable_scope('ddpg_model', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('current'):
                    with tf.variable_scope('vf_model'):
                        # create vf model, default: mlp value network
                        value_network = build_value(obs_space, network, **network_kwargs)
                        self.vf_model = value_network(sess=self.sess)
                with tf.variable_scope('target'):
                    with tf.variable_scope('vf_model'):
                        # create vf model, default: mlp value network
                        value_network = build_value(obs_space, network, **network_kwargs)
                        self.target_vf_model = value_network(sess=self.sess)
        else:
            self.has_shared_vf_model = True
            self.vf_model = vf_model
            self.target_vf_model = target_vf_model

        with tf.variable_scope('ddpg_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('policy_model'):
                    # create policy model, default: mlp policy network
                    policy_network = build_actor(obs_space, ac_space, network, is_determined=True, **network_kwargs)
                    self.policy_model = policy_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('policy_model'):
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

        if self.has_shared_vf_model:
            self.vf_train_variables = tf.trainable_variables('shared_model/current/vf_model')
            self.target_vf_train_variables = tf.trainable_variables('shared_model/target/vf_model')
        else:
            self.vf_train_variables = tf.trainable_variables('ddpg_model/current/vf_model')
            self.target_vf_train_variables = tf.trainable_variables('ddpg_model/target/vf_model')
        self.policy_train_variables = tf.trainable_variables('ddpg_model/current/policy_model')
        self.qf_train_variables = tf.trainable_variables('ddpg_model/current/qf_model')
        self.target_policy_train_variables = tf.trainable_variables('ddpg_model/target/policy_model')
        self.target_qf_train_variables = tf.trainable_variables('ddpg_model/target/qf_model')
        self.ddpg_train_variables = self.policy_train_variables + self.target_policy_train_variables + \
                                    self.qf_train_variables + self.target_qf_train_variables
        if not self.has_shared_vf_model:
            self.ddpg_train_variables += self.vf_train_variables + self.target_vf_train_variables

        # CREATE THE PLACEHOLDERS
        self.adv_ph = tf.placeholder(tf.float32, shape=[None], name='advantages')
        self.target_qvalue_ph = tf.placeholder(tf.float32, shape=[None], name='target_qvalue')
        self.ref_ph = self.policy_model.pdtype.param_placeholder([None])
        self.lr_ph = tf.placeholder(tf.float32, [])

        self.step = self.policy_model.step
        self.value = self.vf_model.step

        # Set up parts
        self.stats_names = []
        self.stats_list = []
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
        logger.info('setting up ddpg actor optimizer')
        # self.actor_loss = -tf.reduce_mean(self.target_qf_model.qf)
        pg_loss = -tf.reduce_mean(self.target_qf_model.qf)
        ref_loss = tf.reduce_mean(tf.square(self.policy_model.pd_param - self.ref_ph))
        self.actor_loss = self.pg_coef * pg_loss + self.ref_coef * ref_loss
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

        self.actor_trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        grads_and_var = self.actor_trainer.compute_gradients(self.actor_loss, self.actor_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.actor_grads = grads
        self.stats_list.extend([pg_loss, ref_loss, self.actor_loss])  # , self.actor_grads
        self.stats_names.extend(['pg_loss', 'ref_loss', 'actor_loss'])  # , 'actor_grads'
        grads_and_var = list(zip(grads, var))
        self.actor_optimizer = self.actor_trainer.apply_gradients(grads_and_var)

    def _setup_critic_optimizer(self):
        logger.info('setting up ddpg critic optimizer')
        self.critic_loss = self.qf_coef * tf.reduce_mean(tf.square(self.qf_model.qf - self.target_qvalue_ph))
        self.critic_train_variables = self.qf_train_variables
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

        self.critic_trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        grads_and_var = self.critic_trainer.compute_gradients(self.critic_loss, self.critic_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.critic_grads = grads
        self.stats_list.extend([self.critic_loss])  # , self.critic_grads
        self.stats_names.extend(['critic_loss'])  # , 'critic_grads'
        grads_and_var = list(zip(grads, var))
        self.critic_optimizer = self.critic_trainer.apply_gradients(grads_and_var)

    def _setup_target_network_updates(self):
        logger.info('setting up ddpg target updates ...')
        logger.info('  {} <- {}'.format('ddpg_target_policy', 'ddpg_current_policy'))
        policy_init_updates, policy_soft_updates = get_target_updates(self.policy_train_variables,
                                                                      self.target_policy_train_variables,
                                                                      self.tau)
        if self.has_shared_vf_model:
            logger.info('  {} <- {}'.format('shared_target_vf', 'shared_current_vf'))
        else:
            logger.info('  {} <- {}'.format('ddpg_target_vf', 'ddpg_current_vf'))
        vf_init_updates, vf_soft_updates = get_target_updates(self.vf_train_variables,
                                                              self.target_vf_train_variables,
                                                              self.tau)
        logger.info('  {} <- {}'.format('ddpg_target_qf', 'ddpg_current_qf'))
        qf_init_updates, qf_soft_updates = get_target_updates(self.qf_train_variables,
                                                              self.target_qf_train_variables,
                                                              self.tau)
        self.target_init_updates = [policy_init_updates, vf_init_updates, qf_init_updates]
        self.target_soft_updates = [policy_soft_updates, vf_soft_updates, qf_soft_updates]

    # def store_transition(self, obs0, action, reward, obs1, terminal1):
    #     B = obs0.shape[0]
    #     for b in range(B):
    #         self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])

    def train(self, progress_fraction, observations, ref_pdparams, advantages, output_stats=True):
        target_qvalue = advantages + self.target_vf_model.step(observations)
        cur_lr = self.lr(progress_fraction)

        train_ops = [self.actor_optimizer, self.critic_optimizer]
        td_map = {
            self.policy_model.obs_ph: observations,
            self.target_policy_model.obs_ph: observations,
            self.vf_model.obs_ph: observations,
            self.target_vf_model.obs_ph: observations,
            self.ref_ph: ref_pdparams,
            self.target_qvalue_ph: target_qvalue,
            self.lr_ph: cur_lr
        }
        self.sess.run(train_ops, feed_dict=td_map)
        self.sess.run(self.target_soft_updates)

        if output_stats:
            return self.sess.run(self.stats_list, td_map)
