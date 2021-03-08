import os
from functools import reduce
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger

from algorithm.RL_algorithm.soppo.actors import build_actor
from algorithm.RL_algorithm.soppo.critics import build_value
from algorithm.RL_algorithm.soppo.utils import get_target_updates, constfn, get_session


class Model(object):
    def __init__(self, *, network, obs_space, ac_space, vf_model=None, target_vf_model=None, sess=None, cliprange=0.2,
                 lr=3e-4, gamma=0.99, lam=0.95, tau=0.01, pg_coef=1.0, vf_coef=0.25, ref_coef=0.1, kl_coef=0.5, ent_coef=0.0,
                 actor_reg_coef=0.1, critic_reg_coef=0.1, max_grad_norm=0.5, **network_kwargs):
        """
        Parameters:
        ----------

        network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies.py

        env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training

        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)


        nminibatches: int                 number of training minibatches per update. For recurrent policies.py,
                                          should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        ent_coef: float                   policy entropy coefficient in the optimization objective

        vf_coef: float                    value function loss coefficient in the optimization objective

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py.py/build_policy and arguments to a particular type of network
                                          For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

        """
        self.initial_state = None  # recurrent architectures not supported yet

        # Model Parameters.
        self.observation_space = obs_space
        self.action_space = ac_space
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.pg_coef = pg_coef
        self.vf_coef = vf_coef
        self.ref_coef = ref_coef
        self.kl_coef = kl_coef
        self.ent_coef = ent_coef
        self.actor_reg_coef = actor_reg_coef
        self.critic_reg_coef = critic_reg_coef
        self.max_grad_norm = max_grad_norm

        if isinstance(lr, float):
            self.lr = constfn(lr)
        else:
            assert callable(lr)

        if isinstance(cliprange, float):
            self.cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        if sess is None:
            self.sess = get_session()
        else:
            self.sess = sess

        if vf_model is None:
            self.has_shared_vf_model = False
            with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
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

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('policy_model'):
                    # create policy model, default: mlp policy network
                    policy_network = build_actor(obs_space, ac_space, network, **network_kwargs)
                    self.policy_model = policy_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('policy_model'):
                    policy_network = build_actor(obs_space, ac_space, network, **network_kwargs)
                    self.target_policy_model = policy_network(sess=self.sess)

        if self.has_shared_vf_model:
            self.vf_train_variables = tf.trainable_variables('shared_model/current/vf_model')
            self.target_vf_train_variables = tf.trainable_variables('shared_model/target/vf_model')
        else:
            self.vf_train_variables = tf.trainable_variables('ppo2_model/current/vf_model')
            self.target_vf_train_variables = tf.trainable_variables('ppo2_model/target/vf_model')
        self.policy_train_variables = tf.trainable_variables('ppo2_model/current/policy_model')
        self.target_policy_train_variables = tf.trainable_variables('ppo2_model/target/policy_model')
        self.ppo_train_variables = self.policy_train_variables + self.target_policy_train_variables
        if not self.has_shared_vf_model:
            self.ppo_train_variables += self.vf_train_variables + self.target_vf_train_variables

        # CREATE THE PLACEHOLDERS
        self.action_ph = self.policy_model.pdtype.sample_placeholder([None])
        self.adv_ph = tf.placeholder(tf.float32, [None])
        self.return_ph = tf.placeholder(tf.float32, [None])
        self.ref_ph = self.policy_model.pdtype.param_placeholder([None])
        self.lr_ph = tf.placeholder(tf.float32, [])
        self.cliprange_ph = tf.placeholder(tf.float32, [])

        self.step = self.policy_model.step
        self.value = self.target_vf_model.step
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/ppo_tmp/'

        # Set up parts
        self.stats_names = []
        self.stats_list = []
        self._setup_actor_optimizer()
        self._setup_critic_optimizer()
        self._setup_referer_optimizer()
        self._setup_target_network_updates()
        self._initialize()

    def _setup_actor_optimizer(self):
        logger.info('setting up ppo actor optimizer')
        self.actor_train_variables = self.policy_train_variables
        actor_shapes = [var.get_shape().as_list() for var in self.actor_train_variables]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))

        # Calculate ratio (pi current policy / pi old policy)
        neglogpac = self.policy_model.cal_action_neglogp(self.action_ph)
        old_neglogpac = self.target_policy_model.cal_action_neglogp(self.action_ph)  # TODO: check no gradient
        ratio = tf.exp(neglogpac - old_neglogpac)
        # Defining Loss = - J is equivalent to max J
        pg_loss1 = self.adv_ph * ratio
        pg_loss2 = self.adv_ph * tf.clip_by_value(ratio, 1.0 - self.cliprange_ph, 1.0 + self.cliprange_ph)
        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

        # Calculate kl diff between the new policy and the old
        kl_loss = tf.reduce_mean(tf.square(neglogpac - old_neglogpac))

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(self.policy_model.cal_entropy())

        # clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.cliprange_ph)))  # 裁剪数据占总数据比例

        self.actor_loss = self.pg_coef * pg_loss + self.kl_coef * kl_loss - self.ent_coef * entropy

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

        self.actor_trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        grads_and_var = self.actor_trainer.compute_gradients(self.actor_loss, self.actor_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.actor_grads = grads
        self.stats_list.extend([pg_loss, kl_loss, entropy, self.actor_loss])  # , self.actor_grads
        self.stats_names.extend(['pg_loss', 'kl_loss', 'entropy', 'actor_loss'])  # , 'actor_grads'
        grads_and_var = list(zip(grads, var))
        self.actor_optimizer = self.actor_trainer.apply_gradients(grads_and_var)

    def _setup_critic_optimizer(self):
        logger.info('setting up ppo critic optimizer')
        self.critic_train_variables = self.vf_train_variables
        critic_shapes = [var.get_shape().as_list() for var in self.critic_train_variables]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = self.vf_model.vf
        old_vpred = self.target_vf_model.vf
        vpredclipped = old_vpred + tf.clip_by_value(vpred - old_vpred, - self.cliprange_ph, self.cliprange_ph)
        # Unclipped value
        vf_loss1 = tf.square(vpred - self.return_ph)
        # Clipped value
        vf_loss2 = tf.square(vpredclipped - self.return_ph)
        vf_loss = tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
        self.critic_loss = self.vf_coef * vf_loss

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

    def _setup_referer_optimizer(self):
        logger.info('setting up ppo refer optimizer')
        self.ref_loss = self.ref_coef * tf.reduce_mean(tf.square(self.policy_model.pd_param - self.ref_ph))
        self.referer_train_variables = self.policy_train_variables

        self.referer_trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        grads_and_var = self.referer_trainer.compute_gradients(self.ref_loss, self.referer_train_variables)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.actor_grads = grads
        self.stats_list.extend([self.ref_loss])  # , self.actor_grads
        self.stats_names.extend(['ref_loss'])  # , 'actor_grads'
        grads_and_var = list(zip(grads, var))
        self.referer_optimizer = self.referer_trainer.apply_gradients(grads_and_var)

    def _setup_target_network_updates(self):
        logger.info('setting up ppo target updates ...')
        logger.info('  {} <- {}'.format('ppo_target_policy', 'ppo_current_policy'))
        policy_init_updates, policy_soft_updates = get_target_updates(self.policy_train_variables,
                                                                      self.target_policy_train_variables,
                                                                      self.tau)
        if self.has_shared_vf_model:
            logger.info('  {} <- {}'.format('shared_target_vf', 'shared_current_vf'))
        else:
            logger.info('  {} <- {}'.format('ppo_target_vf', 'ppo_current_vf'))
        vf_init_updates, vf_soft_updates = get_target_updates(self.vf_train_variables,
                                                              self.target_vf_train_variables,
                                                              self.tau)
        self.target_init_updates = [policy_init_updates, vf_init_updates]
        self.target_soft_updates = [policy_soft_updates, vf_soft_updates]

    def _initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    def train(self, progress_fraction, observations, actions, advantages, ref_pdparams, output_stats=True):
        returns = advantages + self.target_vf_model.step(observations)
        cur_lr = self.lr(progress_fraction)
        cur_cliprange = self.cliprange(progress_fraction)

        train_ops = [self.actor_optimizer, self.critic_optimizer]
        if ref_pdparams is not None:
            train_ops.append(self.referer_optimizer)
        td_map = {
            self.policy_model.obs_ph: observations,
            self.target_policy_model.obs_ph: observations,
            self.vf_model.obs_ph: observations,
            self.target_vf_model.obs_ph: observations,
            self.action_ph: actions,
            self.adv_ph: advantages,
            self.return_ph: returns,
            self.ref_ph: ref_pdparams,
            self.lr_ph: cur_lr,
            self.cliprange_ph: cur_cliprange,
        }
        self.sess.run(train_ops, feed_dict=td_map)
        self.sess.run(self.target_soft_updates)

        if output_stats:
            return self.sess.run(self.stats_list, td_map)
