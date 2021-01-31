import os
import time
from collections import deque
import pickle
from copy import copy
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds

from algorithm.RL_algorithm.soppo.values import build_value
from algorithm.RL_algorithm.soppo.policies import build_policy
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import save_variables, load_variables, get_session

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# def normalize(x, stats):
#     if stats is None:
#         return x
#     return (x - stats.mean) / (stats.std + 1e-8)


# def denormalize(x, stats):
#     if stats is None:
#         return x
#     return x * stats.std + stats.mean


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
    def __init__(self, network, env, obs_ph=None, gamma=1, tau=0.01, total_timesteps=1e6,
                 enable_popart=False,
                 noise_type='adaptive-param_0.2', clip_norm=None, reward_scale=1.,
                 batch_size=128, l2_reg_coef=0.2, actor_lr=1e-4, critic_lr=1e-3,
                 observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 **network_kwargs):
        # logger.info('Using agent with the following configuration:')
        # logger.info(str(self.__dict__.items()))
        self.observation_shape = observation_shape = env.observation_space.shape
        self.action_shape = action_shape = env.action_space.shape
        self.sess = sess = get_session()
        assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

        # Inputs.
        self.obs_ph = obs_ph
        self.action_ph = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.total_timesteps = total_timesteps
        self.enable_popart = enable_popart
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.l2_reg_coef = l2_reg_coef
        self.stats_sample = None
        self.initial_state = None  # recurrent architectures not supported yet
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/tmp/'

        self.memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                             observation_shape=env.observation_space.shape)

        # set action_noise and param_noise
        self.action_noise, self.param_noise = self.init_noise(noise_type)

        with tf.variable_scope('ddpg_model', reuse=tf.AUTO_REUSE):
            # create vf model
            # default: mlp value network
            value_network = build_value(env, network, **network_kwargs)
            vf_model = value_network(obs_ph=self.obs_ph, sess=self.sess)
            # create policy model
            # default: mlp policy network
            policy_network = build_policy(env, network, estimate_q=True, **network_kwargs)
            policy_model = policy_network(obs_ph=self.obs_ph, sess=self.sess, vf_latent=vf_model.vf_latent)

        # Create target networks
        with tf.variable_scope('ddpg_target_model', reuse=tf.AUTO_REUSE):
            # create vf model
            # default: mlp value network
            target_value_network = build_value(env, network, **network_kwargs)
            target_vf_model = target_value_network(obs_ph=self.obs_ph, sess=self.sess)
            # create policy model
            # default: mlp policy network
            target_policy_network = build_policy(env, network, estimate_q=True, **network_kwargs)
            target_policy_model = target_policy_network(obs_ph=self.obs_ph, sess=self.sess, vf_latent=target_vf_model.vf_latent)

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = policy_model.action
        self.critic_tf = self.critic(normalized_obs0, self.action_ph)
        self.critic_with_actor_tf = policy_model.qf
        Q_obs1 = target_policy_model.qf
        self.target_Q = critic_target = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts
        if self.param_noise is not None:
            self.setup_param_noise(self.obs_ph)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        self.setup_stats()
        self.setup_target_network_updates()

    def init_noise(self, noise_type):
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
                    action_noise = NormalActionNoise(mu=np.zeros(self.action_shape[-1]), sigma=float(stddev) * np.ones(self.action_shape[-1]))
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_shape[-1]),
                                                                sigma=float(stddev) * np.ones(self.action_shape[-1]))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
        return action_noise, param_noise

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor,
                                                                       self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
                                       beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - self.critic_target))
        if self.l2_reg_coef > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                               var.name.endswith('/w:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.l2_reg_coef))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.l2_reg_coef),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
                                        beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def train_step(self, obs, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])

        return action, q, None, None

    def step(self, obs, compute_Q=True):
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        if compute_Q:
            action, q = self.sess.run([self.actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(self.actor_tf, feed_dict=feed_dict)
            q = None

        action = np.clip(action, self.action_range[0], self.action_range[1])

        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q],
                                                        feed_dict={
                                                            self.obs1: batch['obs1'],
                                                            self.rewards: batch['rewards'],
                                                            self.terminals1: batch['terminals1'].astype('float32'),
                                                        })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.action_ph: batch['actions'],
            self.critic_target: target_Q,
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.action_ph: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def learn(self, total_timesteps=None,
              seed=None,
              nb_epochs=None,  # with default settings, perform 1M steps total
              nb_epoch_cycles=20,
              nb_rollout_steps=100,
              nb_train_steps=50,  # per epoch cycle and MPI worker,
              batch_size=64,  # per MPI worker
              param_noise_adaption_interval=50,):

        set_global_seeds(seed)

        if total_timesteps is not None:
            assert nb_epochs is None
            nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
        else:
            nb_epochs = 500

        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        sess = U.get_session()
        # Prepare everything.
        self.initialize(sess)
        sess.graph.finalize()
        self.reset()

        obs = self.env.reset()
        # if eval_env is not None:
        #     eval_obs = eval_env.reset()
        nenvs = obs.shape[0]

        episode_reward = np.zeros(nenvs, dtype=np.float32)  # vector
        episode_step = np.zeros(nenvs, dtype=int)  # vector
        episodes = 0  # scalar
        t = 0  # scalar

        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                if nenvs > 1:
                    # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                    # of the environments, so resetting here instead
                    self.reset()
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q, _, _ = self.train_step(obs, apply_noise=True, compute_Q=True)

                    # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                    # new_obs, r, done, info = self.env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, info = self.env.step(action)
                    # note these outputs are batched from vecenv

                    t += 1
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    # the batched data will be unrolled in memory.py's append.
                    self.store_transition(obs, action, r, new_obs, done)

                    obs = new_obs

                    for d in range(len(done)):
                        if done[d]:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward[d])
                            episode_rewards_history.append(episode_reward[d])
                            epoch_episode_steps.append(episode_step[d])
                            episode_reward[d] = 0.
                            episode_step[d] = 0
                            epoch_episodes += 1
                            episodes += 1
                            if nenvs == 1:
                                self.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if self.memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = self.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = self.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    self.update_target_net()

            if MPI is not None:
                mpi_size = MPI.COMM_WORLD.Get_size()
            else:
                mpi_size = 1

            # save trainable variables
            file_name = time.strftime('Y%YM%mD%d_h%Hm%Ms%S', time.localtime(time.time()))
            model_save_path = self.def_path_pre + file_name
            self.save(model_save_path)

            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = self.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            # combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            # combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            # combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            # if eval_env is not None:
            #     combined_stats['eval/return'] = eval_episode_rewards
            #     combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            #     combined_stats['eval/Q'] = eval_qs
            #     combined_stats['eval/episodes'] = len(eval_episode_rewards)

            combined_stats_sums = np.array([np.array(x).flatten()[0] for x in combined_stats.values()])
            if MPI is not None:
                combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

            combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])

            if rank == 0:
                logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(self.env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(self.env.get_state(), f)
                # if eval_env and hasattr(eval_env, 'get_state'):
                #     with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                #         pickle.dump(eval_env.get_state(), f)
        self.sess.graph._unsafe_unfinalize()
        return self

    def save(self, save_path=None):
        save_variables(save_path=save_path, sess=self.sess)
        print('save model variables to', save_path)

    def load_newest(self, load_path=None):
        file_list = os.listdir(self.def_path_pre)
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.def_path_pre, x)))
        if load_path is None:
            load_path = os.path.join(self.def_path_pre, file_list[-1])
        load_variables(load_path=load_path, sess=self.sess)
        print('load_path: ', load_path)

    def load_index(self, index, load_path=None):
        file_list = os.listdir(self.def_path_pre)
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.def_path_pre, x)), reverse=True)
        if load_path is None:
            load_path = os.path.join(self.def_path_pre, file_list[index])
        load_variables(load_path=load_path, sess=self.sess)
        print('load_path: ', load_path)