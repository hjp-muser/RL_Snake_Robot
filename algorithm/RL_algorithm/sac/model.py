import os
import time
from collections import deque

import numpy as np
import tensorflow as tf

from baselines import logger

from algorithm.RL_algorithm.utils.math_utils import safe_mean
from algorithm.RL_algorithm.utils.scheduler import Scheduler
from algorithm.RL_algorithm.utils.tensorflow1.tb_utils import TensorboardWriter
from algorithm.RL_algorithm.sac.buffer import ReplayBuffer
from algorithm.RL_algorithm.sac.policy import MlpPolicy, LnMlpPolicy
from algorithm.RL_algorithm.utils.common_utils import set_global_seeds
from algorithm.RL_algorithm.utils.tensorflow1 import tf_utils
from algorithm.RL_algorithm.utils.tensorflow1 import tb_utils
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import save_variables, load_variables


def build_policy(network, sess=None, ob_space=None, ac_space=None):
    if network == 'mlp':
        return MlpPolicy(sess, ob_space, ac_space)
    elif network == 'lnmlp':
        return LnMlpPolicy(sess, ob_space, ac_space)


class Model(object):
    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model

        learn():
        - control the training and roll-out procedure
    """

    def __init__(self, network, env, *, seed=None, gamma=0.99, learning_rate=0, learning_rate_scheduler='constant',
                 buffer_size=50000, learning_start_threshold=200, train_freq=1, batch_size=128, tau=0.005,
                 ent_coef='auto', target_update_interval=10, gradient_steps=1, target_entropy=10.0,
                 action_noise=None, model_save_path=None, tensorboard_log_path=None):
        """
            Soft Actor-Critic (SAC)
            Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
            This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
            from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
            (https://github.com/rail-berkeley/softlearning/)
            Paper: https://arxiv.org/abs/1801.01290
            Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

            Parameters:
            -----------

            :param network: policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small,
                    conv_only - see policies.py.py for full list) specifying the standard network architecture, or a
                    function that takes tensorflow tensor as input and returns tuple (output_tensor, extra_feed)
                    where output tensor is the last network layer output

            :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)

            :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
                    If None (default), use random seed. Note that if you want completely deterministic
                    results, you must set `n_cpu_tf_sess` to 1.

            :param gamma: (float) the discount factor

            :param learning_rate: (float) learning rate for adam optimizer

            :param learning_rate_scheduler: (callable) schedule of learning rate. Can be 'linear', 'constant',
                    or a function [0..1] -> [0..1] that takes fraction of the training progress as input and returns
                    fraction of the learning rate (specified as lr) as output

            :param buffer_size: (int) size of the replay buffer

            :param batch_size: (int) Minibatch size for each gradient update

            :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)

            :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
                    inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
                    Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)

            :param train_freq: (int) Update the model every `train_freq` steps.

            :param learning_start_threshold: (int) how many steps of the model to collect transitions for before learning starts

            :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.

            :param gradient_steps: (int) How many gradient update after each step

            :param target_entropy: (float) target entropy when learning ent_coef (ent_coef = 'auto')

            :param action_noise: (ActionNoise) the action noise type (None by default), this can help
                    for hard exploration problem. Cf DDPG for the different action noise type.

            :param model_save_path: str, the location to save model parameters (if None, no saving)

            :param tensorboard_log_path: (str) the log location for tensorboard (if None, no logging)

        """

        self.network = network
        self.env = env
        self.n_envs = env.num_envs
        self.gamma = gamma
        # self.learning_rate = learning_rate
        # In the original paper, same learning rate is used for all networks
        self.policy_lr = learning_rate
        self.value_lr = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.buffer_size = buffer_size
        self.learning_start_threshold = learning_start_threshold
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.target_entropy = target_entropy
        self.action_noise = action_noise
        self.model_save_path = model_save_path
        self.tensorboard_log_path = tensorboard_log_path
        self.seed = seed
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/tmp/'

        self.episode_reward = np.zeros((self.n_envs,))
        self.ep_info_buf = deque(maxlen=100)
        self.sess = None
        self.graph = None
        self.replay_buffer = None
        self.value_fn = None
        self.policy = None
        self.target_policy = None
        self.params = None
        self.summary = None
        self.obs_target = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None
        self.deterministic_action = None
        self.step = None

        self.setup_model()

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            set_global_seeds(self.seed)
            self.sess = tf_utils.make_session(graph=self.graph)
            self.replay_buffer = ReplayBuffer(self.buffer_size)

            with tf.variable_scope("input", reuse=False):
                # Create policy and target TF objects
                self.policy = build_policy(self.network, self.sess, self.env.observation_space, self.env.action_space)
                self.target_policy = build_policy(self.network, self.sess, self.env.observation_space,
                                                  self.env.action_space)
                # self.policy = MlpPolicy(self.sess, self.env.observation_space, self.env.action_space)
                # self.target_policy = MlpPolicy(self.sess, self.env.observation_space, self.env.action_space)

                self.step = self.policy.step
                # Initialize Placeholders
                self.observations_ph = self.policy.obs_ph
                # Normalized observation for pixels
                self.processed_obs_ph = self.policy.processed_obs
                self.next_observations_ph = self.target_policy.obs_ph
                self.processed_next_obs_ph = self.target_policy.processed_obs

                # self.action_target = self.target_policy.action_ph
                self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.env.action_space.shape,
                                                 name='actions')
                # self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

            with tf.variable_scope("model", reuse=False):
                # Create the policy
                # first return value corresponds to deterministic actions
                # policy_out corresponds to stochastic actions, used for training
                # logp_pi is the log probability of actions taken by the policy
                self.deterministic_action, policy_out, logp_pi = self.policy.make_actor(self.processed_obs_ph)
                # Monitor the entropy of the policy, but this is not used for training
                self.entropy = tf.reduce_mean(self.policy.entropy)
                #  Use two Q-functions to improve performance by reducing overestimation bias.
                qf1, qf2, value_fn = self.policy.make_critics(self.processed_obs_ph, self.actions_ph,
                                                              create_qf=True, create_vf=True)
                qf1_pi, qf2_pi, _ = self.policy.make_critics(self.processed_obs_ph, policy_out,
                                                             create_qf=True, create_vf=False, reuse=True)

                # The entropy coefficient or entropy can be learned automatically
                # see Automating Entropy Adjustment for Maximum Entropy RL section
                # of https://arxiv.org/abs/1812.05905
                if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                    # Default initial value of ent_coef when learned
                    init_value = 1.0
                    if '_' in self.ent_coef:
                        init_value = float(self.ent_coef.split('_')[1])
                        assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                    self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                        initializer=np.log(init_value).astype(np.float32))
                    self.ent_coef = tf.exp(self.log_ent_coef)
                else:
                    # Force conversion to float
                    # this will throw an error if a malformed string (different from 'auto')
                    # is passed
                    self.ent_coef = float(self.ent_coef)

            with tf.variable_scope("target", reuse=False):
                # Create the value network
                _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                     create_qf=False, create_vf=True)
                self.value_target = value_target

            with tf.variable_scope("loss", reuse=False):
                # Take the min of the two Q-Values (Double-Q Learning)
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                # Target for Q value regression
                q_backup = tf.stop_gradient(
                    self.rewards_ph + (1 - self.terminals_ph) * self.gamma * self.value_target
                )

                # Compute Q-Function loss
                # TODO: test with huber loss (it would avoid too high values)
                qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                # Compute the entropy temperature loss
                # it is used when the entropy coefficient is learned
                ent_coef_loss, entropy_optimizer = None, None
                if not isinstance(self.ent_coef, float):
                    ent_coef_loss = -tf.reduce_mean(
                        self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                    entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)

                # Compute the policy loss
                # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                # NOTE: in the original implementation, they have an additional
                # regularization loss for the Gaussian parameters
                # this is not used for now
                # policy_loss = (policy_kl_loss + policy_regularization_loss)
                policy_loss = policy_kl_loss

                # Target for value fn regression
                # We update the vf towards the min of two Q-functions in order to
                # reduce overestimation bias from function approximation error.
                v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                values_losses = qf1_loss + qf2_loss + value_loss

                # range_center = (self.env.action_space.low + self.env.action_space.high) / 2
                # range_half_len = (self.env.action_space.high - self.env.action_space.low) / 2
                # range_loss = 0.001 * tf.pow((tf.abs(policy_out - range_center) / range_half_len), 5)

                # Policy train op
                # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
                policy_std_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
                policy_train_op = policy_optimizer.minimize(policy_loss,
                                                            var_list=tf_utils.get_trainable_vars('model/pi/mu'))
                policy_std_train_op = policy_std_optimizer.minimize(policy_loss,
                                                            var_list=tf_utils.get_trainable_vars('model/pi/std'))
                policy_grad_mu = tf.gradients(policy_loss, tf_utils.get_trainable_vars('model/pi/mu'))
                policy_grad_std = tf.gradients(policy_loss, tf_utils.get_trainable_vars('model/pi/std'))

                # Value train op
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.value_lr)
                values_params = tf_utils.get_trainable_vars('model/values_fn')
                source_params = tf_utils.get_trainable_vars("model/values_fn/vf")
                target_params = tf_utils.get_trainable_vars("target/values_fn/vf")

                # Polyak averaging for target variables
                self.target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]
                # Initializing target to match source variables
                target_init_op = [
                    tf.assign(target, source)
                    for target, source in zip(target_params, source_params)
                ]

                # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                # and we first need to compute the policy action before computing q values losses
                with tf.control_dependencies([policy_train_op, policy_std_train_op]):
                    train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                    self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                    # All ops to call during one training step
                    self.step_ops = [policy_loss, qf1_loss, qf2_loss, value_loss,
                                     qf1, qf2, value_fn, logp_pi, policy_grad_mu, policy_grad_std,
                                     self.entropy, policy_train_op, policy_std_train_op, train_values_op]

                    # Add entropy coefficient optimization operation if needed
                    if ent_coef_loss is not None:
                        with tf.control_dependencies([train_values_op]):
                            ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                            self.infos_names += ['ent_coef_loss', 'ent_coef']
                            self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                # Monitor losses and entropy in tensorboard
                tf.summary.scalar('policy_loss', policy_loss)
                tf.summary.scalar('qf1_loss', qf1_loss)
                tf.summary.scalar('qf2_loss', qf2_loss)
                tf.summary.scalar('value_loss', value_loss)
                tf.summary.scalar('entropy', self.entropy)
                if ent_coef_loss is not None:
                    tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                    tf.summary.scalar('ent_coef', self.ent_coef)

                # tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

            # Retrieve parameters that must be saved
            self.params = tf_utils.get_trainable_vars("model")
            self.target_params = tf_utils.get_trainable_vars("target/values_fn/vf")

            # Initialize Variables and target network
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(target_init_op)

            self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer):
        # Sample a batch from the replay buffer. The 'obs' and 'rewards' in batch are get normalized.
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]
        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        policy_grad_mu = values[4]
        policy_grad_std = values[5]
        entropy = values[6]
        # print("range_loss = ", range_loss)
        if step % 500 == 0:
            print("policy_grad_mu = ", policy_grad_mu[-1])
            print("policy_grad_std = ", policy_grad_std[-1])

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps=int(1e6), log_interval=4, reset_num_timesteps=True):
        pretrain_load_path = None
        # pretrain_load_path="/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubunt
        # pretrain_load_path="/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubunt
        # pretrain_load_path="/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/programming/RL_Snake_Robot/algorithm/RL_algorithm/sac/tmp/Y2020M07D16_h11m15s56"
        # pretrain_load_path="/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/programming/RL_Snake_Robot/algorithm/RL_algorithm/sac/tmp/Y2020M07D16_h18m36s16"
        if pretrain_load_path is not None:
            variables = self.params + self.target_params
            load_variables(load_path=pretrain_load_path, variables=variables, sess=self.sess)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with TensorboardWriter(self.graph, self.tensorboard_log_path, tb_log_name="SAC",
                               new_tb_log=new_tb_log) as writer:

            # Transform to callable if needed
            # self.learning_rate_scheduler = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
            #                                          schedule=self.learning_rate_scheduler)
            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            # Retrieve unnormalized observation for saving into the buffer
            obs = self.env.reset()

            n_updates = 0
            infos_values = []

            for update in range(total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_start_threshold:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used

                    # action = np.array([self.env.action_space.sample() for _ in range(self.n_envs)])
                    action = np.array([np.random.random(self.env.action_space.shape) for _ in range(self.n_envs)])
                    # action = scale_action(self.env.action_space, unscaled_action)
                else:
                    action, _, _, _ = self.policy.step(obs, deterministic=False)
                    mu, std = self.policy.proba_step(obs)
                    if update % 500 == 0:
                        print("mu = ", mu, " , std = ", std)
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = action + self.action_noise()
                    # inferred actions need to be transformed to environment action_space before stepping
                    # unscaled_action = unscale_action(self.env.action_space, action)
                # print("action = ", action)
                assert action[0].shape == self.env.action_space.shape
                new_obs, reward, done, infos = self.env.step(action)
                self.num_timesteps += 1

                # Store only the unnormalized version
                obs_, new_obs_, reward_ = obs, new_obs, reward

                # Store transition in the replay buffer.
                self.replay_buffer.extend(obs_, action, reward_, new_obs_, done)
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                for info in infos:
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info:
                        self.ep_info_buf.append(maybe_ep_info)

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tb_utils.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                         ep_done, writer, self.num_timesteps)

                if self.num_timesteps % self.train_freq == 0:

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                                or self.num_timesteps < self.learning_start_threshold:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        # current_lr = self.learning_rate_scheduler.value()
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(update, writer))
                        # Update target network
                        if (update + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                # only record the return reward of the first environment
                episode_rewards[-1] += reward_[0]

                if done[0]:
                    #     if self.action_noise is not None:
                    #         self.action_noise.reset()
                    #         obs = self.env.reset()
                    episode_rewards.append(0.0)
                    print("The first env's reward: ", episode_rewards[-2])

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                # Display training infos
                if done[0] and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(update / (time.time() - start_time))
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.record_tabular('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.record_tabular('ep_lenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.record_tabular("n_updates", n_updates)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular('time_elapsed', int(time.time() - start_time))
                    # if len(episode_successes) > 0:
                    #     logger.record_tabular("success rate", episode_successes / num_episodes)
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.record_tabular(name, val)
                    logger.record_tabular("total timesteps", self.num_timesteps)
                    logger.dump_tabular()
                    # Reset infos:
                    infos_values = []

                if update % 1000 == 0 or update == total_timesteps - 1:
                    if self.model_save_path is None:
                        file_name = time.strftime('Y%YM%mD%d_h%Hm%Ms%S', time.localtime(time.time()))
                        model_save_path = self.def_path_pre + file_name
                        self.save(model_save_path)
                    else:
                        self.save(self.model_save_path)

            return self

    def save(self, save_path):
        variables = self.params + self.target_params
        save_variables(save_path=save_path, variables=variables, sess=self.sess)
        print('save model variables to', save_path)

    def load_newest(self, load_path=None):
        file_list = os.listdir(self.def_path_pre)
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.def_path_pre, x)))
        if load_path is None:
            load_path = os.path.join(self.def_path_pre, file_list[-1])
        variables = self.params + self.target_params
        load_variables(load_path=load_path, variables=variables, sess=self.sess)
        print('load_path: ', load_path)

    def load_index(self, index, load_path=None):
        file_list = os.listdir(self.def_path_pre)
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.def_path_pre, x)), reverse=True)
        if load_path is None:
            load_path = os.path.join(self.def_path_pre, file_list[index])
        variables = self.params + self.target_params
        load_variables(load_path=load_path, variables=variables, sess=self.sess)
        print('load_path: ', load_path)

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log
