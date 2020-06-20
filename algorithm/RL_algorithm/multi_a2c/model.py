import time
import tensorflow as tf
import os
from tensorflow import losses
from collections import deque
import numpy as np

from baselines import logger

from algorithm.RL_algorithm.multi_a2c.policy import MlpPolicy, LnMlpPolicy
from algorithm.RL_algorithm.multi_a2c.runner import Runner
from algorithm.RL_algorithm.utils.math_utils import safe_mean, explained_variance
from algorithm.RL_algorithm.utils.common_utils import set_global_seeds
from algorithm.RL_algorithm.utils.scheduler import Scheduler
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import save_variables, load_variables, get_session
from algorithm.RL_algorithm.utils.tensorflow1.tb_utils import TensorboardWriter, total_episode_reward_logger


def build_policy(network, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False):
    if network == 'mlp':
        return MlpPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse)
    elif network == 'lnmlp':
        return LnMlpPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse)


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

    def __init__(self, network, env, *, seed=None, nsteps=5, total_timesteps=int(80e6),
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=5e-4, lrschedule='constant',
                 gamma=0.99, alpha=0.99, epsilon=1e-5, model_save_path=None, tb_log_path=None):

        """
        Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

        Parameters:
        -----------

        :param network: policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see policies.py for full list)
                specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                See policies.py for more details on using recurrent nets in policies

        :param env: RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)

        :param seed: seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

        :param nsteps: int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                nenv is number of environment copies simulated in parallel)

        :param total_timesteps: int, total number of timesteps to train on (default: 80M)

        :param vf_coef: float, coefficient in front of value function loss in the total loss function (default: 0.5)

        :param ent_coef: float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

        :param max_grad_norm: float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

        :param lr: float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

        :param lrschedule: schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of
                the training progress as input and returns fraction of the learning rate (specified as lr) as output

        :param epsilon: float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

        :param alpha: float, RMSProp decay parameter (default: 0.99)

        :param gamma: float, reward discounting parameter (default: 0.99)

        :param model_save_path: str, the location to save model parameters (if None, auto saving)

        :param tb_log_path: str, the log location for tensorboard (if None, no logging)

        """

        self.network = network
        self.env = env
        self.obs_space = env.observation_space
        self.ac_space = env.action_space
        self.nenvs = env.num_envs
        self.nsteps = nsteps
        self.seed = seed
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.total_timesteps = total_timesteps
        self.lrschedule = lrschedule
        self.model_save_path = model_save_path
        self.tb_log_path = tb_log_path
        self.sess = get_session()
        self.graph = self.sess.graph
        self.episode_reward = np.zeros((self.nenvs,))
        self.ac_space_len = []
        for sub_ac_space in self.ac_space:
            self.ac_space_len.append(sub_ac_space.sample().shape[0])

        self.step_model = None
        self.train_model = None
        self.obs_ph = None
        self.actions_ph = None
        self.advances_ph = None
        self.rewards_ph = None
        self.learning_rate_ph = None
        self.policy_loss = None
        self.value_loss = None
        self.entropy = None
        self.total_loss = None
        self.apply_backprop = None
        self.lr_schedule = None
        self.step = None
        self.value = None
        self.initial_state = None
        self.def_path_pre = None
        self.summary = None

        self.setup_model()

    def setup_model(self):
        nbatch = self.nenvs * self.nsteps
        with tf.variable_scope('multi_a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            self.step_model = build_policy(self.network, self.sess, self.obs_space, self.ac_space,
                                           self.nenvs, 1, self.nenvs, reuse=False)

            # train_model is used to train our network
            self.train_model = build_policy(self.network, self.sess, self.obs_space, self.ac_space,
                                            self.nenvs, self.nsteps, nbatch, reuse=True)

        with tf.variable_scope('loss', reuse=False):
            self.obs_ph = self.train_model.obs_ph
            self.actions_ph = self.train_model.action_ph
            self.advances_ph = tf.placeholder(tf.float32, [nbatch])
            self.rewards_ph = tf.placeholder(tf.float32, [nbatch])
            self.learning_rate_ph = tf.placeholder(tf.float32, [])

            # Calculate the loss
            # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

            # Policy loss
            for sub_id in range(len(self.ac_space)):
                if sub_id == 0:
                    neglogpac = self.train_model.pd_list[sub_id].neglogp(self.actions_ph[sub_id])
                else:
                    neglogpac += self.train_model.pd_list[sub_id].neglogp(self.actions_ph[sub_id])
            # L = A(s,a) * -logpi(a|s)
            self.policy_loss = tf.reduce_mean(self.advances_ph * neglogpac)

            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            for sub_id in range(len(self.ac_space)):
                if self.entropy is None:
                    self.entropy = tf.reduce_mean(self.train_model.pd_list[sub_id].entropy())
                else:
                    self.entropy += tf.reduce_mean(self.train_model.pd_list[sub_id].entropy())

            # Value loss
            self.value_loss = losses.mean_squared_error(tf.squeeze(self.train_model.value_fn), self.rewards_ph)

            self.total_loss = self.policy_loss - self.entropy * self.ent_coef + self.value_loss * self.vf_coef

            tf.summary.scalar('lr', self.learning_rate_ph)
            tf.summary.scalar('policy_loss', self.policy_loss)
            tf.summary.scalar('entropy', self.entropy)
            tf.summary.scalar('value_loss', self.value_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.histogram('obs', self.train_model.obs_ph)

            # Update parameters using loss
            # 1. Get the model parameters
            params = tf.trainable_variables("multi_a2c_model")

            # 2. Calculate the gradients
            grads = tf.gradients(self.total_loss, params)
            if self.max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads = list(zip(grads, params))

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha, epsilon=self.epsilon)

        self.apply_backprop = trainer.apply_gradients(grads)

        self.lr_schedule = Scheduler(initial_value=self.lr, n_values=self.total_timesteps, schedule=self.lrschedule)
        self.step = self.step_model.step
        self.value = self.step_model.value
        self.initial_state = self.step_model.initial_state
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/tmp/'  # default path prefix
        self.summary = tf.summary.merge_all()
        tf.global_variables_initializer().run(session=self.sess)

    def train(self, obs, states, rewards, masks, actions, values, update, writer=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # rewards = R + yV(s')
        cur_lr = None
        advs = rewards - values
        for step in range(len(obs)):
            cur_lr = self.lr_schedule.value()

        td_map = {self.obs_ph: obs, self.advances_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        for sub_id in range(len(self.ac_space)):
            td_map[self.actions_ph[sub_id]] = actions[sub_id]

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if writer is not None:
            summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.summary, self.policy_loss, self.value_loss, self.entropy, self.apply_backprop], td_map
            )
            writer.add_summary(summary, update)
        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [self.policy_loss, self.value_loss, self.entropy, self.apply_backprop], td_map
            )
        return policy_loss, value_loss, policy_entropy

    def learn(self, total_timesteps=int(1e6), log_interval=1000, pretrain_load_path=None, learn_frequency=10):

        """
        Parameters:
        -----------

        log_interval: int, specifies how frequently the logs are printed out (default: 100)

        pretrain_load_path: pre-train model load path

        """

        set_global_seeds(self.seed)

        if pretrain_load_path is not None:
            load_variables(load_path=pretrain_load_path, sess=self.sess)

        # Instantiate the runner object
        runner = Runner(self.env, self, nsteps=self.nsteps, gamma=self.gamma)
        epinfobuf = deque(maxlen=100)
        # Calculate the batch_size
        nbatch = self.nenvs * self.nsteps

        # Start total timer
        tstart = time.time()

        with TensorboardWriter(self.graph, self.tb_log_path, 'MULTI_A2C') as writer:
            for update in range(1, total_timesteps):
                if update % learn_frequency != 0:
                    continue
                # Get mini batch of experiences
                obs, states, rewards, masks, actions, values, epinfos = runner.run()
                policy_loss, value_loss, policy_entropy = self.train(obs, states, rewards, masks, actions, values,
                                                                     update, writer)
                epinfobuf.extend(epinfos)
                nseconds = time.time() - tstart
                # Calculate the fps (frame per second)
                fps = int((update * nbatch) / nseconds)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward, rewards.reshape((self.nenvs, self.nsteps)),
                                                masks.reshape((self.nenvs, self.nsteps)), writer, update)

                if update % log_interval == 0 or update == 1:
                    # Calculates if value function is a good predicator of the returns (ev > 1)
                    # or if it's just worse than predicting nothing (ev =< 0)
                    ev = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * nbatch)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("explained_variance", float(ev))
                    logger.record_tabular("eprewmean", safe_mean([epinfo['r'] for epinfo in epinfobuf]))
                    logger.record_tabular("eplenmean", safe_mean([epinfo['l'] for epinfo in epinfobuf]))
                    logger.dump_tabular()

                if update % 2000 == 0 or update == total_timesteps // nbatch:
                    if self.model_save_path is None:
                        file_name = time.strftime('Y%YM%mD%d_h%Hm%Ms%S', time.localtime(time.time()))
                        model_save_path = self.def_path_pre + file_name
                        self.save(model_save_path)
                    else:
                        self.save(self.model_save_path)

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
