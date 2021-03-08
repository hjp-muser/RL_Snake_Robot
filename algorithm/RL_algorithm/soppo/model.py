import os
import time
import numpy as np
from collections import deque

from baselines import logger
from baselines.common import set_global_seeds
import tensorflow as tf

from algorithm.RL_algorithm.soppo.critics import build_value
from algorithm.RL_algorithm.soppo.utils import get_session, GAE
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import save_variables, load_variables

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from algorithm.RL_algorithm.utils.common_utils import set_global_seeds
from algorithm.RL_algorithm.utils.math_utils import safe_mean
from algorithm.RL_algorithm.soppo.ppo import Model as PPO
from algorithm.RL_algorithm.soppo.ddpg import Model as DDPG
from algorithm.RL_algorithm.soppo.memory import Memory
from algorithm.RL_algorithm.soppo.runner import Runner


def check_trajectory_terminal(ppo_tra, ddpg_tra):
    ppo_terminals = ppo_tra[4]
    ddpg_terminals = ddpg_tra[4]

    ppo_tra_num = np.sum(ppo_terminals)
    ddpg_tra_num = np.sum(ddpg_terminals)

    return ppo_tra_num, ddpg_tra_num


class Model(object):
    """
        supervised off-policy ppo
    """

    def __init__(self, *, network, env, lr=5e-4, cliprange=0.2, gamma=0.99, lam=0.95, tau=0.01, pg_coef=1.0,
                 vf_coef=0.5, kl_coef=0.1, ent_coef=0.5, ref_coef=0.1, actor_reg_coef=0.2, critic_reg_coef=0.2,
                 max_grad_norm=0.5, nsteps=40, sess=None, noise_type=None, **network_kwargs):

        self.env = env
        self.nenvs = self.env.num_envs
        self.nsteps = nsteps
        self.gamma = gamma
        self.lam = lam
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/tmp/'
        self.memory = Memory(limit=int(5000))

        if sess is None:
            self.sess = get_session()
        else:
            self.sess = sess

        with tf.variable_scope('shared_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('vf_model'):
                    # create vf model, default: mlp value network
                    value_network = build_value(self.env.observation_space, network, **network_kwargs)
                    self.vf_model = value_network(sess=self.sess)
            with tf.variable_scope('target'):
                with tf.variable_scope('vf_model'):
                    # create vf model, default: mlp value network
                    value_network = build_value(self.env.observation_space, network, **network_kwargs)
                    self.target_vf_model = value_network(sess=self.sess)

        self.ppo_model = PPO(network=network, obs_space=env.observation_space, ac_space=env.action_space,
                             vf_model=self.vf_model, target_vf_model=self.target_vf_model, sess=self.sess,
                             cliprange=cliprange, lr=lr, gamma=gamma, lam=lam, tau=tau, pg_coef=pg_coef,
                             vf_coef=vf_coef, kl_coef=kl_coef, ent_coef=ent_coef, actor_reg_coef=actor_reg_coef,
                             critic_reg_coef=critic_reg_coef, max_grad_norm=max_grad_norm, **network_kwargs)

        self.ddpg_model = DDPG(network=network, obs_space=env.observation_space, ac_space=env.action_space,
                               vf_model=self.vf_model, target_vf_model=self.target_vf_model, memory=self.memory,
                               sess=self.sess, noise_type=noise_type, lr=lr, gamma=gamma, lam=lam, tau=tau,
                               pg_coef=pg_coef, ref_coef=ref_coef, qf_coef=vf_coef,
                               actor_reg_coef=actor_reg_coef, critic_reg_coef=critic_reg_coef,
                               max_grad_norm=max_grad_norm, **network_kwargs)

        self.runner = Runner(env=self.env, models=[self.ppo_model, self.ddpg_model], nsteps=nsteps, gamma=gamma,
                             lam=lam)

    def learn(self, total_timesteps, num_sample_batch=2, minibatch_size=40, num_train_epoch=4, seed=None,
              log_interval=10, save_interval=10):
        set_global_seeds(seed)
        total_timesteps = int(total_timesteps)
        ppo_episode_info_buffer = deque(maxlen=100)
        ddpg_episode_info_buffer = deque(maxlen=100)
        # Start total timer

        for update in range(1, total_timesteps):
            # Start timer
            time_start = time.perf_counter()
            fraction = 1.0 - (update - 1.0) / total_timesteps

            # run
            env_trajectories, epinfos = self.runner.run()
            self.memory.store(*env_trajectories)
            advance_alg = 'ppo'
            if len(epinfos) > 0:
                ppo_tra_num, ddpg_tra_num = check_trajectory_terminal(*env_trajectories)
                ppo_episode_info_buffer.extend(epinfos[0:ppo_tra_num])
                ddpg_episode_info_buffer.extend(epinfos[ppo_tra_num:])
                ppo_eprewmean = safe_mean([epinfo['r'] for epinfo in ppo_episode_info_buffer])
                ddpg_eprewmean = safe_mean([epinfo['r'] for epinfo in ddpg_episode_info_buffer])
                if ppo_eprewmean > ddpg_eprewmean:
                    advance_alg = 'ppo'
                else:
                    advance_alg = 'ddpg'

            ppo_trajectories = env_trajectories[0]
            ppo_train_obs = ppo_trajectories[0]
            ppo_train_ac = ppo_trajectories[1]
            ppo_train_r = ppo_trajectories[2]
            ppo_train_next_obs = ppo_trajectories[3]
            ppo_train_terminal = ppo_trajectories[4]
            ppo_train_adv = GAE(ppo_train_obs, ppo_train_r, ppo_train_next_obs, ppo_train_terminal,
                                self.target_vf_model, self.gamma, self.lam)
            if advance_alg == 'ddpg':
                ppo_ref_pdparam = self.ddpg_model.policy_model.cal_pdparams(ppo_train_obs)
                ppo_train_input = (ppo_train_obs, ppo_train_ac, ppo_train_adv, ppo_ref_pdparam)
            else:
                ppo_train_input = (ppo_train_obs, ppo_train_ac, ppo_train_adv, None)
            # ppo_batch_size = len(ppo_train_obs)
            # ppo_inds = np.arange(ppo_batch_size)

            if self.memory.nb_entries < num_sample_batch:
                ddpg_train_wait = True
                ddpg_train_input = None
                ddpg_batch_size = None
                ddpg_inds = None
            else:
                ddpg_train_wait = False
                ddpg_trajectories = self.memory.sample(batch_size=num_sample_batch)
                ddpg_train_obs = ddpg_trajectories['observations']
                ddpg_train_r = ddpg_trajectories['rewards']
                ddpg_train_next_obs = ddpg_trajectories['next_observations']
                ddpg_train_terminal = ddpg_trajectories['terminals']
                ddpg_train_adv = GAE(ddpg_train_obs, ddpg_train_r, ddpg_train_next_obs, ddpg_train_terminal,
                                     self.target_vf_model, self.gamma, self.lam)
                if advance_alg == 'ppo':
                    ddpg_ref_pdparam = self.ppo_model.policy_model.cal_pdparams(ddpg_train_obs)
                    ddpg_train_input = (ddpg_train_obs, ddpg_train_adv, ddpg_ref_pdparam)
                else:
                    ddpg_train_input = (ddpg_train_obs, ddpg_train_adv)
                ddpg_batch_size = len(ddpg_train_obs)
                ddpg_inds = np.arange(ddpg_batch_size)

            for _ in range(num_train_epoch):
                self.ppo_model.train(fraction, *ppo_train_input, output_stats=False)

            if not ddpg_train_wait:
                for start in range(0, ddpg_batch_size, minibatch_size):
                    end = start + minibatch_size
                    mbinds = ddpg_inds[start:end]
                    slices = (arr[mbinds] for arr in ddpg_train_input)
                    self.ddpg_model.train(fraction, *slices, output_stats=False)

            # End timer
            time_now = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(self.nsteps / (time_now - time_start))

            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                # ev = explained_variance(values, returns)
                logger.record_tabular("fps", fps)
                logger.record_tabular('ppo_eprewmean', safe_mean([epinfo['r'] for epinfo in ppo_episode_info_buffer]))
                logger.record_tabular('ppo_eplenmean', safe_mean([epinfo['l'] for epinfo in ppo_episode_info_buffer]))
                logger.record_tabular('ddpg_eprewmean', safe_mean([epinfo['r'] for epinfo in ddpg_episode_info_buffer]))
                logger.record_tabular('ddpg_eplenmean', safe_mean([epinfo['l'] for epinfo in ddpg_episode_info_buffer]))
                logger.record_tabular("misc/serial_timesteps", update * self.nsteps)
                logger.record_tabular("misc/update_timesteps", update)
                # logger.record_tabular("misc/explained_variance", float(ev))

                # for (stat_val, stat_name) in zip(self.ppo_model.stats_list, self.ppo_model.stats_names):
                #     logger.record_tabular('ppo_loss/' + stat_name, stat_val)
                # for (stat_val, stat_name) in zip(self.ddpg_model.stats_list, self.ddpg_model.stats_names):
                #     logger.record_tabular('ddpg_loss/' + stat_name, stat_val)
                logger.dump_tabular()

            if save_interval and update % save_interval == 0:
                pass
                file_name = time.strftime('Y%YM%mD%d_h%Hm%Ms%S', time.localtime(time.time()))
                model_save_path = self.def_path_pre + file_name
                self.save(model_save_path)

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

    def step(self, obs):
        a = self.ddpg_model.step(obs)
        return a
