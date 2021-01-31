import time
import numpy as np
from collections import deque
from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.common.input import observation_placeholder

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from algorithm.RL_algorithm.utils.common_utils import set_global_seeds
from algorithm.RL_algorithm.utils.math_utils import safe_mean
from algorithm.RL_algorithm.soppo.ppo import Model as PPO


class Model(object):
    """
        supervised off-policy ppo
    """
    def __init__(self, *, network, env, ppo_lr=3e-4, cliprange=0.2, nsteps=128, nminibatches=4,
                 ent_coef=0.0, vf_coef=0.25, max_grad_norm=0.5, gamma=0.99, lam=0.95, mpi_rank_weight=1,
                 comm=None, **network_kwargs):

        self.env = env
        self.obs_ph = observation_placeholder(env.observation_space)
        self.nsteps = nsteps
        self.nminibatches = nminibatches
        self.nenvs = self.env.num_envs
        self.nsteps = nsteps
        self.nbatch = self.nenvs * self.nsteps
        self.nbatch_train = self.nbatch // nminibatches
        self.ppo_model = PPO(network=network, env=env, obs_ph=self.obs_ph, lr=ppo_lr, cliprange=cliprange,
                             nsteps=nsteps, nminibatches=nminibatches,
                             ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gamma=gamma, lam=lam, mpi_rank_weight=mpi_rank_weight,
                             comm=comm, load_path=None, **network_kwargs)

    def learn(self, total_timesteps, noptepochs=4, seed=None, log_interval=10, save_interval=10):

        set_global_seeds(seed)
        total_timesteps = int(total_timesteps)

        # Calculate the batch_size
        is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

        epinfobuf = deque(maxlen=100)

        # Start total timer
        tfirststart = time.perf_counter()

        for update in range(1, total_timesteps):
            assert self.nbatch % self.nminibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (update - 1.0) / total_timesteps
            # Calculate the learning rate
            lrnow = self.ppo_model.lr(frac)
            # Calculate the cliprange
            cliprangenow = self.ppo_model.cliprange(frac)

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Stepping environment...')

            # Get minibatch
            obs, returns, actions, values, neglogpacs, epinfos = self.ppo_model.runner.run()  # pylint: disable=E0632

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Done.')

            epinfobuf.extend(epinfos)

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(self.nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, self.nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                    mblossvals.append(self.ppo_model.train(lrnow, cliprangenow, *slices))

            # TODO: recurrent version
            # else:  # recurrent version
            #     assert self.nenvs % self.nminibatches == 0
            #     envsperbatch = self.nenvs // self.nminibatches
            #     envinds = np.arange(self.nenvs)
            #     flatinds = np.arange(self.nenvs * self.nsteps).reshape(self.nenvs, self.nsteps)
            #     for _ in range(self.noptepochs):
            #         np.random.shuffle(envinds)
            #         for start in range(0, self.nenvs, envsperbatch):
            #             end = start + envsperbatch
            #             mbenvinds = envinds[start:end]
            #             mbflatinds = flatinds[mbenvinds].ravel()
            #             slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
            #             mbstates = states[mbenvinds]
            #             mblossvals.append(self.train(lrnow, cliprangenow, *slices, mbstates))

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(self.nbatch / (tnow - tstart))

            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, returns)
                logger.record_tabular("fps", fps)
                logger.record_tabular('eprewmean', safe_mean([epinfo['r'] for epinfo in epinfobuf]))
                logger.record_tabular('eplenmean', safe_mean([epinfo['l'] for epinfo in epinfobuf]))
                logger.record_tabular("misc/serial_timesteps", update * self.nsteps)
                logger.record_tabular("misc/nupdates", update)
                logger.record_tabular("misc/total_timesteps", update * self.nbatch)
                logger.record_tabular("misc/explained_variance", float(ev))
                logger.record_tabular('misc/time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, self.ppo_model.loss_names):
                    logger.record_tabular('ppo_loss/' + lossname, lossval)

                if is_mpi_root:
                    logger.dump_tabular()

            if save_interval and (update % save_interval == 0 or update == 1) and is_mpi_root:
                file_name = time.strftime('Y%YM%mD%d_h%Hm%Ms%S', time.localtime(time.time()))
                model_save_path = self.ppo_model.def_path_pre + file_name
                self.ppo_model.save(model_save_path)

        return self

    def load_newest(self, load_path=None):
        self.ppo_model.load_newest(load_path)

    def load_index(self, index, load_path=None):
        self.ppo_model.load_index(index, load_path)

    def step(self, obs):
        a, neglogp, v = self.ppo_model.step(obs)
        return a, neglogp, v, None