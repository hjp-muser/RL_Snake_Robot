import numpy as np
from abc import ABC, abstractmethod


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, models, nsteps):
        self.env = env
        self.models = models
        self.nsteps = nsteps
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1

        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.ac = np.zeros((nenv,) + env.action_space.shape, dtype=env.action_space.dtype.name)
        self.rewards = np.zeros((nenv, 1), dtype=np.float32)
        self.terminals = np.zeros((nenv, 1), dtype=np.bool)

        self.obs[:] = env.reset()

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, models, nsteps, gamma, lam):
        super().__init__(env=env, models=models, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_next_obs, mb_dones = [], [], [], [], []
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            if isinstance(self.models, list) is False:
                self.ac[:] = actions = self.models.step(self.obs)
            else:
                assert len(self.models) == self.nenv
                actions = []
                obs_group = np.array_split(self.obs, self.nenv)
                for i in range(self.nenv):
                    action = self.models[i].step(obs_group[i])
                    actions.append(action.squeeze(0))
                actions = np.array(actions)
                self.ac[:] = actions

            mb_obs.append(np.array(self.obs.copy()))
            mb_actions.append(np.array(self.ac.copy()))
            self.obs[:], rewards, terminals, infos = self.env.step(actions)
            rewards = np.reshape(rewards, (self.nenv, 1))
            self.rewards[:] = rewards
            terminals = np.reshape(terminals, (self.nenv, 1))
            self.terminals[:] = terminals
            mb_next_obs.append(np.array(self.obs.copy()))
            mb_rewards.append(np.array(self.rewards.copy()))
            mb_dones.append(np.array(self.terminals.copy()))
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

        # batch of steps to batch of rollouts
        mb_obs = np.array(mb_obs, dtype=self.obs.dtype).swapaxes(0, 1)
        mb_actions = np.array(mb_actions, dtype=self.ac.dtype).swapaxes(0, 1)
        mb_rewards = np.array(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_next_obs = np.array(mb_next_obs, dtype=self.obs.dtype).swapaxes(0, 1)
        mb_dones = np.array(mb_dones, dtype=np.bool).swapaxes(0, 1)

        env_trajectories = []
        for env_id in range(self.nenv):
            env_trajectory = ([mb_obs[env_id], mb_actions[env_id], mb_rewards[env_id], mb_next_obs[env_id], mb_dones[env_id]])
            env_trajectories.append(env_trajectory)

        return tuple(env_trajectories), np.array(epinfos)
