import numpy as np
from abc import ABC, abstractmethod


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, models, nsteps):
        self.env = env
        self.models = models
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

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
                actions = self.models.step(self.obs)
            else:
                assert len(self.models) == self.nenv
                actions = []
                obs_group = np.array_split(self.obs, self.nenv)
                for i in range(self.nenv):
                    action = self.models[i].step(obs_group[i])
                    actions.append(action.squeeze(0))
                actions = np.array(actions)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_next_obs.append(self.obs.copy())
            mb_rewards.append(rewards)
            mb_dones.append(self.dones)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions).swapaxes(0, 1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs.dtype).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)

        env_trajectories = []
        for env_id in range(self.nenv):
            env_trajectory = ([mb_obs[env_id], mb_actions[env_id], mb_rewards[env_id], mb_next_obs[env_id], mb_dones[env_id]])
            env_trajectories.append(env_trajectory)

        return env_trajectories, epinfos
