from collections import deque

import numpy as np
import pickle


class RolloutWorker:

    def __init__(self, venv, policy, dims, logger, T,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        self.venv = venv
        self.policy = policy
        self.dims = dims
        self.logger = logger
        self.T = T
        self.exploit = exploit
        self.use_target_net = use_target_net
        self.compute_Q = compute_Q
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.history_len = history_len
        self.render = render
        self.monitor = monitor
        self.kwargs = kwargs

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self):
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.dims['o']), np.float32)  # observations
        ag = np.empty((self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        reward_sum = 0
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(u)
            reward_sum += reward
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = info.get('is_success', 0.0)

            if done:
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success)
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        print("reward: ", reward_sum)
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        successful = np.array(successes)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += 1
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            episode_batch[key] = val

        # return convert_episode_to_batch_major(episode)
        return episode_batch

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        # logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

