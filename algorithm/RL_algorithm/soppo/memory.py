import numpy as np
from collections import deque
import random


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class FifoBuffer(object):
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    @property
    def length(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.buffer[idx]

    def append(self, data):
        self.buffer.append(data)

    def extend(self, data_list):
        self.buffer.extend(data_list)

    def get_batch(self, batch_size):
        if batch_size > self.length:
            batch_size = self.length
        mini_batch = random.sample(self.buffer, batch_size)
        return mini_batch


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


def decompose_trajectories(batches):
    observations = np.array([batches[i][0] for i in range(len(batches))])
    actions = np.array([batches[i][1] for i in range(len(batches))])
    rewards = np.array([batches[i][2] for i in range(len(batches))])
    next_observations = np.array([batches[i][3] for i in range(len(batches))])
    terminals = np.array([batches[i][4] for i in range(len(batches))])

    return observations, actions, rewards, next_observations, terminals


class Memory(object):
    def __init__(self, limit):
        self.limit = limit
        self.stochastic_trajectories = FifoBuffer(limit)
        self.determined_trajectories = FifoBuffer(limit)
        self.temp_sto_trajectory = {'observations': [], 'actions': [], 'rewards': [],
                                      'next_observations': [], 'terminals': []}
        self.temp_det_trajectory = {'observations': [], 'actions': [], 'rewards': [],
                                      'next_observations': [], 'terminals': []}
        self.sto_next_check_idx = 0
        self.det_next_check_idx = 0

    def store(self, sto_tra, det_tra):
        complete_sto_tra_list = self._store_temp_trajectory(self.temp_sto_trajectory, 'stochastic', *sto_tra)
        complete_det_tra_list = self._store_temp_trajectory(self.temp_det_trajectory, 'determined', *det_tra)

        self.stochastic_trajectories.extend(complete_sto_tra_list)
        self.determined_trajectories.extend(complete_det_tra_list)

    def _store_temp_trajectory(self, temp_trajectory_buff, buff_type, observations, actions, rewards, next_observations, terminals):
        temp_trajectory_buff['observations'].extend(observations)
        temp_trajectory_buff['actions'].extend(actions)
        temp_trajectory_buff['rewards'].extend(rewards)
        temp_trajectory_buff['next_observations'].extend(next_observations)
        temp_trajectory_buff['terminals'].extend(terminals)

        complete_trajectories = []
        terminal_inds = []
        check_len = len(temp_trajectory_buff['terminals'])
        if buff_type == 'stochastic':
            next_check_idx = self.sto_next_check_idx
        else:
            next_check_idx = self.det_next_check_idx
        for idx in range(next_check_idx, check_len):
            if temp_trajectory_buff['terminals'][idx]:
                terminal_inds.append(idx)
        next_check_idx = len(temp_trajectory_buff['terminals'])
        begin_idx = 0
        for idx in terminal_inds:
            complete_trajectory = (temp_trajectory_buff['observations'][begin_idx:idx+1], temp_trajectory_buff['actions'][begin_idx:idx+1],
                                   temp_trajectory_buff['rewards'][begin_idx:idx+1], temp_trajectory_buff['next_observations'][begin_idx:idx+1],
                                   temp_trajectory_buff['terminals'][begin_idx:idx+1])
            begin_idx = idx + 1
            complete_trajectories.append(complete_trajectory)
        temp_trajectory_buff['observations'] = temp_trajectory_buff['observations'][begin_idx:]
        temp_trajectory_buff['actions'] = temp_trajectory_buff['actions'][begin_idx:]
        temp_trajectory_buff['rewards'] = temp_trajectory_buff['rewards'][begin_idx:]
        temp_trajectory_buff['next_observations'] = temp_trajectory_buff['next_observations'][begin_idx:]
        temp_trajectory_buff['terminals'] = temp_trajectory_buff['terminals'][begin_idx:]

        next_check_idx -= begin_idx
        if buff_type == 'stochastic':
            self.sto_next_check_idx = next_check_idx
        else:
            self.det_next_check_idx = next_check_idx

        return complete_trajectories

    def sample(self, batch_size, stochastic_proportion=0.5):
        # Draw such that we always have a proceeding element.
        assert batch_size > 1
        stochastic_batch_size = int(batch_size * stochastic_proportion)
        determined_batch_size = int(batch_size * (1 - stochastic_proportion))
        assert stochastic_batch_size > 0
        assert determined_batch_size > 0
        stochastic_trajectories_batch = self.stochastic_trajectories.get_batch(stochastic_batch_size)
        determined_trajectories_batch = self.determined_trajectories.get_batch(determined_batch_size)
        batches = stochastic_trajectories_batch + determined_trajectories_batch
        observations, actions, rewards, next_observations, terminals = decompose_trajectories(batches)

        result = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'terminals': terminals
        }
        return result

    @property
    def nb_entries(self):
        return len(self.stochastic_trajectories) + len(self.determined_trajectories)
