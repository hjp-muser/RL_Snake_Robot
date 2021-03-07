import os
import multiprocessing
import numpy as np
import tensorflow as tf


def constfn(val):
    def f(_):
        return val

    return f


def flatten_axes01(arr):
    s = arr.shape
    if len(s) > 2:
        return arr.reshape(s[0]*s[1], *s[2:])
    else:
        return arr


def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


# def GAE2(observations_groups, rewards_groups, next_observations_groups, terminals_groups, vf_model, gamma, lam):
#     groups = len(observations_groups)
#     advantages_groups = []
#     for g in range(groups):
#         nsteps = len(observations_groups[g])
#         advs = np.zeros_like(observations_groups[g])
#         last_gae_adv = 0
#         for t in reversed(range(nsteps)):
#             next_nonterminal = 1.0 - terminals_groups[g][t]
#             next_values = vf_model.step(next_observations_groups[g][t])
#             values = vf_model.step(observations_groups[g][t])
#             # GAE (General Advantage Estimation)
#             delta = rewards_groups[g][t] + gamma * next_values * next_nonterminal - values
#             advs[t] = last_gae_adv = delta + gamma * lam * next_nonterminal * last_gae_adv
#         advantages_groups.append(advs)
#         advantages_groups = np.array(advantages_groups)
#     return advantages_groups


def GAE(observations, rewards, next_observations, terminals, vf_model, gamma, lam):
    nsteps = len(observations)
    advantages = np.zeros_like(rewards)
    last_gae_adv = 0
    for t in reversed(range(nsteps)):
        next_nonterminal = 1.0 - terminals[t]
        next_values = vf_model.step(next_observations[t]).squeeze()
        values = vf_model.step(observations[t]).squeeze()
        # GAE (General Advantage Estimation)
        delta = rewards[t] + gamma * next_values * next_nonterminal - values
        advantages[t] = last_gae_adv = delta + gamma * lam * next_nonterminal * last_gae_adv
    return advantages


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess