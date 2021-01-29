import os
import time
import numpy as np
import tensorflow as tf
from collections import deque

from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.common.tf_util import get_session
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from algorithm.RL_algorithm.soppo.policies import build_policy
from algorithm.RL_algorithm.soppo.runner import Runner
from algorithm.RL_algorithm.soppo.values import build_value
from algorithm.RL_algorithm.utils.math_utils import safe_mean
from algorithm.RL_algorithm.utils.tensorflow1.tf_utils import save_variables, load_variables


def constfn(val):
    def f(_):
        return val

    return f


class Model(object):
    def __init__(self, *, network, env, lr=3e-4, cliprange=0.2, nsteps=128, nminibatches=4,
                 ent_coef=0.0, vf_coef=0.25, max_grad_norm=0.5, gamma=0.99, lam=0.95, mpi_rank_weight=1,
                 comm=None, load_path=None, **network_kwargs):

        """
        Parameters:
        ----------

        network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies.py

        env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training

        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)


        nminibatches: int                 number of training minibatches per update. For recurrent policies.py,
                                          should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        ent_coef: float                   policy entropy coefficient in the optimization objective

        vf_coef: float                    value function loss coefficient in the optimization objective

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py.py/build_policy and arguments to a particular type of network
                                          For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

        """

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        self.sess = sess = get_session()

        self.env = env

        if isinstance(lr, float):
            self.lr = constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            self.cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        # Calculate the batch_size
        self.nminibatches = nminibatches
        self.nenvs = self.env.num_envs
        self.nsteps = nsteps
        self.nbatch = self.nenvs * self.nsteps
        self.nbatch_train = self.nbatch // nminibatches

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # default: mlp value network
            value_network = build_value(env, network, **network_kwargs)
            act_vmodel = value_network(self.nenvs, sess)
            train_vmodel = value_network(self.nbatch_train, sess)

            # default: mlp policy network
            policy_network = build_policy(env, network, estimate_q=False, **network_kwargs)

            # CREATE OUR TWO MODELS
            # act_pmodel that is used for sampling
            # act_pmodel = policy(self.nenvs, sess)
            act_pmodel = policy_network(self.nenvs, sess)  # , act_vmodel.vf_latent
            # Train model for training
            # train_pmodel = policy(self.nbatch_train, sess)
            train_pmodel = policy_network(self.nbatch_train, sess)  # , train_vmodel.vf_latent

        # CREATE THE PLACEHOLDERS
        self.A = A = train_pmodel.pdtype.sample_placeholder([None])  # action placeholder
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_pmodel.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_pmodel.pd.entropy())

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_pmodel.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_pmodel.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))  # 裁剪数据占总数据比例

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS

        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_pmodel = train_pmodel
        self.act_pmodel = act_pmodel
        self.step = act_pmodel.step
        self.value = act_pmodel.estimate_v
        self.initial_state = None
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/ppo_tmp/'

        # initialize
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm)  # pylint: disable=E1101

        # load trained model
        if load_path is not None:
            self.load_newest(load_path)

        # Instantiate the runner object
        self.runner = Runner(env=self.env, model=self, nsteps=nsteps, gamma=gamma, lam=lam)

    def train(self, lr, cliprange, obs, returns, actions, values, neglogpacs):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_pmodel.X: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }

        # return stats_list result
        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]

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
            lrnow = self.lr(frac)
            # Calculate the cliprange
            cliprangenow = self.cliprange(frac)

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Stepping environment...')

            # Get minibatch
            obs, returns, actions, values, neglogpacs, epinfos = self.runner.run()  # pylint: disable=E0632

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
                    mblossvals.append(self.train(lrnow, cliprangenow, *slices))

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
                for (lossval, lossname) in zip(lossvals, self.loss_names):
                    logger.record_tabular('loss/' + lossname, lossval)

                if is_mpi_root:
                    logger.dump_tabular()

            if save_interval and (update % save_interval == 0 or update == 1) and is_mpi_root:
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
