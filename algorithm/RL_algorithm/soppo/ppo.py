import os
import tensorflow as tf
from baselines.common.tf_util import get_session, initialize
try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from algorithm.RL_algorithm.soppo.actors import build_actor
from algorithm.RL_algorithm.soppo.critics import build_value


def constfn(val):
    def f(_):
        return val

    return f


class Model(object):
    def __init__(self, *, network, obs_space, ac_space, lr=3e-4, cliprange=0.2, nsteps=128, nminibatches=4,
                 ent_coef=0.0, vf_coef=0.25, max_grad_norm=0.5, mpi_rank_weight=1,
                 comm=None, **network_kwargs):

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

        self.observation_space = obs_space
        self.action_space = ac_space
        self.sess = sess = get_session()

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
        self.nenvs = 1
        self.nsteps = nsteps
        self.nbatch = self.nenvs * self.nsteps
        self.nbatch_train = self.nbatch // nminibatches

        with tf.variable_scope('shared_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('vf_model'):
                    # create vf model, default: mlp value network
                    value_network = build_value(obs_space, network, **network_kwargs)
                    self.vf_model = vf_model = value_network(sess=self.sess)

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('current'):
                with tf.variable_scope('policy'):
                    # create policy model, default: mlp policy network
                    policy_network = build_actor(obs_space, ac_space, network, **network_kwargs)
                    self.policy_model = policy_model = policy_network(sess=self.sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = policy_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = policy_model.cal_action_neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy_model.cal_entropy())

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = vf_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE)
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
        self.train_variables = tf.trainable_variables('ppo2_model') + tf.trainable_variables('vf_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, self.train_variables)
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

        self.policy_model = policy_model
        self.step = policy_model.step
        self.value = vf_model.step
        self.initial_state = None
        self.def_path_pre = os.path.dirname(os.path.abspath(__file__)) + '/ppo_tmp/'

        # initialize
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm)  # pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, actions, values, neglogpacs):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.policy_model.obs_ph: obs,
            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }

        # return stats_list result
        # print (self.sess.run(self.grads, td_map))
        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]
