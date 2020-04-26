import time
import functools
import tensorflow as tf
from tensorflow import losses
from collections import deque

from baselines import logger
from baselines.common import tf_util

from algorithm.RL_algorithm.a2c.policy import MlpPolicy, LnMlpPolicy
from algorithm.RL_algorithm.a2c.runner import Runner
from algorithm.RL_algorithm.utils.math_utils import safemean, explained_variance
from algorithm.RL_algorithm.utils.common_utils import set_global_seeds
from algorithm.RL_algorithm.utils.scheduler import Scheduler

def build_policy(network):
    if network == 'mlp':
        return MlpPolicy
    elif network == 'lnmlp':
        return LnMlpPolicy


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
            vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear',
            gamma=0.99, alpha=0.99, epsilon=1e-5):

        '''
        Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

        Parameters:
        -----------

        network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                            specifying the standard network architecture, or a function that takes tensorflow1 tensor as input and returns
                            tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                            neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                            See baselines.common/policies.py/lstm for more details on using recurrent nets in policies

        env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)

        seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

        nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                            nenv is number of environment copies simulated in parallel)

        total_timesteps:    int, total number of timesteps to train on (default: 80M)

        vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

        ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

        max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

        lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

        lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                            returns fraction of the learning rate (specified as lr) as output

        epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

        alpha:              float, RMSProp decay parameter (default: 0.99)

        gamma:              float, reward discounting parameter (default: 0.99)

        **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                            For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

        '''

        self.env = env
        self.nenvs = env.num_envs
        self.nsteps = nsteps
        nbatch = self.nenvs*nsteps
        self.seed = seed
        self.ent_coed = ent_coef
        self.vf_coed = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.total_timesteps = total_timesteps
        self.lrschedule = lrschedule

        sess = tf_util.get_session()

        policy = build_policy(network)

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(sess, env.observation_space, env.action_space, self.nenvs, 1, self.nenvs, reuse=False)

            # train_model is used to train our network
            train_model = policy(sess, env.observation_space, env.action_space, self.nenvs, self.nsteps, nbatch, reuse=True)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.proba_distribution.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.value_fn), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = tf.trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(initial_value=lr, n_values=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            cur_lr = None
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.obs_ph:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.states_ph] = states
                td_map[train_model.dones_ph] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)

    def learn(self, total_timesteps=int(80e6), log_interval=100, load_path=None):

        """
        Parameters:
        -----------

        log_interval: int, specifies how frequently the logs are printed out (default: 100)

        load_path: pre-train model load path

        """

        set_global_seeds(self.seed)

        if load_path is not None:
            self.load(load_path)

        # Instantiate the runner object
        runner = Runner(self.env, self, nsteps=self.nsteps, gamma=self.gamma)
        epinfobuf = deque(maxlen=100)

        # Calculate the batch_size
        nbatch = self.nenvs*self.nsteps

        # Start total timer
        tstart = time.time()

        for update in range(1, total_timesteps//nbatch+1):
            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values, epinfos = runner.run()
            epinfobuf.extend(epinfos)
            policy_loss, value_loss, policy_entropy = self.train(obs, states, rewards, masks, actions, values)
            nseconds = time.time()-tstart

            # Calculate the fps (frame per second)
            fps = int((update*nbatch)/nseconds)
            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.dump_tabular()
        return self

