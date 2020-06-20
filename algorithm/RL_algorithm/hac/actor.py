import tensorflow as tf
from algorithm.RL_algorithm.hac.utils import layer


class Actor:
    def __init__(self, sess, env, num_layers, layer_number, batch_size, learning_rate=0.0001, tau=0.05):

        self.sess = sess
        self.env = env
        self.batch_size = batch_size
        self.layer_number = layer_number
        self.learning_rate = learning_rate
        # self.exploration_policies = exploration_policies
        self.tau = tau

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_bounds = (self.env.action_space.high - self.env.action_space.low) / 2
            self.action_offset = self.env.action_space.high - self.action_bounds
        else:
            self.action_bounds = self.env.task.subgoal_bounds_symmetric
            self.action_offset = self.env.task.subgoal_offset

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = self.env.action_space.shape[0]
        else:
            self.action_space_size = self.env.task.subgoal_dim

        self.actor_name = 'actor_' + str(layer_number)

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == num_layers - 1:
            self.goal_dim = self.env.task.endgoal_dim
        else:
            self.goal_dim = self.env.task.subgoal_dim

        self.state_dim = self.env.observation_space.shape[0]
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph)

        # Target network code "repurposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)

        # Create target actor network
        self.target = self.create_nn(self.features_ph, name=self.actor_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
            [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                           tf.multiply(self.target_weights[i], 1. - self.tau))
             for i in range(len(self.target_weights))]

        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        self.unnormalized_actor_gradients = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # self.policy_gradient = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))

    def get_action(self, state, goal):
        actions = self.sess.run(self.infer,
                                feed_dict={
                                    self.state_ph: state,
                                    self.goal_ph: goal
                                })

        return actions

    def get_target_action(self, state, goal):
        actions = self.sess.run(self.target,
                                feed_dict={
                                    self.state_ph: state,
                                    self.goal_ph: goal
                                })

        return actions

    def update(self, state, goal, action_derivs):
        weights, policy_grad, _ = self.sess.run([self.weights, self.policy_gradient, self.train],
                                                feed_dict={
                                                    self.state_ph: state,
                                                    self.goal_ph: goal,
                                                    self.action_derivs: action_derivs
                                                })

        return len(weights)

        # self.sess.run(self.update_target_weights)

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_bounds + self.action_offset

        return output


