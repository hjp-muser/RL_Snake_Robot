import numpy as np
from algorithm.RL_algorithm.hac.layer import Layer
import tensorflow as tf
import os
import pickle as cpickle
from collections import deque

# Below class instantiates an agent
class Model:
    def __init__(self, env, network='mlp', seed=None, retrain=True, num_layers=2, num_updates=10, subgoal_test_perc=0.4,
                 attempt_limit=20, subgoal_penalty=-10, atomic_noise=None, subgoal_noise=None, episodes_to_store=500):

        self.env = env
        self.network = network
        self.seed = seed
        self.retrain = retrain
        self.num_layers = num_layers
        self.num_updates = num_updates
        self.subgoal_test_perc = subgoal_test_perc
        self.attempt_limit = attempt_limit
        self.subgoal_penalty = subgoal_penalty
        self.atomic_noise = atomic_noise
        self.subgoal_noise = subgoal_noise
        self.episodes_to_store = episodes_to_store
        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = num_updates
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        # Create agent with number of levels specified by user
        self.layers = [Layer(i, env, self.sess, num_layers, attempt_limit, atomic_noise, subgoal_noise, subgoal_penalty,
                             episodes_to_store, subgoal_test_perc)
                       for i in range(num_layers)]

        # goal_array will store goal for each layer of agent.
        self.goal_array = [[] for _ in range(num_layers)]

        # Track number of low-level actions executed
        self.steps_taken = 0
        self.test = False
        self.verbose = False

        # Below parameters will be used to store performance results
        self.performance_log = []
        self.current_state = None
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        self.initialize_model()

    def initialize_model(self):
        # Below attributes will be used help save network parameter
        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)
        self.model_dir = os.getcwd() + '/models'
        self.model_loc = self.model_dir + '/HAC.ckpt'
        # Set up directory for saving models
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        if not self.retrain:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    # Train agent for an episode
    def train(self, episode_num):
        # Select initial state from in initial state space, defined in environment.py
        self.current_state = self.env.reset()
        # print("Initial State: ", self.current_state)

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.num_layers - 1] = self.env.task.get_goal()
        print("---------------------------------------------")
        print("Next End Goal: ", self.goal_array[self.num_layers - 1])

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode. Start from the top layer.
        goal_status, max_lay_achieved = self.layers[self.num_layers - 1].train(self, episode_num=episode_num)
        # Update actor/critic networks
        for i in range(self.num_layers):
            self.layers[i].learn(self.num_updates)

        # Return whether end goal was achieved
        return goal_status[self.num_layers - 1]

    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)

    # Save performance evaluations
    def log_performance(self, success_rate):

        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log, open("performance_log.p", "wb"))

    # Update actor and critic networks for each layer
    def learn(self, num_episode=int(1e6), log_interval=10):
        successful_episode = 0
        performance_queue = deque(maxlen=100)
        for episode in range(num_episode):
            success = self.train(episode)
            if success:
                print("Episode %d: the task successes" % episode)
                successful_episode += 1
                performance_queue.append(1)
            else:
                performance_queue.append(0)
                print("Episode %d: the task fails" % episode)
            if episode > 100:
                successful_episode -= 1
            if episode % log_interval == 0:
                self.save_model(episode)
                # success_rate = successful_episode / num_episode * 100
                success_rate = performance_queue.count(1)
                print("Training success rate: %.2f%%" % success_rate)
                self.log_performance(success_rate)

    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for _ in range(self.num_layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = self.env.task.project_state_to_subgoal()
        proj_endgoal = self.env.task.project_state_to_endgoal()

        for i in range(self.num_layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.num_layers - 1:

                # # Check dimensions are appropriate
                # assert len(proj_endgoal) == len(self.goal_array[i]) == len(
                #     self.env.task.endgoal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"
                #
                # # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                # for j in range(len(proj_endgoal)):
                #     if np.absolute(self.goal_array[i][j] - proj_endgoal[j]) > self.env.task.endgoal_thresholds[j]:
                #         goal_achieved = False
                #         break
                goal_achieved = self.env.task.goal_achieved

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(
                    self.env.task.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > self.env.task.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved
