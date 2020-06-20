import numpy as np
from algorithm.RL_algorithm.hac.buffer import ExperienceBuffer
from algorithm.RL_algorithm.hac.actor import Actor
from algorithm.RL_algorithm.hac.critic import Critic


class Layer:
    def __init__(self, layer_number, env, sess, num_layers, attempt_limit, atomic_noise, subgoal_noise, subgoal_penalty,
                 episodes_to_store, subgoal_test_perc):
        self.layer_number = layer_number
        self.env = env
        self.sess = sess
        self.num_layers = num_layers
        if layer_number > 0:
            self.attempt_limit = attempt_limit
        else:
            self.attempt_limit = int(self.env.task.episode_len / attempt_limit)
        self.atomic_noise = atomic_noise
        self.subgoal_noise = subgoal_noise
        self.subgoal_penalty = subgoal_penalty
        # Number of transitions in the full episode stored in replay buffer
        self.episodes_to_store = episodes_to_store
        self.subgoal_test_perc = subgoal_test_perc

        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.get_episode_len()).

        self.current_state = None
        self.goal = None

        # Initialize Replay Buffer.  Below variables determine size of replay buffer.
        # Ceiling on buffer size
        self.buffer_size_ceiling = 10 ** 7
        # Set number of transitions to serve as replay goals during goal replay, 抽取三个目标构建三个后见目标转移序列
        self.num_replay_goals = 3
        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.attempt_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.attempt_limit + int(self.attempt_limit / 3)
        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.attempt_limit ** (self.num_layers - 1 - self.layer_number)
                               * self.episodes_to_store, self.buffer_size_ceiling)

        self.batch_size = 1024
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        # Initialize actor and critic networks
        self.actor = Actor(self.sess, self.env, self.num_layers, self.layer_number, self.batch_size)
        self.critic = Critic(self.sess, self.env, self.num_layers, self.layer_number, self.attempt_limit)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = atomic_noise if atomic_noise is not None else 0.1
        else:
            self.noise_perc = subgoal_noise if subgoal_noise is not None else 0.03

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

    # Add noise to provided action
    def add_noise(self, action):

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = (self.env.action_space.high - self.env.action_space.low) / 2
            action_offset = self.env.action_space.high - self.env.action_space.low
        else:
            action_bounds = self.env.task.subgoal_bounds_symmetric
            action_offset = self.env.task.subgoal_offset

        assert len(action) == len(action_bounds), "Action bounds must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(0, self.noise_perc * action_bounds[i])

            action[i] = max(min(action[i], action_bounds[i] + action_offset[i]), -action_bounds[i] + action_offset[i])

        return action

    # Select random action
    def get_random_action(self):

        if self.layer_number == 0:
            action = np.zeros(self.env.action_space.shape[0])
        else:
            action = np.zeros(self.env.task.subgoal_dim)

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                action_bounds = (self.env.action_space.high - self.env.action_space.low) / 2
                action_offset = self.env.action_space.high - self.env.action_space.low
                action[i] = np.random.uniform(-action_bounds[i] + action_offset[i],
                                              action_bounds[i] + action_offset[i])
            else:
                action[i] = np.random.uniform(-self.env.task.subgoal_bounds_symmetric[i] + self.env.task.subgoal_offset[i],
                                              self.env.task.subgoal_bounds_symmetric[i] + self.env.task.subgoal_offset[i])

        return action

    # Function selects action using an epsilon-greedy policy
    def choose_action(self, subgoal_test):

        # If testing subgoals, action is output of actor network without noise
        if subgoal_test:
            return self.actor.get_action(np.reshape(self.current_state, (1, len(self.current_state))),
                                         np.reshape(self.goal, (1, len(self.goal))))[0], "Policy", subgoal_test
        else:

            if np.random.random_sample() > 0.2:
                # Choose noisy action
                action = self.add_noise(
                    self.actor.get_action(np.reshape(self.current_state, (1, len(self.current_state))),
                                          np.reshape(self.goal, (1, len(self.goal))))[0])

                action_type = "Noisy Policy"

            # Otherwise, choose random action
            else:
                action = self.get_random_action()

                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < self.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

            return action, action_type, next_subgoal_test

    # Create action replay transition by evaluating hindsight action given original goal
    def perform_action_replay(self, hindsight_action, next_state, goal_status):

        # Determine reward (0 if goal achieved, -1 otherwise) and finished boolean.  The finished boolean is used for determining the target for Q-value updates
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False

        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None]
        # print("AR Trans: ", transition)

        # Add action replay transition to layer's replay buffer
        self.replay_buffer.add(np.copy(transition))

    # Create initial goal replay transitions
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, total_layers):

        # Create transition evaluating hindsight action for some goal to be determined in future.  Goal will be ultimately be selected from states layer has traversed through.  Transition will be in the form [old state, hindsight action, reward = None, next state, goal = None, finished = None, next state projeted to subgoal/end goal space]

        if self.layer_number == total_layers - 1:
            hindsight_goal = self.env.task.project_state_to_endgoal()
        else:
            hindsight_goal = self.env.task.project_state_to_subgoal()

        # In the process of the hindsight goal's replaying, the fifth element will be filled with hindsight_goal.
        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal]
        # print("\nPrelim GR A: ", transition)

        self.temp_goal_replay_storage.append(np.copy(transition))

        """
        # Designer can create some additional goal replay transitions.  For instance, higher level transitions can be replayed with the subgoal achieved in hindsight as the original goal.
        if self.layer_number > 0:
            transition_b = [self.current_state, hindsight_action, 0, next_state, hindsight_goal, True, None]
            # print("\nGoal Replay B: ", transition_b)
            self.replay_buffer.add(np.copy(transition_b))
        """

    # Return reward given provided goal and goal achieved in hindsight
    @staticmethod
    def get_reward(new_goal, hindsight_goal, goal_thresholds):

        assert len(new_goal) == len(hindsight_goal) == len(goal_thresholds), \
            "Goal, hindsight goal, and goal thresholds do not have same dimensions"

        # If the difference in any dimension is greater than threshold, goal not achieved
        for i in range(len(new_goal)):
            if np.absolute(new_goal[i] - hindsight_goal[i]) > goal_thresholds[i]:
                return -1

        # Else goal is achieved
        return 0

    # Finalize goal replay by filling in goal, reward, and finished boolean for the preliminary goal replay transitions created before
    # 补全后见目标转移序列
    def finalize_goal_replay(self, goal_thresholds):

        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)

        num_replay_goals = self.num_replay_goals
        # If fewer transitions for  ordinary number of replay goals, lower number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        """
        if self.layer_number == 1:
            print("\n\nPerforming Goal Replay\n\n")
            print("Num Trans: ", num_trans, ", Num Replay Goals: ", num_replay_goals)
        """

        indices = np.zeros(num_replay_goals)
        indices[:num_replay_goals - 1] = np.random.randint(num_trans, size=num_replay_goals - 1)
        indices[num_replay_goals - 1] = num_trans - 1  # 每个 episode 最后一个状态必被当作后见目标
        indices = np.sort(indices)

        # For each selected transition, update the goal dimension of the selected transition and all prior transitions
        # by using the next state of the selected transition as the new goal.
        # Given new goal, update the reward and finished boolean as well.
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            # if self.layer_number == 1:
            #   print("GR Iteration: %d, Index %d" % (i, indices[i]))

            new_goal = trans_copy[int(indices[i])][6]
            # for index in range(int(indices[i])+1):
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                # Add finished transition to replay buffer
                # if self.layer_number == 1:
                #   print("\nNew Goal: ", new_goal)
                #   print("Upd Trans %d: " % index, trans_copy[index])

                self.replay_buffer.add(trans_copy[index])

        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []

    # Create transition penalizing subgoal if necessary.  The target Q-value when this transition is used will ignore next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal penalty to depend on the next state.
    def penalize_subgoal(self, subgoal, next_state):

        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]

        self.replay_buffer.add(np.copy(transition))

    # Determine whether layer is finished training
    def return_to_higher_level(self, agent, max_lay_achieved, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached,
        # (ii) maxed out episode time steps (env.get_episode_len()),
        # (iii) not testing and layer is out of attempts, and (iv) testing,
        # layer is not the highest level, and layer is out of attempts.
        # NOTE: during testing, highest level will continue to ouput subgoals until either
        # (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= self.env.task.episode_len:
            return True

        # Return when layer has maxed out attempts
        elif not agent.test and attempts_made >= self.attempt_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.test and self.layer_number < self.num_layers - 1 and attempts_made >= self.attempt_limit:
            return True

        else:
            return False

    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, subgoal_test=False, episode_num=None):

        # print("\nTraining Layer %d" % self.layer_number)

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        # if self.layer_number == 0 and agent.show and self.num_layers > 1:
        #     self.env.display_subgoals(agent.goal_array)
        #     env.sim.data.mocap_pos[3] = env.task.project_state_to_endgoal(env.sim,self.current_state)
        #     print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.attemp_limit attempts to each its goal state.
        attempts_made = 0

        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(subgoal_test)
            # print("primitive action: ", action)
            """
            if self.layer_number == agent.num_layers - 1:
                # print("\nLayer %d Action: " % self.layer_number, action)
                print("Q-Value: ", self.critic.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
            """

            # If next layer is not bottom level, propose subgoal for next layer to achieve
            # and determine whether that subgoal should be tested
            if self.layer_number > 0:

                agent.goal_array[self.layer_number - 1] = action

                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, next_subgoal_test, episode_num)

            # If layer is bottom level, execute low-level action
            else:
                next_state, _, _, _ = self.env.step(action)
                # print("agent.steps_taken: ", agent.steps_taken)

                # Increment steps taken
                agent.steps_taken += 1
                # print("Num Actions Taken: ", agent.steps_taken)

                if agent.steps_taken >= self.env.task.episode_len:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was achieved and,
                # if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals()

            attempts_made += 1

            # Print if goal from current layer as been achieved
            if goal_status[self.layer_number]:
                if self.layer_number < agent.num_layers - 1:
                    print("SUBGOAL ACHIEVED")
                print("Episode %d, Layer %d, Attempt %d Goal Achieved" % (
                    episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == agent.num_layers - 1:
                    print("Hindsight Goal: ", self.env.task.project_state_to_endgoal())
                else:
                    print("Hindsight Goal: ", self.env.task.project_state_to_subgoal())

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number - 1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = self.env.task.project_state_to_subgoal()

            # Next, create hindsight transitions if not testing
            if not agent.test:

                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, agent.num_layers)

                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number - 1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state)

            # Print summary of transition
            if agent.verbose:

                print("\nEpisode %d, Training Layer %d, Attempt %d" % (episode_num, self.layer_number, attempts_made))
                # print("Goal Array: ", agent.goal_array, "Max Lay Achieved: ", max_lay_achieved)
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == agent.num_layers - 1:
                    print("Hindsight Goal: ", self.env.task.project_state_to_endgoal())
                else:
                    print("Hindsight Goal: ", self.env.task.project_state_to_subgoal())
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)

            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or \
                    agent.steps_taken >= self.env.task.episode_len or attempts_made >= self.attempt_limit:

                # if self.layer_number == agent.num_layers - 1:
                #     print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.attempt_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)

                # Finish goal replay by filling in missing goal and reward values before returning to prior level.
                if self.layer_number == agent.num_layers - 1:
                    goal_thresholds = self.env.task.endgoal_thresholds
                else:
                    goal_thresholds = self.env.task.subgoal_thresholds
                self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(agent, max_lay_achieved, attempts_made):
                    return goal_status, max_lay_achieved

    # Update actor and critic networks
    def learn(self, num_updates):

        for _ in range(num_updates):
            # Update weights of non-target networks
            if self.replay_buffer.size >= self.batch_size:
                old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()

                self.critic.update(old_states, actions, rewards, new_states, goals,
                                   self.actor.get_action(new_states, goals), is_terminals)
                action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                self.actor.update(old_states, goals, action_derivs)

        """
        # To use target networks comment for loop above and uncomment for loop below
        for _ in range(num_updates):
            # Update weights of non-target networks
            if self.replay_buffer.size >= self.batch_size:
                old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()


                self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_target_action(new_states,goals), is_terminals)
                action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                self.actor.update(old_states, goals, action_derivs)

        # Update weights of target networks
        self.sess.run(self.critic.update_target_weights)
        self.sess.run(self.actor.update_target_weights)
        """
