from rlbench.environment import Environment
from rlbench.action_config import SnakeRobotActionConfig, ActionConfig
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size
        self.clk = 0

    def act(self, obs):
        joints = np.zeros((self.action_size,))
        for i in range(self.action_size):
            if i % 2 == 0:
                # joints[i] = 0.7 * np.sin(2 * self.clk + (i / 2) * 30) + 0
                joints[i] = 0.7 * np.sin(2 * self.clk + (i/2) * 30 + 60)
            else:
                # joints[i] = 0.7 * np.sin(2 * self.clk + (i/2) * 30 + 12) + 0
                joints[i] = 0.5 * np.sin(4 * self.clk + (i/2) * 30)

        # return (np.random.normal(-0.7, 0.7, size=(self.action_size,))).tolist()
        self.clk += 0.05
        return joints.tolist()


obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.set_all_low_dim(True)

action_mode = ActionConfig(SnakeRobotActionConfig.ABS_JOINT_POSITION)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(action_mode.action_size)

training_steps = 200000
episode_length = 100000
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        agent.clk = 0
        # print(descriptions)
    action = agent.act(obs)
    # print(action)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
