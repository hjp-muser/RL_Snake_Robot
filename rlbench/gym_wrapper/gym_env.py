import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_config import ActionConfig, SnakeRobotActionConfig
from rlbench.observation_config import ObservationConfig

import numpy as np
import glob


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human']}

    def __init__(self, task_class, observation_mode='state', action_mode='joint', multi_action_space=False):
        self.num_skip_control = 20
        self.epi_obs = []
        self.obs_record = True
        self.obs_record_id = None

        self._observation_mode = observation_mode
        self.obs_config = ObservationConfig()
        if observation_mode == 'state':
            self.obs_config.set_all_high_dim(False)
            self.obs_config.set_all_low_dim(True)
        elif observation_mode == 'state-goal':
            self.obs_config.set_all_high_dim(False)
            self.obs_config.set_all_low_dim(True)
            self.obs_config.set_goal_info(True)
        elif observation_mode == 'vision':
            self.obs_config.set_all(False)
            self.obs_config.set_camera_rgb(True)
        elif observation_mode == 'both':
            self.obs_config.set_all(True)
        else:
            raise ValueError('Unrecognised observation_mode: %s.' % observation_mode)

        self._action_mode = action_mode
        self.ac_config = None
        if action_mode == 'joint':
            self.ac_config = ActionConfig(SnakeRobotActionConfig.ABS_JOINT_POSITION)
        elif action_mode == 'trigon':
            self.ac_config = ActionConfig(SnakeRobotActionConfig.TRIGON_MODEL_PARAM)
        else:
            raise ValueError('Unrecognised action_mode: %s.' % action_mode)

        self.env = Environment(action_config=self.ac_config, obs_config=self.obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)
        self.max_episode_steps = self.task.episode_len
        _, obs = self.task.reset()

        if action_mode == 'joint':
            self.action_space = spaces.Box(
                low=-1.7, high=1.7, shape=(self.ac_config.action_size,), dtype=np.float32)
        elif action_mode == 'trigon':
            if multi_action_space:
                # action_space1 = spaces.MultiBinary(n=1)
                low1 = np.array([-0.8, -0.8])
                high1 = np.array([0.8, 0.8])
                action_space1 = spaces.Box(low=low1, high=high1, dtype=np.float32)
                # low = np.array([-0.8, -0.8, 1.0, 3.0, -50, -10, -0.1, -0.1])
                # high = np.array([0.8, 0.8, 3.0, 5.0, 50, 10, 0.1, 0.1])
                low2 = np.array([1.0])
                high2 = np.array([3.0])
                action_space2 = spaces.Box(low=low2, high=high2, dtype=np.float32)
                self.action_space = spaces.Tuple((action_space1, action_space2))
            else:
                # low = np.array([0.0, -0.8, -0.8, 1.0, 3.0, -50, -10, -0.1, -0.1])
                # high = np.array([1.0, 0.8, 0.8, 3.0, 5.0, 50, 10, 0.1, 0.1])
                low = np.array([-1, -1, -1])
                high = np.array([1, 1, 1])
                self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if observation_mode == 'state' or observation_mode == 'state-goal':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs.get_low_dim_data().shape[0]*self.num_skip_control,))
            if observation_mode == 'state-goal':
                self.goal_dim = obs.get_goal_dim()
        elif observation_mode == 'vision':
            self.observation_space = spaces.Box(
                    low=0, high=1, shape=obs.head_camera_rgb.shape)
        elif observation_mode == 'both':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "rattler_eye_rgb": spaces.Box(
                    low=0, high=1, shape=obs.head_camera_rgb.shape),
                "rattler_eye_depth": spaces.Box(
                    low=0, high=1, shape=obs.head_camera_depth.shape),
                })

        self._gym_cam = None

    def _extract_obs(self, obs):
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'state-goal':
            obs_goal = {'observation': obs.get_low_dim_data()}
            obs_goal.update(obs.get_goal_data())
            return obs_goal
        elif self._observation_mode == 'vision':
            return obs.head_camera_rgb
        elif self._observation_mode == 'both':
            return {
                "state": obs.get_low_dim_data(),
                "rattler_eye_rgb": obs.head_camera_rgb,
                "rattle_eye_depth": obs.head_camera_depth,
            }

    def render(self, mode='human'):
        if self._gym_cam is None:
            # # Add the camera to the scene
            self._gym_cam = VisionSensor('monitor')
            self._gym_cam.set_resolution([640, 640])
            self._gym_cam.set_render_mode(RenderMode.EXTERNAL_WINDOWED)

    def reset(self):
        obs_data_group = []
        obs_data_dict = {'observation': [], 'desired_goal': None, 'achieved_goal': None}
        descriptions, obs = self.task.reset()
        self.epi_obs.append(obs)
        obs_data = self._extract_obs(obs)
        for _ in range(self.num_skip_control):
            # obs_data_group.extend(obs_data)
            if isinstance(obs_data, list) or isinstance(obs_data, np.ndarray):
                obs_data_group.extend(obs_data)
            elif isinstance(obs_data, dict):
                obs_data_dict['observation'].extend(obs_data['observation'])
                obs_data_dict['desired_goal'] = obs_data['desired_goal']
                obs_data_dict['achieved_goal'] = obs_data['achieved_goal']
        ret_obs = obs_data_group if len(obs_data_group) else obs_data_dict
        del descriptions  # Not used
        return ret_obs

    def step(self, action):
        obs_data_group = []
        obs_data_dict = {'observation': [], 'desired_goal': None, 'achieved_goal': None}
        reward_group = []
        terminate = False
        for _ in range(self.num_skip_control):
            obs, reward, step_terminate, success = self.task.step(action)
            self.epi_obs.append(obs)
            obs_data = self._extract_obs(obs)
            if isinstance(obs_data, list) or isinstance(obs_data, np.ndarray):
                obs_data_group.extend(obs_data)
            elif isinstance(obs_data, dict):    # used for hierarchical reinforcement algorithm
                obs_data_dict['observation'].extend(obs_data['observation'])
                obs_data_dict['desired_goal'] = obs_data['desired_goal']
                obs_data_dict['achieved_goal'] = obs_data['achieved_goal']
            reward_group.append(reward)
            terminate |= step_terminate
            if terminate:
                if self.obs_record and success:  # record a successful experience
                    self.record_obs("RobotPos")
                self.epi_obs = []
                break
        ret_obs = obs_data_group if len(obs_data_group) else obs_data_dict
        return ret_obs, np.mean(reward_group), terminate, {}

    def close(self):
        self.env.shutdown()

    # def load_env_param(self):
    #     self.env.load_env_param()

    def compute_reward(self, achieved_goal=None, desired_goal=None, info=None):
        assert achieved_goal is not None
        assert desired_goal is not None
        return self.task.compute_reward(achieved_goal, desired_goal)

    def record_obs(self, obs_part):
        if self.obs_record_id is None:
            record_filenames = glob.glob("./obs_record/obs_record_*.txt")
            record_filenames.sort(key=lambda filename: int(filename.split('_')[-1].split('.')[0]))
            if len(record_filenames) == 0:
                self.obs_record_id = 1
            else:
                last_id = int(record_filenames[-1].split('_')[-1].split('.')[0])
                self.obs_record_id = last_id + 1
        else:
            self.obs_record_id += 1
        filename = './obs_record/obs_record_'+str(self.obs_record_id)+'.txt'
        obs_record_file = open(filename, 'w')

        if obs_part == 'All':
            pass
        if obs_part == 'RobotPos':
            robot_pos_arr = []
            for obs in self.epi_obs:
                robot_pos = obs.get_2d_robot_pos()
                robot_pos_arr.append(robot_pos)
            target_pos = self.task.get_goal()
            robot_pos_arr.append(target_pos)        # The last line records the target position
            robot_pos_arr = np.array(robot_pos_arr)
            np.savetxt(obs_record_file, robot_pos_arr, fmt="%f")
        obs_record_file.close()

