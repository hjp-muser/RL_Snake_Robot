import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_config import ActionConfig, SnakeRobotActionConfig
from rlbench.observation_config import ObservationConfig
import numpy as np


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human']}

    def __init__(self, task_class, observation_mode='state', action_mode='joint'):
        self._observation_mode = observation_mode
        self.obs_config = ObservationConfig()
        if observation_mode == 'state':
            self.obs_config.set_all_high_dim(False)
            self.obs_config.set_all_low_dim(True)
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

        self.env = Environment(action_config=self.ac_config, obs_config=self.obs_config, headless=False)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        if action_mode == 'joint':
            self.action_space = spaces.Box(
                low=-1.7, high=1.7, shape=(self.ac_config.action_size,),
                dtype=np.float32)
        elif action_mode == 'trigon':
            low = np.array([1.0, 3.0, -50, -10, -0.8, -0.8, -0.1, -0.1])
            high = np.array([3.0, 5.0, 50, 10, 0.8, 0.8, 0.1, 0.1])
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
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
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return self._extract_obs(obs), reward, terminate, {}

    def close(self):
        self.env.shutdown()
