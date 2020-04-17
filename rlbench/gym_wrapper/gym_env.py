import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, SnakeRobotActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human']}

    def __init__(self, task_class, observation_mode='state'):
        self._observation_mode = observation_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(False)
            obs_config.set_camera_rgb(True)
        elif observation_mode == 'both':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        action_mode = ActionMode(SnakeRobotActionMode.ABS_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.7, high=1.7, shape=(action_mode.action_size,),
            dtype=np.float32)

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
            pass
        #     # Add the camera to the scene
        #     cam_placeholder = Dummy('cam_cinematic_placeholder')
        #     self._gym_cam = VisionSensor.create([640, 360], position=[2.82, -40.43, 2.79], orientation=[-180, -45, 90], far_clipping_plane=100.0)
        #     self._gym_cam = VisionSensor('rattler_eye')
        #     # self._gym_cam.set_pose(cam_placeholder.get_pose())
        #     self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)

    def reset(self):
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return self._extract_obs(obs), reward, terminate, {}

    def close(self):
        self.env.shutdown()
