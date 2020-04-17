from typing import List
from pyrep import PyRep
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.backend.task import Task
from rlbench.backend.robot import Robot
import numpy as np

STEPS_BEFORE_EPISODE_START = 10


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self, pyrep: PyRep, robot: Robot,
                 obs_config=ObservationConfig()):
        self._pyrep = pyrep
        self._snake_robot = robot.robot_body
        self._head_camera = robot.auxiliary_equip
        self._obs_config = obs_config
        self._active_task = None
        self._init_task_state = None
        self._start_joint_pos = robot.robot_body.get_joint_positions()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._initial_robot_state = (self._snake_robot.get_configuration_tree(),
                                     self._head_camera.get_camera_state())

        # Set camera properties from observation config
        self._head_camera.set_camera_properties(self._obs_config.head_camera)

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        self._init_task_state = task.get_state()
        self._active_task = task
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self._active_task is not None:
            self._active_task.unload()
        self._active_task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self._active_task.init_task()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int) -> List[str]:
        """Calls the task init_episode.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        descriptions = self._active_task.init_episode(index)

        # Let objects come to rest
        [self._pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        snake_robot_init_state, head_camera_init_state = self._initial_robot_state
        self._pyrep.set_configuration_tree(snake_robot_init_state)
        self._head_camera.set_camera_state(head_camera_init_state)
        self._snake_robot.set_joint_positions(self._start_joint_pos)
        self._snake_robot.set_joint_target_velocities(
            [0] * len(self._snake_robot.joints))

        if self._active_task is not None and self._has_init_task:
            self._active_task.cleanup_()
            self._active_task.restore_state(self._init_task_state)
        [self._pyrep.step_ui() for _ in range(20)]

    def get_observation(self) -> Observation:
        snake_head = self._snake_robot.get_snake_head()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self._snake_robot.get_joint_forces()
            vels = self._snake_robot.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        hc_ob = self._obs_config.head_camera

        obs = Observation(
            head_camera_rgb=(
                hc_ob.rgb_noise.apply(
                    self._head_camera.capture_rgb())
                if hc_ob.rgb else None),
            head_camera_depth=(
                hc_ob.depth_noise.apply(
                    self._head_camera.capture_depth())
                if hc_ob.depth else None),
            head_camera_mask=(
                self._head_camera.mask().capture_rgb()
                if hc_ob.mask else None),
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self._snake_robot.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self._snake_robot.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=joint_forces,
            snake_head_pose=(
                np.array(snake_head.get_pose())
                if self._obs_config.snake_head_pose else None),
            task_low_dim_state=(
                self._active_task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None))
        obs = self._active_task.decorate_observation(obs)
        return obs

    def step(self):
        self._pyrep.step()
        self._active_task.step()

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config