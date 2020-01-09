from typing import List
from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.observation import Observation
from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.observation_config import ObservationConfig, CameraConfig
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
        self._cam_wrist_mask = None
        self._head_camera.set_camera_properties()

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._init_task_state = task.get_state()
        self._active_task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self._active_task is not None:
            self._robot.gripper.release()
            self._active_task.unload()
        self._active_task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self._active_task.init_task()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self._active_task.init_episode(index)
            try:
                if (randomly_place and
                        not self._active_task.is_static_workspace()):
                    self._place_task()
                self._active_task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self._active_task.cleanup_()
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self._pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        arm, gripper = self._initial_robot_state
        self._pyrep.set_configuration_tree(arm)
        self._pyrep.set_configuration_tree(gripper)
        self._robot.arm.set_joint_positions(self._start_arm_joint_pos)
        self._robot.arm.set_joint_target_velocities(
            [0] * len(self._robot.arm.joints))
        self._robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos)
        self._robot.gripper.set_joint_target_velocities(
            [0] * len(self._robot.gripper.joints))

        if self._active_task is not None and self._has_init_task:
            self._active_task.cleanup_()
            self._active_task.restore_state(self._init_task_state)
        [self._pyrep.step_ui() for _ in range(20)]

    def get_observation(self) -> Observation:
        tip = self._robot.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self._robot.arm.get_joint_forces()
            vels = self._robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self._robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        wc_ob = self._obs_config.wrist_camera

        obs = Observation(
            left_shoulder_rgb=(
                lsc_ob.rgb_noise.apply(
                    self._cam_over_shoulder_left.capture_rgb())
                if lsc_ob.rgb else None),
            left_shoulder_depth=(
                lsc_ob.depth_noise.apply(
                    self._cam_over_shoulder_left.capture_depth())
                if lsc_ob.depth else None),
            right_shoulder_rgb=(
                rsc_ob.rgb_noise.apply(
                    self._cam_over_shoulder_right.capture_rgb())
                if rsc_ob.rgb else None),
            right_shoulder_depth=(
                rsc_ob.depth_noise.apply(
                    self._cam_over_shoulder_right.capture_depth())
                if rsc_ob.depth else None),
            wrist_rgb=(
                wc_ob.rgb_noise.apply(self._cam_wrist.capture_rgb())
                if wc_ob.rgb else None),
            wrist_depth=(
                wc_ob.depth_noise.apply(self._cam_wrist.capture_depth())
                if wc_ob.depth else None),

            left_shoulder_mask=(
                self._cam_shoulder_left_mask.capture_rgb()
                if lsc_ob.mask else None),
            right_shoulder_mask=(
                self._cam_shoulder_right_mask.capture_rgb()
                if rsc_ob.mask else None),
            wrist_mask=(
                self._cam_wrist_mask.capture_rgb()
                if wc_ob.mask else None),

            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self._robot.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self._robot.arm.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=joint_forces,
            gripper_open_amount=(
                1.0 if self._robot.gripper.get_open_amount()[0] > 0.9 else 0.0),
            gripper_pose=(
                np.array(tip.get_pose())
                if self._obs_config.gripper_pose else None),
            gripper_touch_forces=ee_forces_flat,
            gripper_joint_positions=np.array(
                self._robot.gripper.get_joint_positions()),
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

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self._active_task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self._active_task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self._active_task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)
