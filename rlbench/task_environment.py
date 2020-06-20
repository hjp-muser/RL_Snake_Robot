import numpy as np
from pyrep import PyRep
from rlbench.backend.exceptions import BoundaryError, WaypointError
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.backend.robot import Robot
import logging
from typing import List
from rlbench.backend.observation import Observation
from rlbench.action_config import ActionConfig, SnakeRobotActionConfig
from rlbench.observation_config import ObservationConfig

_TORQUE_MAX_VEL = 9999
_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 10


class InvalidActionError(Exception):
    pass


class TaskEnvironmentError(Exception):
    pass


class TaskEnvironment(object):

    def __init__(self, pyrep: PyRep, robot: Robot, scene: Scene, task: Task,
                 action_config: ActionConfig, dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False):
        self._pyrep = pyrep
        self._robot = robot
        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_config = action_config
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._reset_called = False
        self._time_step = 0

        self.goal_achieved = False
        self.endgoal_dim = self._task.endgoal_dim
        # robot's coordinate (x,y) and positions of all joints
        self.subgoal_dim = self._task.subgoal_dim
        self.subgoal_bounds = self._task.subgoal_bounds
        self.subgoal_bounds_symmetric = self._task.subgoal_bounds_symmetric
        self.subgoal_offset = self._task.subgoal_offset
        self.endgoal_thresholds = self._task.endgoal_thresholds
        self.subgoal_thresholds = self._task.subgoal_thresholds
        self.get_goal = self._task.get_goal
        self.episode_len = self._task.episode_len
        self.project_state_to_subgoal = self._task.project_state_to_subgoal
        self.project_state_to_endgoal = self._task.project_state_to_endgoal

        self._scene.load(self._task)
        self._pyrep.start()

    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def reset(self) -> (List[str], Observation):
        print('Resetting task: %s' % self._task.get_name())
        logging.info('Resetting task: %s' % self._task.get_name())
        self._time_step = 0
        self.goal_achieved = False
        self._scene.reset()
        try:
            desc = self._scene.init_episode(self._variation_number)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        ctr_loop = self._robot.robot_body.joints[0].is_control_loop_enabled()
        locked = self._robot.robot_body.joints[0].is_motor_locked_at_zero_velocity()
        self._robot.robot_body.set_control_loop_enabled(False)
        self._robot.robot_body.set_motor_locked_at_zero_velocity(True)

        self._reset_called = True

        self._robot.robot_body.set_control_loop_enabled(ctr_loop)
        self._robot.robot_body.set_motor_locked_at_zero_velocity(locked)

        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    @staticmethod
    def _assert_action_space(action, expected_shape):
        if np.shape(action) != expected_shape:
            raise RuntimeError(
                'Expected the action shape to be: %s, but was shape: %s' % (
                    str(expected_shape), str(np.shape(action))))

    def _torque_action(self, action):
        self._robot.robot_body.set_joint_target_velocities(
            [(_TORQUE_MAX_VEL if t < 0 else -_TORQUE_MAX_VEL)
             for t in action])
        self._robot.robot_body.set_joint_forces(np.abs(action))

    def step(self, action) -> (Observation, int, bool):
        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")

        # robot_action = np.array(action[:-1]) # action should contain 1 extra value for camera open close state
        # auxiliary_action = action[-1]
        robot_action = np.array(action)

        if self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_VELOCITY:

            self._assert_action_space(robot_action, (len(self._robot.robot_body.joints),))
            self._robot.robot_body.set_joint_target_velocities(robot_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_VELOCITY:

            self._assert_action_space(robot_action, (len(self._robot.robot_body.joints),))
            cur = np.array(self._robot.robot_body.get_joint_velocities())
            self._robot.robot_body.set_joint_target_velocities(cur + robot_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_POSITION:

            self._assert_action_space(robot_action, (len(self._robot.robot_body.joints),))
            self._robot.robot_body.set_joint_target_positions(robot_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_POSITION:

            self._assert_action_space(robot_action, (len(self._robot.robot_body.joints),))
            cur = np.array(self._robot.robot_body.get_joint_positions())
            self._robot.robot_body.set_joint_target_positions(cur + robot_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_TORQUE:

            self._assert_action_space(robot_action, (len(self._robot.robot_body.joints),))
            self._torque_action(robot_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_TORQUE:

            cur = np.array(self._robot.robot_body.get_joint_forces())
            new_action = cur + robot_action
            self._torque_action(new_action)

        elif self._action_config.robot_action_config == SnakeRobotActionConfig.TRIGON_MODEL_PARAM:
            self._robot.robot_body.set_trigon_model_params(*robot_action)

        else:
            raise RuntimeError('Unrecognised action mode.')

        # self._robot.auxiliary_equip.set_camera_state(auxiliary_action)

        self._scene.step()
        self._time_step += 1
        success, success_terminate = self._task.success()
        fail, fail_terminate = self._task.failure()
        timeout = self._time_step >= self._task.episode_len
        done = success_terminate | timeout
        long_term_reward = self._task.get_long_term_reward(timeout)
        if self._time_step % 10 == 0 and self._time_step != 0:
            short_term_reward = self._task.get_short_term_reward()
        else:
            short_term_reward = 0

        if success:
            # self._time_step = 0
            self.goal_achieved = True
            print('mission success!')
        elif timeout:
            # self._time_step = 0
            if fail:
                print('mission fail!')
            else:
                print('mission timeout!')

        # if multi_scale_reward:
        #     reward = [short_term_reward, long_term_reward]
        # else:
        #     reward = short_term_reward + long_term_reward * 100
        reward = short_term_reward + long_term_reward * 100

        return self._scene.get_observation(), reward, done
