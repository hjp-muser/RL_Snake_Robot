from pyrep import PyRep
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.robots.snake_head_cameras.rattler_camera import RattlerCamera
from rlbench.robots.snake_robots.rattler import Rattler
from os.path import exists, dirname, abspath, join
import importlib
from typing import Type
from rlbench.observation_config import ObservationConfig
from rlbench.task_environment import TaskEnvironment
from rlbench.action_config import ActionConfig, SnakeRobotActionConfig


DIR_PATH = dirname(abspath(__file__))

# snake robots type
SUPPORTED_ROBOTS = {
    'rattler': (Rattler, RattlerCamera),
}


class Environment(object):
    """Each environment has a scene."""

    def __init__(self, action_config: ActionConfig, dataset_root: str = '',
                 obs_config=ObservationConfig(), headless=False,
                 static_positions: bool = False, robot_configuration='rattler'):

        self._dataset_root = dataset_root
        self._action_config = action_config
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions
        self._robot_configuration = robot_configuration

        if robot_configuration not in SUPPORTED_ROBOTS.keys():
            raise ValueError('robot_configuration must be one of %s' %
                             str(SUPPORTED_ROBOTS.keys()))

        self._check_dataset_structure()

        self._pyrep = None
        self._robot = None
        self._scene = None
        self._prev_task = None

    def _set_control_action(self):
        self._robot.robot_body.set_control_loop_enabled(True)
        if (self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_VELOCITY or
                self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_VELOCITY):
            self._robot.robot_body.set_control_loop_enabled(False)
            self._robot.robot_body.set_motor_locked_at_zero_velocity(True)
        elif (self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_POSITION or
              self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_POSITION or
              self._action_config.robot_action_config == SnakeRobotActionConfig.TRIGON_MODEL_PARAM):
            self._robot.robot_body.set_control_loop_enabled(True)
        elif (self._action_config.robot_action_config == SnakeRobotActionConfig.ABS_JOINT_TORQUE or
              self._action_config.robot_action_config == SnakeRobotActionConfig.DELTA_JOINT_TORQUE):
            self._robot.robot_body.set_control_loop_enabled(False)
        else:
            raise RuntimeError('Unrecognised action configuration.')

    def _check_dataset_structure(self):
        if len(self._dataset_root) > 0 and not exists(self._dataset_root):
            raise RuntimeError(
                'Data set root does not exists: %s' % self._dataset_root)

    @staticmethod
    def _string_to_task(task_name: str):
        task_name = task_name.replace('.py', '')
        try:
            class_name = ''.join(
                [w[0].upper() + w[1:] for w in task_name.split('_')])
            mod = importlib.import_module("rlbench.tasks.%s" % task_name)
        except Exception as e:
            raise RuntimeError(
                'Tried to interpret %s as a task, but failed. Only valid tasks '
                'should belong in the tasks/ folder' % task_name) from e
        return getattr(mod, class_name)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()
        self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)
        self._pyrep.set_simulation_timestep(0.005)

        snake_robot_class, camera_class = SUPPORTED_ROBOTS[self._robot_configuration]

        # We assume the panda is already loaded in the scene.
        if self._robot_configuration != 'rattler':
            raise NotImplementedError("Not implemented the robot")
        else:
            snake_robot, camera = snake_robot_class(), camera_class()

        self._robot = Robot(snake_robot, camera)
        self._scene = Scene(self._pyrep, self._robot, self._obs_config)
        self._set_control_action()

    def shutdown(self):
        if self._pyrep is not None:
            self._pyrep.shutdown()
        self._pyrep = None

    def get_task(self, task_class: Type[Task]) -> TaskEnvironment:
        # Str comparison because sometimes class comparison doesn't work.
        if self._prev_task is not None:
            self._prev_task.unload()
        task = task_class(self._pyrep, self._robot)
        self._prev_task = task
        return TaskEnvironment(
            self._pyrep, self._robot, self._scene, task,
            self._action_config, self._dataset_root, self._obs_config,
            self._static_positions)

    # def load_env_param(self):
    #     self._scene.load_obs_normalizer()
