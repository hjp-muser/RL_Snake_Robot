import os
import re
from os.path import dirname, abspath, join
from typing import List, Tuple

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object

from rlbench.backend.conditions import Condition
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot

TASKS_PATH = join(dirname(abspath(__file__)), '../tasks')


class Task(object):

    def __init__(self, pyrep: PyRep, robot: Robot):
        """Constructor.

        :param pyrep: Instance of PyRep.
        :param robot: Instance of Robot.
        """
        self.pyrep = pyrep
        self.name = self.get_name()
        self.robot = robot
        self._waypoints = None
        self._success_conditions = []
        self._fail_conditions = []
        self._graspable_objects = []
        self._base_object = None
        self._waypoint_additional_inits = {}
        self._waypoint_abilities_start = {}
        self._waypoint_abilities_end = {}
        self._waypoints_should_repeat = lambda: False

        # goal information supplement
        # planar coordinate (x,y) of the target
        self.endgoal_dim = None
        # robot's coordinate (x,y) and positions of all joints
        self.subgoal_dim = None
        self.subgoal_bounds = None
        self.subgoal_bounds_symmetric = None
        self.subgoal_offset = None
        self.endgoal_thresholds = None
        self.subgoal_thresholds = None

    ########################
    # Overriding functions #
    ########################

    def init_task(self) -> None:
        """Initialises the task. Called only once when task loaded.

        Here we can grab references to objects in the task and store them
        as member variables to be used in init_episode. Here we also usually
        set success conditions for the task as well as register what objects
        can be grasped.
        """
        raise NotImplementedError(
            "'init_task' is almost always necessary.")

    def init_episode(self, index: int) -> List[str]:
        """Initialises the episode. Called each time the scene is reset.

        Here we usually define how the task changes across variations. Based on
        this we can change the task descriptions that are returned.

        :param index: The variation index.
        :return: A list of strings describing the task.
        """
        raise NotImplementedError(
            "'init_episode' must be defined and return a list of strings.")

    def variation_count(self) -> int:
        """Number of variations for the task. Can be determined dynamically.

        :return: Number of variations for this task.
        """
        raise NotImplementedError(
            "'variation_count' must be defined and return an int.")

    def get_low_dim_state(self) -> np.ndarray:
        """Gets the pose and various other properties of objects in the task.

        :return: 1D array of low-dimensional task state.
        """
        objs = self.get_base().get_objects_in_tree(
            exclude_base=True, first_generation_only=False)
        state = []
        for obj in objs:
            state.extend(np.array(obj.get_pose()))
            if obj.get_type() == ObjectType.JOINT:
                state.extend([Joint(obj.get_handle()).get_joint_position()])
            elif obj.get_type() == ObjectType.FORCE_SENSOR:
                forces, torques = ForceSensor(obj.get_handle()).read()
                state.extend(forces + torques)
        return np.array(state).flatten()

    def step(self) -> None:
        """Called each time the simulation is stepped. Can usually be left."""
        pass

    def cleanup(self) -> None:
        """Called at the end of the episode. Can usually be left.

        Can be used for complex tasks that spawn many objects.
        """
        pass

    def episode_len(self) -> int:
        """Get the episode length of the specific task"""
        pass

    def get_reward(self) -> int:
        """Get the rewards (duplicate function)"""
        pass

    def get_goal(self) -> list:
        """Get the goal space"""
        pass

    def get_short_term_reward(self) -> int:
        """Get the short-term rewards"""
        pass

    def get_long_term_reward(self, timeout) -> int:
        """Get the long-term rewards"""
        pass

    def project_state_to_subgoal(self) -> list:
        pass

    def project_state_to_endgoal(self) -> list:
        pass

    @staticmethod
    def decorate_observation(observation: Observation) -> Observation:
        """Can be used for tasks that want to modify the observations.

        Usually not used. Perhpas can be used to model

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """
        return observation

    #########################
    # Registering functions #
    #########################

    def register_success_conditions(self, condition: List[Condition]):
        """What conditions need to be met for the task to be a success.

        Note: this replaces any previously registered conditions!

        :param condition: A list of success conditions.
        """
        self._success_conditions = condition

    def register_fail_conditions(self, condition: List[Condition]):
        """What conditions need to be met for the task to be a fail.

        Note: this replaces any previously registered conditions!

        :param condition: A list of fail conditions.
        """
        self._fail_conditions = condition

    def get_name(self) -> str:
        """The name of the task file (without the .py extension).

        :return: The name of the task.
        """
        return re.sub('(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    def should_repeat_waypoints(self):
        return self._waypoints_should_repeat()

    def get_graspable_objects(self):
        return self._graspable_objects

    def success(self):
        all_met = True
        one_terminate = False
        for cond in self._success_conditions:
            met, terminate = cond.condition_met()
            all_met &= met
            one_terminate |= terminate
        return all_met, one_terminate

    def failure(self):
        all_met = True
        one_terminate = False
        for cond in self._fail_conditions:
            met, terminate = cond.condition_met()
            all_met &= met
            one_terminate |= terminate
        return all_met, one_terminate

    def load(self) -> Object:
        if Object.exists(self.get_name()):
            return Dummy(self.get_name())
        ttm_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object

    def unload(self) -> None:
        self.cleanup()
        self._waypoints = None
        self.get_base().remove()
        self.clear_registerings()

    def cleanup_(self) -> None:
        for cond in self._success_conditions:
            cond.reset()
        for cond in self._fail_conditions:
            cond.reset()
        self._waypoints = None
        self.cleanup()

    def clear_registerings(self) -> None:
        self._success_conditions = []
        self._fail_conditions = []
        self._graspable_objects = []
        self._base_object = None
        self._waypoint_additional_inits = {}
        self._waypoint_abilities_start = {}
        self._waypoint_abilities_end = {}

    def get_base(self) -> Dummy:
        self._base_object = Dummy(self.get_name())
        return self._base_object

    def get_state(self) -> Tuple[bytes, int]:
        objs = self.get_base().get_objects_in_tree(exclude_base=False)
        return self.get_base().get_configuration_tree(), len(objs)

    def restore_state(self, state: Tuple[bytes, int]) -> None:
        objs = self.get_base().get_objects_in_tree(exclude_base=False)
        if len(objs) != state[1]:
            raise RuntimeError(
                'Expected to be resetting %d objects, but there were %d.' %
                (state[1], len(objs)))
        self.pyrep.set_configuration_tree(state[0])