from typing import List
import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.robot import Robot
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, OutOfBoundCondition


class ReachTarget(Task):
    def __init__(self, pyrep: PyRep, robot: Robot):
        super().__init__(pyrep, robot)
        self.target = None
        self.success_sensor = None
        self._epi_len = 1.5e3
        self._last_tar_pos = None
        self._last_rob_pos = None
        self._max_dis = 2

        # goal information supplement
        # planar coordinate (x,y) of the target
        self.endgoal_dim = 2
        self.subgoal_dim = 2
        # robot's coordinate (x,y) and positions of all joints
        # self.subgoal_dim = self.robot.robot_body.get_joint_count() + 2
        # self.subgoal_bounds = np.concatenate((np.array([-10, 10]), np.array([-10, 10]),
        #                                       [-np.pi/2, np.pi/2] * self.robot.robot_body.get_joint_count()))
        self.subgoal_bounds = np.concatenate((np.array([0, 3]), np.array([-2, 2])))
        self.subgoal_bounds = np.reshape(self.subgoal_bounds, (-1,2))
        self.subgoal_bounds_symmetric = [(self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
                                         for i in range(self.subgoal_bounds.shape[0])]
        self.subgoal_offset = [self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]
                               for i in range(self.subgoal_bounds.shape[0])]
        self.endgoal_thresholds = np.array([0.1, 0.1])
        # self.subgoal_thresholds = np.concatenate((np.array([0.3, 0.3]),
        #                                           np.array([np.deg2rad(10)]*self.robot.robot_body.get_joint_count())))
        self.subgoal_thresholds = np.array([0.1, 0.1])


    def init_task(self) -> None:
        self.target = Shape('target')
        self.success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.robot_body.get_snake_head(), self.success_sensor)]
        )
        self.register_fail_conditions(
            [OutOfBoundCondition(self.robot.robot_body.get_snake_head(), self.target, self._max_dis)]
        )


    def init_episode(self, index: int) -> List[str]:
        # set color of the target
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)

        # set random position of the target
        rand_x = np.random.rand()
        rand_y = np.random.rand()
        self.target.set_position([rand_x, rand_y, 0.04])
        self.success_sensor.set_position([rand_x, rand_y, 0.04])
        # self.target.set_position([0.2, -0.2, 0.04])
        # self.success_sensor.set_position([0.2, -0.2, 0.04])

        self._last_tar_pos = np.array(self.target.get_position())
        self._last_rob_pos = np.array(self.robot.get_position())

        self.robot.robot_body.init_state()

        return 'reach the %s sphere target' % color_name

    def variation_count(self) -> int:
        return len(colors)

    def get_reward(self) -> int:
        pass

    def get_goal(self) -> list:
        return self.get_target_pos()[:2]

    def get_short_term_reward(self) -> int:
        cur_tar_pos = np.array(self.target.get_position())
        cur_age_pos = np.array(self.robot.get_position())
        cur_dis = np.sqrt(np.sum((cur_tar_pos[:2] - cur_age_pos[:2]) ** 2))
        assert self._last_tar_pos is not None, "The last position of the target is not attached"
        assert self._last_rob_pos is not None, "The last position of the robot is not attached"
        last_dis = np.sqrt(np.sum((self._last_tar_pos[:2] - self._last_rob_pos[:2]) ** 2))
        dis_del = last_dis - cur_dis
        self._last_rob_pos = cur_age_pos
        self._last_tar_pos = cur_tar_pos

        cur_age_head_pos = np.array(self.robot.robot_body.get_snake_head_pos())
        cur_age_tail_pos = np.array(self.robot.robot_body.get_snake_tail_pos())
        cur_age_ht_pos = cur_age_head_pos + cur_age_tail_pos
        cur_block_dis = np.sum(np.abs(cur_tar_pos[:2] - cur_age_ht_pos[:2]))
        # print('cur_block_dis = ', 0.1 / np.power(cur_block_dis, 1.8))
        # print('dis_del = ', dis_del)
        return dis_del + 0.01 / np.power(cur_block_dis, 1.8)

    def get_long_term_reward(self, timeout) -> int:
        success, _ = self.success()
        fail, _ = self.failure()
        if success:
            return 1
        elif fail and timeout:
            return -1
        else:
            return 0

    def project_state_to_subgoal(self):
        robot_position = np.array(self.robot.robot_body.get_snake_head_pos())[:2]
        # joint_position = np.array(self.robot.robot_body.get_joint_positions())
        # subgoal = np.concatenate((robot_position, joint_position))
        subgoal = robot_position
        return subgoal

    def project_state_to_endgoal(self):
        robot_position = np.array(self.robot.robot_body.get_snake_head_pos())[:2]
        endgoal = robot_position
        return endgoal

    @property
    def episode_len(self) -> int:
        return self._epi_len

    def get_target_pos(self) -> List[float]:
        return self.target.get_position()

