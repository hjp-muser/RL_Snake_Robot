from typing import List
import numpy as np
from collections import deque

from numpy.core._multiarray_umath import ndarray
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.robot import Robot
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, OutOfBoundCondition
from rlbench.backend.observation import Observation


class ReachTarget(Task):
    def __init__(self, pyrep: PyRep, robot: Robot):
        super().__init__(pyrep, robot)
        self.target = None
        self.success_sensor = None
        self._epi_len = 1500
        self._tar_pos = None
        self._last_rob_pos_queue = None
        self.init_tar_rob_dis = None
        self._max_dis = 2
        self.dis_del_reward_max = None
        self.dis_del_reward_min = None
        self.angle_reward_min = None
        self.angle_reward_max = None

        # goal information supplement
        # planar coordinate (x,y) of the target
        self.endgoal_dim = 2
        self.subgoal_dim = 2
        # robot's coordinate (x,y) and positions of all joints
        # self.subgoal_dim = self.robot.robot_body.get_joint_count() + 2
        # self.subgoal_bounds = np.concatenate((np.array([-10, 10]), np.array([-10, 10]),
        #                                       [-np.pi/2, np.pi/2] * self.robot.robot_body.get_joint_count()))
        self.subgoal_bounds = np.concatenate((np.array([0, 3]), np.array([-2, 2])))
        self.subgoal_bounds = np.reshape(self.subgoal_bounds, (-1, 2))
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
        # rand_x = np.random.rand()
        # rand_y = np.random.rand()

        rand_x = np.random.uniform(0.0, 1.0)
        arc_angle = 70
        radius = 1.6
        radius_cos = radius * np.cos(np.deg2rad(arc_angle/2))
        if rand_x < radius - radius_cos:
            ymax = np.sqrt(radius**2 - (rand_x-radius)**2)
        else:
            ymax = (radius - rand_x) * np.tan(np.deg2rad(arc_angle/2))
        rand_y = np.random.uniform(-ymax, ymax)

        self.target.set_position([rand_x, rand_y, 0.04])
        self.success_sensor.set_position([rand_x, rand_y, 0.04])
        # self.target.set_position([0.4, -1.0, 0.04])
        # self.success_sensor.set_position([0.4, -1.0, 0.04])

        self._tar_pos = np.array(self.target.get_position())
        self._last_rob_pos_queue = deque(maxlen=50)
        self._last_rob_pos_queue.append(np.array(self.robot.robot_body.get_snake_head_pos()))
        cur_head_pos = np.array(self.robot.robot_body.get_snake_head_pos())
        self.init_tar_rob_dis = np.sqrt(np.sum((self._tar_pos[:2] - cur_head_pos[:2]) ** 2))

        self.robot.robot_body.init_state()

        return 'reach the %s sphere target' % color_name

    def variation_count(self) -> int:
        return len(colors)

    def get_reward(self) -> int:
        pass

    def get_goal(self) -> list:
        return self.get_target_pos()[:2]

    def get_short_term_reward(self) -> int:
        cur_head_pos = np.array(self.robot.robot_body.get_snake_head_pos())
        cur_tail_pos = np.array(self.robot.robot_body.get_snake_tail_pos())
        cur_dis = np.sqrt(np.sum((self._tar_pos[:2] - cur_head_pos[:2]) ** 2))
        body_len = np.sqrt(np.sum((cur_tail_pos[:2] - cur_head_pos[:2]) ** 2))
        dis_reward = - cur_dis / self.init_tar_rob_dis
        dis_reward = np.clip(dis_reward, -1, 0)

        if len(self._last_rob_pos_queue) == 50:
            last_rob_pos = self._last_rob_pos_queue[0]
            last_dis = np.sqrt(np.sum((self._tar_pos[:2] - last_rob_pos[:2]) ** 2))
            dis_del_reward = last_dis - cur_dis
        else:
            dis_del_reward = 0
        self._last_rob_pos_queue.append(cur_head_pos)
        if self.dis_del_reward_min is None:
            self.dis_del_reward_min = dis_del_reward
            self.dis_del_reward_max = dis_del_reward
        elif self.dis_del_reward_min > dis_del_reward:
            self.dis_del_reward_min = dis_del_reward
        elif self.dis_del_reward_max < dis_del_reward:
            self.dis_del_reward_max = dis_del_reward
        if self.dis_del_reward_min != self.dis_del_reward_max:
            dis_del_reward = (dis_del_reward - self.dis_del_reward_min) / (self.dis_del_reward_max - self.dis_del_reward_min) - 1

        k1 = (cur_head_pos[1] - self._tar_pos[1]) / (cur_head_pos[0] - self._tar_pos[0] + 1e-9)
        angle_reward = -np.clip(np.abs(np.arctan(k1)), 0, 1)

        k2 = (cur_tail_pos[1] - cur_head_pos[1]) / (cur_tail_pos[0] - cur_head_pos[0] + 1e-9)
        posture_reward = -np.clip(np.abs(np.arctan(k2)), 0, 1)

        cos_angle = [(self._tar_pos[0] - cur_head_pos[0]) * (cur_tail_pos[0] - cur_head_pos[0]) +
                     (self._tar_pos[1] - cur_head_pos[1]) * (cur_tail_pos[1] - cur_head_pos[1])] / (cur_dis * body_len)
        angle_reward2 = np.abs(np.arccos(cos_angle)) - 1
        # if self.angle_reward_min is None:
        #     self.angle_reward_min = angle_reward
        #     self.angle_reward_max = angle_reward
        # elif self.angle_reward_min > angle_reward:
        #     self.angle_reward_min = angle_reward
        # elif self.angle_reward_max < angle_reward:
        #     self.angle_reward_max = angle_reward
        # if self.angle_reward_min != self.angle_reward_max:
        #     angle_reward = (angle_reward - self.angle_reward_min) / (self.angle_reward_max - self.angle_reward_min)
        # print(dis_del_reward, dis_reward, angle_reward)
        w1 = 1 - dis_reward / (dis_reward + angle_reward + posture_reward)
        w2 = 1 - angle_reward / (dis_reward + angle_reward + posture_reward)
        w3 = 1 - posture_reward / (dis_reward + angle_reward + posture_reward)
        return dis_reward * w1 + angle_reward * w2 + posture_reward * w3
        # return dis_reward + 0.1 * angle_reward + 0.1 * posture_reward
        # return dis_reward + 0.1 * angle_reward2

    def get_long_term_reward(self, timeout) -> int:
        success, _ = self.success()
        fail, _ = self.failure()
        if success:
            return 200
        elif fail:
            return -2
        elif timeout:
            cur_head_pos = np.array(self.robot.robot_body.get_snake_head_pos())
            cur_dis = np.sqrt(np.sum((self._tar_pos[:2] - cur_head_pos[:2]) ** 2))
            timeout_reward = 2 * (1 - cur_dis / self.init_tar_rob_dis) - 1
            timeout_reward = np.clip(timeout_reward, -1, 1)
            return timeout_reward * 100
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

    def get_target_pos(self) -> List[float]:
        return self.target.get_position()

    def get_target_angle(self) -> List[float]:
        robot_pos = np.array(self.robot.robot_body.get_snake_head_pos())[:2]
        target_pos = self._tar_pos[:2]
        k = (robot_pos[1] - target_pos[1]) / (robot_pos[0] - target_pos[0] + 1e-8)
        angle = np.arctan(k)
        # print("angle = ", np.rad2deg(angle))
        return [angle]

    def get_robot_angle(self) -> List[float]:
        robot_angle = self.robot.robot_body.get_snake_angle()
        return [robot_angle]

    def compute_reward(self, achieved_goal, desired_goal) -> np.ndarray:
        goal_dis = np.sqrt(np.sum((achieved_goal[:, :2] - desired_goal[:, :2]) ** 2, axis=1))
        reward = - goal_dis / self.init_tar_rob_dis
        reward = np.clip(reward, -1, 0)
        return reward

    @property
    def episode_len(self) -> int:
        return self._epi_len

    @staticmethod
    def decorate_observation(observation: Observation) -> Observation:
        """Can be used for tasks that want to modify the observations.

        Usually not used. Perhpas can be used to model

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """

        return observation

