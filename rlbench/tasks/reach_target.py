from typing import List, Tuple
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

    def get_epi_len(self) -> int:
        return self._epi_len
