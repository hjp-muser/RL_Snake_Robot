from typing import List, Tuple
import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.robot import Robot
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class ReachTarget(Task):
    def __init__(self, pyrep: PyRep, robot: Robot):
        super().__init__(pyrep, robot)
        self.target = None
        self.success_sensor = None
        self._epi_len = 3e2

    def init_task(self) -> None:
        self.target = Shape('target')
        self.success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.robot_body.get_snake_head(), self.success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)

        return 'reach the %s sphere target' % color_name

    def variation_count(self) -> int:
        return len(colors)

    def get_reward(self) -> int:
        tar_pos = np.array(self.target.get_position())
        agent_pos = np.array(self.robot.get_position())
        # dis = np.sqrt(np.sum((tar_pos[:2]-agent_pos[:2])**2))
        dis = np.sum(np.abs(tar_pos[:2]-agent_pos[:2]))
        return 1/dis

    def get_epi_len(self) -> int:
        return self._epi_len
