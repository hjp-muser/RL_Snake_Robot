from rlbench.robots.snake_robots.rattler import Rattler
from rlbench.robots.snake_head_cameras.rattler_camera import RattlerCamera


class Robot(object):
    """Simple container for the robot components.
    """

    def __init__(self, robot_body: Rattler, auxiliary_equip: RattlerCamera):
        if not (isinstance(robot_body, Rattler) and isinstance(auxiliary_equip, RattlerCamera)):
            raise NotImplementedError("Not implement the other robot except rattler.")
        self.robot_body = robot_body
        self.auxiliary_equip = auxiliary_equip

    def get_position(self):
        return self.robot_body.get_position()
