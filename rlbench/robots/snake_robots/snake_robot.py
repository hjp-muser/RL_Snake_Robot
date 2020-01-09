from rlbench.robots.robot_component import RobotComponent

from pyrep.backend import sim


class SnakeRobot(RobotComponent):
    """Base class representing a snake-liked robot with movement support.
    """

    def __init__(self, count: int, name: str, num_joints: int,
                 base_name: str = None,
                 max_velocity=1.0, max_acceleration=4.0, max_jerk=1000):
        """
        Count is used for when we have multiple copies of arms
        :param max_jerk: The second derivative of the velocity.
        """
        joint_names = ['%s_joint%d' % (name, i + 1) for i in range(num_joints)]
        super().__init__(count, name, joint_names, base_name)

        # Used for movement
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

        # handles
        suffix = '' if count == 1 else '#%d' % (count - 2)
        self._collision_collection = sim.simGetCollectionHandle(
            '%s_collection%s' % (name, suffix))
