from rlbench.robots.snake_robots.snake_robot import SnakeRobot


class Rattler(SnakeRobot):

    def __init__(self, count: int = 1, num_joints: int = 16):
        """
        :param count: the rattler model ID in a scene
        :param num_joints: the number of joints of a rattler
        """
        if count == 0:
            raise ValueError("Robot model number can't be 0.")
        if num_joints == 0:
            raise ValueError("Joints number can't be 0.")
        super().__init__(count, 'rattler', num_joints)
