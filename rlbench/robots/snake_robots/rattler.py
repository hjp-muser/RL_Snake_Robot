from rlbench.robots.snake_robots.snake_robot import SnakeRobot
import numpy as np


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
        self.clk = 0

    def set_trigon_model_params(self, alpha_h, alpha_v, theta_h,
                                theta_v=None, omega=None, delta=None, beta_h=None, beta_v=None):
        # print('sign = ', sign)
        # print('alpha_h = ', alpha_h)
        # print('alpha_v = ', alpha_v)
        # print('theta_h = ', theta_h)
        # print('theta_v = ', theta_v)
        # print('omega = ', omega)
        # print('delta = ', delta)
        # print('beta_h = ', beta_h)
        # print('beta_v = ', beta_v)
        joints = np.zeros((self.get_joint_count(),))
        # sign = -1 if round(sign) == 0 else 1
        # sign = 1 if sign > 0.5 else -1
        for i in range(self.get_joint_count()):
            if i % 2 == 0:
                # joints[i] = alpha_h * np.sin(sign * theta * self.clk + (i/2) * omega + delta) + beta_h
                joints[i] = alpha_h * np.sin(theta_h * self.clk + (i / 2) * 30)
                # joints[i] = alpha_h * np.sin(sign * theta_h * self.clk + (i / 2) * 30)
            else:
                # joints[i] = alpha_v * np.sin(sign * theta * self.clk + (i/2) * omega) + beta_v
                joints[i] = alpha_v * np.sin(4 * self.clk + (i / 2) * 30)
                # joints[i] = alpha_v * np.sin(sign * 4 * self.clk + (i / 2) * 30)
        self.set_joint_target_positions(joints)
        self.clk += 0.05

    def init_state(self):
        self.clk = 0.0