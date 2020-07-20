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

    # def set_trigon_model_params(self, alpha_h=None, alpha_v=None, theta_h=None,
                                # theta_v=None, omega=None, delta=None, beta_h=None, beta_v=None):
    def set_trigon_model_params(self, alpha_h=None, alpha_v=None, theta_h=None, theta_v=None):
    # def set_trigon_model_params(self, theta_h=None, theta_v=None):
    #     print('sign = ', sign)
    #     print('alpha_h = ', alpha_h)
    #     print('alpha_v = ', alpha_v)
    #     print('theta_h = ', theta_h)
    #     print('theta_v = ', theta_v)
    #     print('beta = ', beta)
        num_joints = self.get_joint_count()
        joints = np.zeros((num_joints,))
        alpha_h = 0.2 * alpha_h + 0.5
        alpha_v = 0.2 * alpha_v + 0.5
        theta_h = theta_h + 4
        theta_v = theta_h + 4
        # print("------------------------------------------------")
        # print('sign = ', sign)
        # print('alpha_h = ', alpha_h)
        # print('alpha_v = ', alpha_v)
        # print('theta_h = ', theta_h)
        # print('theta_v = ', theta_v)
        # print('beta = ', beta)
        # print("================================================")
        for i in range(self.get_joint_count()):
            if i % 2 == 0:
                # joints[i] = alpha_h * np.sin(sign * theta * self.clk + (i/2) * omega + delta) + beta_h
                # joints[i] = alpha_h * np.sin(theta_h * self.clk + (i / 2) * 30)
                # print("alpha_h = ", alpha_h)
                joints[i] = (alpha_h - i/2*(alpha_h/self.get_joint_count()/5)) * np.sin(theta_h * self.clk + (i / 2) * 30)

                # joints[i] = alpha_h * np.sin(theta_h * self.clk + (i / 2) * 30)
            else:
                # joints[i] = alpha_v * np.sin(sign * theta * self.clk + (i/2) * omega) + beta_v
                # joints[i] = alpha_v * np.sin(4 * self.clk + (i / 2) * 30)
                # print("alpha_v = ", alpha_v)
                joints[i] = (alpha_v - i/2*(alpha_v/self.get_joint_count()/5)) * np.sin(theta_v * self.clk + (i / 2) * 30)

                # joints[i] = alpha_v * np.sin(theta_v * self.clk + (i / 2) * 30)
        pre_joints = self.get_joint_positions()
        for i in range(num_joints):
            if joints[i] - pre_joints[i] > np.deg2rad(8):
                joints[i] = pre_joints[i] + np.deg2rad(8)
            elif joints[i] - pre_joints[i] < -np.deg2rad(8):
                joints[i] = pre_joints[i] - np.deg2rad(8)
        self.set_joint_target_positions(joints)
        self.clk += 0.05

    def init_state(self):
        self.clk = 0.0