import numpy as np
from rlbench.observation_config import ObservationConfig


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 head_camera_rgb: np.ndarray,
                 head_camera_depth: np.ndarray,
                 head_camera_mask: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions:  np.ndarray,
                 joint_forces: np.ndarray,
                 robot_pos:  np.ndarray,
                 target_pos: np.ndarray,
                 target_angle: np.ndarray,
                 robot_angle: np.ndarray,
                 desired_goal: np.ndarray,
                 achieved_goal: np.ndarray):
        self.head_camera_rgb = head_camera_rgb
        self.head_camera_depth = head_camera_depth
        self.head_camera_mask = head_camera_mask
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.robot_pos = robot_pos
        self.target_pos = target_pos
        self.target_angle = target_angle                    # 目标与机器人头部连线的全局角度
        self.robot_angle = robot_angle                      # 机器人全局姿态角度（机器人朝向）
        self.desired_goal = desired_goal
        self.achieved_goal = achieved_goal

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional observations.

        :return: 1D array of observations.
        """
        low_dim_data = []
        for data in [self.joint_velocities, self.joint_positions, self.joint_forces, self.robot_pos,
                     self.target_pos, self.target_angle, self.robot_angle]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data)

    def get_goal_data(self) -> dict:
        goal_data = {'achieved_goal': np.array(self.achieved_goal), 'desired_goal': np.array(self.desired_goal)}
        return goal_data

    def get_goal_dim(self) -> int:
        return np.array(self.achieved_goal).shape[0]

    def get_flatten_data(self) -> np.ndarray:
        flatten_data = []
        for data in [self.head_camera_rgb, self.head_camera_depth, self.head_camera_mask,
                     self.joint_velocities, self.joint_positions, self.joint_forces,
                     self.robot_pos, self.target_pos, self.target_angle, self.robot_angle]:
            if data is not None:
                flatten_data.append(data)
        flatten_data = np.concatenate(flatten_data)
        return flatten_data

    @property
    def shape(self) -> tuple:
        flatten_data = self.get_flatten_data()
        return flatten_data.shape

