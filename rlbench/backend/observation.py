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
                 task_low_dim_state: np.ndarray):
        self.head_camera_rgb = head_camera_rgb
        self.head_camera_depth = head_camera_depth
        self.head_camera_mask = head_camera_mask
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.robot_pos = robot_pos
        self.target_pos = target_pos
        self.target_angle = target_angle
        self.robot_angle = robot_angle
        self.task_low_dim_state = task_low_dim_state

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

    def get_flatten_data(self) -> np.ndarray:
        flatten_data = []
        for data in [self.head_camera_rgb, self.head_camera_depth, self.head_camera_mask,
                     self.joint_velocities, self.joint_positions, self.joint_forces,
                     self.robot_pos, self.target_pos, self.target_angle, self.robot_angle, self.task_low_dim_state]:
            if data is not None:
                flatten_data.append(data)
        flatten_data = np.concatenate(flatten_data)
        return flatten_data

    @property
    def shape(self) -> tuple:
        flatten_data = self.get_flatten_data()
        return flatten_data.shape

