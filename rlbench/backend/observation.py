import numpy as np


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
                 task_low_dim_state: np.ndarray):
        self.head_camera_rgb = head_camera_rgb
        self.head_camera_depth = head_camera_depth
        self.head_camera_mask = head_camera_mask
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.robot_pos = robot_pos
        self.target_pos = target_pos
        self.task_low_dim_state = task_low_dim_state

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional observations.

        :return: 1D array of observations.
        """
        low_dim_data = []
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.robot_pos, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data)
