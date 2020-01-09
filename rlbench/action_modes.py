from enum import Enum

from rlbench.robots.snake_head_cameras.rattler_camera import RattlerCamera
from rlbench.robots.snake_robots.rattler import Rattler

SNAKE_ROBOT_JOINTS = 16


class SnakeRobotActionMode(Enum):

    # Absolute arm joint velocities
    ABS_JOINT_VELOCITY = (0, SNAKE_ROBOT_JOINTS,)

    # Change in arm joint velocities
    DELTA_JOINT_VELOCITY = (1, SNAKE_ROBOT_JOINTS,)

    # Absolute arm joint positions/angles (in radians)
    ABS_JOINT_POSITION = (2, SNAKE_ROBOT_JOINTS,)

    # Change in arm joint positions/angles (in radians)
    DELTA_JOINT_POSITION = (3, SNAKE_ROBOT_JOINTS,)

    # Absolute arm joint forces/torques
    ABS_JOINT_TORQUE = (4, SNAKE_ROBOT_JOINTS,)

    # Change in arm joint forces/torques
    DELTA_JOINT_TORQUE = (5, SNAKE_ROBOT_JOINTS,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


class CameraActionMode(Enum):
    # The open amount (0 >= x <= 1) of the camera. 0 is close, 1 is open.
    OPEN_AMOUNT = (0, 1,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


EE_SIZE = 7
ARM_JOINTS = 7


class ArmActionMode(Enum):

    # Absolute arm joint velocities
    ABS_JOINT_VELOCITY = (0, ARM_JOINTS,)

    # Change in arm joint velocities
    DELTA_JOINT_VELOCITY = (1, ARM_JOINTS,)

    # Absolute arm joint positions/angles (in radians)
    ABS_JOINT_POSITION = (2, ARM_JOINTS,)

    # Change in arm joint positions/angles (in radians)
    DELTA_JOINT_POSITION = (3, ARM_JOINTS,)

    # Absolute arm joint forces/torques
    ABS_JOINT_TORQUE = (4, ARM_JOINTS,)

    # Change in arm joint forces/torques
    DELTA_JOINT_TORQUE = (5, ARM_JOINTS,)

    # Absolute end-effector velocity (position (3) and quaternion (4))
    ABS_EE_VELOCITY = (6, EE_SIZE,)

    # Change in end-effector velocity (position (3) and quaternion (4))
    DELTA_EE_VELOCITY = (7, EE_SIZE,)

    # Absolute end-effector pose (position (3) and quaternion (4))
    ABS_EE_POSE = (8, EE_SIZE,)

    # Change in end-effector pose (position (3) and quaternion (4))
    DELTA_EE_POSE = (9, EE_SIZE,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


class GripperActionMode(Enum):

    # The open amount (0 >= x <= 1) of the gripper. 0 is close, 1 is open.
    OPEN_AMOUNT = (0, 1,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


class ActionMode(object):

    def __init__(self,
                 robot_body: SnakeRobotActionMode = SnakeRobotActionMode.ABS_JOINT_VELOCITY,
                 auxiliary_equip: CameraActionMode = CameraActionMode.OPEN_AMOUNT):
        if isinstance(robot_body, Rattler) and isinstance(auxiliary_equip, RattlerCamera):
            raise NotImplementedError("Not implement the other robot except rattler.")
        self.robot_body = robot_body
        self.auxiliary_equip = auxiliary_equip
        self.action_size = self.robot_body.action_size
