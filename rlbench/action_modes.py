from enum import Enum

SNAKE_ROBOT_JOINTS = 17


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
                 robot_action_mode: SnakeRobotActionMode = SnakeRobotActionMode.ABS_JOINT_VELOCITY,
                 aux_equip_action_mode: CameraActionMode = CameraActionMode.OPEN_AMOUNT):
        if not (isinstance(robot_action_mode, SnakeRobotActionMode) and
                isinstance(aux_equip_action_mode, CameraActionMode)):
            raise NotImplementedError("Not implement for the other action mode except snake robot action mode.")
        self.robot_action_mode = robot_action_mode
        self.aux_equip_action_mode = aux_equip_action_mode
        self.action_size = self.robot_action_mode.action_size
