from enum import Enum

SNAKE_ROBOT_JOINTS = 16
TRIGON_MODEL_PARAMS = 3


class SnakeRobotActionConfig(Enum):

    # Absolute snake robot joint velocities
    ABS_JOINT_VELOCITY = (0, SNAKE_ROBOT_JOINTS,)

    # Change in snake robot joint velocities
    DELTA_JOINT_VELOCITY = (1, SNAKE_ROBOT_JOINTS,)

    # Absolute snake robot joint positions/angles (in radians)
    ABS_JOINT_POSITION = (2, SNAKE_ROBOT_JOINTS,)

    # Change in snake robot joint positions/angles (in radians)
    DELTA_JOINT_POSITION = (3, SNAKE_ROBOT_JOINTS,)

    # Absolute snake robot joint forces/torques
    ABS_JOINT_TORQUE = (4, SNAKE_ROBOT_JOINTS,)

    # Change snake robot arm joint forces/torques
    DELTA_JOINT_TORQUE = (5, SNAKE_ROBOT_JOINTS,)

    # Trigonometric function model parameters
    TRIGON_MODEL_PARAM = (6, TRIGON_MODEL_PARAMS,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


class CameraActionConfig(Enum):
    # The open amount (0 >= x <= 1) of the camera. 0 is close, 1 is open.
    OPEN_AMOUNT = (0, 1,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


EE_SIZE = 7
ARM_JOINTS = 7


class ArmActionConfig(Enum):

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


class GripperActionConfig(Enum):

    # The open amount (0 >= x <= 1) of the gripper. 0 is close, 1 is open.
    OPEN_AMOUNT = (0, 1,)

    def __init__(self, id, action_size):
        self.id = id
        self.action_size = action_size


class ActionConfig(object):

    def __init__(self,
                 robot_action_config: SnakeRobotActionConfig = SnakeRobotActionConfig.ABS_JOINT_VELOCITY,
                 aux_equip_action_config: CameraActionConfig = CameraActionConfig.OPEN_AMOUNT):
        if not (isinstance(robot_action_config, SnakeRobotActionConfig) and
                isinstance(aux_equip_action_config, CameraActionConfig)):
            raise NotImplementedError("Not implement for the other action configuration except snake robot action configuration.")
        self.robot_action_config = robot_action_config
        self.aux_equip_action_config = aux_equip_action_config
        self.action_size = self.robot_action_config.action_size
