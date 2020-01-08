from enum import Enum

JOINTS = 16


class ArmActionMode(Enum):

    # Absolute arm joint velocities
    ABS_JOINT_VELOCITY = (0, JOINTS,)

    # Change in arm joint velocities
    DELTA_JOINT_VELOCITY = (1, JOINTS,)

    # Absolute arm joint positions/angles (in radians)
    ABS_JOINT_POSITION = (2, JOINTS,)

    # Change in arm joint positions/angles (in radians)
    DELTA_JOINT_POSITION = (3, JOINTS,)

    # Absolute arm joint forces/torques
    ABS_JOINT_TORQUE = (4, JOINTS,)

    # Change in arm joint forces/torques
    DELTA_JOINT_TORQUE = (5, JOINTS,)

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
                 arm: ArmActionMode=ArmActionMode.ABS_JOINT_VELOCITY,
                 gripper: GripperActionMode=GripperActionMode.OPEN_AMOUNT):
        self.arm = arm
        self.gripper = gripper
        self.action_size = self.arm.action_size + self.gripper.action_size
