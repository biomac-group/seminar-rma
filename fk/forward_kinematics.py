import numpy as np

def forward_kinematics(q, key, kintree):
    """Compute the forward kinematics for the given joint angles and kinematic tree.

    Args:
        q (ndarray): Generalized coordinates (joint angles) of shape (N,).
        key (list): List of joint names corresponding to the angles in 'q'.
        kintree (dict): Kinematic tree as in `model/kintree.py`.

    Returns:
        joints (dict): 3D positions of each joint after applying the forward kinematics, (J: 3).
        markers (dict): 3D positions of each marker after applying the forward kinematics, (M: 3).
    """

    ## ToDo: Your implementation here, hint: use recursion to traverse the kinematic tree
    # Check when an axis is written with [-1, 0, 0] instead of [1, 0, 0] - then you need to invert the angle
    pass

def get_connections(kintree, joints):
    """Get connections between joints for visualization.

    Args:
        kintree (dict): Kinematic tree as in `model/kintree.py`.
        joints (dict): 3D positions of each joint after applying the forward kinematics, (J: 3).

    Returns:
        connections (list): List of tuples representing connection lines between joints.
        e.g [([x1, y1, z1], [x2, y2, z2]), ([x3, y3, z3], [x4, y4, z4]), ...] for lines between joint1 and joint2, joint3 and joint4.
    """
    pass

