import numpy as np


def forward_kinematics_second_order(q, qdot, qddot, key, kintree):
    """Compute second-order forward kinematics for the given generalized coordinates.

    The goal of this task is to propagate segment kinematics through the tree and compute
    global quantities for each segment:
    - position
    - orientation
    - linear velocity
    - linear acceleration
    - angular velocity
    - angular acceleration

    Args:
        q (ndarray): Generalized coordinates of shape (N,).
        qdot (ndarray): Generalized velocities of shape (N,).
        qddot (ndarray): Generalized accelerations of shape (N,).
        key (list): List of generalized coordinate names corresponding to q, qdot, qddot.
        kintree (dict): Kinematic tree as in model/kintree.py.

    Returns:
        segments (dict): Dictionary with one entry per segment containing the propagated
            global kinematic quantities.
        markers (dict): Dictionary with one entry per marker containing at least global
            position, velocity and acceleration.
    """

    # ToDo:
    # 1. Reuse your recursive traversal from assignment 1.
    # 2. For each joint, map q, qdot and qddot to the corresponding local motion.
    # 3. Propagate linear and angular quantities from parent to child.
    # 4. Store a consistent dictionary of segment states and marker states.
    pass


def get_marker_kinematics(segments, kintree):
    """Compute marker kinematics from segment kinematics.

    Args:
        segments (dict): Output of forward_kinematics_second_order.
        kintree (dict): Kinematic tree as in model/kintree.py.

    Returns:
        markers (dict): Marker kinematics in global coordinates.
    """

    # ToDo: rotate each marker offset into the global frame and propagate
    # position, velocity and acceleration from its parent segment.
    pass