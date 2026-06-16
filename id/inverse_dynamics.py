import numpy as np



from utils.gait2d_model import Gait2DModel


def inverse_dynamics(q, qdot, qddot, grf_row=None, gravity=np.array([0.0, -9.81, 0.0])):
    """Compute inverse dynamics forces and moments at each joint using a SymPy-based 2D gait model.

    Args:
        q (ndarray): Joint positions of shape (9,).
        qdot (ndarray): Joint velocities of shape (9,).
        qddot (ndarray): Joint accelerations of shape (9,).
        grf_row (pandas.Series or dict, optional): Row of ground reaction force data.
        gravity (ndarray): Gravity vector (default is [0, -9.81, 0]).

    Returns:
        joint_forces (dict): Dict mapping segment names to 3D joint forces [F_x, F_y, F_z]
        joint_moments (dict): Dict mapping segment names to 3D joint moments [M_x, M_y, M_z]
    """
    model = Gait2DModel.get_instance()

    grf_dict = {}
    if grf_row is not None:
        grf_dict = grf_row if isinstance(grf_row, dict) else grf_row.to_dict()

    g_val = float(np.abs(gravity[1])) if len(gravity) > 1 else 9.81

    # Evaluate mass matrix M and forcing vector F_0
    M_val, F_val = model.evaluate(q, qdot, g_val=g_val, grf_dict=grf_dict)

    # Compute joint moments and residual forces/moments: tau = M * qddot - F_0
    tau = M_val @ qddot - F_val

    # Format the outputs to maintain compatibility with the plotting and test suites.
    # tau[0] and tau[1] represent horizontal and vertical residual forces at the pelvis.
    # tau[2] represents pelvis residual moment.
    # tau[3:9] represent the Z-moments (flexion/extension) for femur_r, tibia_r, foot_r, femur_l, tibia_l, foot_l.
    joint_forces = {
        "pelvis": np.array([tau[0], tau[1], 0.0]),
        "femur_r": np.zeros(3),
        "tibia_r": np.zeros(3),
        "foot_r": np.zeros(3),
        "femur_l": np.zeros(3),
        "tibia_l": np.zeros(3),
        "foot_l": np.zeros(3),
    }

    joint_moments = {
        "pelvis": np.array([0.0, 0.0, tau[2]]),
        "femur_r": np.array([0.0, 0.0, tau[3]]),
        "tibia_r": np.array([0.0, 0.0, tau[4]]),
        "foot_r": np.array([0.0, 0.0, tau[5]]),
        "femur_l": np.array([0.0, 0.0, tau[6]]),
        "tibia_l": np.array([0.0, 0.0, tau[7]]),
        "foot_l": np.array([0.0, 0.0, tau[8]]),
    }

    return joint_forces, joint_moments

