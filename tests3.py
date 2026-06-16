import numpy as np
import pandas as pd
import pytest
from model.kintree import get_model_dictionary
from fk.forward_kinematics_second_order import forward_kinematics_second_order
from id.inverse_dynamics import inverse_dynamics



def test_inverse_dynamics_static_multi_body_equilibrium():
    # Set zero state (static posture, zero motion)
    q = np.zeros(9)
    qdot = np.zeros(9)
    qddot = np.zeros(9)

    gravity = np.array([0.0, -9.81, 0.0])
    forces, moments = inverse_dynamics(q, qdot, qddot, grf_row=None, gravity=gravity)

    # Total Mass Calculation from XML/model:
    # pelvis (68.61472966) + 2 * (femur (10.12016662) + tibia (4.705877477) + foot (1.467424159))
    total_mass = 101.201666172
    expected_root_force = np.array([0.0, total_mass * 9.81, 0.0])

    np.testing.assert_allclose(forces["pelvis"], expected_root_force, atol=1e-3)
    np.testing.assert_allclose(moments["pelvis"][2], 2.33397921, atol=1e-3)



def test_inverse_dynamics_gait_cycle():
    q_all = pd.read_csv("data/angles_clean.csv")
    grf_data = pd.read_csv("data/grf.csv")

    key = list(q_all.columns[1:])
    time = q_all["time"].to_numpy()
    q = q_all[key].to_numpy()
    qdot = np.gradient(q, time, axis=0)
    qddot = np.gradient(qdot, time, axis=0)

    # Run for frame 100
    frame = 100
    grf_row = grf_data.iloc[frame]
    forces, moments = inverse_dynamics(q[frame], qdot[frame], qddot[frame], grf_row=grf_row)

    expected_segments = ["pelvis", "femur_r", "tibia_r", "foot_r", "femur_l", "tibia_l", "foot_l"]
    for seg in expected_segments:
        assert seg in forces
        assert seg in moments
        assert forces[seg].shape == (3,)
        assert moments[seg].shape == (3,)
        assert np.all(np.isfinite(forces[seg]))
        assert np.all(np.isfinite(moments[seg]))

    assert np.abs(forces["pelvis"][1]) < 10.0, f"Vertical residual force {forces['pelvis'][1]} N exceeds 10 N"
    assert np.abs(moments["pelvis"][2]) < 40.0, f"Sagittal residual moment {moments['pelvis'][2]} N-m exceeds 40 N-m"

    # Assert that joint moments are within realistic physical boundaries (< 300 N-m)
    for seg in ["femur_r", "tibia_r", "foot_r", "femur_l", "tibia_l", "foot_l"]:
        assert np.abs(moments[seg][2]) < 300.0, f"Moment for {seg} is {moments[seg][2]} N-m, which is physically unrealistic"

