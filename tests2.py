import numpy as np

from fk.forward_kinematics_second_order import forward_kinematics_second_order
from ik.inverse_kinematics import compute_ik_gradient, ik_target_function


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _as_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(value)


def _simple_marker_fk(q, key, kintree):
    return {}, {"marker_test": [q[0], q[1], q[2]]}


def _root_tree_with_marker(marker_offset):
    return {
        "pelvis": {
            "parent": None,
            "joint_location": np.zeros(3, dtype=float),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "joints": {},
            "markers": {"marker_test": np.asarray(marker_offset, dtype=float)},
            "children": {},
        }
    }


def _slider_tree():
    return {
        "pelvis": {
            "parent": None,
            "joint_location": np.zeros(3, dtype=float),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "joints": {
                "pelvis_tx": {
                    "type": "slider",
                    "axis": np.array([1.0, 0.0, 0.0], dtype=float),
                    "limits": None,
                    "pos": np.zeros(3, dtype=float),
                }
            },
            "markers": {"marker_test": np.array([0.5, 0.0, 0.0], dtype=float)},
            "children": {},
        }
    }


def _hinge_tree():
    return {
        "pelvis": {
            "parent": None,
            "joint_location": np.zeros(3, dtype=float),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "joints": {
                "pelvis_tilt": {
                    "type": "hinge",
                    "axis": np.array([0.0, 0.0, 1.0], dtype=float),
                    "limits": None,
                    "pos": np.zeros(3, dtype=float),
                }
            },
            "markers": {"marker_test": np.array([1.0, 0.0, 0.0], dtype=float)},
            "children": {},
        }
    }


def test_ik_target_function_is_zero_for_perfect_marker_match():
    q = np.array([1.0, -2.0, 3.0], dtype=float)
    marker_positions = {
        "marker_test_x": 1.0,
        "marker_test_y": -2.0,
        "marker_test_z": 3.0,
    }
    bounds = (np.full(3, -10.0), np.full(3, 10.0))

    loss = ik_target_function(_simple_marker_fk, q, ["q0", "q1", "q2"], {}, marker_positions, bounds)
    assert _as_float(loss) == 0.0


def test_compute_ik_gradient_points_toward_marker_target():
    q = np.array([0.0, 0.0, 0.0], dtype=float)
    marker_positions = {
        "marker_test_x": 1.0,
        "marker_test_y": -2.0,
        "marker_test_z": 3.0,
    }
    bounds = (np.full(3, -10.0), np.full(3, 10.0))

    gradient = compute_ik_gradient(_simple_marker_fk, q, ["q0", "q1", "q2"], {}, marker_positions, bounds)
    gradient = _as_numpy(gradient)

    assert gradient.shape == q.shape
    assert np.all(np.isfinite(gradient))
    assert gradient[0] < 0.0
    assert gradient[1] > 0.0
    assert gradient[2] < 0.0


def test_second_order_forward_kinematics_returns_expected_structure():
    kintree = _root_tree_with_marker([1.0, 0.0, 0.0])
    segments, markers = forward_kinematics_second_order(
        q=np.array([], dtype=float),
        qdot=np.array([], dtype=float),
        qddot=np.array([], dtype=float),
        key=[],
        kintree=kintree,
    )

    assert isinstance(segments, dict)
    assert isinstance(markers, dict)
    assert "pelvis" in segments
    assert "marker_test" in markers

    segment_state = segments["pelvis"]
    marker_state = markers["marker_test"]

    expected_segment_keys = {
        "position",
        "orientation",
        "linear_velocity",
        "linear_acceleration",
        "angular_velocity",
        "angular_acceleration",
    }
    expected_marker_keys = {"position", "velocity", "acceleration"}

    assert expected_segment_keys.issubset(segment_state.keys())
    assert expected_marker_keys.issubset(marker_state.keys())


def test_second_order_slider_joint_propagates_linear_kinematics():
    segments, markers = forward_kinematics_second_order(
        q=np.array([1.0], dtype=float),
        qdot=np.array([2.0], dtype=float),
        qddot=np.array([3.0], dtype=float),
        key=["pelvis_tx"],
        kintree=_slider_tree(),
    )

    pelvis = segments["pelvis"]
    marker = markers["marker_test"]

    np.testing.assert_allclose(_as_numpy(pelvis["position"]), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(pelvis["linear_velocity"]), [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(pelvis["linear_acceleration"]), [3.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(pelvis["angular_velocity"]), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(pelvis["angular_acceleration"]), [0.0, 0.0, 0.0], atol=1e-6)

    np.testing.assert_allclose(_as_numpy(marker["position"]), [1.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(marker["velocity"]), [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(marker["acceleration"]), [3.0, 0.0, 0.0], atol=1e-6)


def test_second_order_hinge_joint_propagates_angular_kinematics():
    segments, _ = forward_kinematics_second_order(
        q=np.array([np.pi / 2], dtype=float),
        qdot=np.array([1.5], dtype=float),
        qddot=np.array([-0.5], dtype=float),
        key=["pelvis_tilt"],
        kintree=_hinge_tree(),
    )

    pelvis = segments["pelvis"]
    np.testing.assert_allclose(_as_numpy(pelvis["angular_velocity"]), [0.0, 0.0, 1.5], atol=1e-6)
    np.testing.assert_allclose(_as_numpy(pelvis["angular_acceleration"]), [0.0, 0.0, -0.5], atol=1e-6)
