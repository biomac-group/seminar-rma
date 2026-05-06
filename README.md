# Assignment 2: Inverse Kinematics and Second-Order Forward Kinematics

This branch is the continuation of assignment 1.

Assignment 1 is assumed to be completed already. In particular, you should already have a working implementation of:
- `utils/filter.py`
- `utils/segment.py`
- `fk/forward_kinematics.py`

If your assignment-1 solution lives in a different branch or repository, pull or copy it into this branch before starting. The new tasks in this assignment build directly on that implementation.

## Setup

You need:

1. A Python environment created with conda, venv, or virtualenv.
2. The required packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest`.
3. One autodiff library of your choice, for example `torch`, `jax`, or `tensorflow`.

## Scope of this assignment

This assignment introduces two new topics:

1. Inverse kinematics.
2. Second-order forward kinematics.

The output of this assignment will later be used for inverse dynamics, but inverse dynamics is not part of this assignment yet.

## Files you will work on

You should mainly work in:
- `ik/inverse_kinematics.py`
- `fk/forward_kinematics_second_order.py`
- `task03.ipynb`
- `task04.ipynb`

The file `fk/forward_kinematics.py` from assignment 1 remains the foundation for both tasks.

## Notebook order

1. Open `introduction.ipynb` if you want a quick refresher on the data and model structure.
2. Open `task03.ipynb` and work through the inverse-kinematics introduction and implementation.
3. Open `task04.ipynb` and extend your forward kinematics to second order.

## Task 3: Inverse Kinematics

The goal of inverse kinematics is to estimate generalized coordinates from experimental marker trajectories.

You will:
- formulate a marker-based objective function,
- compute gradients with an autodiff framework,
- optimize one frame first,
- then solve the full trial frame by frame.

The notebook starts with a small optimization warm-up using the Rosenbrock function and LBFGS. After that, you transfer the same idea to the movement-analysis problem.

In `ik/inverse_kinematics.py`, you are expected to implement templates for:
- `ik_target_function`
- `compute_ik_gradient`
- `ik_solver`

Your implementation should:
- reuse your forward kinematics from assignment 1,
- accept `numpy.ndarray` inputs,
- handle missing markers robustly,
- and optionally penalize joint-limit violations.

## Task 4: Second-Order Forward Kinematics

The goal of second-order forward kinematics is to propagate not only positions and orientations, but also velocities and accelerations through the kinematic tree.

For each segment, you should compute global:
- position,
- orientation,
- linear velocity,
- linear acceleration,
- angular velocity,
- angular acceleration.

This task starts from `q`, `qdot`, and `qddot` and extends your recursive forward-kinematics solution from assignment 1.

In `fk/forward_kinematics_second_order.py`, you are expected to implement templates for:
- `forward_kinematics_second_order`
- `get_marker_kinematics`

The expected structure is the same as in assignment 1: traverse the tree recursively, but now propagate first- and second-order kinematic quantities as well.

## Deliverables

At the end of this assignment, you should be able to show:

1. A working inverse-kinematics pipeline in `task03.ipynb`.
2. A working second-order forward-kinematics pipeline in `task04.ipynb`.
3. Clean, readable implementations in `ik/inverse_kinematics.py` and `fk/forward_kinematics_second_order.py`.
4. Clear plots and short explanations in the notebooks.

## Notes

- Keep your code compatible with `numpy` arrays even if you use an autodiff library internally.
- Reuse as much as possible from assignment 1 instead of rewriting the full forward-kinematics logic from scratch.
- Focus on clear equations and a consistent recursive implementation. That will matter again in the inverse-dynamics assignment.


