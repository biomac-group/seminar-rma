import xml.etree.ElementTree as ET
import numpy as np
import sympy as sp
from sympy.physics.mechanics import (
    ReferenceFrame,
    Point,
    Inertia,
    RigidBody,
    KanesMethod,
    dynamicsymbols,
)
from pathlib import Path

class Gait2DModel:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.build_model()

    def build_model(self):
        from model.kintree import get_model_dictionary
        kintree = get_model_dictionary()
        
        # 1. Collect all joints in depth-first order to ensure coordinate mapping is deterministic
        def collect_joints(node):
            joints_list = []
            for joint_name in node.get('joints', {}):
                joints_list.append(joint_name)
            for child_name, child_node in node.get('children', {}).items():
                joints_list.extend(collect_joints(child_node))
            return joints_list

        root_name = list(kintree.keys())[0]  # typically "pelvis"
        all_joints = collect_joints(kintree[root_name])
        
        # 2. Define coordinates and speeds
        num_coords = len(all_joints)
        q = dynamicsymbols(f'q0:{num_coords}')
        u = dynamicsymbols(f'u0:{num_coords}')
        kd = [q[i].diff() - u[i] for i in range(num_coords)]
        joint_map = {name: (q[i], u[i]) for i, name in enumerate(all_joints)}
        
        # 3. Setup Newtonian Reference Frame and Origin Point
        N = ReferenceFrame('N')
        O = Point('O')
        O.set_vel(N, 0)
        
        rigid_bodies = []
        loads = []
        g = sp.symbols('g')
        
        # Define Ground Reaction Force and COP symbols
        fx_r, fy_r, copx_r = sp.symbols('fx_r fy_r copx_r')
        fx_l, fy_l, copx_l = sp.symbols('fx_l fy_l copx_l')
        
        # 4. Recursively build points, frames, bodies and loads
        def build_branch(name, node, parent_frame, parent_point):
            # joint_location is position of current body origin relative to parent body origin
            pos = node.get('joint_location', np.zeros(3))
            pos_vector = pos[0]*parent_frame.x + pos[1]*parent_frame.y + pos[2]*parent_frame.z
            
            # Accumulate displacements from slider joints (translations)
            trans_expr = 0
            for joint_name, joint_info in node.get('joints', {}).items():
                if joint_info['type'] == 'slider':
                    q_j, _ = joint_map[joint_name]
                    axis = joint_info.get('axis', np.array([1.0, 0.0, 0.0]))
                    trans_expr += q_j * (axis[0]*parent_frame.x + axis[1]*parent_frame.y + axis[2]*parent_frame.z)
            
            joint_origin = Point(name + '_joint_origin')
            joint_origin.set_pos(parent_point, pos_vector + trans_expr)
            
            # Determine velocity
            if parent_frame == N:
                # For root segment, linear velocity is set by its translation speeds (sliders)
                vel_expr = 0
                for joint_name, joint_info in node.get('joints', {}).items():
                    if joint_info['type'] == 'slider':
                        _, u_j = joint_map[joint_name]
                        axis = joint_info.get('axis', np.array([1.0, 0.0, 0.0]))
                        vel_expr += u_j * (axis[0]*parent_frame.x + axis[1]*parent_frame.y + axis[2]*parent_frame.z)
                joint_origin.set_vel(N, vel_expr)
            else:
                # Child bodies propagate velocity via v2pt_theory
                joint_origin.v2pt_theory(parent_point, N, parent_frame)
                
            # Define orientation
            R = ReferenceFrame('R_' + name)
            hinge_joint = None
            for joint_name, joint_info in node.get('joints', {}).items():
                if joint_info['type'] == 'hinge':
                    hinge_joint = (joint_name, joint_info)
                    break
                    
            if hinge_joint is not None:
                joint_name, joint_info = hinge_joint
                q_j, _ = joint_map[joint_name]
                axis = joint_info.get('axis', np.array([0.0, 0.0, 1.0]))
                axis_vector = axis[0]*parent_frame.x + axis[1]*parent_frame.y + axis[2]*parent_frame.z
                R.orient(parent_frame, 'Axis', (q_j, axis_vector))
            else:
                # Orientation matches parent's if no hinge joint exists
                R.orient(parent_frame, 'Axis', (0, parent_frame.z))
                
            # Define COM point of current body
            com = node.get('com', np.zeros(3))
            com_pt = Point('com_' + name)
            com_pt.set_pos(joint_origin, com[0]*R.x + com[1]*R.y + com[2]*R.z)
            com_pt.v2pt_theory(joint_origin, N, R)
            
            # Define RigidBody with mass and inertia properties
            mass = node.get('mass', 0.0)
            ixx, iyy, izz, ixy, ixz, iyz = node.get('inertia', np.zeros(6))
            inertia = Inertia.from_inertia_scalars(com_pt, R, ixx, iyy, izz, ixy, ixz, iyz)
            body_rb = RigidBody(name, com_pt, R, mass, inertia)
            rigid_bodies.append(body_rb)
            
            # Apply gravitational force at COM
            loads.append((com_pt, -mass * g * N.y))
            
            # Apply external GRF to right and left foot segments
            if name == 'foot_r':
                F_ext_r = fx_r * N.x + fy_r * N.y
                M_ext_r = (copx_r * N.x - joint_origin.pos_from(O)).cross(F_ext_r)
                loads.append((joint_origin, F_ext_r))
                loads.append((R, M_ext_r))
            elif name == 'foot_l':
                F_ext_l = fx_l * N.x + fy_l * N.y
                M_ext_l = (copx_l * N.x - joint_origin.pos_from(O)).cross(F_ext_l)
                loads.append((joint_origin, F_ext_l))
                loads.append((R, M_ext_l))
                
            # Recurse for children
            for child_name, child_node in node.get('children', {}).items():
                build_branch(child_name, child_node, R, joint_origin)

        # Start dynamic recursive building from the root node
        build_branch(root_name, kintree[root_name], N, O)
        
        # 5. Formulate dynamic equations of motion using Kane's Method
        km = KanesMethod(N, q_ind=q, u_ind=u, kd_eqs=kd)
        km.kanes_equations(rigid_bodies, loads)
        
        M_sym = km.mass_matrix
        forcing_sym = km.forcing
        
        # Lambdify for fast numerical evaluation
        args = list(q) + list(u) + [g, fx_r, fy_r, copx_r, fx_l, fy_l, copx_l]
        self._M_fn = sp.lambdify(args, M_sym, modules='numpy')
        self._forcing_fn = sp.lambdify(args, forcing_sym, modules='numpy')

    def evaluate(self, q_val, u_val, g_val=9.81, grf_dict=None):
        if grf_dict is None:
            grf_dict = {}
        
        fx_r_val = grf_dict.get("force_r_x", 0.0)
        fy_r_val = grf_dict.get("force_r_y", 0.0)
        copx_r_val = grf_dict.get("cop_r_x", 0.0)
        
        fx_l_val = grf_dict.get("force_l_x", 0.0)
        fy_l_val = grf_dict.get("force_l_y", 0.0)
        copx_l_val = grf_dict.get("cop_l_x", 0.0)
        
        eval_args = (
            list(q_val) +
            list(u_val) +
            [
                g_val,
                fx_r_val,
                fy_r_val,
                copx_r_val,
                fx_l_val,
                fy_l_val,
                copx_l_val
            ]
        )
        
        M_val = self._M_fn(*eval_args)
        F_val = self._forcing_fn(*eval_args).squeeze()
        return M_val, F_val
