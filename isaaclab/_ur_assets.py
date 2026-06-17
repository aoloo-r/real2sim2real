"""UR5e + Robotiq Hand-E configuration for Isaac Lab.

Loads the URDF-converted unified USD produced by
`scripts/convert_ur5e_handE_urdf.py`, which converts the local URDF at
`local_assets/ur5e_handE.urdf` into a single-articulation USD via Isaac Lab's
UrdfConverter. One articulation root, 9 bodies (7 UR5e + 2 Hand-E fingers),
8 driven joints (6 revolute arm + 2 prismatic finger sliders).

If the unified USD is missing, run:
    ./isaaclab.sh -p scripts/convert_ur5e_handE_urdf.py
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# --- Real-world constants -----------------------------------------------------
PEDESTAL_HEIGHT_M = 0.7
PEDESTAL_RADIUS_M = 0.07

# URDF-converted unified USD: 8 joints (6 arm + 2 prismatic fingers) that the
# Isaac Lab articulation parser keeps. The composed/flat USDs drop the gripper
# sliders at load (PhysX merge quirk), exposing only the 6 arm joints.
UR5E_HANDE_USD_PATH = "/home/aoloo/IsaacLab/local_assets/ur5e_handE_urdf/ur5e_handE.usd"

# Hand-E finger joints (URDF-converted names) + 50 mm parallel stroke (~25 mm/finger).
HAND_E_FINGER_JOINTS = ("finger_joint_left", "finger_joint_right")
HAND_E_FINGER_OPEN_M = 0.0
HAND_E_FINGER_CLOSED_M = 0.025

# Gripper-DOWN home pose (tool0 points at the table, like the real arm's working
# pose). Verified by render (/tmp/ur5e_pose_down_B.png): UR5e bent, 2F-85 fingers
# pointing straight down. The previous pose pointed the gripper forward.
UR5E_HOME_JOINTS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.0472,   # -60 deg
    "elbow_joint": 1.5708,            # +90 deg
    "wrist_1_joint": -2.0944,         # -120 deg
    "wrist_2_joint": -1.5708,         # -90 deg
    "wrist_3_joint": 0.0,
}


# --- Helpers ------------------------------------------------------------------
def spawn_pedestal(prim_path: str, height: float = PEDESTAL_HEIGHT_M,
                   radius: float = PEDESTAL_RADIUS_M):
    """Static cylinder placeholder for the column the UR5e is mounted on."""
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.55)),
    )
    cfg.func(prim_path, cfg, translation=(0.0, 0.0, height / 2.0))


# --- UR5e + Hand-E unified config --------------------------------------------
if not os.path.isfile(UR5E_HANDE_USD_PATH):
    raise FileNotFoundError(
        f"Unified UR5e+HandE USD not found at {UR5E_HANDE_USD_PATH}. "
        f"Run: ./isaaclab.sh -p scripts/convert_ur5e_handE_urdf.py"
    )

UR5e_HAND_E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5E_HANDE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **UR5E_HOME_JOINTS,
            "finger_joint_left": 0.0,
            "finger_joint_right": 0.0,
        },
        pos=(0.0, 0.0, PEDESTAL_HEIGHT_M),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"], stiffness=1320.0, damping=72.66,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"], stiffness=600.0, damping=34.64,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"], stiffness=216.0, damping=29.39,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint_.*"],
            effort_limit_sim=20.0,
            velocity_limit_sim=0.15,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
)


# --- UR5e + Robotiq 2F-85 (CLEAN bundled NVIDIA asset) -----------------------
# The hand-composed/URDF UR5e+Hand-E USDs either drop the gripper joints at load
# or spawn with disconnected links. The bundled ur5e.usd with the Robotiq_2f_85
# gripper VARIANT loads as one clean, connected articulation (verified by
# ur5e_spawn_test.py: 6 arm joints + a working finger_joint drive). This is the
# robot used for sim-first physics-grasp validation. Both Hand-E and 2F-85 are
# parallel grippers, so the wall-straddle grasp transfers.
ROBOTIQ_2F85_OPEN = 0.0      # finger_joint (rad): 0 = fully open
ROBOTIQ_2F85_CLOSED = 0.70   # ~ firm parallel close (max ~0.8)

UR5E_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
        variants={"Gripper": "Robotiq_2f_85"},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **UR5E_HOME_JOINTS,
            "finger_joint": 0.0,
            ".*_inner_finger_joint": 0.0,
            ".*_inner_finger_knuckle_joint": 0.0,
            ".*_outer_.*_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"], stiffness=1320.0, damping=72.66),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"], stiffness=600.0, damping=34.64),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"], stiffness=216.0, damping=29.39),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=40.0, velocity_limit_sim=1.0,
            stiffness=80.0, damping=2.0),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=1.0, velocity_limit_sim=1.0,
            stiffness=0.2, damping=0.001),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
            effort_limit_sim=1.0, velocity_limit_sim=1.0,
            stiffness=0.0, damping=0.0),
    },
)
