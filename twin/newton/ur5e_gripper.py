"""Build a UR5e + Robotiq 2F-85 articulation in Newton from mujoco_menagerie.

The menagerie ships the arm (universal_robots_ur5e/ur5e.xml) and the gripper
(robotiq_2f85/2f85.xml) separately; this bolts the gripper onto the arm's flange
at its `attachment_site` and returns the wrist body index + a gripper-down home
config. Closest match to the real UR5e + Robotiq Hand-E.

NOTE: the 2F-85 is a CLOSED-LOOP 4-bar linkage (equality constraints). The fold's
cloth coupling uses SolverFeatherstone (open trees only), so for that use the
gripper joints should be frozen (fixed open pose) — the fold's PIN grasp holds the
cloth, so the gripper is a visual + fingertip reference, not a load-bearing actuator.
"""
from __future__ import annotations

import glob

import numpy as np
import warp as wp

import newton
from newton._src.utils.download_assets import download_git_folder, MENAGERIE_URL, MENAGERIE_REF

UR5E_HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]   # menagerie 'home' key (gripper down)


def _menagerie(folder, fname):
    return glob.glob(str(download_git_folder(MENAGERIE_URL, folder, ref=MENAGERIE_REF)) + f"/{fname}")[0]


def add_ur5e_gripper(builder: newton.ModelBuilder, xform: wp.transform, home=True):
    """Add a UR5e + Robotiq 2F-85 to `builder` at `xform`. Returns (wrist3_body, first_arm_dof)."""
    ur = _menagerie("universal_robots_ur5e", "ur5e.xml")
    gr = _menagerie("robotiq_2f85", "2f85.xml")
    dof0 = builder.joint_dof_count
    builder.add_mjcf(ur, xform=xform, floating=False, collapse_fixed_joints=True)
    wrist3 = builder.body_count - 1
    q = np.array([1.0, 0.0, 0.0, -1.0]); q /= np.linalg.norm(q)   # attachment_site quat (mjcf wxyz -1,1,0,0)
    att = wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat(q[0], q[1], q[2], q[3]))
    builder.add_mjcf(gr, parent_body=wrist3, xform=att, floating=False, collapse_fixed_joints=True)
    if home:
        for i in range(6):
            builder.joint_q[dof0 + i] = UR5E_HOME[i]
    return wrist3, dof0
