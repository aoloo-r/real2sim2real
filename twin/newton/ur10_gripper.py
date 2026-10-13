"""Build a UR10e + Robotiq 2F-85 articulation in Newton (mirror of ur5e_gripper.py).

Same recipe as the UR5e: the menagerie ships the arm (universal_robots_ur10e/ur10e.xml)
and the gripper (robotiq_2f85/2f85.xml) separately; this bolts the 2F-85 onto the UR10e
flange (the UR10e shares the same ISO-9409 flange as the UR5e, so the attachment is
identical) and returns the wrist body index + a gripper-down home config + the fingertip
pinch point. The fold's PIN grasp holds the cloth at the pinch point; the gripper fingers
are posed kinematically (they visibly close on the fabric).
"""
from __future__ import annotations

import glob

import numpy as np
import warp as wp

import newton
from newton._src.utils.download_assets import download_git_folder, MENAGERIE_URL, MENAGERIE_REF

# UR home key that points the flange (and gripper) straight down at the table — same joint
# structure as the UR5e, so the same home config applies.
UR10E_HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


def _menagerie(folder, fname):
    return glob.glob(str(download_git_folder(MENAGERIE_URL, folder, ref=MENAGERIE_REF)) + f"/{fname}")[0]


def add_ur10_gripper(builder: newton.ModelBuilder, xform: wp.transform, scale=1.0, home=True):
    """Add a UR10e + Robotiq 2F-85 to `builder` at `xform`, scaled by `scale` (100 for cm-scale sims).
    Returns (wrist3_body, first_arm_dof, n_arm_dof, pinch_offset_local) matching add_ur5e_gripper."""
    ur = _menagerie("universal_robots_ur10e", "ur10e.xml")
    gr = _menagerie("robotiq_2f85", "2f85.xml")
    dof0 = builder.joint_dof_count
    builder.add_mjcf(ur, xform=xform, scale=scale, floating=False, collapse_fixed_joints=True)
    wrist3 = builder.body_count - 1
    n_arm = builder.joint_dof_count - dof0                        # 6 for UR10e
    q = np.array([1.0, 0.0, 0.0, -1.0]); q /= np.linalg.norm(q)   # attachment_site quat (mjcf wxyz -1,1,0,0)
    att = wp.transform(wp.vec3(0.0, 0.1 * scale, 0.0), wp.quat(q[0], q[1], q[2], q[3]))
    builder.add_mjcf(gr, parent_body=wrist3, xform=att, scale=scale, floating=False, collapse_fixed_joints=True)
    if home:
        for i in range(6):
            builder.joint_q[dof0 + i] = UR10E_HOME[i]
    # pinch point = where the CLOSED 2F-85 finger pads meet, in the wrist_3 frame (same gripper as UR5e)
    pinch_local = wp.vec3(0.0, 0.223 * scale, 0.0)
    return wrist3, dof0, n_arm, pinch_local
