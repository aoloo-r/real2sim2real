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


def add_ur5e_gripper(builder: newton.ModelBuilder, xform: wp.transform, scale=1.0, home=True, gripper_armature=0.0,
                     loop_joints=True, finger_cloth_contact=True):
    """Add a UR5e + Robotiq 2F-85 to `builder` at `xform`, scaled by `scale` (use 100 for cm-scale sims).
    Returns (wrist3_body, first_arm_dof, n_arm_dof, pinch_offset_local) where pinch_offset_local is the
    gripper pinch point (fingertip) in the wrist_3 body frame (scaled units)."""
    ur = _menagerie("universal_robots_ur5e", "ur5e.xml")
    gr = _menagerie("robotiq_2f85", "2f85.xml")
    dof0 = builder.joint_dof_count
    builder.add_mjcf(ur, xform=xform, scale=scale, floating=False, collapse_fixed_joints=True)
    wrist3 = builder.body_count - 1
    n_arm = builder.joint_dof_count - dof0                        # 6 for UR5e
    q = np.array([1.0, 0.0, 0.0, -1.0]); q /= np.linalg.norm(q)   # attachment_site quat (mjcf wxyz -1,1,0,0)
    att = wp.transform(wp.vec3(0.0, 0.1 * scale, 0.0), wp.quat(q[0], q[1], q[2], q[3]))
    g0 = builder.joint_dof_count
    s0 = builder.shape_count
    # loop_joints=False drops the 2F-85's <equality><connect> tags. Newton imports them as LOOP
    # joints - synthesized ball joints, the 2 extra coordinates per arm - and Featherstone cannot
    # integrate them on a SECOND articulation: a 0.5 rad/s shoulder command put 8e33 rad/s on the
    # linkage and NaN by substep 16, where the same gripper without loop joints peaks at 0.02 and
    # coords == dofs. The twin drives the 4-bar kinematically and pins the cloth, so the loop
    # closure adds nothing. The single-arm path keeps its validated model (loop_joints=True).
    builder.add_mjcf(gr, parent_body=wrist3, xform=att, scale=scale, floating=False, collapse_fixed_joints=True,
                     **({} if loop_joints else {"skip_equality_constraints": True}))
    if not finger_cloth_contact:
        # The grasp is a KINEMATIC PIN at the pinch point; the fingers are what the user sees.
        # With finger colliders active, the pinned particles sit inside the closed pads' volume,
        # and after the jaws open and the arm lifts, particles still penetrating the finger
        # geometry are carried up by the contact solver - both towel corners rose with the
        # lift on every side-layout run, hooked on the open finger tips. Cloth never needs to
        # collide with the fingers in this grasp model, so it does not.
        try:
            from newton import ShapeFlags
        except ImportError:
            from newton._src.geometry.types import ShapeFlags
        for i in range(s0, builder.shape_count):
            builder.shape_flags[i] = int(builder.shape_flags[i]) & ~int(ShapeFlags.COLLIDE_PARTICLES)
    if gripper_armature > 0.0:
        # Reflected motor inertia on the 2F-85's dofs. The 4-bar's links are near-massless and
        # Newton 1.5's Featherstone leaves the SECOND articulation's joint-space inertia
        # unregularized there: a 0.5 rad/s shoulder command put 3e4 rad/s on the follower joints
        # in one substep and NaN by substep 16, while an identical first arm was fine. Armature
        # on these dofs is the physically meaningful regularizer and makes both arms stable.
        for d in range(g0, builder.joint_dof_count):
            builder.joint_armature[d] = max(float(builder.joint_armature[d]), float(gripper_armature))
    if home:
        for i in range(6):
            builder.joint_q[dof0 + i] = UR5E_HOME[i]
    # PINCH POINT = where the CLOSED finger pads actually meet, measured by FK-ing the closed 2F-85
    # (pad tips at wrist-local (0, 0.223*scale, 0)). The old 0.245 was 2.2cm PAST the tips, so the arm
    # drove a phantom point to the cloth while the real pads pinched air above it. 0.223 lands the pads
    # ON the cloth.
    pinch_local = wp.vec3(0.0, 0.223 * scale, 0.0)
    return wrist3, dof0, n_arm, pinch_local
