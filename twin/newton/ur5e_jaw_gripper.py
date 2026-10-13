"""Build a UR5e + a SIMPLE 2-JAW PARALLEL gripper in Newton.

Unlike the Robotiq 2F-85 (a closed-loop 4-bar linkage that SolverFeatherstone can't
actuate — see ur5e_gripper.py), this gripper is two box fingers on PRISMATIC (slide)
joints that Newton CAN physically drive. The jaws actually close on the cloth and hold
it by CONTACT + FRICTION (high geom friction), the way the shipped cloth_franka example
clamps fabric — NO software pin. The fingers visibly pinch the cloth and lift it.

Built as a tiny MJCF attached with add_mjcf(parent_body=wrist3) — the SAME proven path
the 2F-85 uses (handles articulation grouping; a raw add_link+add_joint can't extend the
UR5e's already-closed articulation). Tool axis = wrist_3 +y. Fingers separate along x and
slide along x to open/close.

Returns (wrist3, dof0, n_arm, pinch_local, grip_dofs, open_pos, close_pos): grip_dofs are
the two finger slide dof indices; open_pos/close_pos are their target slide positions (cm).
"""
from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import warp as wp

import newton
from newton._src.utils.download_assets import download_git_folder, MENAGERIE_URL, MENAGERIE_REF

UR5E_HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# 2-jaw gripper MJCF (metres; scale=100 -> cm). jaw_base is fixed to the flange (collapsed);
# the two fingers slide along x to close. High geom friction so the closed jaws hold cloth.
_JAW_MJCF = """<mujoco model="jaw2">
  <compiler angle="radian"/>
  <worldbody>
    <body name="jaw_base" pos="0 0.02 0">
      <geom type="box" size="0.045 0.012 0.02" pos="0 0 0" rgba="0.13 0.13 0.16 1"/>
      <body name="jaw_L" pos="-{ho} 0.012 0">
        <joint name="jaw_L" type="slide" axis="1 0 0" range="-0.005 {trav}" damping="5"/>
        <geom type="box" size="0.005 {hl} 0.016" pos="0 {hl} 0" rgba="0.13 0.13 0.16 1" friction="1.7 0.05 0.01"/>
      </body>
      <body name="jaw_R" pos="{ho} 0.012 0">
        <joint name="jaw_R" type="slide" axis="-1 0 0" range="-0.005 {trav}" damping="5"/>
        <geom type="box" size="0.005 {hl} 0.016" pos="0 {hl} 0" rgba="0.13 0.13 0.16 1" friction="1.7 0.05 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="jaw_L" joint="jaw_L" kp="80"/>
    <position name="jaw_R" joint="jaw_R" kp="80"/>
  </actuator>
</mujoco>
"""


def _menagerie(folder, fname):
    return glob.glob(str(download_git_folder(MENAGERIE_URL, folder, ref=MENAGERIE_REF)) + f"/{fname}")[0]


def add_ur5e_jaw_gripper(builder: newton.ModelBuilder, xform: wp.transform, scale=1.0, home=True,
                         open_gap=0.08, finger_len=0.09):
    """Add a UR5e + 2-jaw parallel gripper at `xform` (scale=100 for cm sims).
    open_gap/finger_len in METRES (mjcf units). Returns
    (wrist3, dof0, n_arm, pinch_local, grip_dofs, open_pos, close_pos)."""
    ur = _menagerie("universal_robots_ur5e", "ur5e.xml")
    dof0 = builder.joint_dof_count
    builder.add_mjcf(ur, xform=xform, scale=scale, floating=False, collapse_fixed_joints=True)
    wrist3 = builder.body_count - 1
    n_arm = builder.joint_dof_count - dof0                        # 6 for UR5e
    if home:
        for i in range(6):
            builder.joint_q[dof0 + i] = UR5E_HOME[i]

    ho = open_gap / 2.0                                           # half-open (m)
    hl = finger_len / 2.0
    mjcf = _JAW_MJCF.format(ho=ho, hl=hl, trav=ho + 0.005)
    tf = tempfile.NamedTemporaryFile("w", suffix="_jaw2.xml", delete=False)
    tf.write(mjcf); tf.close()
    g0 = builder.joint_dof_count
    # attach at the flange, fingers along wrist_3 +y (identity keeps mjcf +y -> flange +y)
    att = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    builder.add_mjcf(str(tf.name), parent_body=wrist3, xform=att, scale=scale,
                     floating=False, collapse_fixed_joints=True)
    os.unlink(tf.name)
    grip_dofs = list(range(g0, builder.joint_dof_count))          # the 2 slide dofs
    open_pos = 0.0
    close_pos = (ho - 0.006) * scale                              # inner faces ~ few mm apart (pinch)
    pinch_local = wp.vec3(0.0, (0.02 + finger_len * 0.82) * scale, 0.0)
    return wrist3, dof0, n_arm, pinch_local, grip_dofs, open_pos, close_pos
