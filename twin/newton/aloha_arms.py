"""Build the bimanual ALOHA 2 (two ViperX 300 6-DOF arms) in Newton from mujoco_menagerie.

The menagerie ships both arms in ONE MJCF (aloha/aloha.xml): bases at x = -/+46.9cm, y = -1.9cm,
z = +2cm above the table top, the right arm yawed 180deg. Newton loads it as 16 bodies / 16 dofs:
per arm 6 arm joints (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate) then 2
finger joints. Unlike the UR5e path there is no separate gripper model to attach.

Everything below was measured by FK on this model at scale 100 (cm), not copied from the XML:

  PINCH_LOCAL   (11.62, 0, -0.08)cm in gripper_link = where the two finger PADS meet (the point
                that actually holds cloth). The MJCF's own "gripper" site sits at 13cm, between
                the OPEN jaw tips, which is 1.4cm past the fabric.
  HOME_DOWN     a tool-straight-down pose over the table, solved to 0.00cm / 0.0deg.
  reach         at table height each arm covers its own ~50cm of table with a straight-down
                approach and the two overlap in a ~20cm band about x=0 - so a 30-36cm garment
                centred between the arms has each side in its own arm's workspace.
"""
from __future__ import annotations

import glob

import numpy as np
import warp as wp

import newton
from newton._src.utils.download_assets import download_git_folder, MENAGERIE_URL, MENAGERIE_REF

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
FINGER_JOINTS = ("left_finger", "right_finger")
SIDES = ("left", "right")

# menagerie 'neutral_pose' keyframe (both arms), arm joints only
NEUTRAL = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
# solved here: pinch at (-/+15, -20, table+0)cm with the approach axis straight down
HOME_DOWN = {
    "left": (-0.518, 0.317, 0.448, 0.0, 0.806, 1.873),
    "right": (0.518, 0.318, 0.446, 0.0, 0.807, -1.306),
}
JOINT_LIMITS = np.array([[-3.14158, 3.14158], [-1.85005, 1.25664], [-1.76278, 1.6057],
                         [-3.14158, 3.14158], [-1.8675, 2.23402], [-3.14158, 3.14158]])
# finger pad sites (aloha.xml), in the finger body frames, metres
PAD_LOCAL = {"left_finger_link": (0.015, -0.06, 0.02), "right_finger_link": (0.015, 0.06, 0.02)}
BASE_XY_CM = {"left": (-46.9, -1.9), "right": (46.9, -1.9)}   # at scale 100, relative to the rig xform
PINCH_LOCAL_CM = (11.62, 0.0, -0.08)


def _aloha_xml():
    return glob.glob(str(download_git_folder(MENAGERIE_URL, "aloha", ref=MENAGERIE_REF)) + "/aloha.xml")[0]


def add_aloha(builder: newton.ModelBuilder, xform: wp.transform, scale: float = 1.0, home: bool = True):
    """Add both ALOHA arms to `builder` at `xform`, scaled by `scale` (100.0 for the cm-scale twin).

    Returns {"left": arm, "right": arm, "pinch_local": wp.vec3, "n_arm": 6} where each arm is
    {"ee_body": int, "arm_dofs": [6 ints], "finger_dofs": [2 ints], "arm_coords": [6 ints]}.
    Indices are absolute in `builder`, so the cloth/table/other robots may be added before or after.
    """
    j0 = builder.joint_count
    builder.add_mjcf(_aloha_xml(), xform=xform, scale=float(scale), floating=False,
                     collapse_fixed_joints=True)
    keys = getattr(builder, "joint_key", None) or getattr(builder, "joint_label", None)
    out = {"pinch_local": wp.vec3(*[v * (scale / 100.0) for v in PINCH_LOCAL_CM]), "n_arm": len(ARM_JOINTS)}
    for s in SIDES:
        out[s] = {"ee_body": None, "arm_dofs": [None] * len(ARM_JOINTS), "arm_coords": [None] * len(ARM_JOINTS),
                  "finger_dofs": [], "bodies": {}}
    for j in range(j0, builder.joint_count):
        # ".../<side>/<body>/<side>/<joint>": the model prefixes every name with its arm
        parts = str(keys[j]).split("/")
        name, child_body = parts[-1], parts[-3] if len(parts) >= 3 else ""
        side = "left" if "/left/" in str(keys[j]) else ("right" if "/right/" in str(keys[j]) else None)
        if side is None:
            continue
        arm = out[side]
        arm["bodies"][child_body] = int(builder.joint_child[j])
        if name in ARM_JOINTS:
            i = ARM_JOINTS.index(name)
            arm["arm_dofs"][i] = int(builder.joint_qd_start[j])
            arm["arm_coords"][i] = int(builder.joint_q_start[j])
        elif name in FINGER_JOINTS:
            arm["finger_dofs"].append(int(builder.joint_qd_start[j]))
    for s in SIDES:
        arm = out[s]
        if any(d is None for d in arm["arm_dofs"]):
            raise RuntimeError(f"ALOHA {s} arm: missing joints in the MJCF ({arm['arm_dofs']})")
        arm["ee_body"] = arm["bodies"].get("gripper_link")
        q = HOME_DOWN[s] if home else NEUTRAL
        for c, v in zip(arm["arm_coords"], q):
            builder.joint_q[c] = float(v)
    return out


def pinch_world(body_q, arm, scale: float = 1.0):
    """Where this arm's finger pads meet, in world units - the FK truth behind PINCH_LOCAL."""
    pts = []
    for nm, local in PAD_LOCAL.items():
        i = arm["bodies"].get(nm)
        if i is None:
            continue
        p = np.asarray(body_q[i][:3], float)
        R = np.array(wp.quat_to_matrix(wp.quat(*[float(v) for v in body_q[i][3:7]]))).reshape(3, 3)
        pts.append(p + R @ (np.asarray(local, float) * scale))
    return np.mean(pts, axis=0) if pts else None


if __name__ == "__main__":     # self-test: build it, FK it, check the pinch point and the home pose
    S = 100.0
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))
    rig = add_aloha(b, wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), scale=S, home=True)
    m = b.finalize(device="cpu"); st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    bq = st.body_q.numpy()
    print(f"bodies {b.body_count}  dofs {b.joint_dof_count}")
    for s in SIDES:
        arm = rig[s]
        ee = arm["ee_body"]
        p = bq[ee][:3]
        R = np.array(wp.quat_to_matrix(wp.quat(*[float(v) for v in bq[ee][3:7]]))).reshape(3, 3)
        w = pinch_world(bq, arm, scale=S)
        down = np.degrees(np.arccos(np.clip(float(R[:, 0] @ np.array([0, 0, -1.0])), -1, 1)))
        print(f"  {s:5s} ee body {ee:3d}  arm dofs {arm['arm_dofs']}  fingers {arm['finger_dofs']}")
        print(f"        pinch {np.round(w, 2)}cm   local {np.round(R.T @ (w - p), 2)}cm   "
              f"approach {down:.1f}deg from straight down")
