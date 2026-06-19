"""Resolve a grasp's orientation ONCE against the REAL UR5e's kinematics and bake it
into the EE trajectory — so the sim twin and the real robot execute the IDENTICAL
grasp (no runtime improvisation).

THE BUG THIS FIXES: the real executor (ur5e_ee_executor.py) used to search a list of
orientations at runtime (commanded straight-down, then LEAN toward base, then tilts)
and let MoveIt pick the first reachable one. The sim executed the clean commanded
orientation; the real robot, unable to reach it straight-down, tilted -> the two
grasped DIFFERENT faces of the object (sim "side", real "back"). The trajectory was
NOT the single source of truth.

THE FIX: do that orientation search HERE, once, offline, using cuRobo `ur5e.yml` IK
(the real robot's kinematics), pick the reachable orientation, and write it into every
pose waypoint. The twin replays this resolved trajectory; the real executor runs it
verbatim. Same TCP poses + orientation on both -> identical grasp.

The candidate set + order mirror the executor's old candidate_quats() exactly, so the
resolved orientation is the one the robot would have chosen — just decided up front.

Run (isaaclab conda env; cuRobo only, no Isaac Sim app needed):
  python resolve_orientation.py --traj ee.json --out ee_resolved.json [--tcp_offset 0.187]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


# ---- quaternion helpers ([w,x,y,z]) — identical convention to ur5e_ee_executor ----
def quat_from_axis_angle(axis, ang):
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-12)
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), a[0] * s, a[1] * s, a[2] * s]


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def quat_to_R(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]]


def candidate_quats(base_q, gx, gy):
    """Orientation candidates in increasing wrist tilt — MUST mirror the executor's
    candidate_quats() order: commanded-down first, then lean toward the robot base,
    then ±side tilts."""
    cands = [("down", list(base_q))]
    n = math.hypot(gx, gy) or 1.0
    lean_axis = [-gy / n, gx / n, 0.0]               # rotate down-axis toward base
    for deg in (12, 20, 28, 35, 45, 55, 60):
        cands.append(("lean%d" % deg,
                      quat_mul(quat_from_axis_angle(lean_axis, math.radians(deg)), base_q)))
    for ax, nm in (([1, 0, 0], "tx"), ([0, 1, 0], "ty")):
        for deg in (20, -20, 35, -35, 50, -50):
            cands.append(("%s%+d" % (nm, deg),
                          quat_mul(quat_from_axis_angle(ax, math.radians(deg)), base_q)))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="EE-trajectory JSON (ur5e_base_link frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tcp_offset", type=float, default=0.187,
                    help="tool0 -> fingertip distance (m); UR5e+HandE ~0.187")
    ap.add_argument("--grasp_label", default="descend_to_grasp",
                    help="waypoint whose reachability binds the orientation")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    wps = traj.get("waypoints", [])
    gwp = next((w for w in wps if w.get("label") == args.grasp_label and w.get("position")), None)
    if gwp is None:
        raise SystemExit(f"no '{args.grasp_label}' waypoint with a position in {args.traj}")
    tip = gwp["position"]
    base_q = gwp.get("quaternion") or [0.0, 1.0, 0.0, 0.0]
    cands = candidate_quats(base_q, tip[0], tip[1])

    # tool0 target per candidate: fingertip - R[:,2]*tcp_offset (mirror executor)
    pos, quat = [], []
    for _nm, q in cands:
        R = quat_to_R(q)
        pos.append([tip[0] - R[0][2] * args.tcp_offset,
                    tip[1] - R[1][2] * args.tcp_offset,
                    tip[2] - R[2][2] * args.tcp_offset])
        quat.append(q)

    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    tdt = TensorDeviceType()
    # Scene collision (table + other objects) from the trajectory, so IK reachability
    # accounts for the same obstacles the real robot's MoveIt sees.
    cubs = []
    for c in traj.get("collision_objects", []):
        if c.get("type") == "box" and c.get("center") and c.get("size"):
            cubs.append(Cuboid(name=c.get("name", "box"), dims=list(c["size"]),
                               pose=[c["center"][0], c["center"][1], c["center"][2], 1, 0, 0, 0]))
    world = WorldConfig(cuboid=cubs) if cubs else None
    print(f"[RESOLVE] scene collision: {len(cubs)} box(es) from trajectory")
    ik = IKSolver(IKSolverConfig.load_from_robot_config(
        "ur5e.yml", world, tensor_args=tdt, num_seeds=30,
        self_collision_check=True, self_collision_opt=True))
    goal = Pose(position=torch.tensor(pos, dtype=torch.float32, device="cuda"),
                quaternion=torch.tensor(quat, dtype=torch.float32, device="cuda"))
    res = ik.solve_batch(goal)
    succ = res.success.view(-1).tolist()

    idx = next((i for i, s in enumerate(succ) if s), None)
    if idx is None:
        print("[RESOLVE] WARNING: NONE of the candidate orientations is UR5e-reachable; "
              "keeping commanded (real grasp may fail — re-plan needed).")
        idx = 0
    rname, rq = cands[idx]
    print(f"[RESOLVE] {sum(succ)}/{len(cands)} orientations UR5e-reachable; "
          f"chosen = '{rname}'  q={[round(v,4) for v in rq]}")

    # bake the resolved orientation into EVERY pose waypoint
    n_baked = 0
    for w in wps:
        if w.get("quaternion") is not None:
            w["quaternion"] = [float(v) for v in rq]
            n_baked += 1
    traj["resolved_orientation"] = {"name": rname, "quaternion": [float(v) for v in rq],
                                    "tcp_offset": args.tcp_offset,
                                    "n_reachable": int(sum(succ)), "n_candidates": len(cands)}
    json.dump(traj, open(args.out, "w"), indent=2)
    print(f"[RESOLVE] baked into {n_baked} pose waypoints -> {args.out}  "
          f"(sim twin + real executor now run this VERBATIM)")


if __name__ == "__main__":
    main()
