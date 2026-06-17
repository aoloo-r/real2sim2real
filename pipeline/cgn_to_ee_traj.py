"""Convert a Contact-GraspNet grasp (cgn_*_grasp.json, pose in ur5e_base_link) into
an EE trajectory the sim replay / real executor understands: approach -> descend ->
close -> lift -> retreat, with the grasp's own 6-DoF orientation.

The pre-grasp/retreat back off along the grasp's APPROACH axis (the learned grasp
direction), so this respects whatever 6-DoF pose the net proposed.
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp", required=True, help="cgn grasp json")
    ap.add_argument("--out", required=True, help="output ee_trajectory.json")
    ap.add_argument("--backoff", type=float, default=0.15, help="pre-grasp standoff along approach (m)")
    ap.add_argument("--lift", type=float, default=0.15, help="lift height after grasp (m)")
    ap.add_argument("--table_z", type=float, default=-0.06, help="table height (base frame) for collision box")
    args = ap.parse_args()

    g = json.load(open(args.grasp))
    p = np.asarray(g["position"], float)
    q = g["quaternion_wxyz"]                          # [w,x,y,z]
    a = np.asarray(g["approach"], float)
    a = a / (np.linalg.norm(a) + 1e-9)               # unit approach (points toward object)

    pre = (p - args.backoff * a).tolist()            # backed off along approach
    hov = (p - 0.05 * a).tolist()
    grasp = p.tolist()
    lift = (p + np.array([0, 0, args.lift])).tolist()
    retreat = (p - args.backoff * a + np.array([0, 0, 0.08])).tolist()

    wps = [
        {"label": "open_pregrasp",    "position": None,  "quaternion": None, "gripper": "open"},
        {"label": "approach_above",   "position": pre,   "quaternion": q, "gripper": "none"},
        {"label": "hover_object",     "position": hov,   "quaternion": q, "gripper": "none"},
        {"label": "descend_to_grasp", "position": grasp, "quaternion": q, "gripper": "none"},
        {"label": "close_gripper",    "position": None,  "quaternion": None, "gripper": "close"},
        {"label": "lift",             "position": lift,  "quaternion": q, "gripper": "none"},
        {"label": "retreat",          "position": retreat,"quaternion": q, "gripper": "none"},
        {"label": "open_gripper",     "position": None,  "quaternion": None, "gripper": "open"},
    ]
    out = {
        "frame": g.get("frame", "ur5e_base_link"),
        "source": "contact_graspnet",
        "grasp_score": g.get("score"),
        "pick_label": g.get("target", "object"),
        "default_vel_scale": 0.1,
        "waypoints": wps,
        "collision_objects": [{
            "name": "table", "type": "box",
            "center": [round(float(p[0]), 3), round(float(p[1]), 3), round(args.table_z - 0.04, 3)],
            "size": [1.2, 1.2, 0.04]}],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[CGN->EE] wrote {args.out}")
    print(f"  grasp  ({grasp[0]:.3f},{grasp[1]:.3f},{grasp[2]:.3f}) quat_wxyz={[round(x,3) for x in q]}")
    print(f"  approach standoff -> ({pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f})")


if __name__ == "__main__":
    main()
