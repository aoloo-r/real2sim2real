"""TAMP MOTION COMPILER — task plan -> EE trajectories (the motion half of TAMP).

Reads a task_plan.json (from tamp_plan.py) and compiles EACH pick-and-place action
into an EE-trajectory JSON using the SHARED geometry (ee_geometry.py) — the exact
same grasp/place math the sim twin uses, so trajectories are twin-identical. Runs
WITHOUT launching Isaac (cheap -> parallel-trial friendly).

Outputs, into --out_dir:
  step_01_<object>.json, step_02_<object>.json, ...   (one EE-traj per action)
  plan_manifest.json                                  (ordered list + metadata)

Usage:
  python tamp_to_ee.py \
    --task_plan /tmp/task_plan.json \
    --scene_dir outputs/robot_20260604_174345 \
    --capture_dir captures/robot_20260604_174345 \
    --out_dir /tmp/plan_cup
"""
from __future__ import annotations

import argparse
import json
import os
import re

import ee_geometry


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "obj").lower()).strip("_") or "obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_plan", required=True, help="task_plan.json from tamp_plan.py")
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--out_dir", default="/tmp/tamp_plan")
    ap.add_argument("--rim_frac", type=float, default=0.92,
                    help="rim-grasp height fraction (0.92 = high on the wall, like the cup demo)")
    ap.add_argument("--lift_clearance", type=float, default=0.28,
                    help="travel/lift height above table (m); higher = cleaner carry in sim")
    ap.add_argument("--grip_max", type=float, default=0.06,
                    help="max width for a center grasp (else rim); 0.06 center-grasps small fruit")
    args = ap.parse_args()

    plan = json.load(open(args.task_plan))
    layout = json.load(open(os.path.join(args.scene_dir, "scene_layout.json")))
    T = (layout.get("camera_extrinsics") or {}).get("T_base_cam")
    if T is None:
        # fall back to extrinsics.json (same source tamp_plan/ee export use)
        exj = json.load(open(os.path.join(args.capture_dir, "extrinsics.json")))
        frame = plan.get("frame", "ur5e_base_link")
        T = exj["transforms"][frame]["T_base_cam"]
        layout.setdefault("camera_extrinsics", {})["T_base_cam"] = T
        layout["camera_extrinsics"]["frame"] = frame

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = ee_geometry.GraspCfg(rim_frac=args.rim_frac,
                               lift_clearance=args.lift_clearance,
                               grip_max=args.grip_max)

    actions = plan.get("actions", [])
    print(f"[TAMP->EE] compiling {len(actions)} action(s) from {args.task_plan}")
    manifest = {"instruction": plan.get("instruction"),
                "frame": plan.get("frame", "ur5e_base_link"),
                "rim_frac": args.rim_frac, "steps": []}

    for i, a in enumerate(actions, 1):
        obj_label = a.get("object_label") or ""
        tgt_label = a.get("target_label")
        relation = a.get("relation", "on")
        print(f"\n[TAMP->EE] step {i}: pick '{obj_label}' "
              f"-> {relation} '{tgt_label}'")
        traj = ee_geometry.compute_pick_place_trajectory(
            layout, T, args.scene_dir, args.capture_dir,
            pick_label=obj_label, place_on=tgt_label, relation=relation,
            cfg=cfg, verbose=True)
        if traj is None:
            print(f"[TAMP->EE] step {i} FAILED to compile (no grasp); skipping")
            manifest["steps"].append({"index": i, "status": "failed",
                                      "object": obj_label, "target": tgt_label,
                                      "relation": relation})
            continue
        # carry the task-level metadata into the trajectory
        traj["tamp_step"] = i
        traj["tamp_relation"] = relation
        traj["pick_label"] = obj_label or traj.get("pick_label")
        traj["place_label"] = tgt_label
        fname = f"step_{i:02d}_{_slug(obj_label)}.json"
        fpath = os.path.join(args.out_dir, fname)
        json.dump(traj, open(fpath, "w"), indent=2)
        gp = traj.get("pick_position_base")
        print(f"[TAMP->EE] step {i} -> {fname}  grasp={gp} "
              f"strategy={traj.get('grasp_strategy')} ({len(traj['waypoints'])} wp)")
        manifest["steps"].append({"index": i, "status": "ok", "file": fname,
                                  "object": obj_label, "target": tgt_label,
                                  "relation": relation,
                                  "grasp_strategy": traj.get("grasp_strategy"),
                                  "pick_position_base": gp})

    mpath = os.path.join(args.out_dir, "plan_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    ok = sum(1 for s in manifest["steps"] if s["status"] == "ok")
    print(f"\n[TAMP->EE] wrote {ok}/{len(actions)} EE trajectories to {args.out_dir}")
    print(f"[TAMP->EE] manifest: {mpath}")


if __name__ == "__main__":
    main()
