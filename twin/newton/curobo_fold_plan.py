"""Stage 2 of the cuRobo fold pipeline (runs in the ISAACLAB conda env, where cuRobo lives).

Reads the fold's keyframe fingertip poses + obstacles (Franka-base frame, metres) exported by
newton_fold.py --export_plan, and uses cuRobo MotionGen to plan a COLLISION-FREE joint trajectory
through them (table + box as obstacles). Writes the dense joint trajectory + gripper events for
newton_fold.py --exec_plan to execute in the coupled cloth sim.

Run:
  /home/aoloo/miniforge3/envs/isaaclab/bin/python twin/newton/curobo_fold_plan.py \
      --plan /tmp/fold_plan.json --out /tmp/fold_traj.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuPose
from curobo.types.state import JointState as CuJointState
from curobo.geom.types import WorldConfig, Cuboid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

JN = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
      "panda_joint5", "panda_joint6", "panda_joint7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_box_obstacle", action="store_true", help="ignore the box (for place-inside)")
    ap.add_argument("--table_offset", type=float, default=0.03,
                    help="lower the table collision top (m) so fingers can descend to the cloth")
    args = ap.parse_args()
    P = json.load(open(args.plan))

    ta = TensorDeviceType(device=torch.device("cuda:0"))
    cuboids = []
    for o in P["obstacles"]:
        if o["name"] == "box" and args.no_box_obstacle:
            continue
        c = list(o["center_m"])
        if o["name"] == "table":
            c[2] -= args.table_offset          # drop collision boundary below the cloth surface
        cuboids.append(Cuboid(name=o["name"], dims=o["dims_m"], pose=c + [1, 0, 0, 0]))
    world = WorldConfig(cuboid=cuboids)

    print("[cuRobo] setting up MotionGen (franka)...", flush=True)
    cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml", world_model=world, tensor_args=ta,
        num_trajopt_seeds=12, num_graph_seeds=12, interpolation_dt=0.02, use_cuda_graph=False)
    mg = MotionGen(cfg); mg.warmup(warmup_js_trajopt=False)
    pc = MotionGenPlanConfig(enable_graph=True, max_attempts=30, enable_finetune_trajopt=True)

    down = torch.tensor([P["down_quat_wxyz"]], device="cuda")
    off = float(P["ee_offset_z"])
    cur = torch.tensor([P["home_js"]], device="cuda")

    segments = []                      # one dense joint trajectory per keyframe transition
    prev_pos = None
    for i, kf in enumerate(P["keyframes"]):
        ft = np.array(kf["fingertip_m"], float)
        hand = ft + np.array([0.0, 0.0, off])          # cuRobo plans the hand link; hand is above fingertip
        if prev_pos is not None and np.linalg.norm(hand - prev_pos) < 0.005:
            segments.append({"q": [cur.cpu().numpy()[0].tolist()], "gripper": kf["gripper"], "dur": kf["dur"]})
            prev_pos = hand
            continue
        js = CuJointState(position=cur, velocity=torch.zeros_like(cur), acceleration=torch.zeros_like(cur),
                          joint_names=JN)
        goal = CuPose(position=torch.tensor([hand.tolist()], device="cuda"), quaternion=down)
        res = mg.plan_single(js, goal, pc)
        if res.success.item():
            traj = res.get_interpolated_plan().position.cpu().numpy()   # (T,7)
            cur = torch.tensor([traj[-1].tolist()], device="cuda")
            segments.append({"q": traj.tolist(), "gripper": kf["gripper"], "dur": kf["dur"]})
            print(f"  kf{i:2d} {kf['gripper']:6s} -> planned {len(traj)} wpts", flush=True)
        else:
            segments.append({"q": [cur.cpu().numpy()[0].tolist()], "gripper": kf["gripper"], "dur": kf["dur"]})
            print(f"  kf{i:2d} {kf['gripper']:6s} -> FAILED (hold)", flush=True)
        prev_pos = hand

    n_ok = sum(1 for s in segments if len(s["q"]) > 1)
    json.dump({"joint_names": JN, "segments": segments}, open(args.out, "w"))
    print(f"[cuRobo] {n_ok}/{len(segments)} keyframes planned with motion -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
