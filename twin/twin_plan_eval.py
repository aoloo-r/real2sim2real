"""Parallel task-plan search engine (Stage 2) — the twin as a many-plans-at-once
validator. Clones the REAL multi-object twin into N envs, rolls out N candidate
PLANS (from twin_plan_search.py) in parallel under physics, and scores each by task
success: GOALS achieved + STABILITY + no COLLISIONS/knockovers + EFFICIENCY. The
best plan is selected (Simify-style evolution is stage 3).

Object-agnostic: objects are primitives SIZED from the real SAM3D reconstruction at
auto-leveled base positions (consistent with ee_geometry's grasp frame). Each env
runs a different candidate; all candidates share K placements x a fixed grasp/place
skeleton, so the N rollouts step in lockstep with batched cuRobo planning.

Run:
  DISPLAY=:1 ./isaaclab.sh -p scripts/twin_plan_eval.py \
    --scene_dir <out> --capture_dir <cap> --candidates /tmp/candidates.json
  (add --scene_only for the 2a scene-clone check)
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel task-plan evaluation engine")
parser.add_argument("--scene_dir", required=True)
parser.add_argument("--capture_dir", required=True)
parser.add_argument("--candidates", default=None)
parser.add_argument("--num_envs", type=int, default=None, help="default = #candidates")
parser.add_argument("--base_frame", default="ur5e_base_link")
parser.add_argument("--scene_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

sys.path.insert(0, "/home/aoloo/real2sim2real/tamp")
import ee_geometry

VESSEL_KW = ("cup", "bowl", "plate", "mug", "glass", "container", "dish")
EE_OFFSET_Z = 0.105
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
DOWN = [0.0, 1.0, 0.0, 0.0]


def load_scene_objects(scene_dir, capture_dir, base_frame):
    layout = json.load(open(f"{scene_dir}/scene_layout.json"))
    T = (layout.get("camera_extrinsics") or {}).get("T_base_cam")
    if T is None:
        ex = json.load(open(f"{capture_dir}/extrinsics.json"))
        T = ex["transforms"][base_frame]["T_base_cam"]
        layout.setdefault("camera_extrinsics", {})["T_base_cam"] = T
        layout["camera_extrinsics"]["frame"] = base_frame
    Tn = np.asarray(T, float); cam = Tn[:3, 3]
    plane = ee_geometry._fit_table_plane_base(capture_dir, T)
    if plane is not None:
        R = ee_geometry._align_rotation(plane["normal"], np.array([0.0, 0.0, 1.0]))
        Z_table = float((R @ (plane["centroid"] - cam) + cam)[2])
    else:
        R = np.eye(3); Z_table = -0.05
    objs = []
    for o in layout.get("objects", []):
        di = o.get("depth_info") or {}
        pc = di.get("position_cam") or (o.get("icp_pose") or {}).get("position_cam")
        if not pc:
            continue
        P = Tn @ np.array([pc[0], pc[1], pc[2], 1.0])
        Pb = (R @ (P[:3] - cam) + cam) if plane is not None else P[:3]
        w = max(0.02, min(0.40, float(di.get("physical_width_m") or o.get("physical_size_m") or 0.06)))
        h = max(0.02, min(0.40, float(di.get("physical_height_m") or o.get("physical_size_m") or 0.06)))
        label = str(o.get("label", "")).lower()
        hollow = any(k in label for k in VESSEL_KW)
        mass = max(0.03, min(1.0, (w * w * h) * (60.0 if hollow else 200.0)))
        if hollow:
            mass = min(mass, 0.12)
        objs.append({"id": o["id"], "label": o.get("label"), "x": float(Pb[0]),
                     "y": float(Pb[1]), "radius": w / 2.0, "height": h,
                     "mass": mass, "hollow": hollow, "real_z": float(Pb[2])})

    # STACKING: an object whose footprint sits inside a larger one AND whose real
    # reconstructed height is above it (e.g. kiwi ON the plate) must rest ON TOP of
    # that object, not at table level (where it spawns embedded in the lower disc).
    for o in objs:
        o["spawn_z"] = Z_table + o["height"] / 2.0 + 0.005
    for a in objs:
        for b in objs:
            if a is b or b["radius"] <= a["radius"]:
                continue
            d = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            inside = d < (b["radius"] - 0.4 * a["radius"])      # a mostly within b
            higher = a["real_z"] > b["real_z"] + 0.005           # a really sits above b
            if inside and higher:
                b_top = (Z_table + b["height"] / 2.0 + 0.005) + b["height"] / 2.0
                a["spawn_z"] = b_top + a["height"] / 2.0 + 0.003
                print(f"[PLAN-EVAL] STACK: '{a['label']}' rests ON '{b['label']}' "
                      f"-> spawn z={a['spawn_z']:+.3f}")
    return objs, Z_table, layout, T


@configclass
class PlanEvalSceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/Light",
                        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def build_scene_cfg(objs, Z_table, num_envs, scene_dir):
    cfg = PlanEvalSceneCfg(num_envs=num_envs, env_spacing=3.0)
    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, Z_table)))
    palette = [(0.2, 0.5, 0.85), (0.3, 0.7, 0.3), (0.85, 0.4, 0.6),
               (0.85, 0.75, 0.2), (0.6, 0.4, 0.8), (0.4, 0.7, 0.7)]
    n_real = 0
    for i, o in enumerate(objs):
        # Spawn the SAME prepared, category-appropriate USD that real2sim uses
        # (object_<id>/mesh_obb.usd: real visual + collision + mass) so the parallel
        # twin matches the reconstructed scene. Fall back to a sized cylinder only if
        # the prepared USD is missing.
        usd = os.path.join(scene_dir, f"object_{o['id']}", "mesh_obb.usd")
        z0 = o.get("spawn_z", Z_table + o["height"] / 2.0 + 0.02)
        if os.path.isfile(usd):
            n_real += 1
            spawn = UsdFileCfg(usd_path=usd)
        else:
            spawn = sim_utils.CylinderCfg(
                radius=o["radius"], height=o["height"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=palette[i % len(palette)]))
        setattr(cfg, f"obj_{o['id']}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obj_%d" % o["id"],
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(o["x"], o["y"], z0))))
    print(f"[PLAN-EVAL] spawning {n_real}/{len(objs)} objects from real prepared USDs "
          f"(mesh_obb.usd); {len(objs)-n_real} cylinder fallback")
    return cfg


def precompute(cands, layout, T, scene_dir, capture_dir, N):
    """Per-env (=candidate) ordered actions: each {pick_id, target_id, relation, wps}."""
    per_env = []
    for e in range(N):
        c = cands[e % len(cands)]
        acts = []
        for a in c["actions"]:
            g = a.get("grasp", {}) or {}
            cfg = ee_geometry.GraspCfg(rim_frac=g.get("rim_frac", 0.92),
                                       grip_max=g.get("grip_max", 0.06), lift_clearance=0.28)
            po = a.get("place_offset", {}) or {}
            tr = ee_geometry.compute_pick_place_trajectory(
                layout, T, scene_dir, capture_dir,
                pick_label=a.get("object_label") or "", place_on=a.get("target_label"),
                relation=a.get("relation", "on"), place_dx=po.get("dx", 0.0),
                place_dy=po.get("dy", 0.0), cfg=cfg, verbose=False)
            acts.append({"pick_id": a.get("object"), "target_id": a.get("target"),
                         "relation": a.get("relation"), "wps": (tr or {}).get("waypoints", [])})
        per_env.append({"plan_id": c.get("plan_id", e), "actions": acts})
    return per_env


def main():
    objs, Z_table, layout, T = load_scene_objects(args_cli.scene_dir, args_cli.capture_dir, args_cli.base_frame)
    obj_by_id = {o["id"]: o for o in objs}
    print(f"\n[PLAN-EVAL] scene: {len(objs)} objects, table z={Z_table:+.3f}")
    for o in objs:
        print(f"   id={o['id']} '{o['label']}' @({o['x']:+.3f},{o['y']:+.3f}) "
              f"r={o['radius']:.3f} h={o['height']:.3f} m={o['mass']:.3f}kg")

    cands = json.load(open(args_cli.candidates)).get("candidates", []) if args_cli.candidates else []
    N = args_cli.num_envs or max(1, len(cands))
    if cands:
        cands = cands[:N] if len(cands) >= N else cands

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view(eye=[2.6, 2.6, 2.0], target=[0.5, 0.0, Z_table])
    scene = InteractiveScene(build_scene_cfg(objs, Z_table, N, args_cli.scene_dir))
    sim.reset()
    dt = 1.0 / 120.0
    origins = scene.env_origins                              # (N,3)

    def settle(sec):
        for _ in range(int(sec / dt)):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

    settle(1.5)
    if args_cli.scene_only:
        print("[PLAN-EVAL] 2a scene check done."); settle(2.0); return

    robot = scene["robot"]
    arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
    fin_ids, _ = robot.find_joints("panda_finger_joint.*", preserve_order=True)
    hand_id = robot.find_bodies("panda_hand")[0][0]

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.state import JointState as CuJointState
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    tdt = TensorDeviceType(device=torch.device("cuda:0"))
    world = WorldConfig(cuboid=[Cuboid(name="table", dims=[1.4, 1.4, 0.10],
                                       pose=[0.5, 0.0, Z_table - 0.05, 1, 0, 0, 0])])
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        "franka.yml", world_model=world, tensor_args=tdt, num_trajopt_seeds=4,
        num_graph_seeds=1, interpolation_dt=dt, use_cuda_graph=False))
    mg.warmup(warmup_js_trajopt=False, batch=N)
    JN = ["panda_joint%d" % i for i in range(1, 8)]
    pcfg = MotionGenPlanConfig(enable_graph=False, max_attempts=6)

    HOME = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], device="cuda")
    cur_js = HOME.unsqueeze(0).repeat(N, 1).clone()
    plan_fail = torch.zeros(N, device="cuda")

    # per-env attach state
    att_obj = [None] * N
    att_off = torch.zeros((N, 3), device="cuda")
    att_quat = torch.zeros((N, 4), device="cuda")

    def step_once():
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
        for e in range(N):
            if att_obj[e] is not None:
                ro = scene[f"obj_{att_obj[e]}"]
                ee = robot.data.body_pose_w[e, hand_id, :3]
                newp = (ee + att_off[e]).unsqueeze(0)
                ro.write_root_pose_to_sim(torch.cat([newp, att_quat[e].unsqueeze(0)], dim=1),
                                          env_ids=torch.tensor([e], device="cuda"))
                ro.write_root_velocity_to_sim(torch.zeros((1, 6), device="cuda"),
                                              env_ids=torch.tensor([e], device="cuda"))

    def batch_plan(goal_pos, goal_quat):
        js = CuJointState(position=cur_js.clone(), velocity=torch.zeros((N, 7), device="cuda"),
                          acceleration=torch.zeros((N, 7), device="cuda"), joint_names=JN)
        res = mg.plan_batch(js, CuPose(position=goal_pos, quaternion=goal_quat), pcfg)
        succ = res.success.view(-1)
        pos = res.interpolated_plan.position
        if pos.dim() == 2:
            pos = pos.unsqueeze(0).repeat(N, 1, 1)
        last = res.path_buffer_last_tstep
        clamped = pos.clone()
        for i in range(pos.shape[0]):
            if not bool(succ[i]):
                clamped[i, :, :] = cur_js[i]
            else:
                Li = max(0, min(int(last[i]) if last is not None else pos.shape[1] - 1, pos.shape[1] - 1))
                clamped[i, Li + 1:, :] = pos[i, Li:Li + 1, :]
        return succ, clamped

    def run_traj(pos, grip):
        nonlocal cur_js
        ft = torch.full((N, len(fin_ids)), grip, device="cuda")
        for t in range(pos.shape[1]):
            robot.set_joint_position_target(pos[:, t, :], joint_ids=arm_ids)
            robot.set_joint_position_target(ft, joint_ids=fin_ids)
            step_once()
        cur_js = pos[:, -1, :].clone()

    def hold(grip, sec):
        ft = torch.full((N, len(fin_ids)), grip, device="cuda")
        for _ in range(int(sec / dt)):
            robot.set_joint_position_target(cur_js, joint_ids=arm_ids)
            robot.set_joint_position_target(ft, joint_ids=fin_ids)
            step_once()

    def local_xy(obj_id):
        ro = scene[f"obj_{obj_id}"]
        return (ro.data.root_pos_w[:, :2] - origins[:, :2])      # (N,2)

    init_local = {o["id"]: local_xy(o["id"]).clone() for o in objs}

    per_env = precompute(cands, layout, T, args_cli.scene_dir, args_cli.capture_dir, N)
    K = max(len(p["actions"]) for p in per_env)
    print(f"\n[PLAN-EVAL] rolling out {N} candidate plans in parallel, K={K} actions each")

    cur_grip = GRIP_OPEN
    for k in range(K):
        print(f"\n[PLAN-EVAL] === action slot {k+1}/{K} ===", flush=True)
        # waypoint skeleton aligned across envs (use env0 action k as template)
        templ = per_env[0]["actions"][k]["wps"] if k < len(per_env[0]["actions"]) else []
        for wi in range(len(templ)):
            w0 = templ[wi]
            if w0.get("position") is None:                      # gripper segment
                is_close = (w0.get("gripper") == "close")
                if is_close:
                    for e in range(N):
                        a = per_env[e]["actions"][k]
                        pid = a["pick_id"]
                        if pid is None:
                            continue
                        ee = robot.data.body_pose_w[e, hand_id, :3]
                        op = scene[f"obj_{pid}"].data.root_pos_w[e, :3]
                        att_obj[e] = pid
                        att_off[e] = (op - ee)
                        att_quat[e] = scene[f"obj_{pid}"].data.root_quat_w[e, :4]
                    cur_grip = GRIP_CLOSE
                    hold(GRIP_CLOSE, 1.5)
                else:                                           # open -> release
                    for e in range(N):
                        att_obj[e] = None
                    cur_grip = GRIP_OPEN
                    hold(GRIP_OPEN, 1.0)
            else:                                               # pose segment
                gp = torch.zeros((N, 3), device="cuda")
                gq = torch.zeros((N, 4), device="cuda")
                for e in range(N):
                    a = per_env[e]["actions"][k]
                    w = a["wps"][wi] if wi < len(a["wps"]) else a["wps"][-1]
                    p = w.get("position") or [0.5, 0.0, Z_table + 0.2]
                    q = w.get("quaternion") or DOWN
                    gp[e] = torch.tensor([p[0], p[1], p[2] + EE_OFFSET_Z], device="cuda")
                    gq[e] = torch.tensor(q, device="cuda")
                succ, pos = batch_plan(gp, gq)
                plan_fail += (~succ.bool()).float()
                run_traj(pos, cur_grip)
        hold(cur_grip, 0.3)

    settle(1.0)

    # -------- SCORING (goals + stability + collisions + efficiency) --------
    print("\n" + "=" * 70)
    print("  PARALLEL TASK-PLAN RESULTS (composite score)")
    print("=" * 70)
    results = []
    fin_local = {o["id"]: local_xy(o["id"]) for o in objs}
    vel = {o["id"]: scene[f"obj_{o['id']}"].data.root_lin_vel_w for o in objs}
    for e in range(N):
        c = per_env[e]
        goals = c["actions"]
        involved = set()
        gmet = 0
        for a in goals:
            pid, tid = a["pick_id"], a["target_id"]
            involved.add(pid)
            if tid is None or pid is None:
                continue
            d = float((fin_local[pid][e] - fin_local[tid][e]).norm())
            tol = obj_by_id[tid]["radius"] + obj_by_id[pid]["radius"] + 0.03
            if d < tol:
                gmet += 1
        goal_frac = gmet / max(1, len(goals))
        # stability: involved objects settled (low speed)
        spd = max(float(vel[a["pick_id"]][e].norm()) for a in goals if a["pick_id"] is not None)
        stable = 1.0 if spd < 0.05 else max(0.0, 1.0 - spd)
        # collisions: bystander objects displaced from initial
        byst = 0.0
        for o in objs:
            if o["id"] in involved:
                continue
            byst += float((fin_local[o["id"]][e] - init_local[o["id"]][e]).norm())
        coll_pen = min(1.0, byst / 0.10)
        # efficiency: fewer plan failures
        eff = 1.0 - min(1.0, float(plan_fail[e]) / max(1.0, K * 4.0))
        score = 1.0 * goal_frac + 0.2 * stable - 0.4 * coll_pen + 0.1 * eff
        results.append((e, c["plan_id"], goal_frac, gmet, len(goals), stable, coll_pen, eff, score))

    results.sort(key=lambda r: r[8], reverse=True)
    for rank, (e, pid, gf, gm, gtot, st, cp, ef, sc) in enumerate(results, 1):
        print(f"  #{rank}  plan {pid} (env {e}): score={sc:+.3f}  "
              f"goals {gm}/{gtot}  stable={st:.2f}  bystander_pen={cp:.2f}  eff={ef:.2f}")
    win = results[0]
    print("-" * 70)
    print(f"  WINNER: plan {win[1]} (env {win[0]})  score={win[8]:+.3f}  "
          f"goals {win[3]}/{win[4]}")
    print("=" * 70, flush=True)
    settle(3.0)


if __name__ == "__main__":
    main()
    simulation_app.close()
