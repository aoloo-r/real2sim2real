"""Parallel-trial grasp-evaluation engine — the twin as a many-trials-at-once
validator (the meeting's "run many trials in sim in parallel").

Clones the scene into N environments, places ONE graspable target per env, and runs
N DIFFERENT grasp candidates simultaneously under physics. cuRobo plan_batch plans
all N approaches in one GPU call; the N physics rollouts step in lockstep. Each env
is scored by how far it lifts the object (continuous z-rise — robust to marginal
grasps where a binary pass/fail would be all-zero), then candidates are RANKED and
the winner reported. This is what an open-loop TAMP/VLA baseline cannot do.

v1 uses a solid cylinder target so physics cleanly discriminates good vs bad grasps
(a centered grasp holds; an off-center one slips). Plugging in the real SAM3D mesh +
Contact-GraspNet candidates is the next refinement.

Run (GUI on :1):
  DISPLAY=:1 ./isaaclab.sh -p scripts/twin_grasp_eval.py --num_envs 8
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel grasp-evaluation engine")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--obj_radius", type=float, default=0.028, help="target cylinder radius (m)")
parser.add_argument("--obj_height", type=float, default=0.12, help="target cylinder height (m)")
parser.add_argument("--obj_mass", type=float, default=0.10, help="target mass (kg)")
parser.add_argument("--obj_x", type=float, default=0.50)
parser.add_argument("--obj_y", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- after-app imports -------------------------------------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)     # gripper points down (wxyz)
EE_OFFSET_Z = 0.105                  # panda_hand frame -> fingertip
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
TABLE_Z = 0.0                        # target rests on z=0 plane


@configclass
class EvalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/Light",
                        spawn=sim_utils.DomeLightCfg(intensity=3000.0,
                                                     color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot")
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CylinderCfg(
            radius=args_cli.obj_radius, height=args_cli.obj_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.obj_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(args_cli.obj_x, args_cli.obj_y, TABLE_Z + args_cli.obj_height / 2.0)),
    )


def make_candidates(n: int):
    """N grasp candidates for the target: vary lateral X-offset from center and
    grasp height. The centered, mid-height grasp should hold; off-center ones slip.
    Returns list of dicts {dx, dz_frac, label}."""
    import math
    cands = []
    # spread lateral offsets symmetrically; vary height a little too
    offs = [(-0.04 + 0.08 * i / max(1, n - 1)) for i in range(n)]   # -4cm .. +4cm
    for i, dx in enumerate(offs):
        cands.append({"dx": float(dx),
                      "dz_frac": 0.6,                  # grasp at 60% of object height
                      "label": f"dx={dx*100:+.1f}cm"})
    return cands


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view(eye=[2.2, 2.2, 1.8], target=[0.4, 0.0, 0.2])
    scene = InteractiveScene(EvalSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    sim.reset()
    dt = 1.0 / 120.0
    N = args_cli.num_envs

    robot = scene["robot"]
    target = scene["target"]
    arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
    fin_ids, _ = robot.find_joints("panda_finger_joint.*", preserve_order=True)
    hand_id = robot.find_bodies("panda_hand")[0][0]

    # ---- cuRobo (batched) ----
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.state import JointState as CuJointState
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import (
        MotionGen, MotionGenConfig, MotionGenPlanConfig)
    tdt = TensorDeviceType(device=torch.device("cuda:0"))
    world = WorldConfig(cuboid=[Cuboid(name="table", dims=[1.2, 1.2, 0.10],
                                       pose=[0.5, 0.0, -0.05, 1, 0, 0, 0])])
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        "franka.yml", world_model=world, tensor_args=tdt,
        num_trajopt_seeds=4, num_graph_seeds=1, interpolation_dt=dt,
        use_cuda_graph=False))
    mg.warmup(warmup_js_trajopt=False, batch=N)
    JNAMES = ["panda_joint%d" % i for i in range(1, 8)]
    plan_cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=8)

    cands = make_candidates(N)
    print(f"\n[EVAL] {N} grasp candidates on the target:")
    for i, c in enumerate(cands):
        print(f"   env {i}: {c['label']} (grasp at {c['dz_frac']*100:.0f}% height)")

    HOME = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], device="cuda")
    cur_js = HOME.unsqueeze(0).repeat(N, 1).clone()      # (N,7) per-env arm config

    ox, oy = args_cli.obj_x, args_cli.obj_y
    gz = TABLE_Z + args_cli.obj_height * cands[0]["dz_frac"]
    # per-env goal XYZ for each phase
    gx = torch.tensor([ox + c["dx"] for c in cands], device="cuda")
    gy = torch.full((N,), oy, device="cuda")
    quat = torch.tensor(DOWN_QUAT, device="cuda").repeat(N, 1)

    def goalset(zs):
        pos = torch.stack([gx, gy, torch.tensor(zs, device="cuda")], dim=1)
        return CuPose(position=pos, quaternion=quat)

    def batch_plan(goal_pos):
        js = CuJointState(position=cur_js.clone(),
                          velocity=torch.zeros((N, 7), device="cuda"),
                          acceleration=torch.zeros((N, 7), device="cuda"),
                          joint_names=JNAMES)
        res = mg.plan_batch(js, goal_pos, plan_cfg)
        succ = res.success.view(-1)                       # (N,)
        pos = res.interpolated_plan.position              # (N, T, 7)
        if pos.dim() == 2:                                # safety: (T,dof) -> batch
            pos = pos.unsqueeze(0).repeat(N, 1, 1)
        last = res.path_buffer_last_tstep                 # per-env valid length or None
        Nn, Tt, _ = pos.shape
        clamped = pos.clone()
        for i in range(Nn):
            if not bool(succ[i]):
                clamped[i, :, :] = cur_js[i]              # plan failed -> hold start
                continue
            Li = (last[i] if last is not None else Tt - 1)
            Li = max(0, min(int(Li), Tt - 1))
            clamped[i, Li + 1:, :] = pos[i, Li:Li + 1, :]  # hold goal after reaching it
        return succ, clamped

    def run_traj(pos, grip):
        """Execute a (N,T,7) batched arm trajectory in lockstep; fingers=grip."""
        T = pos.shape[1]
        ft = torch.full((N, len(fin_ids)), grip, device="cuda")
        for t in range(T):
            robot.set_joint_position_target(pos[:, t, :], joint_ids=arm_ids)
            robot.set_joint_position_target(ft, joint_ids=fin_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
        nonlocal cur_js
        cur_js = pos[:, -1, :].clone()

    def hold(grip, seconds):
        ft = torch.full((N, len(fin_ids)), grip, device="cuda")
        for _ in range(int(seconds / dt)):
            robot.set_joint_position_target(cur_js, joint_ids=arm_ids)
            robot.set_joint_position_target(ft, joint_ids=fin_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

    # settle the targets
    for _ in range(int(1.0 / dt)):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
    z_rest = target.data.root_pos_w[:, 2].clone()

    AP = EE_OFFSET_Z + TABLE_Z + 0.22
    print("\n[EVAL] phase: approach (batched plan)")
    ok, pos = batch_plan(goalset([AP] * N)); run_traj(pos, GRIP_OPEN)
    print(f"   planned {int(ok.sum())}/{N}")
    print("[EVAL] phase: descend")
    ok, pos = batch_plan(goalset([gz + EE_OFFSET_Z] * N)); run_traj(pos, GRIP_OPEN)
    print(f"   planned {int(ok.sum())}/{N}")
    print("[EVAL] phase: close")
    hold(GRIP_CLOSE, 2.0)
    print("[EVAL] phase: lift (batched plan)")
    ok, pos = batch_plan(goalset([AP] * N)); run_traj(pos, GRIP_CLOSE)
    print(f"   planned {int(ok.sum())}/{N}")
    hold(GRIP_CLOSE, 0.5)
    z_lift = target.data.root_pos_w[:, 2].clone()

    rise = (z_lift - z_rest)
    print("\n" + "=" * 60)
    print("  PARALLEL GRASP-EVAL RESULTS (rank by lift height)")
    print("=" * 60)
    order = torch.argsort(rise, descending=True)
    results = []
    for rank, i in enumerate(order.tolist(), 1):
        r = float(rise[i])
        good = "HOLD" if r > 0.04 else ("partial" if r > 0.01 else "slip")
        results.append((i, cands[i]["label"], r, good))
        print(f"   #{rank}  env {i}  {cands[i]['label']:>12s}  "
              f"rose {r*100:+5.1f}cm   [{good}]")
    best_i = int(order[0])
    print("-" * 60)
    print(f"  WINNER: env {best_i}  {cands[best_i]['label']}  "
          f"(+{float(rise[best_i])*100:.1f}cm)")
    print(f"  {int((rise > 0.04).sum())}/{N} candidates held (>4cm); "
          f"{int((rise > 0.01).sum())}/{N} partial")
    print("=" * 60, flush=True)

    # hold so the user can see the final state
    for _ in range(int(3.0 / dt)):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
