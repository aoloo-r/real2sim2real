"""De-risking spike for the parallel digital-twin grasp-evaluation harness.

Goal: prove our installed Isaac Lab + cuRobo can (1) clone a scene into N parallel
environments and step them with BATCHED tensors, and (2) batch-plan N goals with
cuRobo in one GPU call. If both pass, the full N-trial twin harness is buildable
on this stack.

Run (GUI on :1, small N):
  DISPLAY=:1 ./isaaclab.sh -p scripts/twin_parallel_spike.py --num_envs 8 --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel twin spike")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=150)
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


@configclass
class TwinSpikeSceneCfg(InteractiveSceneCfg):
    """Minimal twin: ground + light + Franka + a falling cube, per env."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3)),
    )


def test_curobo_batch(num_envs: int):
    """Confirm cuRobo can plan N goals in one batched call on this install."""
    print("\n[SPIKE] --- cuRobo batch-planning test ---", flush=True)
    try:
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose as CuPose
        from curobo.types.state import JointState as CuJointState
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.wrap.reacher.motion_gen import (
            MotionGen, MotionGenConfig, MotionGenPlanConfig)

        tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        world_config = WorldConfig(cuboid=[Cuboid(
            name="table", dims=[1.2, 1.2, 1.05],
            pose=[0.5, 0.0, -0.545, 1, 0, 0, 0])])
        mg_cfg = MotionGenConfig.load_from_robot_config(
            "franka.yml", world_model=world_config, tensor_args=tensor_args,
            num_trajopt_seeds=4, num_graph_seeds=4, interpolation_dt=0.02,
            use_cuda_graph=False)
        mg = MotionGen(mg_cfg)
        mg.warmup(warmup_js_trajopt=False, batch=num_envs)

        home = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
                            device="cuda")
        start = CuJointState(
            position=home.unsqueeze(0).repeat(num_envs, 1),
            velocity=torch.zeros((num_envs, 7), device="cuda"),
            acceleration=torch.zeros((num_envs, 7), device="cuda"),
            joint_names=["panda_joint1", "panda_joint2", "panda_joint3",
                         "panda_joint4", "panda_joint5", "panda_joint6",
                         "panda_joint7"])
        # N different goal positions spread across the workspace
        xs = torch.linspace(0.40, 0.60, num_envs)
        ys = torch.linspace(-0.20, 0.20, num_envs)
        goal_pos = torch.stack([xs, ys, torch.full((num_envs,), 0.25)], dim=1).cuda()
        goal_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda").repeat(num_envs, 1)
        goals = CuPose(position=goal_pos, quaternion=goal_quat)

        plan_cfg = MotionGenPlanConfig(enable_graph=True, max_attempts=10)
        result = mg.plan_batch(start, goals, plan_cfg)
        succ = result.success
        print(f"[SPIKE] cuRobo plan_batch: {int(succ.sum().item())}/{num_envs} goals "
              f"planned in one batched call", flush=True)
        return int(succ.sum().item())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[SPIKE] cuRobo batch test FAILED: {e}", flush=True)
        return -1


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])

    scene_cfg = TwinSpikeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    cube = scene["cube"]
    print("\n[SPIKE] --- multi-env tensor shapes ---", flush=True)
    print(f"[SPIKE] num_envs requested = {args_cli.num_envs}", flush=True)
    print(f"[SPIKE] robot.data.joint_pos shape = {tuple(robot.data.joint_pos.shape)} "
          f"(expect [{args_cli.num_envs}, dof])", flush=True)
    print(f"[SPIKE] cube.data.root_pos_w  shape = {tuple(cube.data.root_pos_w.shape)} "
          f"(expect [{args_cli.num_envs}, 3])", flush=True)

    z0 = cube.data.root_pos_w[:, 2].clone()
    dt = 1.0 / 120.0
    for _ in range(args_cli.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    z1 = cube.data.root_pos_w[:, 2]
    print(f"\n[SPIKE] cube z per env after {args_cli.steps} steps "
          f"(fell under gravity, should be ~table level & similar):", flush=True)
    print("[SPIKE]   " + ", ".join(f"{v:.3f}" for v in z1.tolist()), flush=True)
    parallel_ok = (robot.data.joint_pos.shape[0] == args_cli.num_envs
                   and cube.data.root_pos_w.shape[0] == args_cli.num_envs
                   and bool((z1 < z0).all()))
    print(f"[SPIKE] PARALLEL STEP OK: {parallel_ok}", flush=True)

    n_planned = test_curobo_batch(args_cli.num_envs)

    print("\n[SPIKE] ===== SUMMARY =====", flush=True)
    print(f"[SPIKE] N-env clone + batched step : {'PASS' if parallel_ok else 'FAIL'}",
          flush=True)
    print(f"[SPIKE] cuRobo batch planning      : "
          f"{'PASS (' + str(n_planned) + '/' + str(args_cli.num_envs) + ')' if n_planned > 0 else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
