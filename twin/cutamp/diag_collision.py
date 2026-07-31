"""Pin the movable_to_world collision: build the real Santa-scene env + world exactly as
run_cutamp does, then for each movable check its FLAT and FOLDED collision spheres against
the static world collider at (a) the object's initial pose and (b) a representative in-box
placement — printing which object/geometry penetrates a static and by how much.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scene_tamp import load_real_scene_env
from cutamp.tamp_world import TAMPWorld
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.robots import get_q_home, load_robot_container
from cutamp.utils.common import transform_spheres, pose_list_to_mat4x4
from cutamp.fold_registry import FOLDABLE
from curobo.types.base import TensorDeviceType

ta = TensorDeviceType()
env = load_real_scene_env("/home/aoloo/sam-3d-objects/outputs/robot_20260731_143933", box_cx=0.38)
cfg = TAMPConfiguration(num_particles=8, robot="panda", grasp_dof=4, approach="optimization",
                        num_opt_steps=1, num_initial_plans=1, curobo_plan=False,
                        warmup_motion_gen=False, enable_visualizer=False,
                        enable_experiment_logging=False)
validate_tamp_config(cfg)
robot = load_robot_container(cfg.robot, ta)
q0 = ta.to_device(get_q_home(cfg.robot))
world = TAMPWorld(env, ta, robot=robot, q_init=q0,
                  collision_activation_distance=cfg.world_activation_distance,
                  coll_n_spheres=cfg.coll_n_spheres, coll_sphere_radius=cfg.coll_sphere_radius)

box = next(s for s in env.statics if s.name == "box_floor")
bx, by = box.pose[0], box.pose[1]
print(f"[diag] statics: {[s.name for s in env.statics]}")
print(f"[diag] box_floor centre ({bx:.3f},{by:.3f}) dims {[round(d,3) for d in box.dims]}")

def check(name, sph, x, y, z, tag):
    import torch
    mat = pose_list_to_mat4x4([x, y, z, 1, 0, 0, 0]).to(world.device)
    s = transform_spheres(sph.to(world.device), mat)[None, None].contiguous()
    cost = float(world.collision_fn(s).sum())
    print(f"[diag]   {name:14} {tag:16} @({x:.2f},{y:.2f},{z:.2f}) -> cost={cost:.4f} "
          f"{'<-- COLLIDES' if cost > 1e-4 else 'ok'}")

for obj in world.movables:
    ip = obj.pose
    check(obj.name, world.get_collision_spheres(obj.name), ip[0], ip[1], ip[2], "flat@initial")
    if obj.name in FOLDABLE:
        fs = FOLDABLE[obj.name]["folded_spheres"]
        # folded bundle placed in the box (rest on box floor top)
        zrest = -(fs[:, 2] - fs[:, 3]).min().item() + box.pose[2] + box.dims[2] / 2 + 0.001
        check(obj.name, fs, bx, by, zrest, "folded@box-centre")
