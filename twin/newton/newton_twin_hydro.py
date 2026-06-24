"""Newton-native twin — increment 3: HYDROELASTIC grasp (grasps that hold).

Ports the working contact recipe from Newton's `robot_panda_hydro` example (the one
that grasps a cup) into our twin: gripper PAD meshes on the fingers, SDF + hydroelastic
contact on hand/fingers/object, an explicit CollisionPipeline (broad_phase="explicit",
sdf_hydroelastic_config), and SolverMuJoCo(use_mujoco_contacts=False). Combined with our
per-world grasp search (each world a different lateral grasp offset), scored by the
object's MAX lift height.

Fixes increment-2's "fingers pass through the object" — bare fr3 fingers have no usable
contact geometry; pads + hydroelastic contact give a real grip.

Run (newton-spike env):
  .../newton-spike/bin/python twin/newton/newton_twin_hydro.py \
    --scene_dir <out> --capture_dir <cap> --target_id 2 --world_count 8
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import replace

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.ik as ik
import newton.usd
from newton.geometry import HydroelasticSDF


def load_scene_objects(scene_dir, capture_dir, base_frame="ur5e_base_link"):
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    T = (layout.get("camera_extrinsics") or {}).get("T_base_cam")
    if T is None:
        ex = json.load(open(os.path.join(capture_dir, "extrinsics.json")))
        T = ex["transforms"][base_frame]["T_base_cam"]
    T = np.asarray(T, float)
    objs = []
    for o in layout.get("objects", []):
        di = o.get("depth_info") or {}
        pc = di.get("position_cam") or (o.get("icp_pose") or {}).get("position_cam")
        if not pc:
            continue
        P = T @ np.array([pc[0], pc[1], pc[2], 1.0])
        w = max(0.02, min(0.40, float(di.get("physical_width_m") or o.get("physical_size_m") or 0.06)))
        h = max(0.02, min(0.40, float(di.get("physical_height_m") or o.get("physical_size_m") or 0.06)))
        objs.append({"id": o["id"], "label": str(o.get("label", "")),
                     "x": float(P[0]), "y": float(P[1]),
                     "radius": w / 2.0, "height": h})
    return objs


def quat_to_vec4(q):
    return wp.vec4(q[0], q[1], q[2], q[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--target_id", type=int, default=2)
    ap.add_argument("--world_count", type=int, default=8)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    args = ap.parse_args()

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    tgt = next((o for o in objs if o["id"] == args.target_id), objs[-1])
    print(f"[HYDRO] target: id={tgt['id']} '{tgt['label']}' r={tgt['radius']:.3f} h={tgt['height']:.3f}")

    wp.init()
    SDF_RES = 64
    SDF_BAND = (-0.01, 0.01)

    shape_cfg = newton.ModelBuilder.ShapeConfig(kh=1e11, gap=0.01,
                                                mu_torsional=0.0, mu_rolling=0.0)
    cfg_mesh = replace(shape_cfg, is_hydroelastic=True)
    cfg_prim = replace(shape_cfg, is_hydroelastic=True,
                       sdf_max_resolution=SDF_RES, sdf_narrow_band_range=SDF_BAND)

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg = shape_cfg

    ROBOT_BASE = wp.vec3(tgt["x"] - 0.50, tgt["y"], 0.0)
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(ROBOT_BASE, wp.quat_identity()),
        enable_self_collisions=False, parse_visuals_as_colliders=True)

    def find_body(name):
        return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

    finger_bodies = {find_body("fr3_leftfinger"), find_body("fr3_rightfinger"), find_body("fr3_hand")}
    non_finger = []
    for sidx, bidx in enumerate(builder.shape_body):
        if bidx in finger_bodies and builder.shape_type[sidx] == newton.GeoType.MESH:
            mesh = builder.shape_source[sidx]
            if mesh is not None and mesh.sdf is None:
                sc = np.asarray(builder.shape_scale[sidx], dtype=np.float32)
                if not np.allclose(sc, 1.0):
                    mesh = mesh.copy(vertices=mesh.vertices * sc, recompute_inertia=True)
                    builder.shape_source[sidx] = mesh
                    builder.shape_scale[sidx] = (1.0, 1.0, 1.0)
                mesh.build_sdf(max_resolution=SDF_RES, narrow_band_range=SDF_BAND, margin=shape_cfg.gap)
            builder.shape_flags[sidx] |= newton.ShapeFlags.HYDROELASTIC
        elif bidx not in finger_bodies:
            non_finger.append(sidx)
    builder.approximate_meshes(method="convex_hull", shape_indices=non_finger, keep_visual_shapes=True)

    init_q = [-3.68e-03, 2.39e-02, 3.68e-03, -2.3683, -1.29e-04, 2.3922, 0.7855]
    builder.joint_q[:9] = [*init_q, 0.04, 0.04]
    builder.joint_target_q[:9] = [*init_q, 0.04, 0.04]
    builder.joint_target_ke[:9] = [650.0] * 9
    builder.joint_target_kd[:9] = [100.0] * 9
    builder.joint_effort_limit[:7] = [80.0] * 7
    builder.joint_effort_limit[7:9] = [20.0] * 2
    builder.joint_armature[:7] = [0.1] * 7
    builder.joint_armature[7:9] = [0.5] * 2

    lfi, rfi = find_body("fr3_leftfinger"), find_body("fr3_rightfinger")
    pad_path = newton.utils.download_asset("manipulation_objects/pad")
    pad_stage = Usd.Stage.Open(str(pad_path / "model.usda"))
    pad_mesh = newton.usd.get_mesh(pad_stage.GetPrimAtPath("/root/Model/Model"),
                                   load_normals=True, face_varying_normal_conversion="vertex_splitting")
    pad_scale = np.asarray(newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
    if not np.allclose(pad_scale, 1.0):
        pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
    pad_mesh.build_sdf(max_resolution=SDF_RES, narrow_band_range=SDF_BAND, margin=shape_cfg.gap)
    pad_xform = wp.transform(wp.vec3(0.0, 0.005, 0.045),
                             wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi))
    builder.add_shape_mesh(body=lfi, mesh=pad_mesh, xform=pad_xform, cfg=cfg_mesh)
    builder.add_shape_mesh(body=rfi, mesh=pad_mesh, xform=pad_xform, cfg=cfg_mesh)

    # IK model = ROBOT ONLY (deepcopy BEFORE adding the object) -> joint_coord_count = 9
    model_single = copy.deepcopy(builder).finalize()

    # now add the target object (a free body) to the per-world builder
    obj_z0 = tgt["height"] / 2.0 + 0.002
    obj_body = builder.add_body(label="target",
                               xform=wp.transform(wp.vec3(tgt["x"], tgt["y"], obj_z0), wp.quat_identity()))
    builder.add_shape_cylinder(obj_body, radius=tgt["radius"], half_height=tgt["height"] / 2.0, cfg=cfg_prim)
    object_body_local = obj_body

    bodies_per_world = builder.body_count
    scene = newton.ModelBuilder()
    scene.replicate(builder, args.world_count)
    scene.add_ground_plane(cfg=shape_cfg)
    model = scene.finalize()
    print(f"[HYDRO] {args.world_count} worlds x {bodies_per_world} bodies; object_body_local={object_body_local}")

    state_0 = model.state(); state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    coll = newton.CollisionPipeline(model, reduce_contacts=True, broad_phase="explicit",
                                    sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=False))
    contacts = coll.contacts()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, solver="newton",
                                         integrator="implicitfast", cone="elliptic",
                                         njmax=500, nconmax=500, iterations=15,
                                         ls_iterations=100, impratio=1000.0)
    control = model.control()
    ctrl_size = control.joint_target_q.shape[0]
    ctrl_per_world = ctrl_size // args.world_count
    wp.copy(dest=control.joint_target_q, src=model.joint_q, count=ctrl_size)

    EE = (find_body("fr3_hand_tcp") if any(l.endswith("/fr3_hand_tcp") for l in builder.body_label)
          else find_body("fr3_hand"))
    st = model_single.state(); newton.eval_fk(model_single, model_single.joint_q, model_single.joint_qd, st)
    ee_tf = wp.transform(*st.body_q.numpy()[EE])
    DOWN = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
    pos_obj = ik.IKObjectivePosition(link_index=EE, link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.transform_get_translation(ee_tf)] * args.world_count, dtype=wp.vec3))
    rot_obj = ik.IKObjectiveRotation(link_index=EE, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([quat_to_vec4(DOWN)] * args.world_count, dtype=wp.vec4))
    ik_dofs = model_single.joint_coord_count           # 9 (robot only)
    # per-problem joint limits: tile the single robot's limits across worlds (numpy ->
    # warp, avoiding warp 2D slicing).
    jl_lo = model_single.joint_limit_lower.numpy()
    jl_hi = model_single.joint_limit_upper.numpy()
    jll = wp.array(np.tile(jl_lo, args.world_count).astype(np.float32), dtype=wp.float32)
    jul = wp.array(np.tile(jl_hi, args.world_count).astype(np.float32), dtype=wp.float32)
    lim_obj = ik.IKObjectiveJointLimit(joint_limit_lower=jll, joint_limit_upper=jul)
    # initial IK joint state: first ik_dofs coords of each world's full joint block
    jpw = model.joint_q.shape[0] // args.world_count   # full joint coords per world (16)
    jq = model.joint_q.numpy().reshape(args.world_count, jpw)[:, :ik_dofs]
    joint_q_ik = wp.array(jq.copy().astype(np.float32), dtype=wp.float32)
    ik_solver = ik.IKSolver(model=model_single, n_problems=args.world_count,
                             objectives=[pos_obj, rot_obj, lim_obj], lambda_initial=0.05,
                             jacobian_mode=ik.IKJacobianType.ANALYTIC)

    GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0

    def set_targets(positions, grip):
        pos_obj.set_target_positions(wp.array(positions, dtype=wp.vec3))
        rot_obj.set_target_rotations(wp.array([quat_to_vec4(DOWN)] * args.world_count, dtype=wp.vec4))
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=48)
        jt = control.joint_target_q.reshape((args.world_count, ctrl_per_world))
        wp.copy(dest=jt[:, :7], src=joint_q_ik[:, :7])
        gv = wp.full((args.world_count, 2), value=grip, dtype=wp.float32)
        wp.copy(dest=jt[:, 7:9], src=gv)

    def set_grip(grip):
        jt = control.joint_target_q.reshape((args.world_count, ctrl_per_world))
        gv = wp.full((args.world_count, 2), value=grip, dtype=wp.float32)
        wp.copy(dest=jt[:, 7:9], src=gv)

    dt = 1.0 / 600.0
    collide_every = 2
    max_z = np.array([obj_z0] * args.world_count)

    def sim(n):
        nonlocal state_0, state_1, max_z
        for k in range(n):
            if k % collide_every == 0:
                coll.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy()
        zs = np.array([bq[w * bodies_per_world + object_body_local, 2] for w in range(args.world_count)])
        max_z = np.maximum(max_z, zs)
        return zs

    def obj_z():
        bq = state_0.body_q.numpy()
        return np.array([bq[w * bodies_per_world + object_body_local, 2] for w in range(args.world_count)])

    offs = np.linspace(-tgt["radius"] * 0.6, tgt["radius"] * 0.6, args.world_count)
    gz = tgt["height"] / 2.0 + 0.002

    print("[HYDRO] settle..."); set_grip(GRIP_OPEN); sim(120)
    z_rest = obj_z().copy()
    print("[HYDRO] approach...")
    set_targets([wp.vec3(tgt["x"] + float(d), tgt["y"], gz + 0.16) for d in offs], GRIP_OPEN)
    sim(300)
    print("[HYDRO] descend...")
    set_targets([wp.vec3(tgt["x"] + float(d), tgt["y"], gz) for d in offs], GRIP_OPEN)
    sim(240)
    print("[HYDRO] close...")
    set_grip(GRIP_CLOSE); sim(180)
    print("[HYDRO] lift...")
    set_targets([wp.vec3(tgt["x"] + float(d), tgt["y"], gz + 0.22) for d in offs], GRIP_CLOSE)
    t0 = time.time(); sim(360); t_lift = time.time() - t0
    z_lift = obj_z()

    order = np.argsort(max_z - z_rest)[::-1]
    print("\n" + "=" * 64)
    print(f"  NEWTON HYDROELASTIC GRASP  (N={args.world_count}, target '{tgt['label']}', "
          f"lift phase {t_lift:.1f}s)")
    print("=" * 64)
    for rank, w in enumerate(order, 1):
        pk = float(max_z[w] - z_rest[w]); fin = float(z_lift[w] - z_rest[w])
        tag = "HELD" if pk > 0.04 else ("partial" if pk > 0.01 else "slip")
        print(f"  #{rank}  world {w}  dx={offs[w]*100:+.1f}cm  peak +{pk*100:4.1f}cm  "
              f"final +{fin*100:4.1f}cm  [{tag}]")
    best = int(order[0])
    print("-" * 64)
    print(f"  WINNER: world {best} dx={offs[best]*100:+.1f}cm peak +{float(max_z[best]-z_rest[best])*100:.1f}cm")
    print(f"  {int((max_z-z_rest > 0.04).sum())}/{args.world_count} held (peak>4cm)")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
