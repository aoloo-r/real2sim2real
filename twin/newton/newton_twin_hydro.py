"""Newton-native twin — increment 4: HYDROELASTIC grasp on the ACTUAL SAM3D meshes.

Increment 3 grasped a cylinder stand-in; this loads the real reconstructed mesh.obj for
the target (free, hydroelastic, SDF-backed) and for the other scene objects (real meshes
as context — physical convex colliders by default, or visual-only via --obstacles). Meshes
are metric (physical_scale_baked) and recentred to a z-up rest pose; stacking is inferred
so an object resting on another (e.g. sphere on tray) starts at the right height.


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
import trimesh
import warp as wp
from pxr import Usd

import newton
import newton.ik as ik
import newton.usd
from newton.geometry import HydroelasticSDF
from newton.viewer import ViewerGL, ViewerNull


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
        mp = o.get("mesh_path")
        mesh_abs = os.path.join(scene_dir, mp) if mp else None
        col = o.get("display_color") or [0.6, 0.6, 0.6]
        radius, height = w / 2.0, h
        if mesh_abs and os.path.exists(mesh_abs):
            try:                       # mesh extent is ground truth; depth height can be wrong
                ext = _mesh_extent(mesh_abs)
                radius, height = float(max(ext[0], ext[1]) / 2.0), float(ext[2])
            except Exception:
                pass
        objs.append({"id": o["id"], "label": str(o.get("label", "")),
                     "x": float(P[0]), "y": float(P[1]), "real_z": float(P[2]),
                     "radius": radius, "height": height,
                     "mesh_path": mesh_abs, "color": [float(c) for c in col[:3]]})
    # stacking: a small object whose footprint sits inside a larger, flatter one AND is
    # physically higher (larger base-frame z) rests ON it -> its base_z is that support's top.
    for o in objs:
        o["base_z"] = 0.0
    for a in objs:
        for b in objs:
            if a is b or b["radius"] <= a["radius"]:
                continue
            d = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            if d < (b["radius"] - 0.4 * a["radius"]) and a["real_z"] > b["real_z"] + 0.005:
                a["base_z"] = b["height"]
    return objs


def _mesh_extent(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    v = np.asarray(m.vertices, float)
    return v.max(0) - v.min(0)


def load_obj_mesh(path, recenter=True):
    """SAM3D mesh.obj -> newton.Mesh. Vertices are already metric (physical_scale_baked).
    Recenter to XY-center + base at local z=0 so placement is deterministic (z-up rest pose)."""
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    V = np.asarray(m.vertices, dtype=np.float32).copy()
    F = np.asarray(m.faces, dtype=np.int32).reshape(-1)
    if recenter and len(V):
        V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) * 0.5
        V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) * 0.5
        V[:, 2] -= V[:, 2].min()
    return newton.Mesh(V, F), V


def quat_to_vec4(q):
    return wp.vec4(q[0], q[1], q[2], q[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--target_id", type=int, default=2)
    ap.add_argument("--world_count", type=int, default=8)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"],
                    help="gl = live GUI window (DISPLAY :1); null = headless")
    ap.add_argument("--hold", type=float, default=20.0,
                    help="seconds to keep the GUI open at the end for inspection")
    ap.add_argument("--obstacles", default="collide", choices=["collide", "visual", "none"],
                    help="other scene objects: physical colliders, visual-only, or omitted")
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

    # IK model = ROBOT ONLY (deepcopy BEFORE adding any scene object) -> joint_coord_count = 9
    model_single = copy.deepcopy(builder).finalize()

    # --- TARGET: the actual SAM3D mesh as a free, hydroelastic body ---
    tgt_mesh, tgt_V = load_obj_mesh(tgt["mesh_path"])
    ext = (tgt_V.max(0) - tgt_V.min(0)).astype(float)
    tgt["radius"] = float(max(ext[0], ext[1]) / 2.0)   # actual mesh extent drives the grasp
    tgt["height"] = float(ext[2])
    tgt_mesh.build_sdf(max_resolution=SDF_RES, narrow_band_range=SDF_BAND, margin=shape_cfg.gap)
    obj_z0 = tgt["base_z"] + 0.002                       # mesh base rests on its support
    obj_body = builder.add_body(label="target",
                               xform=wp.transform(wp.vec3(tgt["x"], tgt["y"], obj_z0), wp.quat_identity()))
    builder.add_shape_mesh(obj_body, mesh=tgt_mesh, cfg=cfg_mesh,
                           color=wp.vec3(*tgt["color"]), label="target_mesh")
    object_body_local = obj_body
    print(f"[HYDRO] target mesh '{tgt['label']}': {len(tgt_V)} verts, "
          f"extent {ext.round(3)}, base_z={tgt['base_z']:.3f}")

    # --- OTHER objects: their real SAM3D meshes as scene context ---
    obstacle_shapes = []
    if args.obstacles != "none":
        ocfg = replace(shape_cfg, has_shape_collision=(args.obstacles == "collide"))
        for o in objs:
            if o["id"] == tgt["id"] or not o["mesh_path"] or not os.path.exists(o["mesh_path"]):
                continue
            om, _ = load_obj_mesh(o["mesh_path"])
            sidx = builder.add_shape_mesh(
                -1, xform=wp.transform(wp.vec3(o["x"], o["y"], o["base_z"]), wp.quat_identity()),
                mesh=om, cfg=ocfg, color=wp.vec3(*o["color"]), label="obs_%d" % o["id"])
            obstacle_shapes.append(sidx)
        if args.obstacles == "collide" and obstacle_shapes:
            builder.approximate_meshes(method="convex_hull", shape_indices=obstacle_shapes,
                                       keep_visual_shapes=True)
        print(f"[HYDRO] loaded {len(obstacle_shapes)} context object meshes "
              f"({args.obstacles})")

    bodies_per_world = builder.body_count
    scene = newton.ModelBuilder()
    scene.replicate(builder, args.world_count)
    scene.add_ground_plane(cfg=shape_cfg)
    model = scene.finalize()
    print(f"[HYDRO] {args.world_count} worlds x {bodies_per_world} bodies; object_body_local={object_body_local}")

    state_0 = model.state(); state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # live GUI viewer (shows on DISPLAY :1) or headless
    if args.viewer == "gl":
        viewer = ViewerGL()
        viewer.set_model(model)
        import math as _m
        ncol = int(_m.ceil(_m.sqrt(args.world_count)))
        nrow = int(_m.ceil(args.world_count / ncol))
        viewer.set_world_offsets(wp.vec3(1.2, 1.2, 0.0))
        # frame the whole grid: aim at its center, stand back ~2.5m up ~1.8m
        cx = tgt["x"] + 1.2 * (ncol - 1) / 2.0
        cy = tgt["y"] + 1.2 * (nrow - 1) / 2.0
        viewer.set_camera(wp.vec3(cx - 1.8, cy - 2.4, 1.8), -28, 60)
    else:
        viewer = ViewerNull(num_frames=10**9)
        viewer.set_model(model)

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

    sim_clock = [0.0]
    render_every = 10   # render one frame per 10 sub-steps (~60 fps view)

    def sim(n):
        nonlocal state_0, state_1, max_z
        for k in range(n):
            if k % collide_every == 0:
                coll.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
            sim_clock[0] += dt
            if k % render_every == 0:
                viewer.begin_frame(sim_clock[0])
                viewer.log_state(state_0)
                try:
                    viewer.log_contacts(contacts, state_0)
                except Exception:
                    pass
                viewer.end_frame()
        bq = state_0.body_q.numpy()
        zs = np.array([bq[w * bodies_per_world + object_body_local, 2] for w in range(args.world_count)])
        max_z = np.maximum(max_z, zs)
        return zs

    def obj_z():
        bq = state_0.body_q.numpy()
        return np.array([bq[w * bodies_per_world + object_body_local, 2] for w in range(args.world_count)])

    offs = np.linspace(-tgt["radius"] * 0.6, tgt["radius"] * 0.6, args.world_count)
    gz = tgt["base_z"] + tgt["height"] / 2.0          # grasp at object mid-height

    print("[HYDRO] settle..."); set_grip(GRIP_OPEN); sim(120)
    z_rest = obj_z().copy()
    max_z = z_rest.copy()          # measure peak lift from the SETTLED rest pose, not spawn
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

    # keep the GUI open so the final grasped state can be inspected / orbited
    if args.viewer == "gl" and args.hold > 0:
        print(f"[HYDRO] holding GUI open {args.hold:.0f}s — orbit to inspect (Ctrl-C to close)", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            sim(render_every)


if __name__ == "__main__":
    main()
