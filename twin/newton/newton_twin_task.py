"""Newton twin — MULTI-STEP manipulation (a pick-and-place SEQUENCE).

Runs a sequence of pick-place steps in one rollout. Default task:
    1) pick the kiwi, put it IN the bowl
    2) pick the bowl (now holding the kiwi), put it ON the plate
All grasped objects (kiwi, bowl) are free hydroelastic bodies; the rest (plate, cup)
are static coacd colliders. Same gripper/contact recipe as newton_twin_hydro.

Single environment (you watch the whole task). Reuses helpers from newton_twin_hydro.

Run (newton-spike env):
  .../python twin/newton/newton_twin_task.py --scene_dir <out> --capture_dir <cap>
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import warp as wp
from dataclasses import replace
from pxr import Usd

import newton
import newton.ik as ik
import newton.usd
from newton.geometry import HydroelasticSDF
from newton.viewer import ViewerGL, ViewerNull

from newton_twin_hydro import load_scene_objects, load_obj_mesh, quat_to_vec4, qmul

SDF_RES = 64
SDF_BAND = (-0.01, 0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=1800.0)
    ap.add_argument("--grip_max", type=float, default=0.08)
    ap.add_argument("--steps", default="2:3:center:in,3:1:rim:on",
                    help="task steps 'tid:did:mode:rel,...' (default: kiwi->cup->plate on 174345)")
    args = ap.parse_args()
    grip_open = args.grip_max / 2.0
    N = 1

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    by_id = {o["id"]: o for o in objs}

    # TASK: list of (target_id, dest_id, grasp_mode, relation)
    TASK = [(int(a), int(b), m, rel) for a, b, m, rel in (s.split(":") for s in args.steps.split(","))]
    grasp_ids = sorted({t[0] for t in TASK})            # grasped -> free bodies
    static_ids = [oid for oid in by_id if oid not in grasp_ids]
    print("[TASK] steps:")
    for tid, did, m, rel in TASK:
        print(f"   pick '{by_id[tid]['label']}' ({m}) -> put {rel} '{by_id[did]['label']}'")

    wp.init()
    shape_cfg = newton.ModelBuilder.ShapeConfig(kh=1e11, gap=0.01, mu_torsional=0.0, mu_rolling=0.0)
    cfg_mesh = replace(shape_cfg, is_hydroelastic=True)
    # grasped objects: a bit LESS friction than the default mu=1.0 so they don't adhere to
    # the sticky hydroelastic pads and ride up on release (still grips fine: high pad normal
    # force + the rim straddle is form-closure). Fixes flaky "object stuck to gripper".
    cfg_obj = replace(shape_cfg, is_hydroelastic=True, mu=0.8)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg = shape_cfg

    t0 = by_id[TASK[0][0]]
    ROBOT_BASE = wp.vec3(t0["x"] - 0.50, t0["y"], 0.0)
    builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
                     xform=wp.transform(ROBOT_BASE, wp.quat_identity()),
                     enable_self_collisions=False, parse_visuals_as_colliders=True)

    def find_body(name):
        return next(i for i, l in enumerate(builder.body_label) if l.endswith(f"/{name}"))

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
    builder.joint_q[:9] = [*init_q, grip_open, grip_open]
    builder.joint_target_q[:9] = [*init_q, grip_open, grip_open]
    builder.joint_limit_upper[7] = grip_open
    builder.joint_limit_upper[8] = grip_open
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
    # The asset pad is ~2cm THICK (y), which shrank the effective gripper opening from 8cm
    # to ~6cm ("something inside the gripper") and left a 5.6cm object almost no release
    # clearance. Thin it to ~0.4cm so the gripper opens nearly fully and releases cleanly.
    pad_scale = pad_scale * np.array([1.0, 0.2, 1.0], dtype=np.float32)
    pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
    pad_mesh.build_sdf(max_resolution=SDF_RES, narrow_band_range=SDF_BAND, margin=shape_cfg.gap)
    pad_xform = wp.transform(wp.vec3(0.0, 0.005, 0.045),
                             wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi))
    builder.add_shape_mesh(body=lfi, mesh=pad_mesh, xform=pad_xform, cfg=cfg_mesh)
    builder.add_shape_mesh(body=rfi, mesh=pad_mesh, xform=pad_xform, cfg=cfg_mesh)

    model_single = copy.deepcopy(builder).finalize()

    # free bodies for grasped objects (kiwi, bowl)
    body_local = {}
    for oid in grasp_ids:
        o = by_id[oid]
        m, V = load_obj_mesh(o["mesh_path"])
        ext = (V.max(0) - V.min(0)).astype(float)
        o["radius"] = float(max(ext[0], ext[1]) / 2.0)
        o["ext"] = ext
        o["height"] = float(ext[2])
        m.build_sdf(max_resolution=SDF_RES, narrow_band_range=SDF_BAND, margin=shape_cfg.gap)
        b = builder.add_body(label=f"obj_{oid}",
                             xform=wp.transform(wp.vec3(o["x"], o["y"], o["base_z"] + 0.002), wp.quat_identity()))
        builder.add_shape_mesh(b, mesh=m, cfg=cfg_obj, color=wp.vec3(*o["color"]), label=f"obj_{oid}_mesh")
        body_local[oid] = b
        print(f"[TASK] free body '{o['label']}' id={oid} r={o['radius']:.3f} h={o['height']:.3f} base_z={o['base_z']:.3f}")

    # static obstacles (plate, cup) as hollow coacd colliders + visual
    ocfg = replace(shape_cfg, has_shape_collision=True)
    obs = []
    for oid in static_ids:
        o = by_id[oid]
        if not o["mesh_path"]:
            continue
        m, V = load_obj_mesh(o["mesh_path"])
        ext = (V.max(0) - V.min(0)).astype(float)
        o["radius"] = float(max(ext[0], ext[1]) / 2.0)
        o["height"] = float(ext[2])
        s = builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(o["x"], o["y"], o["base_z"]), wp.quat_identity()),
                                   mesh=m, cfg=ocfg, color=wp.vec3(*o["color"]), label=f"obs_{oid}")
        obs.append(s)
    if obs:
        builder.approximate_meshes(method="coacd", shape_indices=obs, keep_visual_shapes=True)

    bodies_per_world = builder.body_count
    scene = newton.ModelBuilder()
    scene.replicate(builder, N)
    scene.add_ground_plane(cfg=shape_cfg)
    model = scene.finalize()

    state_0 = model.state(); state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(t0["x"] - 0.55, t0["y"] - 0.75, 0.55), -30, 55)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    coll = newton.CollisionPipeline(model, reduce_contacts=True, broad_phase="explicit",
                                    sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=False))
    contacts = coll.contacts()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, solver="newton",
                                         integrator="implicitfast", cone="elliptic",
                                         njmax=700, nconmax=700, iterations=15, ls_iterations=100, impratio=1000.0)
    control = model.control()
    ctrl_size = control.joint_target_q.shape[0]; ctrl_per_world = ctrl_size // N
    wp.copy(dest=control.joint_target_q, src=model.joint_q, count=ctrl_size)

    EE = (find_body("fr3_hand_tcp") if any(l.endswith("/fr3_hand_tcp") for l in builder.body_label)
          else find_body("fr3_hand"))
    sts = model_single.state(); newton.eval_fk(model_single, model_single.joint_q, model_single.joint_qd, sts)
    ee_tf = wp.transform(*sts.body_q.numpy()[EE])
    DOWN = np.array([1.0, 0.0, 0.0, 0.0])
    pos_obj = ik.IKObjectivePosition(link_index=EE, link_offset=wp.vec3(0, 0, 0),
        target_positions=wp.array([wp.transform_get_translation(ee_tf)] * N, dtype=wp.vec3))
    rot_obj = ik.IKObjectiveRotation(link_index=EE, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([quat_to_vec4(DOWN)] * N, dtype=wp.vec4))
    ik_dofs = model_single.joint_coord_count
    jll = wp.array(np.tile(model_single.joint_limit_lower.numpy(), N).astype(np.float32), dtype=wp.float32)
    jul = wp.array(np.tile(model_single.joint_limit_upper.numpy(), N).astype(np.float32), dtype=wp.float32)
    lim_obj = ik.IKObjectiveJointLimit(joint_limit_lower=jll, joint_limit_upper=jul)
    jpw = model.joint_q.shape[0] // N
    jq = model.joint_q.numpy().reshape(N, jpw)[:, :ik_dofs]
    joint_q_ik = wp.array(jq.copy().astype(np.float32), dtype=wp.float32)
    ik_solver = ik.IKSolver(model=model_single, n_problems=N,
                            objectives=[pos_obj, rot_obj, lim_obj], lambda_initial=0.05,
                            jacobian_mode=ik.IKJacobianType.ANALYTIC)

    GRIP_OPEN, GRIP_CLOSE = grip_open, 0.0

    def yaw_q(yaw):
        return quat_to_vec4(qmul(np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]), DOWN))

    cur_grip = [GRIP_OPEN]

    def ik_solve(x, y, z, yaw):
        pos_obj.set_target_positions(wp.array([wp.vec3(x, y, z)] * N, dtype=wp.vec3))
        rot_obj.set_target_rotations(wp.array([yaw_q(yaw)] * N, dtype=wp.vec4))
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=48)
        return joint_q_ik.numpy()[:, :7].copy()

    def set_grip(grip):
        cur_grip[0] = grip
        jt = control.joint_target_q.reshape((N, ctrl_per_world))
        wp.copy(dest=jt[:, 7:9], src=wp.full((N, 2), value=grip, dtype=wp.float32))

    dt = 1.0 / 600.0; clock = [0.0]; stt = [state_0, state_1]

    def sim(n):
        for k in range(n):
            if k % 2 == 0:
                coll.collide(stt[0], contacts)
            stt[0].clear_forces()
            solver.step(stt[0], stt[1], control, contacts, dt)
            stt[0], stt[1] = stt[1], stt[0]
            clock[0] += dt
            if k % 10 == 0:
                viewer.begin_frame(clock[0]); viewer.log_state(stt[0])
                try: viewer.log_contacts(contacts, stt[0])
                except Exception: pass
                viewer.end_frame()

    def move(x, y, z, yaw, steps=240):
        """Smoothly drive the arm: ramp joint targets from the current pose to the IK
        goal with a smoothstep profile (ease in/out) so the motion isn't jerky."""
        goal = ik_solve(x, y, z, yaw)
        jt = control.joint_target_q.reshape((N, ctrl_per_world))
        cur = jt.numpy()[:, :7].copy()
        for i in range(1, steps + 1):
            s = i / steps; s = s * s * (3 - 2 * s)           # smoothstep ease in/out
            interp = (cur * (1 - s) + goal * s).astype(np.float32)
            wp.copy(dest=jt[:, :7], src=wp.array(interp, dtype=wp.float32))
            wp.copy(dest=jt[:, 7:9], src=wp.full((N, 2), value=cur_grip[0], dtype=wp.float32))
            sim(1)

    def grip_to(target, steps=140):
        """Close/open the gripper gradually (no jolt that flicks the object)."""
        jt = control.joint_target_q.reshape((N, ctrl_per_world))
        g0 = cur_grip[0]
        for i in range(1, steps + 1):
            g = float(g0 + (target - g0) * (i / steps))
            wp.copy(dest=jt[:, 7:9], src=wp.full((N, 2), value=g, dtype=wp.float32))
            sim(1)
        cur_grip[0] = target

    def pose(oid):
        return stt[0].body_q.numpy()[body_local[oid]][:3]

    print("[TASK] settle..."); set_grip(GRIP_OPEN); sim(150)

    for si, (tid, did, mode, rel) in enumerate(TASK, 1):
        o = by_id[tid]; d = by_id[did]
        tp = pose(tid)
        if mode == "rim":
            wall = o["ext"][0] / 2.0                  # near (-x) wall distance on the approach axis
            yaw = np.pi / 2; gx = tp[0] - wall; gy = tp[1]
            gz = tp[2] + 0.55 * o["height"]          # TCP==fingertip -> grip mid-wall at the rim
            place_dx = -wall                          # vessel center sits +wall from the TCP
        else:
            yaw = 0.0; gx = tp[0] + 0.3 * o["radius"]; gy = tp[1]   # +offset = known-good grip
            gz = tp[2] + 0.5 * o["height"]
            place_dx = 0.0
        grasp_off = gz - tp[2]
        carry_z = max(gz, tp[2] + o["height"]) + 0.30
        dbase = pose(did)[2] if did in body_local else d["base_z"]
        dxy = pose(did)[:2] if did in body_local else np.array([d["x"], d["y"]])
        # Placement height of the object base:
        #   "in"  -> just above the container RIM, centred, so it drops to the bottom and
        #            settles low/centred (the gripper can't reach a deep cup's floor without
        #            hitting the walls; thin pads now release cleanly so a short drop is fine)
        #   "on"  -> rest right on top of the destination (set down, don't drop)
        obj_base_target = (dbase + d["height"] + 0.025) if rel == "in" else (dbase + d["height"] - 0.006)

        print(f"[TASK] step {si}: pick '{o['label']}' ({mode}) @({gx:+.3f},{gy:+.3f},{gz:+.3f})")
        move(gx, gy, gz + 0.16, yaw, 300)         # approach above
        move(gx, gy, gz, yaw, 240)                # descend
        grip_to(GRIP_CLOSE, 160); sim(60)         # close gently
        move(gx, gy, carry_z, yaw, 340)           # lift
        zlift = pose(tid)[2]
        held = zlift - tp[2] > 0.04
        print(f"       lift: {o['label']} rose {(zlift-tp[2])*100:+.1f}cm  [{'HELD' if held else 'SLIP'}]")
        print(f"[TASK] step {si}: place {rel} '{d['label']}' @({dxy[0]:+.3f},{dxy[1]:+.3f})")
        # Measure where the object ACTUALLY sits relative to the TCP (it slips/shifts in the
        # grip during lift+carry) in BOTH xy and z, so we can CENTER it on the destination
        # and set it down to rest -- instead of placing it off-centre near the edge (tips off)
        # or releasing it from a stale height (drops).
        op = pose(tid)
        off_xy = op[:2] - np.array([gx, gy])      # object centre relative to the TCP, world xy
        off_below = carry_z - op[2]               # object base below the TCP
        tx, ty = float(dxy[0] - off_xy[0]), float(dxy[1] - off_xy[1])   # centre object on dest
        rel_tcp_z = obj_base_target + off_below
        move(tx, ty, carry_z, yaw, 380)           # traverse, centred over dest
        move(tx, ty, rel_tcp_z, yaw, 300)         # lower until the object rests on dest
        sim(200)                                  # settle on the surface while still gripped
        grip_to(GRIP_OPEN, 160)                   # THEN open to release
        sim(300)                                  # long dwell: let it fully separate/settle
        move(tx, ty, carry_z + 0.06, yaw, 240)    # retreat straight up, clear
        sim(180)
        rp = pose(tid)
        print(f"       released: {o['label']} now at z={rp[2]*100:.1f}cm, "
              f"{np.linalg.norm(rp[:2]-dxy)*100:.1f}cm from dest center")

    # report
    print("\n" + "=" * 64); print("  NEWTON MULTI-STEP TASK RESULT"); print("=" * 64)
    for tid, did, mode, rel in TASK:
        o = by_id[tid]; d = by_id[did]
        op = pose(tid)
        dxy = pose(did)[:2] if did in body_local else np.array([d["x"], d["y"]])
        dist = float(np.linalg.norm(op[:2] - dxy))
        ok = dist < (d["radius"] + o["radius"])
        print(f"  '{o['label']}' {rel} '{d['label']}': xy-offset {dist*100:.1f}cm  z={op[2]*100:.1f}cm  "
              f"[{'OK' if ok else 'MISSED'}]")
    print("=" * 64, flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[TASK] holding GUI open {args.hold:.0f}s", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            sim(10)


if __name__ == "__main__":
    main()
