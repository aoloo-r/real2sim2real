"""Newton-native twin — increment 2: robot + parallel grasp execution.

N parallel Newton worlds, each with a Franka Panda + the real SAM3D scene objects.
Each world tries a DIFFERENT grasp offset on the target object (cup/cylinder) —
the Newton-native equivalent of twin_grasp_eval.py on PhysX. The robot is driven by
Newton's built-in IK solver (no cuRobo/IsaacLab). Scores by object z-rise. Reports
timing vs PhysX (6.4 ms/step @8 envs) and the grasp ranking.

Runs in the `newton-spike` conda env (Newton 1.3 / Warp 1.14).

Run:
  /home/aoloo/miniforge3/envs/newton-spike/bin/python twin/newton/newton_twin_grasp.py \
    --scene_dir <out> --capture_dir <cap> [--target_id 3] [--world_count 8]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
import newton.ik as ik


VESSEL_KW = ("cup", "bowl", "plate", "mug", "glass", "container", "dish")


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
                     "x": float(P[0]), "y": float(P[1]), "real_z": float(P[2]),
                     "radius": w / 2.0, "height": h})
    for o in objs:
        o["z"] = o["height"] / 2.0 + 0.002
    for a in objs:
        for b in objs:
            if a is b or b["radius"] <= a["radius"]:
                continue
            d = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            if d < (b["radius"] - 0.4 * a["radius"]) and a["real_z"] > b["real_z"] + 0.005:
                a["z"] = b["height"] + a["height"] / 2.0 + 0.005
    return objs


def download_franka():
    p = newton.utils.download_asset("franka_emika_panda")
    return str(p / "urdf" / "fr3_franka_hand.urdf")


def build_robot(base_pos, use_mujoco=True):
    """Single-world Franka subbuilder with gravity compensation."""
    rb = newton.ModelBuilder()
    if use_mujoco:
        newton.solvers.SolverMuJoCo.register_custom_attributes(rb)
    rb.add_urdf(
        download_franka(),
        xform=wp.transform(base_pos, wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )
    n_dof = 9  # 7 arm + 2 finger
    home = [-0.004, 0.024, 0.004, -2.368, -0.000, 2.392, 0.785, 0.05, 0.05]
    rb.joint_q[:n_dof] = home
    rb.joint_target_q[:n_dof] = home
    rb.joint_target_ke[:n_dof] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
    rb.joint_target_kd[:n_dof] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
    rb.joint_effort_limit[:n_dof] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
    rb.joint_armature[:n_dof] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
    if use_mujoco:
        gj = rb.custom_attributes["mujoco:jnt_actgravcomp"]
        if gj.values is None:
            gj.values = {}
        for i in range(7):
            gj.values[i] = True
        gb = rb.custom_attributes["mujoco:gravcomp"]
        if gb.values is None:
            gb.values = {}
        for i in range(2, 14):
            gb.values[i] = 1.0
        # torsional friction (condim=4) on the finger shapes so they grip, not slip
        cd = rb.custom_attributes["mujoco:condim"]
        if cd.values is None:
            cd.values = {}
        for si in range(rb.shape_count):
            if rb.shape_body[si] in (12, 13):   # left/right finger bodies
                cd.values[si] = 4
    return rb


def grasp_candidates(tgt, N):
    """N grasp positions spread across the object (lateral dx offsets, same height).
    Grasp the object BODY: TCP at object centre height (fr3_hand_tcp ~ at fingertips,
    so centre-height TCP puts the fingers around the object's mid-body)."""
    offs = np.linspace(-tgt["radius"] * 0.8, tgt["radius"] * 0.8, N)
    gz = tgt["z"]                                   # object centre height
    return [{"dx": float(dx), "gz": gz,
             "label": f"dx={dx*100:+.1f}cm"} for dx in offs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--target_id", type=int, default=3, help="id of the object to grasp (0-indexed)")
    ap.add_argument("--world_count", type=int, default=8)
    ap.add_argument("--steps_settle", type=int, default=60)
    ap.add_argument("--steps_approach", type=int, default=180)
    ap.add_argument("--steps_grasp", type=int, default=120)
    ap.add_argument("--steps_lift", type=int, default=180)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    args = ap.parse_args()

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    tgt = next((o for o in objs if o["id"] == args.target_id), objs[-1])
    print(f"[NEWTON-GRASP] target: id={tgt['id']} '{tgt['label']}' "
          f"@({tgt['x']:+.3f},{tgt['y']:+.3f}) r={tgt['radius']:.3f}")
    cands = grasp_candidates(tgt, args.world_count)
    print(f"[NEWTON-GRASP] {args.world_count} candidate grasps:")
    for i, c in enumerate(cands):
        print(f"   world {i}: {c['label']} gz={c['gz']:.3f}")

    wp.init()

    # --- Franka subbuilder (single world) ---
    # Place robot base so the arm reaches the target — EE at home is ~0.5m in X,
    # so the base sits at target_x - 0.5; and roughly behind in Y.
    ROBOT_BASE = wp.vec3(tgt["x"] - 0.50, tgt["y"], 0.0)
    franka_sub = build_robot(ROBOT_BASE, use_mujoco=True)
    model_single = franka_sub.finalize()
    robot_body_count = model_single.body_count

    # --- Build N heterogeneous worlds (begin_world/end_world) ---
    t_build = time.time()
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    SPACING = 2.5
    # object shape config: high friction, zero margin, torsional friction (condim=4)
    # so the gripper fingers actually grip (mirrors the cube-stacking example).
    obj_cfg = newton.ModelBuilder.ShapeConfig(density=200.0, margin=0.0)
    obj_cfg.mu = 1.0
    for wi in range(args.world_count):
        ox = wi * SPACING   # world X offset for the grid
        builder.begin_world()
        rb = build_robot(wp.vec3(ox + ROBOT_BASE.x, ROBOT_BASE.y, ROBOT_BASE.z), use_mujoco=True)
        builder.add_builder(rb)
        for o in objs:
            body = builder.add_body(
                xform=wp.transform(wp.vec3(ox + o["x"], o["y"], o["z"]), wp.quat_identity()),
                label=f"w{wi}_obj{o['id']}")
            sidx = builder.shape_count
            builder.add_shape_cylinder(body, radius=o["radius"], half_height=o["height"] / 2.0,
                                       cfg=obj_cfg)
            cd = builder.custom_attributes["mujoco:condim"]
            if cd.values is None:
                cd.values = {}
            cd.values[sidx] = 4
        builder.end_world()
    builder.add_ground_plane()
    model = builder.finalize()
    n_bodies_per_world = model.body_count // args.world_count
    n_obj = len(objs)
    print(f"[NEWTON-GRASP] model: {args.world_count} worlds x {n_bodies_per_world} bodies "
          f"({robot_body_count} robot + {n_obj} objects) — built in {time.time()-t_build:.2f}s")

    # --- Solver ---
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton", integrator="implicitfast",
        iterations=20, ls_iterations=100, nconmax=1000, njmax=2000,
        cone="elliptic", impratio=1000.0, use_mujoco_contacts=True)
    # Use a list so inner functions can swap states without scoping issues
    st = [model.state(), model.state()]
    control = model.control()
    contacts = model.contacts()

    # joint_target_q covers only ACTUATED joints (objects are free bodies, no target).
    # model.joint_q includes free-body dofs — sizes differ. Copy only the actuated slice.
    ctrl_size = control.joint_target_q.shape[0]
    ctrl_per_world = ctrl_size // args.world_count
    wp.copy(dest=control.joint_target_q, src=model.joint_q, count=ctrl_size)
    newton.eval_fk(model, model.joint_q, model.joint_qd, st[0])

    # --- IK setup ---
    EE_BODY = 11   # fr3_hand_tcp (Franka hand TCP link, index within one world)
    body_q_np = st[0].body_q.numpy()
    DOWN_QUAT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    pos_obj = ik.IKObjectivePosition(
        link_index=EE_BODY,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*body_q_np[EE_BODY][:3])] * args.world_count, dtype=wp.vec3))
    rot_obj = ik.IKObjectiveRotation(
        link_index=EE_BODY,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([DOWN_QUAT] * args.world_count, dtype=wp.vec4))

    ik_dofs = model_single.joint_coord_count   # 9 per robot (7 arm + 2 finger)
    jll = wp.clone(model.joint_limit_lower.reshape((args.world_count, -1))[:, :ik_dofs])
    jul = wp.clone(model.joint_limit_upper.reshape((args.world_count, -1))[:, :ik_dofs])
    lim_obj = ik.IKObjectiveJointLimit(joint_limit_lower=jll.flatten(),
                                       joint_limit_upper=jul.flatten())

    joint_q_np = model.joint_q.numpy()[:ctrl_size].reshape((args.world_count, ctrl_per_world))
    joint_q_ik = wp.array(joint_q_np[:, :ik_dofs].copy(), dtype=wp.float32)

    ik_solver = ik.IKSolver(model=model_single, n_problems=args.world_count,
                             objectives=[pos_obj, rot_obj, lim_obj],
                             lambda_initial=0.01, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def update_ik():
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=8)
        jt_view = control.joint_target_q.reshape((args.world_count, ctrl_per_world))
        wp.copy(dest=jt_view[:, :7], src=joint_q_ik[:, :7])

    def set_ee_targets(positions, rotation=None):
        pos_obj.set_target_positions(wp.array(positions, dtype=wp.vec3))
        if rotation is not None:
            rot_obj.set_target_rotations(wp.array([rotation] * args.world_count, dtype=wp.vec4))
        # Warm-start IK to a good solution before simulating
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=64)
        update_ik()

    def set_gripper(val):
        jt_view = control.joint_target_q.reshape((args.world_count, ctrl_per_world))
        gv = wp.full(shape=(args.world_count, 2), value=val, dtype=wp.float32)
        wp.copy(dest=jt_view[:, 7:9], src=gv)

    _gripper_val = [0.05]   # mutable so update_gripper_in_loop can track

    def sim_steps(n, run_ik=True, gripper=None):
        """Step N times; update IK target every step if run_ik; always end with state in st[0]."""
        if gripper is not None:
            set_gripper(gripper)
        for _ in range(n):
            if run_ik:
                update_ik()
            model.collide(st[0], contacts)
            st[0].clear_forces()
            solver.step(st[0], st[1], control, contacts, 1.0 / 120.0)
            st[0], st[1] = st[1], st[0]

    def obj_z():
        bq = st[0].body_q.numpy()
        return np.array([bq[wi * n_bodies_per_world + robot_body_count + args.target_id, 2]
                         for wi in range(args.world_count)])

    def ee_pos(wi=0):
        bq = st[0].body_q.numpy()
        b = bq[wi * n_bodies_per_world + EE_BODY]
        return (float(b[0]), float(b[1]), float(b[2]))

    def finger_w0():
        # finger joint dofs are indices 7,8 of world-0's actuated block
        jq = model.joint_q.numpy() if False else st[0].body_q  # placeholder
        bq = st[0].body_q.numpy()
        lf = bq[0 * n_bodies_per_world + 12, :3]
        rf = bq[0 * n_bodies_per_world + 13, :3]
        return float(np.linalg.norm(lf - rf))   # finger separation

    def dbg(tag):
        ex, ey, ez = ee_pos(0)
        oz = obj_z()
        print(f"   [dbg {tag}] world0 EE=({ex:.3f},{ey:.3f},{ez:.3f})  "
              f"finger_gap={finger_w0()*100:.1f}cm  "
              f"cup_z(w0)={oz[0]:.3f}  cup_z range=[{oz.min():.3f},{oz.max():.3f}]", flush=True)

    # --- Phase: settle ---
    print(f"\n[NEWTON-GRASP] settling ({args.steps_settle} steps)...", flush=True)
    t_start = time.time()
    sim_steps(args.steps_settle, run_ik=False, gripper=0.05)
    z_rest = obj_z().copy()
    t_settle = time.time() - t_start

    # --- Phase: approach (above each candidate grasp) ---
    # IMPORTANT: IK uses model_single (world-0 frame). The world-grid offset (wi*SPACING)
    # is purely visual; ALL worlds have the SAME relative geometry (robot-base-to-cup).
    # So IK targets are in model_single / world-0 coordinates (NO wi*SPACING).
    # World-1's robot base is at (SPACING+ROBOT_BASE.x, ...) but reaches the same
    # local cup position as world-0.
    print(f"[NEWTON-GRASP] approach ({args.steps_approach} steps)...", flush=True)
    approach_pos = [
        wp.vec3(tgt["x"] + cands[wi]["dx"], tgt["y"], cands[wi]["gz"] + 0.18)
        for wi in range(args.world_count)]
    print(f"   approach target world0: ({approach_pos[0][0]:.3f},{approach_pos[0][1]:.3f},{approach_pos[0][2]:.3f})")
    set_ee_targets(approach_pos, DOWN_QUAT)
    t0 = time.time()
    sim_steps(args.steps_approach, run_ik=True, gripper=0.05)
    t_approach = time.time() - t0
    dbg("after-approach")

    # --- Phase: descend to grasp height ---
    print(f"[NEWTON-GRASP] descend ({args.steps_grasp} steps)...", flush=True)
    grasp_pos = [
        wp.vec3(tgt["x"] + cands[wi]["dx"], tgt["y"], cands[wi]["gz"])
        for wi in range(args.world_count)]
    set_ee_targets(grasp_pos, DOWN_QUAT)
    sim_steps(args.steps_grasp, run_ik=True, gripper=0.05)
    dbg("after-descend")

    # --- Phase: close gripper ---
    print(f"[NEWTON-GRASP] closing gripper...", flush=True)
    sim_steps(72, run_ik=False, gripper=0.0)
    dbg("after-close")

    # --- Phase: lift ---
    print(f"[NEWTON-GRASP] lift ({args.steps_lift} steps)...", flush=True)
    lift_pos = [
        wp.vec3(tgt["x"] + cands[wi]["dx"], tgt["y"], cands[wi]["gz"] + 0.22)
        for wi in range(args.world_count)]
    set_ee_targets(lift_pos, DOWN_QUAT)
    t_lift0 = time.time()
    sim_steps(args.steps_lift, run_ik=True, gripper=0.0)
    t_lift = time.time() - t_lift0
    dbg("after-lift")

    z_lift = obj_z()
    wp.synchronize()

    total_steps = args.steps_settle + args.steps_approach + args.steps_grasp + 72 + args.steps_lift
    total_time = t_settle + t_approach + t_lift   # core phases
    ms_step = (total_time / total_steps) * 1000.0

    rise = z_lift - z_rest
    order = np.argsort(rise)[::-1]

    print("\n" + "=" * 68)
    print(f"  NEWTON PARALLEL GRASP-EVAL  (N={args.world_count} worlds, {total_steps} steps)")
    print(f"  {total_time:.2f}s total | {ms_step:.2f} ms/step vs PhysX 6.4 ms/step @8 envs")
    print("=" * 68)
    for rank, wi in enumerate(order, 1):
        r = float(rise[wi])
        state = "HOLD" if r > 0.04 else ("partial" if r > 0.01 else "slip")
        print(f"  #{rank}  world {wi}  {cands[wi]['label']:>10s}  "
              f"rose {r * 100:+5.1f}cm   [{state}]")
    best = int(order[0])
    print("-" * 68)
    print(f"  WINNER: world {best}  {cands[best]['label']}  "
          f"(+{float(rise[best]) * 100:.1f}cm)")
    held = int((rise > 0.04).sum())
    print(f"  {held}/{args.world_count} held (>4cm)  |  "
          f"{int((rise > 0.01).sum())}/{args.world_count} partial")
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
