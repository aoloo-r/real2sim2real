"""EXECUTE a cuTAMP plan on the robot in Newton: a Franka arm steps through the plan
JSON's action skeleton — moving to each object, FOLDING garments (flat -> compact bundle),
and PICK-PLACING every bundle/rigid into the box at cuTAMP's placement, in the planner's
order. Kinematic execution (arm driven by Newton IK, objects carried kinematically) so the
whole fold+pack plan is watchable end-to-end. Frame = metres (matches the plan + URDF).

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_execute_plan.py --plan /tmp/santa_plan.json \
      --scene_dir .../outputs/robot_20260731_143933 --capture_dir .../captures/...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

import newton
from newton import ik
from newton.viewer import ViewerGL, ViewerNull
from newton._src.utils.texture import load_texture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloth_mesh import cloth_from_mask

GARMENT_KW = ("shirt", "cloth", "pants", "trouser", "legging", "garment", "towel", "fabric", "tee")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    initial = plan["initial"]
    place = {p["name"]: p for p in plan["placements"]}
    statics = plan["statics"]
    bundles = {n: tuple(plan["bundles_m"].get(n, (0.11, 0.11, 0.09))) for n in initial}
    # parse the skeleton into (verb, obj) actions
    actions = []
    for a in plan["skeleton"]:
        verb = a.split("(")[0]
        obj = a.split("(")[1].split(",")[0].strip()
        if verb in ("Pick", "Place", "Fold"):
            actions.append((verb, obj))

    layout = json.load(open(os.path.join(args.scene_dir, "scene_layout.json")))
    meta = {o["label"].lower().replace(" ", "_"): o for o in layout["objects"]}
    mask_dir = args.scene_dir.rstrip("/") + "_sam3d_raw/masks"

    wp.init()
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"

    # ROBOT-ONLY model for IK (joint_coord_count must be just the arm's 9, not incl. the
    # free objects — the newton_twin_hydro gotcha). Build the same URDF twice.
    rbuilder = newton.ModelBuilder()
    rbuilder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                      floating=False, enable_self_collisions=False)
    model_single = rbuilder.finalize()

    builder = newton.ModelBuilder()
    # --- Franka fr3 at origin (reaches the plan's ~0.4-0.6m workspace) ---
    builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                     floating=False, enable_self_collisions=False)
    robot_dof = builder.joint_dof_count

    def find_body(name):
        for i, lbl in enumerate(builder.body_label):
            if lbl.endswith("/" + name) or lbl == name:
                return i
        return -1

    # --- box floor + walls from the plan statics ---
    for sn, s in statics.items():
        if sn in ("table", "staging_zone"):
            continue
        p, d = s["pose"], s["dims"]
        builder.add_shape_box(-1, wp.transform(wp.vec3(p[0], p[1], p[2]), wp.quat_identity()),
                              hx=d[0] / 2, hy=d[1] / 2, hz=d[2] / 2, cfg=cfg,
                              color=wp.vec3(0.72, 0.60, 0.42))

    # --- movable free bodies at initial plan poses; garments carry real texture ---
    obj_body, garment_render, start, is_g = {}, {}, {}, {}
    rgb_path = os.path.join(args.capture_dir, "rgb.png")
    for nm, pose in initial.items():
        o = meta.get(nm, {})
        col = o.get("display_color", [0.6, 0.6, 0.65])
        g = any(k in nm for k in GARMENT_KW)
        is_g[nm] = g
        b = builder.add_body(xform=wp.transform(wp.vec3(*pose[:3]), wp.quat_identity()), label=nm)
        if g:
            oid = o.get("id")
            v_cm, faces, uvs, tex = cloth_from_mask(os.path.join(mask_dir, f"{oid}.png"),
                                                    args.capture_dir, step=7, base_frame=args.base_frame)
            v_m = v_cm * 0.01; v_m -= v_m.mean(0)
            garment_render[nm] = {"v0": v_m.astype(np.float32), "f": faces.reshape(-1).astype(np.int32),
                                  "uv": uvs.astype(np.float32), "tex": load_texture(rgb_path), "scale": 1.0}
            builder.add_shape_sphere(b, radius=0.001, cfg=cfg, color=wp.vec3(*col))
        else:
            builder.add_shape_box(b, hx=0.045, hy=0.025, hz=0.015, cfg=cfg, color=wp.vec3(*col))
        obj_body[nm] = b
        start[nm] = np.array(pose[:3])

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    bq = state.body_q.numpy()

    # --- IK on the ROBOT-ONLY model ---
    def find_body_s(name):
        for i, lbl in enumerate(rbuilder.body_label):
            if lbl.endswith("/" + name) or lbl == name:
                return i
        return -1
    EE_s = find_body_s("fr3_hand_tcp")
    if EE_s < 0:
        EE_s = find_body_s("fr3_hand")
    EE = find_body("fr3_hand_tcp")
    if EE < 0:
        EE = find_body("fr3_hand")
    DOWN = np.array([1.0, 0.0, 0.0, 0.0])                 # Rx(pi): tool points down, as vec4 xyzw
    ik_dofs = model_single.joint_coord_count
    st0 = model_single.state(); newton.eval_fk(model_single, model_single.joint_q, model_single.joint_qd, st0)
    ee0 = st0.body_q.numpy()[EE_s][:3]
    pos_obj = ik.IKObjectivePosition(link_index=EE_s, link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([ee0], dtype=wp.vec3))
    rot_obj = ik.IKObjectiveRotation(link_index=EE_s, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([DOWN], dtype=wp.vec4))
    jll = wp.array(model_single.joint_limit_lower.numpy().astype(np.float32), dtype=wp.float32)
    jul = wp.array(model_single.joint_limit_upper.numpy().astype(np.float32), dtype=wp.float32)
    lim_obj = ik.IKObjectiveJointLimit(joint_limit_lower=jll, joint_limit_upper=jul)
    joint_q_ik = wp.array(model_single.joint_q.numpy().reshape(1, ik_dofs).copy().astype(np.float32),
                          dtype=wp.float32)
    solver = ik.IKSolver(model=model_single, n_problems=1, objectives=[pos_obj, rot_obj, lim_obj],
                         lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(0.30, -0.55, 0.55), -38, 70)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    fps = 60.0

    # object positions are held HERE (eval_fk recomputes free-body poses from joint_q each
    # step and would otherwise reset them); re-applied to body_q after every IK solve.
    obj_pos = {nm: start[nm].copy() for nm in obj_body}

    def ik_to(pos):
        pos_obj.set_target_positions(wp.array([np.asarray(pos, np.float32)], dtype=wp.vec3))
        solver.step(joint_q_ik, joint_q_ik, iterations=40)
        jq = joint_q_ik.numpy().reshape(-1)[:9]           # arm's 9 coords -> full model's first 9
        full = model.joint_q.numpy(); full[:9] = jq
        model.joint_q = wp.array(full, dtype=wp.float32)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        nonlocal bq
        bq = state.body_q.numpy()
        for onm, opos in obj_pos.items():                 # keep objects where WE put them
            bq[obj_body[onm], :3] = opos
        state.body_q = wp.array(bq, dtype=wp.transform)

    def ee_now():
        return state.body_q.numpy()[EE][:3]

    def render(t):
        viewer.begin_frame(t)
        viewer.log_state(state)
        for nm, gr in garment_render.items():
            bp = bq[obj_body[nm], :3]
            v = gr["v0"] * gr["scale"] + bp
            viewer.log_mesh(f"g_{nm}", wp.array(v, dtype=wp.vec3),
                            wp.array(gr["f"], dtype=wp.int32), uvs=wp.array(gr["uv"], dtype=wp.vec2),
                            texture=gr["tex"], color=(1, 1, 1), backface_culling=False)
            ob = getattr(viewer, "objects", {}).get(f"g_{nm}")
            if ob is not None:
                r, m, c, _ = ob.material; ob.material = (r, m, c, 1.0)
        viewer.end_frame()

    def move_ee(p0, p1, dur, carry=None):
        nonlocal bq
        n = int(dur * fps)
        for k in range(n):
            s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            if carry is not None:
                obj_pos[carry] = ee_now() + [0, 0, -0.03]   # object rides under the TCP (before IK)
            ik_to(p0 + (p1 - p0) * s)
            if carry is not None:
                obj_pos[carry] = ee_now() + [0, 0, -0.03]   # and track the new TCP after IK
                bq[obj_body[carry], :3] = obj_pos[carry]
                state.body_q = wp.array(bq, dtype=wp.transform)
            render((k + 1) / fps)

    APPROACH = 0.16
    t = 0.0
    render(t)
    for verb, nm in actions:
        if nm not in obj_body:
            continue
        if verb == "Fold" and is_g.get(nm):
            p = start[nm]
            move_ee(ee_now(), p + [0, 0, APPROACH], 0.7)          # go above garment
            move_ee(ee_now(), p + [0, 0, 0.02], 0.4)              # descend
            gr = garment_render[nm]                               # fold: shrink the sheet in place
            for k in range(int(0.8 * fps)):
                gr["scale"] = max(0.28, gr["scale"] - 0.9 / (0.8 * fps))
                render(0.0)
            move_ee(ee_now(), p + [0, 0, APPROACH], 0.4)
            print(f"[exec] folded {nm} -> bundle", flush=True)
        elif verb == "Pick":
            p = start[nm]
            move_ee(ee_now(), p + [0, 0, APPROACH], 0.6)
            move_ee(ee_now(), p + [0, 0, 0.03], 0.4)
            print(f"[exec] picked {nm}", flush=True)
        elif verb == "Place" and nm in place:
            tgt = np.array(place[nm]["pos"])
            move_ee(ee_now(), ee_now() + [0, 0, APPROACH], 0.4, carry=nm)   # lift
            move_ee(ee_now(), np.array([tgt[0], tgt[1], tgt[2] + APPROACH]), 1.3, carry=nm)  # carry
            move_ee(ee_now(), tgt + [0, 0, 0.03], 0.5, carry=nm)           # lower into box
            obj_pos[nm] = tgt.copy()                                        # release AT the placement
            bq[obj_body[nm], :3] = tgt; state.body_q = wp.array(bq, dtype=wp.transform)
            move_ee(ee_now(), ee_now() + [0, 0, APPROACH], 0.4)             # retreat (object stays)
            print(f"[exec] placed {nm} in box at {tgt.round(3)}", flush=True)

    print("[exec] PLAN EXECUTED — all objects folded and packed in the box", flush=True)
    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[exec] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        render(0.0)


if __name__ == "__main__":
    main()
