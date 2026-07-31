"""PLAN PLAYBACK: animate a cuTAMP plan (from run_scene_tamp.py --export) over the REAL
reconstructed meshes in Newton — each movable lifts, carries, and lowers to its planned
placement in the planner's chosen order. KINEMATIC visualization of the plan (no robot,
no cloth sim) — the physics validation lives in newton_pack_settle / the fold pipeline.

Frames: scene meshes live in the robot-base frame (metres); the cuTAMP env used an
image-derived frame anchored at the box. Placements transfer as BOX-RELATIVE deltas.

Run (newton-spike env, GUI):
  python newton_plan_playback.py --plan /tmp/scene_plan.json \
      --order block_1,green_cloth,block_2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL, ViewerNull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_inspect_scene import load_scene_objects, load_obj_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default="/home/aoloo/sam-3d-objects/outputs/robot_20260729_131708")
    ap.add_argument("--capture_dir", default="/home/aoloo/sam-3d-objects/captures/robot_20260729_131708")
    ap.add_argument("--plan", default="/tmp/scene_plan.json")
    ap.add_argument("--order", default="block_1,green_cloth,block_2",
                    help="action sequence: 'name' = pick-place to its planned target; "
                         "'fold:name' = fold action (garment swaps to its compact bundle)")
    ap.add_argument("--bundles", default="shirt:0.24,0.29,0.12;green_cloth:0.19,0.15,0.03",
                    help="bundle dims per foldable, 'name:w,d,h;...' (m)")
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--flip_x", action="store_true", help="mirror cuTAMP dx (frame fix)")
    ap.add_argument("--flip_y", action="store_true", help="mirror cuTAMP dy (frame fix)")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    place = {p["name"]: p for p in plan["placements"]}
    ct_box = plan["statics"]["box_floor"]["pose"][:2]

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    by_label = {}
    for o in objs:
        by_label.setdefault(o["label"], []).append(o)
    box_o = by_label["cardboard box"][0]
    shirt_o = by_label["yellow t-shirt"][0]
    cloth_o = by_label["green cloth"][0]
    blocks = by_label.get("black rectangular object", [])
    name_to_obj = {"green_cloth": cloth_o, "shirt": shirt_o}
    for i, b in enumerate(blocks):
        name_to_obj[f"block_{i+1}"] = b
    bundles = {}
    for spec in args.bundles.split(";"):
        nm, d = spec.split(":")
        bundles[nm] = tuple(float(v) for v in d.split(","))
    foldable_names = {a.split(":", 1)[1] for a in args.order.split(",") if a.startswith("fold:")}

    wp.init()
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

    # statics: box at its reconstructed pose (shirt is a movable now if it's in the plan)
    statics = [box_o] if "shirt" in name_to_obj else [box_o, shirt_o]
    for o in statics:
        mesh, _ = load_obj_mesh(o["mesh_path"])
        builder.add_shape_mesh(-1, xform=wp.transform(
            wp.vec3(o["x"], o["y"], o["base_z"] + 0.002), wp.quat_identity()),
            mesh=mesh, cfg=cfg, color=wp.vec3(*o["color"]), label=o["label"])

    # movables: free bodies, kinematically driven. Cloth starts ON the shirt (its real
    # configuration — the loader table-snaps everything, so re-stack it). Each FOLDABLE
    # also gets a hidden BUNDLE body (parked below ground) that swaps in at its Fold.
    PARK_Z = -0.6
    body_ids, start, bundle_ids = {}, {}, {}
    for name, o in name_to_obj.items():
        mesh, _ = load_obj_mesh(o["mesh_path"])
        z0 = o["base_z"] + 0.002
        if name == "green_cloth":
            z0 = shirt_o["base_z"] + shirt_o["height"] + 0.004
        b = builder.add_body(xform=wp.transform(wp.vec3(o["x"], o["y"], z0),
                                                wp.quat_identity()), label=name)
        builder.add_shape_mesh(b, mesh=mesh, cfg=cfg, color=wp.vec3(*o["color"]), label=name)
        body_ids[name] = b
        start[name] = np.array([o["x"], o["y"], z0])
        if name in foldable_names and name in bundles:
            w, d, h = bundles[name]
            bb = builder.add_body(xform=wp.transform(wp.vec3(o["x"], o["y"], PARK_Z),
                                                     wp.quat_identity()), label=f"{name}_bundle")
            builder.add_shape_box(bb, hx=w / 2, hy=d / 2, hz=h / 2, cfg=cfg,
                                  color=wp.vec3(*o["color"]), label=f"{name}_bundle")
            bundle_ids[name] = bb

    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # targets: box-relative delta transfer between frames
    sx = -1.0 if args.flip_x else 1.0
    sy = -1.0 if args.flip_y else 1.0
    targets = {}
    for name, p in place.items():
        dx, dy = p["pos"][0] - ct_box[0], p["pos"][1] - ct_box[1]
        tz = p["pos"][2]
        targets[name] = np.array([box_o["x"] + sx * dx, box_o["y"] + sy * dy,
                                  box_o["base_z"] + max(0.01, tz)])
        print(f"[playback] {name}: start {start[name].round(3)} -> target "
              f"{targets[name].round(3)}", flush=True)

    cx = float(np.mean([o["x"] for o in objs])); cy = float(np.mean([o["y"] for o in objs]))
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(cx - 0.28, cy - 0.45, 0.38), -32, 55)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    fps = 60.0
    bq = state.body_q.numpy()

    def set_pose(name, pos):
        bq[body_ids[name], :3] = pos
        state.body_q = wp.array(bq, dtype=wp.transform)

    def render(t):
        viewer.begin_frame(t); viewer.log_state(state); viewer.end_frame()

    def set_pose_id(body, pos):
        bq[body, :3] = pos
        # (state.body_q reassigned in set_pose below)

    t = 0.0
    render(t)
    LIFT = 0.14
    folded = set()
    for action in args.order.split(","):
        if action.startswith("fold:"):
            name = action.split(":", 1)[1]
            # FOLD: flat garment sinks away while its compact bundle rises in place
            p_flat = start[name].copy()
            bz = bundles[name][2] / 2 + p_flat[2] - 0.005
            n = int(1.2 * fps)
            for k in range(n):
                s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
                bq[body_ids[name], :3] = p_flat + np.array([0, 0, (PARK_Z - p_flat[2])]) * s
                bq[bundle_ids[name], :3] = np.array([p_flat[0], p_flat[1],
                                                     PARK_Z + (bz - PARK_Z) * s])
                state.body_q = wp.array(bq, dtype=wp.transform)
                t += 1.0 / fps
                render(t)
            start[name] = np.array([p_flat[0], p_flat[1], bz])
            body_ids[name] = bundle_ids[name]    # subsequent pick-place moves the bundle
            folded.add(name)
            print(f"[playback] folded {name}", flush=True)
            for _ in range(int(0.4 * fps)):
                t += 1.0 / fps; render(t)
            continue
        name = action
        if name not in targets:
            print(f"[playback] no placement for {name}, skipping", flush=True)
            continue
        p0, p1 = start[name].copy(), targets[name]
        hi0 = p0 + [0, 0, LIFT]; hi1 = np.array([p1[0], p1[1], p0[2] + LIFT])
        for (a, b, dur) in ((p0, hi0, 0.6), (hi0, hi1, 1.6), (hi1, p1, 0.7)):
            n = int(dur * fps)
            for k in range(n):
                s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)   # smoothstep
                set_pose(name, a + (b - a) * s)
                t += 1.0 / fps
                render(t)
        for _ in range(int(0.5 * fps)):                        # pause between actions
            t += 1.0 / fps; render(t)
        print(f"[playback] placed {name}", flush=True)

    print("[playback] plan complete — holding GUI", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        t += 1.0 / fps
        render(t)


if __name__ == "__main__":
    main()
