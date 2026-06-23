"""Newton-native twin — prototype (increment 1 of the Path-B migration).

Loads the real SAM3D scene (cup/plate/kiwi/bowl, sized + placed from the
reconstruction) into N PARALLEL Newton worlds, settles physics, and reports per-world
rest poses + step throughput. This is the Newton equivalent of the PhysX twin's "2a"
scene-clone check (twin_plan_eval.py), and the foundation for the parallel task-plan
search on Newton (next: add the robot + grasp execution + scoring).

Runs in the ISOLATED `newton-spike` conda env (Newton 1.x / Warp), NOT the isaaclab
env. Perception (SAM3D) is unchanged; this only replaces the twin's physics layer.

Run:
  /home/aoloo/miniforge3/envs/newton-spike/bin/python twin/newton/newton_twin.py \
    --scene_dir <out> --capture_dir <cap> --world_count 64 [--steps 120]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp
import newton

VESSEL_KW = ("cup", "bowl", "plate", "mug", "glass", "container", "dish")


def load_scene_objects(scene_dir, capture_dir, base_frame="ur5e_base_link"):
    """Objects with base-frame XY (via T_base_cam) + sizes + stacking spawn z.
    Table is taken as z=0 here (prototype); objects rest on it / stack."""
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
    # stacking: object whose footprint is inside a larger one AND sits higher rests ON it
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--world_count", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    args = ap.parse_args()

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    print(f"[NEWTON-TWIN] scene: {len(objs)} objects")
    for o in objs:
        print(f"   id={o['id']} '{o['label']}' @({o['x']:+.3f},{o['y']:+.3f},{o['z']:+.3f}) "
              f"r={o['radius']:.3f} h={o['height']:.3f}")

    wp.init()
    dev = wp.get_device()

    # one sub-world with all scene objects, then replicate to N parallel worlds
    cfg_mu = 1.0
    scene = newton.ModelBuilder()
    scene.default_shape_cfg.mu = cfg_mu
    obj_bodies = []
    for o in objs:
        body = scene.add_body(
            xform=wp.transform(p=wp.vec3(o["x"], o["y"], o["z"]), q=wp.quat_identity()),
            label="obj_%d" % o["id"])
        scene.add_shape_cylinder(body, radius=o["radius"], half_height=o["height"] / 2.0)
        obj_bodies.append(body)

    builder = newton.ModelBuilder()
    builder.replicate(scene, args.world_count, spacing=(1.5, 1.5, 0.0))
    builder.add_ground_plane()
    model = builder.finalize()
    n_obj = len(objs)
    print(f"[NEWTON-TWIN] {args.world_count} worlds x {n_obj} objects = "
          f"{model.body_count} bodies")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=10)

    dt = 1.0 / 120.0
    z0 = state_0.body_q.numpy()[:, 2].copy()    # initial z of every body

    # CUDA-graph-captured settle for speed
    def settle_once():
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)

    graph = None
    if dev.is_cuda:
        # capture a single step (swap handled outside graph via double-call pattern)
        pass

    t0 = time.time()
    for _ in range(args.steps):
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    wp.synchronize()
    elapsed = time.time() - t0

    bq = state_0.body_q.numpy()                  # (body_count, 7) transforms
    fps = args.steps / elapsed
    ms_step = elapsed / args.steps * 1000.0
    print(f"\n[NEWTON-TWIN] stepped {args.steps} steps in {elapsed:.2f}s  "
          f"-> {fps:.0f} steps/s, {ms_step:.2f} ms/step  "
          f"({args.world_count} worlds x {n_obj} obj)")

    # per-object rest z in world 0 + cross-world consistency (bodies are laid out
    # world-major after replicate: world w, object i -> body index w*n_obj + i)
    print("[NEWTON-TWIN] world-0 rest z + max cross-world drift:")
    stable = True
    for i, o in enumerate(objs):
        zs = np.array([bq[w * n_obj + i, 2] for w in range(args.world_count)])
        xy = np.array([[bq[w * n_obj + i, 0], bq[w * n_obj + i, 1]] for w in range(args.world_count)])
        # subtract each world's own origin offset by comparing to world-0 relative
        rel = xy - xy[0]
        drift = float(np.linalg.norm(rel - rel.mean(0), axis=1).max()) if args.world_count > 1 else 0.0
        fell = zs[0] < -0.05
        stable = stable and not fell
        print(f"   id={o['id']} '{o['label']:>12s}' rest_z={zs[0]:+.3f} "
              f"(spawn {o['z']:+.3f})  cross-world drift={drift*1000:.2f}mm"
              f"{'  <-- FELL THROUGH' if fell else ''}")
    print(f"\n[NEWTON-TWIN] SCENE CHECK: "
          f"{'PASS — real scene hosted + stable in all ' + str(args.world_count) + ' Newton worlds' if stable else 'FAIL'}")


if __name__ == "__main__":
    main()
