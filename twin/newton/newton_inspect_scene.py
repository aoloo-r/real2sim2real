"""Newton scene inspector — verify SAM3D object IMPORT quality (single environment).

Loads every reconstructed object from a pipeline scene into ONE Newton world as its
real mesh.obj (full visual mesh + a physical collider), drops them under gravity to
settle, and holds a tight, robot-free camera on the object cluster so the import can be
judged directly: shape, scale, colour, placement, stacking, and whether vessels are
actually hollow.

Collider fidelity is selectable so we can iterate toward the "best" import:
  --collision coacd        convex DECOMPOSITION (hollow bowls/cups; default)
  --collision convex_hull  single convex hull (filled vessels; fast)
  --collision none         visual only (no physics; exact reconstructed pose)

Run (newton-spike env):
  .../newton-spike/bin/python twin/newton/newton_inspect_scene.py \
    --scene_dir <out> --capture_dir <cap> [--collision coacd] [--settle 90]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import warp as wp
from dataclasses import replace

import newton
from newton.viewer import ViewerGL, ViewerNull

from newton_twin_hydro import load_scene_objects, load_obj_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--collision", default="coacd",
                    choices=["coacd", "convex_hull", "none"])
    ap.add_argument("--settle", type=int, default=90,
                    help="physics steps to relax the scene (0 = freeze at reconstructed pose)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    args = ap.parse_args()

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    print(f"[INSPECT] scene: {len(objs)} objects")
    for o in objs:
        print(f"   id={o['id']:>2} '{o['label']:<18}' @({o['x']:+.3f},{o['y']:+.3f}) "
              f"r={o['radius']:.3f} h={o['height']:.3f} base_z={o['base_z']:.3f}")

    wp.init()
    shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.005, mu=0.8)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg = shape_cfg

    freeze = args.settle == 0 or args.collision == "none"
    coll_shapes = []
    for o in objs:
        if not o["mesh_path"] or not os.path.exists(o["mesh_path"]):
            print(f"   [skip] id={o['id']} no mesh"); continue
        mesh, _ = load_obj_mesh(o["mesh_path"])
        xf = wp.transform(wp.vec3(o["x"], o["y"], o["base_z"] + 0.002), wp.quat_identity())
        ocfg = replace(shape_cfg, has_shape_collision=(args.collision != "none"))
        if freeze:                       # static: shows the exact reconstructed pose
            sidx = builder.add_shape_mesh(-1, xform=xf, mesh=mesh, cfg=ocfg,
                                          color=wp.vec3(*o["color"]), label=o["label"])
        else:                            # free body: settles under gravity
            b = builder.add_body(xform=xf, label=o["label"])
            sidx = builder.add_shape_mesh(b, mesh=mesh, cfg=ocfg,
                                          color=wp.vec3(*o["color"]), label=o["label"])
        coll_shapes.append(sidx)

    if args.collision in ("coacd", "convex_hull") and coll_shapes:
        t0 = time.time()
        builder.approximate_meshes(method=args.collision, shape_indices=coll_shapes,
                                   keep_visual_shapes=True)
        print(f"[INSPECT] {args.collision} colliders built in {time.time()-t0:.1f}s")

    builder.add_ground_plane(cfg=shape_cfg)
    model = builder.finalize()

    state_0 = model.state(); state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()

    # tight camera on the object cluster
    cx = float(np.mean([o["x"] for o in objs])); cy = float(np.mean([o["y"] for o in objs]))
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(cx - 0.28, cy - 0.40, 0.34), -30, 55)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    contacts = None if freeze else model.contacts()
    solver = (None if freeze else
              newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True,
                                          iterations=20, ls_iterations=20,
                                          njmax=600, nconmax=600))
    dt = 1.0 / 240.0

    def render(state, t):
        viewer.begin_frame(t); viewer.log_state(state); viewer.end_frame()

    st = [state_0, state_1]
    if not freeze:
        print(f"[INSPECT] settling {args.settle} steps ({args.collision} colliders)...")
        for k in range(args.settle):
            model.collide(st[0], contacts)
            st[0].clear_forces()
            solver.step(st[0], st[1], control, contacts, dt)
            st[0], st[1] = st[1], st[0]
            if k % 4 == 0:
                render(st[0], k * dt)
    print("[INSPECT] scene ready", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[INSPECT] holding GUI open {args.hold:.0f}s — orbit to inspect", flush=True)
        t_end = time.time() + args.hold; t = 0.0
        while time.time() < t_end and viewer.is_running():
            render(st[0], t); t += dt


if __name__ == "__main__":
    main()
