"""Newton twin — import a captured CLOTH as a simulated, foldable cloth.

A real fabric is captured + reconstructed (depth-carve). That raw mesh is a noisy, ~4cm
thick depth blob -> degenerate/unstable for cloth sim. So we import the cloth the standard
real->sim way: read its MEASURED footprint (physical_width x physical_height) and colour
from the scene, and build a clean parametric CLOTH GRID of that size. It drapes under
gravity with the VBD solver (cm scale, particle self-contact). Foundation for robot folding.

Run (newton-spike env):
  .../python twin/newton/newton_cloth.py --scene_dir <out> [--cell 0.02]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton.solvers import SolverVBD
from newton.viewer import ViewerGL, ViewerNull

CLOTH_KW = ("fabric", "cloth", "towel", "shirt", "blanket", "napkin", "garment", "sheet", "scarf", "rag")


def load_cloth_spec(scene_dir):
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    objs = layout.get("objects", [])
    o = next((o for o in objs if any(k in str(o.get("label", "")).lower() for k in CLOTH_KW)), objs[0])
    di = o.get("depth_info") or {}
    w = float(di.get("physical_width_m") or o.get("physical_size_m") or 0.35)
    h = float(di.get("physical_height_m") or w)
    col = [float(c) for c in (o.get("display_color") or [0.25, 0.22, 0.28])[:3]]
    return w, h, col, str(o.get("label", "cloth"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--cell", type=float, default=0.02, help="cloth cell size (m)")
    ap.add_argument("--drop", type=float, default=0.04, help="m above ground to start")
    args = ap.parse_args()

    w, h, col, label = load_cloth_spec(args.scene_dir)
    S = 100.0                                    # m -> cm (VBD prefers cm scale)
    cell = args.cell * S                          # cm
    dim_x = max(2, round(w / args.cell))
    dim_y = max(2, round(h / args.cell))
    print(f"[CLOTH] '{label}': {w*100:.0f}x{h*100:.0f}cm -> grid {dim_x}x{dim_y} "
          f"({(dim_x+1)*(dim_y+1)} particles), colour {[round(c,2) for c in col]}")

    wp.init()
    builder = ModelBuilder(gravity=-981.0)        # cm/s^2
    builder.add_ground_plane()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * dim_x * cell, -0.5 * dim_y * cell, float(args.drop * S)),
        rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x, dim_y=dim_y, cell_x=cell, cell_y=cell, mass=0.1,
        tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.0e-1, edge_ke=10.0, edge_kd=0.0,
        particle_radius=0.3 * cell)
    builder.color(include_bending=True)           # graph colouring required by VBD
    model = builder.finalize()
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 1.0
    print(f"[CLOTH] {model.particle_count} particles, {model.tri_count} triangles")

    state_0 = model.state(); state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(model, iterations=10, particle_enable_self_contact=True,
                       particle_self_contact_radius=0.2, particle_self_contact_margin=0.2)

    span = max(dim_x, dim_y) * cell
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(-0.5 * span, -1.1 * span, 0.8 * span), -32, 55)
    else:
        viewer = ViewerNull(num_frames=10 ** 9); viewer.set_model(model)

    frame_dt = 1.0 / 60.0; substeps = 10; dt = frame_dt / substeps
    clock = [0.0]; sm = {"s0": state_0, "s1": state_1}

    def step_and_render():
        for _ in range(substeps):
            sm["s0"].clear_forces()
            model.collide(sm["s0"], contacts)
            solver.step(sm["s0"], sm["s1"], control, contacts, dt)
            sm["s0"], sm["s1"] = sm["s1"], sm["s0"]
        clock[0] += frame_dt
        viewer.begin_frame(clock[0])
        viewer.log_state(sm["s0"])
        try:
            viewer.log_contacts(contacts, sm["s0"])
        except Exception:
            pass
        viewer.end_frame()

    print("[CLOTH] draping under gravity...", flush=True)
    t0 = time.time()
    for _ in range(150):
        step_and_render()
    zmin = float(sm["s0"].particle_q.numpy()[:, 2].min())
    print(f"[CLOTH] settled in {time.time()-t0:.1f}s; lowest z={zmin:.2f}cm "
          f"({'NaN/unstable' if not np.isfinite(zmin) else 'on ground' if zmin < 1.0 else 'draping'})", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[CLOTH] holding GUI open {args.hold:.0f}s — orbit to inspect", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            step_and_render()


if __name__ == "__main__":
    main()
