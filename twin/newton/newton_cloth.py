"""Newton twin — import a captured CLOTH as a simulated, foldable, TEXTURED cloth.

A real fabric is captured + reconstructed. The raw depth-carved mesh is a noisy ~4cm thick
blob (degenerate -> VBD NaN), so we import the standard real->sim way: read the MEASURED
footprint (physical_width x physical_height) from the scene and build a clean parametric
CLOTH GRID of that size, then render it with the ACTUAL captured fabric image as a texture
(UV-mapped from the fabric's pixel box) so it looks like the real cloth. Drapes/folds with
the VBD solver (cm scale, particle self-contact).

Run (newton-spike env):
  .../python twin/newton/newton_cloth.py --scene_dir <out> [--capture_dir <cap>]
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
    img = layout.get("image_size_px", [640, 480])
    box = o.get("box_px") or [0, 0, img[0], img[1]]
    return w, h, col, str(o.get("label", "cloth")), box, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", default=None, help="for the rgb.png texture (default: derive)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--drop", type=float, default=0.04)
    ap.add_argument("--no_texture", action="store_true")
    ap.add_argument("--screenshot", default=None, help="save a rendered frame to this PNG then continue")
    args = ap.parse_args()

    cap = args.capture_dir or args.scene_dir.replace("/outputs/", "/captures/")
    rgb_path = os.path.join(cap, "rgb.png")
    w, h, col, label, box, img = load_cloth_spec(args.scene_dir)

    S = 100.0
    cell = args.cell * S
    dim_x = max(2, round(w / args.cell))
    dim_y = max(2, round(h / args.cell))
    print(f"[CLOTH] '{label}': {w*100:.0f}x{h*100:.0f}cm -> grid {dim_x}x{dim_y} "
          f"({(dim_x+1)*(dim_y+1)} particles)")

    wp.init()
    builder = ModelBuilder(gravity=-981.0)
    builder.add_ground_plane()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * dim_x * cell, -0.5 * dim_y * cell, float(args.drop * S)),
        rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x, dim_y=dim_y, cell_x=cell, cell_y=cell, mass=0.1,
        tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.0e-1, edge_ke=10.0, edge_kd=0.0,
        particle_radius=0.3 * cell)
    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_ke = 1.0e4; model.soft_contact_kd = 1.0e-2; model.soft_contact_mu = 1.0
    print(f"[CLOTH] {model.particle_count} particles, {model.tri_count} triangles")

    state_0 = model.state(); state_1 = model.state()
    control = model.control(); contacts = model.contacts()
    solver = SolverVBD(model, iterations=10, particle_enable_self_contact=True,
                       particle_self_contact_radius=0.2, particle_self_contact_margin=0.2)

    # ---- texture set-up: UV-map the cloth grid to the fabric's pixel box in the photo ----
    use_tex = (not args.no_texture) and os.path.exists(rgb_path)
    tex_img = None
    if use_tex:
        from newton._src.utils.texture import load_texture
        tex_img = load_texture(rgb_path)            # preload once (avoid per-frame disk decode)
    tri_flat = wp.array(model.tri_indices.numpy().reshape(-1).astype(np.int32), dtype=wp.int32)
    pq0 = state_0.particle_q.numpy()
    nx = (pq0[:, 0] - pq0[:, 0].min()) / max(1e-6, np.ptp(pq0[:, 0]))   # 0..1 across cloth
    ny = (pq0[:, 1] - pq0[:, 1].min()) / max(1e-6, np.ptp(pq0[:, 1]))
    bx0, by0, bx1, by1 = box
    uu = (bx0 + nx * (bx1 - bx0)) / img[0]
    vv = (by0 + (1.0 - ny) * (by1 - by0)) / img[1]                       # flip V (image y is down)
    uvs = wp.array(np.stack([uu, vv], 1).astype(np.float32), dtype=wp.vec2)
    # ground quad (so we don't double-draw via log_state)
    L = 1.2 * max(dim_x, dim_y) * cell
    g_pts = wp.array(np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]], np.float32), dtype=wp.vec3)
    g_idx = wp.array(np.array([0, 1, 2, 0, 2, 3], np.int32), dtype=wp.int32)
    print(f"[CLOTH] texture: {'rgb.png box '+str(box) if use_tex else 'OFF (flat colour)'}")

    span = max(dim_x, dim_y) * cell
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(-0.5 * span, -1.1 * span, 0.8 * span), -32, 55)
    else:
        viewer = ViewerNull(num_frames=10 ** 9); viewer.set_model(model)

    frame_dt = 1.0 / 60.0; substeps = 10; dt = frame_dt / substeps
    clock = [0.0]; sm = {"s0": state_0, "s1": state_1}

    def render():
        viewer.begin_frame(clock[0])
        viewer.log_mesh("ground", g_pts, g_idx, color=(0.55, 0.55, 0.58))
        if use_tex:
            # color WHITE so the texture shows true colors (shader multiplies tex x ObjectColor)
            viewer.log_mesh("cloth", sm["s0"].particle_q, tri_flat, uvs=uvs, texture=tex_img,
                            color=(1.0, 1.0, 1.0), backface_culling=False)
            o = viewer.objects.get("cloth")
            if o is not None:                       # flip Material.w -> enable texture sampling
                r, m, c, _ = o.material
                o.material = (r, m, c, 1.0)
        else:
            viewer.log_mesh("cloth", sm["s0"].particle_q, tri_flat,
                            color=tuple(col), backface_culling=False)
        viewer.end_frame()

    def step():
        for _ in range(substeps):
            sm["s0"].clear_forces()
            model.collide(sm["s0"], contacts)
            solver.step(sm["s0"], sm["s1"], control, contacts, dt)
            sm["s0"], sm["s1"] = sm["s1"], sm["s0"]
        clock[0] += frame_dt
        render()

    print("[CLOTH] draping under gravity...", flush=True)
    t0 = time.time()
    for _ in range(150):
        step()
    zmin = float(sm["s0"].particle_q.numpy()[:, 2].min())
    print(f"[CLOTH] settled in {time.time()-t0:.1f}s; lowest z={zmin:.2f}cm "
          f"({'NaN' if not np.isfinite(zmin) else 'on ground' if zmin < 1.5 else 'draping'})", flush=True)

    if args.screenshot and args.viewer == "gl":
        for _ in range(3):
            step()
        from PIL import Image
        fb = viewer.get_frame().numpy()
        Image.fromarray(fb).save(args.screenshot)
        print(f"[CLOTH] screenshot -> {args.screenshot}", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[CLOTH] holding GUI open {args.hold:.0f}s — orbit to inspect", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            step()


if __name__ == "__main__":
    main()
