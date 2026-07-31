"""Multi-item pack settle: drop the WHOLE scene's packables — folded garment bundles
AND rigid items — into the REAL reconstructed box and settle under physics, to SHOW how
the five objects fit in the box and report the space they use.

Unlike newton_pack_settle (one bundle geometry for all placements), this takes a list of
heterogeneous items with their own dims, arranges them non-overlapping above the box
opening, drops them, settles, and reports per-item in-box + the floor-area utilisation
(how full the box is).

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_pack_all.py --scene_dir .../outputs/robot_20260731_143933 \
      --capture_dir .../captures/robot_20260731_143933 \
      --items "shirt:24,20,10:200,120,120;pants:20,16,8:60,180,150;eraser1:9,4,2:40,40,45;eraser2:9,4,2:40,40,45"
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverXPBD
from newton.viewer import ViewerGL, ViewerNull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_pack_settle import load_real_box_cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--items", required=True,
                    help="name:w,d,h:r,g,b;... (cm, colour 0-255) — folded bundles + rigids")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--sim_seconds", type=float, default=4.0)
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--density", type=float, default=200.0)
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    items = []
    for spec in args.items.split(";"):
        nm, dims, col = spec.split(":")
        w, d, h = (float(v) for v in dims.split(","))
        r, g, b = (float(v) / 255.0 for v in col.split(","))
        items.append({"name": nm, "w": w, "d": d, "h": h, "col": (r, g, b)})

    rb = load_real_box_cm(args.scene_dir, args.capture_dir)
    iw = float(rb["bbox"][1][0] - rb["bbox"][0][0])
    ih = float(rb["bbox"][1][1] - rb["bbox"][0][1])
    cx, cy = float(rb["cxy"][0]), float(rb["cxy"][1])
    zt = float(rb["zt"])
    floor_area = iw * ih
    used_area = sum(it["w"] * it["d"] for it in items)
    print(f"[pack_all] real box interior ~{iw:.0f}x{ih:.0f}cm (area {floor_area:.0f}cm^2), "
          f"top_z={zt:.0f}cm | {len(items)} items, footprint sum {used_area:.0f}cm^2 "
          f"({100*used_area/floor_area:.0f}% of floor)", flush=True)

    builder = newton.ModelBuilder(gravity=-981.0)
    stat_cfg = builder.ShapeConfig(density=0.0, mu=args.mu, ke=3.0e4, kd=1.0e2)
    # real box mesh as static open-top SDF collider (robust vs raw triangles)
    cmesh = newton.Mesh(rb["verts"], rb["open_faces"])
    cmesh.build_sdf(target_voxel_size=0.6)
    builder.add_shape_mesh(-1, wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                           mesh=cmesh, cfg=stat_cfg, color=wp.vec3(*rb["color"]))

    # arrange items non-overlapping on a shelf grid above the box opening, largest first
    order = sorted(range(len(items)), key=lambda i: -items[i]["w"] * items[i]["d"])
    dyn_cfg = builder.ShapeConfig(density=args.density, mu=args.mu, restitution=0.0,
                                  ke=2.0e4, kd=1.0e2)
    body_ids = []
    # simple 2-column staging inside the footprint so they drop roughly where they fit
    xs = [cx - iw / 4, cx + iw / 4]
    ycur = [cy + ih / 3, cy + ih / 3]
    col = 0
    for k, idx in enumerate(order):
        it = items[idx]
        sx = xs[col]
        sy = ycur[col] - it["d"] / 2
        ycur[col] -= it["d"] + 1.0
        sz = zt + 5.0 + 3.0 * k                       # stagger drop heights (avoid mid-air overlap)
        bod = builder.add_body(xform=wp.transform((sx, sy, sz), wp.quat_identity()), label=it["name"])
        builder.add_shape_box(bod, hx=it["w"] / 2, hy=it["d"] / 2, hz=it["h"] / 2,
                              cfg=dyn_cfg, color=wp.vec3(*it["col"]))
        body_ids.append((idx, bod))
        col = 1 - col

    model = builder.finalize()
    solver = SolverXPBD(model, iterations=12)
    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    contacts = model.collide(state0)

    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(cx - 0.0, cy - 42.0, zt + 34.0), -40, 90)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    dt = 1.0 / 60.0
    st = [state0, state1]
    n_steps = int(args.sim_seconds / dt)
    for i in range(n_steps):
        for _ in range(args.substeps):
            st[0].clear_forces()
            contacts = model.collide(st[0])
            solver.step(st[0], st[1], None, contacts, dt / args.substeps)
            st = [st[1], st[0]]
        viewer.begin_frame(i * dt); viewer.log_state(st[0]); viewer.end_frame()

    # verdict: each item's final xy inside the box footprint, and how full the box is
    bq = st[0].body_q.numpy()
    x0, y0 = rb["bbox"][0][0], rb["bbox"][0][1]
    x1, y1 = rb["bbox"][1][0], rb["bbox"][1][1]
    inbox = 0
    print("[pack_all] --- SETTLE VERDICT ---", flush=True)
    for idx, bod in body_ids:
        it = items[idx]
        p = bq[bod, :3]
        ok = (x0 - 2 <= p[0] <= x1 + 2) and (y0 - 2 <= p[1] <= y1 + 2) and (p[2] < zt + 2)
        inbox += ok
        print(f"[pack_all]   {it['name']:10} final=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})cm  "
              f"{'IN BOX' if ok else 'OUT/ABOVE'}", flush=True)
    print(f"[pack_all] {inbox}/{len(items)} items packed inside the box | box floor "
          f"{100*used_area/floor_area:.0f}% covered by footprints", flush=True)

    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[pack_all] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    t = n_steps * dt
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        t += dt
        viewer.begin_frame(t); viewer.log_state(st[0]); viewer.end_frame()


if __name__ == "__main__":
    main()
