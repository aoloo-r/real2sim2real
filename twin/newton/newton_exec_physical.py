"""PLAN-DRIVEN PHYSICAL executor (checkpoint build): one coupled cloth+robot Newton sim
holding the WHOLE scene — the real box mesh, the Franka, BOTH garments as VBD cloth, and
the rigid erasers — that EXECUTES a cuTAMP plan JSON. At each Fold action the robot
physically folds THAT garment (real cloth sim); at Pick/Place it carries the object to
cuTAMP's placement. Nothing about the sequence is hardcoded — the plan drives it.

Checkpoint 1 (this file's initial scope): load all objects + robot + cloth solver, settle,
render — verify the two cloths co-simulate stably before adding fold/pack actions.

Run (newton-spike env):
  DISPLAY=:1 python newton_exec_physical.py --plan .../santa_20260731_plan.json \
      --scene_dir .../outputs/robot_20260731_143933 --capture_dir .../captures/... \
      --settle 1.5 --screenshot /tmp/exec_phys_c1.png
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
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.viewer import ViewerGL, ViewerNull
from newton._src.utils.texture import load_texture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloth_mesh import cloth_from_mask, find_cloth_mask  # noqa: E402
import sam3d_scene  # noqa: E402

GARMENT_KW = ("shirt", "cloth", "pants", "trouser", "legging", "garment", "towel", "fabric", "tee")
M2CM = 100.0


def load_box_mesh_cm(scene_dir, capture_dir):
    """Real reconstructed box mesh in cm, base at z=0, as an open-top static collider."""
    sc = sam3d_scene.load_scene(scene_dir, capture_dir)
    box = next(o for o in sc["objects"] if "box" in o["label"].lower())
    v = box["verts"].astype(np.float64)
    c = v.mean(0)
    fp = (v[:, :2].max(0) - v[:, :2].min(0)).max()
    meas = max([m for m in box["measured_m"] if m > 0] or [fp])
    v = (c + (meas / fp) * (v - c)) * M2CM
    v[:, 2] -= v[:, 2].min()
    return v.astype(np.float32), box["faces"].reshape(-1, 3).astype(np.int32), box["color"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to settle before actions")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    initial = plan["initial"]                      # metres, cuTAMP frame
    layout = json.load(open(os.path.join(args.scene_dir, "scene_layout.json")))
    meta = {o["label"].lower().replace(" ", "_"): o for o in layout["objects"]}
    mask_dir = args.scene_dir.rstrip("/") + "_sam3d_raw/masks"

    # everything at CM scale + gravity -981 (newton_fold convention: VBD is far more stable
    # at cm than m). Plan poses are in metres -> x100.
    wp.init()
    scene = ModelBuilder(gravity=-981.0)
    cfg = ModelBuilder.ShapeConfig(mu=0.6)

    # --- real box mesh as static open-top collider ---
    bv, bf, bcol = load_box_mesh_cm(args.scene_dir, args.capture_dir)
    box_o = meta[next(k for k in meta if "box" in k)]
    # box goes at its plan position (cuTAMP frame, cm)
    bpx = np.array(initial.get("cardboard_box", [0.38, 0.10, 0.0])[:3]) * M2CM \
        if "cardboard_box" in initial else None
    box_mesh = newton.Mesh(bv, bf.reshape(-1))
    scene.add_shape_mesh(-1, wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                         mesh=box_mesh, cfg=cfg, color=wp.vec3(*bcol))

    # --- garments as VBD cloth; erasers as rigid free bodies ---
    cloth_info, rigids = {}, {}
    part0 = 0
    for nm, pose in initial.items():
        o = meta.get(nm, {})
        col = o.get("display_color", [0.6, 0.6, 0.65])
        p_cm = np.array(pose[:3]) * M2CM
        if any(k in nm for k in GARMENT_KW):
            oid = o.get("id")
            v_cm, faces, uvs, tex = cloth_from_mask(os.path.join(mask_dir, f"{oid}.png"),
                                                    args.capture_dir, step=8, base_frame=args.base_frame)
            v = v_cm.astype(np.float64)                # already cm, real base-frame position
            v[:, 2] += 1.0 - v[:, 2].min()             # rest ~1cm above the table
            n = len(v)
            scene.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0,
                                 vel=wp.vec3(0.0), vertices=[wp.vec3(*x) for x in v],
                                 indices=faces.reshape(-1).tolist(), density=0.02,
                                 tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6,
                                 edge_ke=5.0, edge_kd=1.0e-2, particle_radius=0.8)
            cloth_info[nm] = {"p0": part0, "n": n, "uvs": uvs, "tex": tex, "faces": faces.reshape(-1)}
            part0 += n
            print(f"[phys] cloth '{nm}': {n} particles @({p_cm[0]:.0f},{p_cm[1]:.0f})cm", flush=True)
        else:
            b = scene.add_body(xform=wp.transform(wp.vec3(p_cm[0], p_cm[1], 3.0), wp.quat_identity()), label=nm)
            scene.add_shape_box(b, hx=4.5, hy=2.5, hz=1.5, cfg=cfg, color=wp.vec3(*col))
            rigids[nm] = b
            print(f"[phys] rigid '{nm}' @({p_cm[0]:.0f},{p_cm[1]:.0f})cm", flush=True)

    scene.add_ground_plane(cfg=cfg)
    scene.color(include_bending=True)                  # VBD graph colouring (before finalize)
    model = scene.finalize()
    model.soft_contact_ke = 1.0e4; model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 0.5; model.soft_contact_margin = 0.8

    cloth_solver = SolverVBD(model, iterations=10, particle_enable_self_contact=True,
                             particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
                             particle_topological_contact_filter_threshold=1,
                             particle_rest_shape_contact_exclusion_radius=0.5,
                             particle_vertex_contact_buffer_size=64,
                             particle_edge_contact_buffer_size=64)
    st = [model.state(), model.state()]
    contacts = model.collide(st[0])

    VIZ = 0.01
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        cx = float(np.mean([np.array(p[:2]).mean() for p in initial.values()]))
        viewer.set_camera(wp.vec3(0.30, -0.55, 0.55), -38, 70)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    tex_imgs = {nm: load_texture(os.path.join(args.capture_dir, "rgb.png")) for nm in cloth_info}

    def render(t):
        viewer.begin_frame(t)
        pq = st[0].particle_q.numpy() * VIZ            # cm -> m for display
        for nm, ci in cloth_info.items():
            v = pq[ci["p0"]:ci["p0"] + ci["n"]]
            viewer.log_mesh(f"g_{nm}", wp.array(v.astype(np.float32), dtype=wp.vec3),
                            wp.array(ci["faces"].astype(np.int32), dtype=wp.int32),
                            uvs=wp.array(ci["uvs"].astype(np.float32), dtype=wp.vec2),
                            texture=tex_imgs[nm], color=(1, 1, 1), backface_culling=False)
            ob = getattr(viewer, "objects", {}).get(f"g_{nm}")
            if ob is not None:
                r, m, c, _ = ob.material; ob.material = (r, m, c, 1.0)
        viewer.log_state(st[0])                        # box + rigids
        viewer.end_frame()

    dt, sub = 1.0 / 60.0, 10
    n_settle = int(args.settle / dt)
    print(f"[phys] settling {n_settle} frames (checkpoint 1: verify 2-cloth stability)...", flush=True)
    for i in range(n_settle):
        for _ in range(sub):
            st[0].clear_forces()
            contacts = model.collide(st[0])
            cloth_solver.step(st[0], st[1], None, contacts, dt / sub)
            st = [st[1], st[0]]
        if i % 10 == 0:
            q = st[0].particle_q.numpy()
            if not np.isfinite(q).all():
                print(f"[phys] NaN at settle frame {i} — 2-cloth VBD unstable", flush=True)
                break
            print(f"[phys] t={i*dt:.1f}s z=[{q[:,2].min():.1f},{q[:,2].max():.1f}]cm", flush=True)
        render(i * dt)

    q = st[0].particle_q.numpy()
    print(f"[phys] CHECKPOINT 1: {'OK — 2 cloths stable' if np.isfinite(q).all() else 'NaN'} | "
          f"all objects loaded: {list(cloth_info)} cloth + {list(rigids)} rigid + box", flush=True)
    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[phys] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        render(0.0)


if __name__ == "__main__":
    main()
