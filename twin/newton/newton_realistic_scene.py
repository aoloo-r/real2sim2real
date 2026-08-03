"""REALISTIC scene view: render each garment from its REAL SAM2 mask silhouette +
depth + photo texture (sleeves/legs and the actual print show), and each rigid from
its reconstructed mesh. Fixes the "blobby garment" look of the raw-SAM3D inspector —
the garments now match the RGBD image shape and appearance.

Garment vs rigid is decided by label keyword. Masks live in <scene>_sam3d_raw/masks/
numbered by object id.

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_realistic_scene.py \
      --scene_dir .../outputs/robot_20260731_143933 \
      --capture_dir .../captures/robot_20260731_143933
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL, ViewerNull
from newton._src.utils.texture import load_texture

from newton_twin_hydro import load_scene_objects, load_obj_mesh
from cloth_mesh import cloth_from_mask

GARMENT_KW = ("shirt", "cloth", "garment", "t-shirt", "tee", "pants", "trouser",
              "legging", "towel", "fabric", "sock", "sleeve")


def is_garment(label):
    return any(k in label.lower() for k in GARMENT_KW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--step", type=int, default=6, help="mask sampling stride (px)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    raw = args.scene_dir.rstrip("/") + "_sam3d_raw"
    mask_dir = os.path.join(raw, "masks")
    layout = json.load(open(os.path.join(args.scene_dir, "scene_layout.json")))
    id_to_label = {o["id"]: o["label"] for o in layout["objects"]}

    objs = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
    VIZ = 0.01   # cm -> m

    wp.init()
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

    garments = []   # (label, verts_m[N,3], faces, uvs, texture)
    rgb_path = os.path.join(args.capture_dir, "rgb.png")
    # measured physical size per object id (ground truth for scale — the raw SAM3D mesh is
    # often oversized, esp. small dark objects)
    meas = {o["id"]: float(o.get("physical_size_m") or 0.1) for o in layout["objects"]}

    for o in objs:
        label = o["label"]
        oid = o["id"]
        mpath = os.path.join(mask_dir, f"{oid}.png")
        if is_garment(label) and os.path.exists(mpath):
            # REAL silhouette + depth + photo texture
            v_cm, faces, uvs, tex = cloth_from_mask(mpath, args.capture_dir,
                                                    step=args.step, base_frame=args.base_frame)
            v_m = v_cm * VIZ
            # MOST RGBD-FAITHFUL: cloth_from_mask back-projects the real mask pixels through
            # the real depth + extrinsics, so the verts are ALREADY at the true base-frame
            # position AND real size. Keep them exactly there (re-centring onto SAM3D's
            # position_cam estimate drifts each garment ~2cm off); only ground z. Optionally
            # shrink about the CENTROID (position-preserving) if depth grossly over-inflated.
            ext = max(np.ptp(v_m[:,0]), np.ptp(v_m[:,1]))
            if ext > meas.get(oid, 0.5) * 1.6:              # only if depth badly over-inflated
                c = v_m[:, :2].mean(0)
                v_m[:, :2] = c + (v_m[:, :2] - c) * (meas[oid] * 1.4 / ext)
            v_m[:, 2] -= v_m[:, 2].min()                    # rest on the floor plane (z=0)
            garments.append((label, v_m, faces.reshape(-1), uvs, tex))
            print(f"[real] garment '{label}' @({o['x']:.2f},{o['y']:.2f}) size~{ext*100:.0f}cm "
                  f"(measured {meas.get(oid,0)*100:.0f}cm)", flush=True)
        else:
            if not o.get("mesh_path") or not os.path.exists(o["mesh_path"]):
                print(f"[real] skip '{label}' (no mesh)", flush=True)
                continue
            mesh, _ = load_obj_mesh(o["mesh_path"])
            # SCALE the raw mesh to its measured size (fixes SAM3D's oversized eraser blobs)
            mv = np.asarray(mesh.vertices)
            ext = float(max(np.ptp(mv[:,0]), np.ptp(mv[:,1]), np.ptp(mv[:,2])))
            scale = meas.get(oid, ext) / ext if ext > 1e-6 else 1.0
            scale = min(scale, 1.0) if is_garment(label) else scale   # only shrink rigids up-capped
            builder.add_shape_mesh(-1, xform=wp.transform(
                wp.vec3(o["x"], o["y"], 0.002), wp.quat_identity()),   # GROUNDED at z=0 (no float)
                mesh=mesh, cfg=cfg, color=wp.vec3(*o["color"]), label=label,
                scale=wp.vec3(scale, scale, scale))
            print(f"[real] rigid '{label}' @({o['x']:.2f},{o['y']:.2f}) scaled x{scale:.2f} "
                  f"-> {meas.get(oid,0)*100:.0f}cm, grounded", flush=True)

    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    tex_cache = {}
    gtri, gverts, guv, gtex = [], [], [], []
    for label, v_m, faces, uvs, tex in garments:
        gverts.append(wp.array(v_m.astype(np.float32), dtype=wp.vec3))
        gtri.append(wp.array(faces.astype(np.int32), dtype=wp.int32))
        guv.append(wp.array(uvs.astype(np.float32), dtype=wp.vec2))
        if id(tex) not in tex_cache:
            tex_cache[id(tex)] = load_texture(rgb_path)
        gtex.append(tex_cache[id(tex)])

    allx = [o["x"] for o in objs]; ally = [o["y"] for o in objs]
    cx, cy = float(np.mean(allx)), float(np.mean(ally))
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(cx - 0.30, cy - 0.42, 0.36), -32, 55)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    def render(t):
        viewer.begin_frame(t)
        viewer.log_state(state)     # rigids (box, erasers)
        for i, (label, *_rest) in enumerate(garments):
            viewer.log_mesh(f"garment_{i}", gverts[i], gtri[i], uvs=guv[i],
                            texture=gtex[i], color=(1.0, 1.0, 1.0), backface_culling=False)
            ob = getattr(viewer, "objects", {}).get(f"garment_{i}")
            if ob is not None:
                r, m, c, _ = ob.material; ob.material = (r, m, c, 1.0)
        viewer.end_frame()

    render(0.0)
    print("[real] scene ready — realistic garments (mask silhouette + photo texture)", flush=True)
    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[real] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    t = 0.0
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        t += 1 / 60.0
        render(t)


if __name__ == "__main__":
    main()
