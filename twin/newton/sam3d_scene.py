"""Load a SAM3D-reconstructed scene FAITHFULLY for Newton.

No fabricated geometry. For each object in scene_layout.json we use the actual SAM3D mesh.obj at
its reconstructed pose (icp_pose: canonical -> camera; extrinsics: camera -> ur5e_base_link), and
texture it by PROJECTING the captured rgb.png onto the geometry (each vertex -> its pixel via the
camera intrinsics) so the sim shows the real photo on the real shape. Newton's log_mesh can't do
per-vertex colours (UV interpolation breaks a 1D colour index), so projective texture is how we get
faithful colour.

Returns per-object dicts with base-frame vertices (metres), faces, projective UVs, and the rgb path.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh


def _quat2mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def load_scene(scene_dir, capture_dir):
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    img = layout.get("image_size_px", [640, 480])
    intr = json.load(open(os.path.join(capture_dir, "intrinsics.json")))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ext = json.load(open(os.path.join(capture_dir, "extrinsics.json")))
    T_base_cam = np.array(ext["transforms"]["ur5e_base_link"]["T_base_cam"], np.float64)
    rgb_path = os.path.join(capture_dir, "rgb.png")

    objs = []
    for i, o in enumerate(layout.get("objects", [])):
        mp = os.path.join(scene_dir, o.get("mesh_path") or f"object_{i}/mesh.obj")
        m = trimesh.load(mp, force="mesh")
        try:                                              # denoise the raw reconstruction (keeps shape)
            trimesh.smoothing.filter_humphrey(m, alpha=0.1, beta=0.5, iterations=12)
        except Exception:
            pass
        vcanon = np.asarray(m.vertices, np.float64)
        faces = np.asarray(m.faces, np.int64)
        ic = o["icp_pose"]
        R = _quat2mat(ic["rotation_cam"]); s = float(ic["scale"]); t = np.asarray(ic["position_cam"], np.float64)
        v_cam = (s * (R @ vcanon.T)).T + t                       # canonical -> camera frame
        # projective UVs from the camera (vertex -> pixel -> [0,1], V flipped for GL)
        z = np.clip(v_cam[:, 2], 1e-4, None)
        u = (fx * v_cam[:, 0] / z + cx) / img[0]
        vv = (fy * v_cam[:, 1] / z + cy) / img[1]          # image row -> V (texture V=0 at top)
        uvs = np.stack([u, vv], 1).astype(np.float32)
        v_base = (T_base_cam @ np.c_[v_cam, np.ones(len(v_cam))].T).T[:, :3]   # -> ur5e_base frame (m)
        # faithful colour = MEDIAN of the actual per-vertex sampled colours (robust to shadow/
        # highlight); scene_layout display_color is a single derived value that can be way off
        vc_path = os.path.join(scene_dir, f"object_{i}/vertex_colors.npy")
        if os.path.exists(vc_path):
            col = list(np.median(np.load(vc_path).reshape(-1, 3), axis=0)[:3])
        else:
            col = o.get("display_color") or [0.6, 0.6, 0.6]
        di = o.get("depth_info") or {}
        meas = [float(di.get("physical_width_m") or 0.0), float(di.get("physical_height_m") or 0.0)]
        objs.append({
            "label": str(o.get("label", f"object_{i}")),
            "idx": i, "verts": v_base.astype(np.float32), "faces": faces.astype(np.int32),
            "uvs": uvs, "rgb": rgb_path, "color": tuple(float(c) for c in col[:3]),
            "measured_m": meas,
        })
    # tabletop scene: everything rests on one table plane -> snap each object's base to it
    # (removes reconstruction z-noise that leaves objects floating)
    table_z = min(o["verts"][:, 2].min() for o in objs)
    for o in objs:
        o["verts"][:, 2] += table_z - o["verts"][:, 2].min()
    return {"objects": objs, "rgb": rgb_path, "img": img, "table_z": table_z}


# ---- standalone faithful render (no robot, no sim): verify it looks like reality ----
if __name__ == "__main__":
    import argparse, time
    import warp as wp
    from newton.viewer import ViewerGL

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", default=None)
    ap.add_argument("--hold", type=float, default=1800.0)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--cam", default=None, help="x,y,z,pitch,yaw (m)")
    args = ap.parse_args()
    cap = args.capture_dir or args.scene_dir.replace("/outputs/", "/captures/")

    sc = load_scene(args.scene_dir, cap)
    wp.init()
    viewer = ViewerGL()
    allv = np.concatenate([o["verts"] for o in sc["objects"]], 0)
    ctr = allv.mean(0); zmin = allv[:, 2].min()
    print(f"[SCENE] {len(sc['objects'])} objects; centre={np.round(ctr,3)} z[{allv[:,2].min():.3f},{allv[:,2].max():.3f}]")
    for o in sc["objects"]:
        e = o["verts"].max(0) - o["verts"].min(0)
        print(f"  {o['label']}: {len(o['verts'])} verts, extent(m)={np.round(e,3)}, centre={np.round(o['verts'].mean(0),3)}")

    # ground quad at the support plane
    L = 1.0
    g = wp.array(np.array([[ctr[0]-L, ctr[1]-L, zmin], [ctr[0]+L, ctr[1]-L, zmin],
                           [ctr[0]+L, ctr[1]+L, zmin], [ctr[0]-L, ctr[1]+L, zmin]], np.float32), dtype=wp.vec3)
    gi = wp.array(np.array([0, 1, 2, 0, 2, 3], np.int32), dtype=wp.int32)

    pts = [wp.array(o["verts"], dtype=wp.vec3) for o in sc["objects"]]
    idx = [wp.array(o["faces"].reshape(-1), dtype=wp.int32) for o in sc["objects"]]
    uvs = [wp.array(o["uvs"], dtype=wp.vec2) for o in sc["objects"]]

    if args.cam:
        cxv, cyv, czv, cp, cyaw = (float(v) for v in args.cam.split(","))
        viewer.set_camera(wp.vec3(cxv, cyv, czv), cp, cyaw)
    else:
        viewer.set_camera(wp.vec3(ctr[0] + 0.6, ctr[1] - 0.9, zmin + 0.7), -32, 115)

    def draw(t):
        viewer.begin_frame(t)
        viewer.log_mesh("ground", g, gi, color=(0.25, 0.25, 0.27))
        for k, o in enumerate(sc["objects"]):
            # solid per-object colour on the real SAM3D mesh (what looked good before), shaded
            viewer.log_mesh(o["label"], pts[k], idx[k], color=o["color"], backface_culling=False)
        viewer.end_frame()

    for f in range(4):
        draw(0.0)
    if args.screenshot:
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[SCENE] screenshot -> {args.screenshot}")
    t0 = time.time()
    while time.time() - t0 < args.hold and viewer.is_running():
        draw(0.0)
