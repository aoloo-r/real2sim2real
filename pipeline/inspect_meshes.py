# -*- coding: utf-8 -*-
"""Render reconstructed colored meshes from several angles for visual QA.

For each object_N/mesh.obj in a scene dir, render 3 views with vertex colors and
compose them next to the input crop (input_rgba.png) into object_N/recon_views.png
and a combined scene montage recon_montage.png.
"""
import argparse
import glob
import os

import numpy as np
import cv2
import trimesh
import open3d as o3d


def render_views(mesh_path, size=320):
    mesh = trimesh.load(mesh_path, process=False)
    v = np.asarray(mesh.vertices, dtype=float)
    v = v - v.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(v, axis=1).max()) or 0.1

    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(v)
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    om.compute_vertex_normals()
    has_color = getattr(mesh.visual, "vertex_colors", None) is not None
    if has_color:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        om.vertex_colors = o3d.utility.Vector3dVector(vc)

    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit" if has_color else "defaultLit"
    r.scene.add_geometry("m", om, mat)

    views = []
    d = radius * 3.0
    eyes = [(0, -d, d * 0.6), (d, -d * 0.6, d * 0.5), (-d, -d * 0.6, d * 0.5)]
    for eye in eyes:
        r.setup_camera(60.0, [0, 0, 0], list(eye), [0, 0, 1])
        img = np.asarray(r.render_to_image())
        views.append(img)
    del r
    return np.hstack(views)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    args = ap.parse_args()
    obj_dirs = sorted(glob.glob(os.path.join(args.scene_dir, "object_*")))
    rows = []
    for od in obj_dirs:
        mp = os.path.join(od, "mesh.obj")
        if not os.path.isfile(mp):
            continue
        views = render_views(mp)
        H = views.shape[0]
        inp_p = os.path.join(od, "input_rgba.png")
        if os.path.isfile(inp_p):
            inp = cv2.cvtColor(cv2.imread(inp_p, cv2.IMREAD_UNCHANGED)[:, :, :3], cv2.COLOR_BGR2RGB)
            inp = cv2.resize(inp, (H, H))
        else:
            inp = np.full((H, H, 3), 240, np.uint8)
        row = np.hstack([inp, views])
        cv2.putText(row, os.path.basename(od), (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)
        out_p = os.path.join(od, "recon_views.png")
        cv2.imwrite(out_p, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        rows.append(row)
        print("wrote", out_p)
    if rows:
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        montage = np.vstack(rows)
        mp = os.path.join(args.scene_dir, "recon_montage.png")
        cv2.imwrite(mp, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print("wrote", mp)


if __name__ == "__main__":
    main()
