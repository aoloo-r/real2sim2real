# -*- coding: utf-8 -*-
"""Render-and-compare QA for reconstructed objects.

Render the reconstructed mesh from the REAL camera (intrinsics + pose) and
compare its silhouette to the object's SAM-2 mask. A low IoU flags a bad
reconstruction (e.g. a shallow dish hallucinated into a tall funnel) so the
caller can retry with a different seed and keep the best result.

Pose: the mesh comes in its local frame (Z up, bottom at z=0, centered in XY).
We need where "up" points in the camera optical frame. With the calibrated
camera->robot extrinsic this is exact: world up [0,0,1] expressed in the camera
frame is R_cam_base @ [0,0,1] = R_base_cam.T @ [0,0,1]. Without an extrinsic we
fall back to a downward-looking-camera assumption.

Renderer: Open3D 0.19 EGL OffscreenRenderer (pyrender is not installed).
"""

from __future__ import annotations

import numpy as np


def up_in_camera(T_base_cam=None):
    """Unit "up" direction expressed in the camera optical frame."""
    if T_base_cam is not None:
        R = np.asarray(T_base_cam, dtype=float)[:3, :3]
        up = R.T @ np.array([0.0, 0.0, 1.0])
        n = np.linalg.norm(up)
        if n > 1e-6:
            return up / n
    # Fallback: a camera looking down/forward sees world-up as roughly -Y_cam.
    return np.array([0.0, -1.0, 0.0])


def _rot_about(axis, angle):
    """Rotation matrix about unit `axis` by `angle` radians (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _mask_centroid(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _aligned_iou(rmask, real, do_align=True):
    """IoU after optionally shifting the rendered mask so its centroid matches
    the real mask centroid (factors out the known position offset, leaving a
    shape-quality score). Returns (iou, (dy, dx))."""
    if not do_align:
        inter = np.logical_and(rmask, real).sum()
        union = np.logical_or(rmask, real).sum()
        return (float(inter) / float(union) if union else 0.0), (0, 0)
    cr, cre = _mask_centroid(rmask), _mask_centroid(real)
    if cr is None or cre is None:
        return 0.0, (0, 0)
    dy, dx = int(round(cre[0] - cr[0])), int(round(cre[1] - cr[1]))
    shifted = np.roll(np.roll(rmask, dy, axis=0), dx, axis=1)
    inter = np.logical_and(shifted, real).sum()
    union = np.logical_or(shifted, real).sum()
    return (float(inter) / float(union) if union else 0.0), (dy, dx)


def _rotation_z_to(target):
    """Rotation mapping local +Z onto unit vector `target` (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    t = np.asarray(target, dtype=float)
    t = t / (np.linalg.norm(t) + 1e-12)
    v = np.cross(z, t)
    c = float(np.dot(z, t))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def mesh_to_camera_pose(position_cam, T_base_cam=None):
    """Return (R, t) placing a local Z-up mesh into the camera optical frame:
    p_cam = R @ p_local + t."""
    R = _rotation_z_to(up_in_camera(T_base_cam))
    t = np.asarray(position_cam, dtype=float).reshape(-1)[:3]
    return R, t


def rotation_to_quat_wxyz(R):
    """3x3 rotation -> quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return (q / (np.linalg.norm(q) + 1e-12)).tolist()


def render_silhouette(mesh, intrinsics, R, t):
    """Render the mesh (placed by R,t into camera frame) and return
    (rendered_mask bool HxW, rendered_color HxWx3 uint8 or None)."""
    import open3d as o3d

    W, H = int(intrinsics["width"]), int(intrinsics["height"])
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])

    # Center the mesh at its own centroid so the placement target `t` aligns
    # the object's center (not its bottom corner) with the depth centroid.
    verts = np.asarray(mesh.vertices, dtype=float)
    verts = verts - verts.mean(axis=0, keepdims=True)
    verts_cam = (np.asarray(R) @ verts.T).T + np.asarray(t).reshape(1, 3)

    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(verts_cam)
    om.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    om.compute_vertex_normals()
    has_color = getattr(mesh.visual, "vertex_colors", None) is not None
    if has_color:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        om.vertex_colors = o3d.utility.Vector3dVector(vc)

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0, 0, 0, 0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit" if has_color else "defaultLit"
    renderer.scene.add_geometry("obj", om, mat)

    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    renderer.setup_camera(intr, np.eye(4))  # mesh already in camera frame
    color = np.asarray(renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
    rendered_mask = np.isfinite(depth) & (depth > 1e-6) & (depth < 1e3)
    del renderer
    return rendered_mask, color


def score_reconstruction(mesh, mask_u8, intrinsics, position_cam,
                         T_base_cam=None, rgb_image=None, out_overlay=None,
                         azimuths=(0, 45, 90, 135, 180, 225, 270, 315),
                         align_2d=True):
    """Render the mesh from the real camera and score SHAPE agreement vs mask.

    To make the IoU a *shape*-quality score (not a position/azimuth error), we
    search rotation about the up-axis (`azimuths`) and shift the rendered mask
    to the real mask centroid (`align_2d`) before scoring; the best is kept.

    Returns {iou, coverage, color_err, azimuth_deg, shift, rendered_px, real_px}.
    Never raises on render failure — returns iou=0.0.
    """
    real = mask_u8 > 127
    real_px = int(real.sum())
    up = up_in_camera(T_base_cam)
    base_R = _rotation_z_to(up)
    t = np.asarray(position_cam, dtype=float).reshape(-1)[:3]

    best = {"iou": -1.0, "az": 0, "shift": (0, 0), "rmask": None, "rcolor": None}
    for az in azimuths:
        try:
            R = _rot_about(up, np.radians(az)) @ base_R
            rmask, rcolor = render_silhouette(mesh, intrinsics, R, t)
        except Exception as e:
            print(f"    [QA] render failed (az={az}): {e!r}")
            continue
        iou, shift = _aligned_iou(rmask, real, do_align=align_2d)
        if iou > best["iou"]:
            best = {"iou": iou, "az": az, "shift": shift,
                    "rmask": rmask, "rcolor": rcolor}

    if best["rmask"] is None:
        return {"iou": 0.0, "coverage": 0.0, "color_err": None,
                "azimuth_deg": 0, "shift": (0, 0), "rendered_px": 0,
                "real_px": real_px}

    dy, dx = best["shift"]
    rmask = np.roll(np.roll(best["rmask"], dy, axis=0), dx, axis=1)
    rcolor = np.roll(np.roll(best["rcolor"], dy, axis=0), dx, axis=1)
    inter = np.logical_and(rmask, real).sum()
    coverage = float(inter) / real_px if real_px > 0 else 0.0

    color_err = None
    if rgb_image is not None and inter > 0:
        img = np.asarray(rgb_image)
        if img.ndim == 3 and img.shape[2] >= 3:
            both = np.logical_and(rmask, real)
            real_mean = img[both][:, :3].astype(np.float64).mean(axis=0)
            rend_mean = rcolor[both][:, :3].astype(np.float64).mean(axis=0)
            color_err = float(np.abs(real_mean - rend_mean).mean() / 255.0)

    if out_overlay is not None:
        try:
            _save_overlay(out_overlay, rmask, rcolor, real, rgb_image, best["iou"])
        except Exception as e:
            print(f"    [QA] overlay save failed: {e!r}")

    return {"iou": round(best["iou"], 4), "coverage": round(coverage, 4),
            "color_err": (round(color_err, 4) if color_err is not None else None),
            "azimuth_deg": int(best["az"]), "shift": [int(dy), int(dx)],
            "rendered_px": int(rmask.sum()), "real_px": real_px}


def _save_overlay(path, rmask, rcolor, real, rgb_image, iou):
    """Side-by-side: real masked crop | rendered, with IoU annotation."""
    import cv2

    H, W = real.shape
    if rgb_image is not None:
        base = np.asarray(rgb_image)[:, :, :3].copy()
        if base.shape[:2] != (H, W):
            base = cv2.resize(base, (W, H))
        left = base.copy()
        left[~real] = (left[~real] * 0.25).astype(left.dtype)  # dim background
    else:
        left = np.zeros((H, W, 3), np.uint8)
        left[real] = (255, 255, 255)

    right = rcolor[:, :, :3].copy() if rcolor is not None else np.zeros((H, W, 3), np.uint8)
    # green outline of rendered silhouette on the right
    edges = (rmask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, cnts, -1, (0, 255, 0), 2)

    combo = np.hstack([left, right])
    cv2.putText(combo, f"IoU={iou:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)
    # left is RGB (from rgb_image) -> convert to BGR for cv2 write
    cv2.imwrite(path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
