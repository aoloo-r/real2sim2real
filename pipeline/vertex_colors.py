"""Project RGB colors from the input image onto mesh vertices.

Uses the camera intrinsics + depth to project each vertex of the Hunyuan
mesh back to the 2D image plane, then samples the RGB color at that pixel.
This gives per-vertex color without needing the heavy Hunyuan texture pipeline.

The result: meshes that visually match the real objects when rendered in
Isaac Lab, including color gradients and patterns.

Usage:
    from vertex_colors import bake_vertex_colors
    colored_mesh = bake_vertex_colors(mesh, rgb_image, depth, mask, intrinsics, pose_cam)
"""

from __future__ import annotations

import numpy as np


def bake_vertex_colors(mesh, rgb_image: np.ndarray, depth: np.ndarray,
                       mask: np.ndarray, intrinsics: dict,
                       position_cam: list, rotation_cam: list = None,
                       scale: float = 1.0) -> None:
    """Project image colors onto mesh vertices (in-place).

    For each vertex:
      1. Transform from mesh local frame to camera frame (using ICP pose)
      2. Project to 2D pixel using camera intrinsics
      3. Sample RGB color at that pixel
      4. Store as vertex color

    Args:
        mesh: trimesh.Trimesh to color (modified in place)
        rgb_image: (H, W, 3) uint8 BGR image from OpenCV (or RGB)
        depth: (H, W) float32 depth in meters
        mask: (H, W) uint8 mask
        intrinsics: dict with fx, fy, cx, cy
        position_cam: [x, y, z] mesh center in camera frame
        rotation_cam: [w, x, y, z] mesh rotation in camera frame
        scale: mesh scale factor from ICP
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # Ensure RGB is (H, W, 3) — guard against grayscale / single-channel inputs
    # that would make per-vertex color sampling collapse to a scalar.
    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim == 2:
        rgb_image = np.repeat(rgb_image[:, :, None], 3, axis=2)
    elif rgb_image.ndim == 3 and rgb_image.shape[2] == 1:
        rgb_image = np.repeat(rgb_image, 3, axis=2)
    elif rgb_image.ndim == 3 and rgb_image.shape[2] >= 4:
        rgb_image = rgb_image[:, :, :3]
    H, W = rgb_image.shape[:2]

    # Build transform: mesh local → camera frame.
    # position_cam / rotation_cam may arrive nested (e.g. [[x,y,z]] or
    # [[w,x,y,z]]) from scene metadata — flatten before unpacking.
    t = np.asarray(position_cam, dtype=float).reshape(-1)[:3]
    if rotation_cam is not None:
        quat = np.asarray(rotation_cam, dtype=float).reshape(-1)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
    else:
        R = np.eye(3)

    # Transform vertices to camera frame. Center at the mesh centroid first so
    # `t` (the depth-surface centroid) aligns the object's middle with the
    # camera ray — matching the pose used by the render-compare QA.
    verts = np.array(mesh.vertices) * scale
    verts = verts - verts.mean(axis=0, keepdims=True)
    verts_cam = (R @ verts.T).T + t

    # Project to pixel coordinates
    u = (verts_cam[:, 0] * fx / verts_cam[:, 2] + cx).astype(int)
    v = (verts_cam[:, 1] * fy / verts_cam[:, 2] + cy).astype(int)

    # Clamp to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # Sample colors
    colors = rgb_image[v, u].astype(np.float64) / 255.0

    # For vertices that project outside the mask, use the mean color
    in_mask = mask[v, u] > 127
    if in_mask.any():
        mean_color = colors[in_mask].mean(axis=0)
        colors[~in_mask] = mean_color

    # Set vertex colors (RGBA)
    vertex_colors = np.ones((len(colors), 4))
    vertex_colors[:, :3] = colors
    mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

    n_colored = int(in_mask.sum())
    print(f"    [COLOR] baked {n_colored}/{len(verts)} vertices from image projection")


def bake_uniform_color(mesh, rgb_tuple: tuple) -> None:
    """Set a uniform color on all vertices (fallback when no depth/pose available).

    Args:
        mesh: trimesh.Trimesh
        rgb_tuple: (r, g, b) floats 0..1
    """
    r, g, b = rgb_tuple
    n_verts = len(mesh.vertices)
    colors = np.ones((n_verts, 4), dtype=np.uint8) * 255
    colors[:, 0] = int(r * 255)
    colors[:, 1] = int(g * 255)
    colors[:, 2] = int(b * 255)
    mesh.visual.vertex_colors = colors
    print(f"    [COLOR] uniform RGB({r:.2f},{g:.2f},{b:.2f}) on {n_verts} vertices")
