"""Build 3D meshes directly from RealSense depth point clouds.

Instead of using Hunyuan3D-2 (which needs high-res images and hallucinates
artifacts from low-res RealSense crops), this extracts the ACTUAL visible
geometry from the depth sensor per-object mask.

Pipeline:
  1. Extract per-object depth point cloud (using SAM-2 mask)
  2. Estimate normals
  3. Poisson surface reconstruction → watertight mesh
  4. Trim to the actual object bounds
  5. Optionally complete the back side with a convex hull

This gives meshes that match the REAL observed geometry — no generative
model needed, no artifacts, correct scale by construction.

Usage:
    from depth_mesh import build_mesh_from_depth
    mesh = build_mesh_from_depth(depth, mask, intrinsics, rgb_image)
"""

from __future__ import annotations

import numpy as np
import trimesh


def depth_to_colored_pointcloud(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    rgb_image: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract a colored 3D point cloud from depth within a mask.

    Returns:
        points: (N, 3) float32 in camera frame
        colors: (N, 3) float32 RGB 0..1, or None if no rgb_image
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    binary = mask > 127
    valid = binary & (depth > 0.05) & (depth < 3.0)

    vs, us = np.where(valid)
    zs = depth[vs, us]

    xs = (us.astype(np.float32) - cx) * zs / fx
    ys = (vs.astype(np.float32) - cy) * zs / fy

    points = np.stack([xs, ys, zs], axis=-1)

    colors = None
    if rgb_image is not None:
        colors = rgb_image[vs, us].astype(np.float32) / 255.0

    return points, colors


def build_mesh_from_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    rgb_image: np.ndarray = None,
    poisson_depth: int = 7,
    min_points: int = 100,
) -> trimesh.Trimesh | None:
    """Build a watertight mesh from depth data within a mask.

    Steps:
      1. Extract colored point cloud from depth
      2. Center and normalize
      3. Estimate normals
      4. Poisson surface reconstruction (produces watertight mesh)
      5. Trim excess geometry beyond the point cloud bounds
      6. Apply vertex colors from the RGB image

    Args:
        depth: (H, W) float32 depth in meters
        mask: (H, W) uint8 mask (255=foreground)
        intrinsics: camera intrinsics dict
        rgb_image: (H, W, 3) uint8 RGB image for coloring
        poisson_depth: Poisson octree depth (higher = more detail, slower)
        min_points: minimum points needed for reconstruction

    Returns:
        trimesh.Trimesh with vertex colors, or None if insufficient data
    """
    try:
        import open3d as o3d
    except ImportError:
        print("    [DEPTH-MESH] open3d not available, falling back to convex hull")
        return _fallback_convex_hull(depth, mask, intrinsics, rgb_image)

    # Extract point cloud
    points, colors = depth_to_colored_pointcloud(depth, mask, intrinsics, rgb_image)

    if len(points) < min_points:
        print(f"    [DEPTH-MESH] only {len(points)} points, need {min_points}")
        return None

    # Center the point cloud (makes processing easier)
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals (required for Poisson reconstruction)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.02, max_nn=30
    ))
    # Orient normals toward the camera (which is at -centroid in centered coords)
    pcd.orient_normals_towards_camera_location(-centroid)

    # Poisson surface reconstruction
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=True
    )

    # Trim low-density vertices (removes the "skirt" around the actual object)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Also crop to the bounding box of the original point cloud + padding
    bbox = pcd.get_axis_aligned_bounding_box()
    pad = 0.01  # 1cm padding
    # Reassign (newer Open3D makes min_bound/max_bound read-only -> no in-place -=)
    bbox.min_bound = np.asarray(bbox.min_bound) - pad
    bbox.max_bound = np.asarray(bbox.max_bound) + pad
    mesh_o3d = mesh_o3d.crop(bbox)

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    if len(faces) < 10:
        print(f"    [DEPTH-MESH] reconstruction produced only {len(faces)} faces")
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Transfer colors from the point cloud to mesh vertices
    if colors is not None:
        from scipy.spatial import KDTree
        tree = KDTree(points_centered)
        _, indices = tree.query(vertices)
        vertex_colors = np.ones((len(vertices), 4))
        vertex_colors[:, :3] = colors[indices.clip(0, len(colors) - 1)]
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

    # Translate back to world coordinates (centered at centroid)
    # Note: we keep the mesh centered at origin for consistency with the
    # rest of the pipeline. The centroid is stored separately.
    mesh.metadata["centroid_cam"] = centroid.tolist()

    # Fix normals
    trimesh.repair.fix_normals(mesh)

    print(f"    [DEPTH-MESH] built mesh: {len(faces)} faces, "
          f"{len(vertices)} verts, watertight={mesh.is_watertight}")

    return mesh


def _fallback_convex_hull(depth, mask, intrinsics, rgb_image):
    """Simple fallback when open3d is not available."""
    points, colors = depth_to_colored_pointcloud(depth, mask, intrinsics, rgb_image)
    if len(points) < 10:
        return None

    centroid = points.mean(axis=0)
    points_centered = points - centroid

    try:
        cloud = trimesh.PointCloud(points_centered)
        mesh = cloud.convex_hull
        mesh.metadata["centroid_cam"] = centroid.tolist()
        if colors is not None:
            from scipy.spatial import KDTree
            tree = KDTree(points_centered)
            _, indices = tree.query(np.array(mesh.vertices))
            vc = np.ones((len(mesh.vertices), 4))
            vc[:, :3] = colors[indices.clip(0, len(colors)-1)]
            mesh.visual.vertex_colors = (vc * 255).astype(np.uint8)
        print(f"    [DEPTH-MESH] convex hull: {len(mesh.faces)} faces")
        return mesh
    except Exception as e:
        print(f"    [DEPTH-MESH] convex hull failed: {e}")
        return None
