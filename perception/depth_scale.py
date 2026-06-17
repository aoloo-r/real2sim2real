"""Depth-based object sizing from RealSense RGB-D.

Given an RGB image, its aligned depth map, camera intrinsics, and SAM-2 masks,
computes the REAL physical size of each object using the depth data.

This replaces the bbox-based estimation with millimeter-accurate measurements:
  physical_width = mask_pixel_width × depth / fx

Usage (standalone test):
    python depth_scale.py --capture_dir captures/capture_001 \
                          --masks_dir outputs/scene_realsense/masks

Or import and use in hunyuan_demo.py:
    from depth_scale import compute_physical_sizes
    sizes = compute_physical_sizes(depth_npy, intrinsics, masks, boxes)
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


def compute_physical_sizes(
    depth: np.ndarray,
    intrinsics: dict,
    masks: list[np.ndarray],
    gemini_boxes: list[dict],
    max_size_m: float = 0.40,
) -> list[dict]:
    """Compute physical size of each object using depth + mask.

    For each masked object:
      1. Extract depth pixels within the mask
      2. Compute median depth Z (robust to noise/outliers)
      3. Compute the object's physical extent from its mask pixel span:
         physical_size = max(mask_pixel_width, mask_pixel_height) × Z / fx

    Args:
        depth: (H, W) float32 depth in meters (0 = invalid)
        intrinsics: dict with fx, fy, cx, cy, width, height
        masks: list of (H, W) uint8 binary masks (255=foreground)
        gemini_boxes: list of dicts with "box_px" and "label"

    Returns:
        list of dicts with:
            label, physical_size_m, median_depth_m, mask_width_px, mask_height_px
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    results = []
    for i, mask in enumerate(masks):
        label = gemini_boxes[i].get("label", f"object_{i}") if i < len(gemini_boxes) else f"object_{i}"
        binary = mask > 127

        # Get depth values within the mask
        mask_depths = depth[binary]
        valid_depths = mask_depths[mask_depths > 0.05]  # filter invalid (< 5cm)

        if len(valid_depths) < 10:
            print(f"  [DEPTH] {label}: insufficient valid depth pixels ({len(valid_depths)})")
            results.append({
                "label": label,
                "physical_size_m": None,
                "median_depth_m": None,
            })
            continue

        median_z = float(np.median(valid_depths))

        # Compute mask tight bounding box (pixel extent)
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        mask_w_px = c1 - c0
        mask_h_px = r1 - r0

        # ROBUST size from the masked 3D point cloud (percentile extent) —
        # resistant to stray/bleeding mask pixels and tilt that inflate the raw
        # pixel-bbox span. Falls back to the pixel-bbox×Z method if too sparse.
        cx_i, cy_i = intrinsics["cx"], intrinsics["cy"]
        ys_m, xs_m = np.where(binary)
        z_m = depth[ys_m, xs_m]
        sel = z_m > 0.05
        x3 = y3 = None
        if sel.sum() >= 30:
            x3 = (xs_m[sel] - cx_i) * z_m[sel] / fx
            y3 = (ys_m[sel] - cy_i) * z_m[sel] / fy
            phys_w = float(np.percentile(x3, 98) - np.percentile(x3, 2))
            phys_h = float(np.percentile(y3, 98) - np.percentile(y3, 2))
        else:
            # fallback: pixel-bbox span × Z / f
            phys_w = mask_w_px * median_z / fx
            phys_h = mask_h_px * median_z / fy
        phys_size = max(phys_w, phys_h)

        # Cap only as a last-resort sanity bound (now rarely needed).
        if phys_size > max_size_m:
            print(f"  [DEPTH] WARNING: {label} computed size {phys_size*100:.1f}cm "
                  f"exceeds max {max_size_m*100:.1f}cm — capping")
            phys_size = max_size_m

        # Grasp point = ROBUST 3D centroid of the masked point cloud (median of
        # the back-projected points = the object's central axis). The old method
        # back-projected the pixel-bbox MIDPOINT, which drifts off-axis for
        # asymmetric/partially-occluded masks and biased the grasp laterally
        # (gripper landing to one side of the object). Median is robust to stray
        # mask pixels and depth noise. Falls back to bbox-center only when sparse.
        if x3 is not None:
            x_cam = float(np.median(x3))
            y_cam = float(np.median(y3))
            z_cam = float(np.median(z_m[sel]))
        else:
            mask_center_u = (c0 + c1) / 2.0
            mask_center_v = (r0 + r1) / 2.0
            x_cam = (mask_center_u - intrinsics["cx"]) * median_z / fx
            y_cam = (mask_center_v - intrinsics["cy"]) * median_z / fy
            z_cam = median_z

        results.append({
            "label": label,
            "physical_size_m": round(phys_size, 4),
            "physical_width_m": round(phys_w, 4),
            "physical_height_m": round(phys_h, 4),
            "median_depth_m": round(median_z, 4),
            "mask_width_px": int(mask_w_px),
            "mask_height_px": int(mask_h_px),
            "position_cam": [round(x_cam, 4), round(y_cam, 4), round(z_cam, 4)],
        })

        print(f"  [DEPTH] {label}: {phys_size*100:.1f}cm "
              f"(depth={median_z*100:.1f}cm, {mask_w_px}×{mask_h_px}px)")

    return results


def load_capture(capture_dir: str):
    """Load RGB, depth, and intrinsics from a capture directory."""
    rgb = cv2.imread(os.path.join(capture_dir, "rgb.png"))
    depth = np.load(os.path.join(capture_dir, "depth.npy"))
    with open(os.path.join(capture_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    return rgb, depth, intrinsics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_dir", required=True)
    parser.add_argument("--masks_dir", default=None,
                        help="Directory with mask PNGs (0.png, 1.png, ...)")
    args = parser.parse_args()

    rgb, depth, intrinsics = load_capture(args.capture_dir)
    print(f"RGB: {rgb.shape}, Depth: {depth.shape}")
    print(f"Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
    print(f"Depth range: {depth[depth>0].min():.3f}m - {depth.max():.3f}m")

    if args.masks_dir:
        from PIL import Image
        masks = []
        i = 0
        while True:
            p = os.path.join(args.masks_dir, f"{i}.png")
            if not os.path.exists(p):
                break
            masks.append(np.array(Image.open(p).convert("L")))
            i += 1

        boxes = [{"label": f"object_{j}"} for j in range(len(masks))]
        results = compute_physical_sizes(depth, intrinsics, masks, boxes)
        for r in results:
            print(f"  {r['label']}: {r['physical_size_m']*100:.1f}cm at depth {r['median_depth_m']*100:.1f}cm")
