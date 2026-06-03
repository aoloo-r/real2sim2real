# -*- coding: utf-8 -*-
"""Measure each kept object's REAL footprint + height from depth (table frame)."""
import json, os, sys
import numpy as np, cv2
from depth_scale import compute_physical_sizes
from object_focus import select_target_objects
from hunyuan_demo import gemini_guided_segment_sam2
import render_compare as rc

cap = sys.argv[1]
depth = np.load(os.path.join(cap, "depth.npy"))
intr = json.load(open(os.path.join(cap, "intrinsics.json")))
fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
ext = json.load(open(os.path.join(cap, "extrinsics.json")))
T = np.array(ext["transforms"]["ur5e_base_link"]["T_base_cam"])

masks, labels, boxes, image_size = gemini_guided_segment_sam2(os.path.join(cap, "rgb.png"))
dr = compute_physical_sizes(depth, intr, masks, boxes)
pos = [(r.get("position_cam") if r else None) for r in dr]
keep = select_target_objects(boxes, pos, image_size, T)

print("\n=== REAL object dimensions from depth (ur5e_base_link frame) ===")
for i in keep:
    m = masks[i] > 127
    ys, xs = np.nonzero(m & (depth > 0.05) & (depth < 3.0))
    if len(xs) < 30:
        print(f"  {labels[i]:24s} too few depth pts"); continue
    z = depth[ys, xs]
    X = (xs - cx) * z / fx; Y = (ys - cy) * z / fy
    P = (T @ np.vstack([X, Y, z, np.ones_like(z)]))[:3].T  # Nx3 in base frame
    ext_xyz = P.max(0) - P.min(0)
    # footprint shape: mask fill ratio (area / bbox area) -> ~0.78 round, ~1.0 square
    x0, y0, x1, y1 = boxes[i]["box_px"]
    fill = m.sum() / max((x1 - x0) * (y1 - y0), 1)
    shape = "square-ish" if fill > 0.85 else ("round-ish" if fill < 0.82 else "mixed")
    print(f"  {labels[i]:24s} footprint={ext_xyz[0]*100:5.1f} x {ext_xyz[1]*100:5.1f} cm  "
          f"height={ext_xyz[2]*100:4.1f} cm  mask_fill={fill:.2f} -> {shape}")
