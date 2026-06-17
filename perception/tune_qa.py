# -*- coding: utf-8 -*-
"""Re-score already-reconstructed meshes with render_compare (QA tuning harness).

Reuses the saved meshes in an output dir + re-derives masks for the same capture,
so we can iterate on render_compare (azimuth search, 2D alignment, threshold)
WITHOUT re-running Hunyuan.

Usage:
  python tune_qa.py --scene_dir outputs/focus_test --capture_dir captures/robot_XXXX
"""
import argparse
import json
import os

import numpy as np
import trimesh

import render_compare as rc
from depth_scale import compute_physical_sizes
from object_focus import select_target_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--robot_base_frame", default="ur5e_base_link")
    ap.add_argument("--max_objects", type=int, default=4)
    args = ap.parse_args()

    cap = args.capture_dir
    rgb = __import__("cv2").cvtColor(
        __import__("cv2").imread(os.path.join(cap, "rgb.png")),
        __import__("cv2").COLOR_BGR2RGB)
    depth = np.load(os.path.join(cap, "depth.npy"))
    intr = json.load(open(os.path.join(cap, "intrinsics.json")))
    T_base_cam = None
    ext_p = os.path.join(cap, "extrinsics.json")
    if os.path.isfile(ext_p):
        tf = (json.load(open(ext_p)).get("transforms") or {}).get(args.robot_base_frame)
        if tf:
            T_base_cam = tf["T_base_cam"]

    # Re-derive masks for the SAME kept objects as the pipeline.
    from hunyuan_demo import gemini_guided_segment_sam2
    masks, labels, boxes, image_size = gemini_guided_segment_sam2(
        os.path.join(cap, "rgb.png"))
    dr = compute_physical_sizes(depth, intr, masks, boxes)
    positions = [(r.get("position_cam") if r else None) for r in dr]
    keep = select_target_objects(boxes, positions, image_size, T_base_cam,
                                 max_objects=args.max_objects)
    masks = [masks[i] for i in keep]
    labels = [labels[i] for i in keep]
    boxes = [boxes[i] for i in keep]

    print("\n=== QA re-score (azimuth search + 2D centroid align) ===")
    for i, (m, lab, box) in enumerate(zip(masks, labels, boxes)):
        mesh_p = os.path.join(args.scene_dir, f"object_{i}", "mesh.obj")
        if not os.path.isfile(mesh_p):
            print(f"  object_{i} ({lab}): no mesh at {mesh_p}")
            continue
        mesh = trimesh.load(mesh_p, process=False)
        dr_i = compute_physical_sizes(depth, intr, [m], [box])
        pos = dr_i[0].get("position_cam") if dr_i and dr_i[0] else None
        if pos is None:
            print(f"  object_{i} ({lab}): no depth position")
            continue
        s = rc.score_reconstruction(
            mesh, m, intr, pos, T_base_cam=T_base_cam, rgb_image=rgb,
            out_overlay=os.path.join(args.scene_dir, f"object_{i}", "qa_overlay_tuned.png"))
        print(f"  object_{i:>1} {lab:<16} IoU={s['iou']:.3f}  cov={s['coverage']:.3f}  "
              f"az={s['azimuth_deg']:>3}deg  color_err={s['color_err']}")


if __name__ == "__main__":
    main()
