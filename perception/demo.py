# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for autonomous pipeline: image → segment → reconstruct → export
import sys
import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask


def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3D Objects — Autonomous Pipeline")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image (JPEG/PNG)")
    parser.add_argument("--output_dir", type=str, default="outputs/scene",
                        help="Directory to save output files")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory with pre-made masks (0.png, 1.png, ...). "
                             "If not provided, auto-segments with SAM-2.")
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="Min mask area as fraction of image (filters tiny detections)")
    parser.add_argument("--max_area_ratio", type=float, default=0.6,
                        help="Max mask area as fraction of image (filters background)")
    parser.add_argument("--max_objects", type=int, default=15,
                        help="Max number of objects to reconstruct")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default="hf", help="Checkpoint tag")
    parser.add_argument("--skip_mesh", action="store_true",
                        help="Skip mesh export (only export Gaussian splats)")
    parser.add_argument("--use_gemini", action="store_true",
                        help="Use Gemini Vision to identify manipulable objects, "
                             "then prompt SAM-2 with those bounding boxes (instead "
                             "of running SAM-2's automatic mask generator). "
                             "Requires GEMINI_API_KEY env var.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Automatic segmentation
# ──────────────────────────────────────────────────────────────────────

def auto_segment_sam2(image_path, min_area_ratio=0.005, max_area_ratio=0.6,
                      max_objects=15):
    """
    Auto-segment objects using SAM-2's automatic mask generator.
    Returns list of binary masks (H, W) as numpy arrays.

    Install: pip install segment-anything-2
    Checkpoint: download sam2_hiera_large.pt from
    https://github.com/facebookresearch/sam2
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        # Adjust checkpoint path to where you downloaded it
        sam2_checkpoint = os.environ.get(
            "SAM2_CHECKPOINT", "checkpoints/sam2_hiera_large.pt"
        )
        sam2_config = os.environ.get(
            "SAM2_CONFIG", "sam2_hiera_l.yaml"
        )

        sam2 = build_sam2(sam2_config, sam2_checkpoint, device="cuda")
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Downscale large images to fit in GPU memory (24GB)
        h_orig, w_orig = image_rgb.shape[:2]
        max_dim = 2048
        if max(h_orig, w_orig) > max_dim:
            scale_factor = max_dim / max(h_orig, w_orig)
            new_w = int(w_orig * scale_factor)
            new_h = int(h_orig * scale_factor)
            image_small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Resized {w_orig}x{h_orig} → {new_w}x{new_h} for SAM-2 (GPU memory)")
        else:
            image_small = image_rgb
            scale_factor = 1.0

        masks_data = mask_generator.generate(image_small)

        # Scale masks back to original resolution if we downscaled
        if scale_factor < 1.0:
            for m in masks_data:
                m["segmentation"] = cv2.resize(
                    m["segmentation"].astype(np.uint8), (w_orig, h_orig),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                m["area"] = int(m["segmentation"].sum())

    except ImportError:
        print("SAM-2 not installed. Falling back to simple segmentation.")
        print("Install with: pip install segment-anything-2")
        print("Or provide pre-made masks with --mask_dir")
        return auto_segment_fallback(image_path, min_area_ratio,
                                     max_area_ratio, max_objects)

    # Filter and sort by area (largest first, skip background)
    h, w = masks_data[0]["segmentation"].shape
    total_pixels = h * w

    valid_masks = []
    for m in masks_data:
        area_ratio = m["area"] / total_pixels
        if min_area_ratio <= area_ratio <= max_area_ratio:
            valid_masks.append(m)

    # Sort by area descending, take top N
    valid_masks.sort(key=lambda x: x["area"], reverse=True)
    valid_masks = valid_masks[:max_objects]

    masks = [m["segmentation"].astype(np.uint8) * 255 for m in valid_masks]
    print(f"  SAM-2 found {len(masks_data)} segments, "
          f"kept {len(masks)} after filtering")
    return masks


def gemini_guided_segment_sam2(image_path, max_dim=2560):
    """Gemini-prompted SAM-2 segmentation with multi-prompts.

    1. Ask Gemini Vision for bounding boxes of every manipulable object.
    2. For each object, run SAM2ImagePredictor with:
        - its bounding box (spatial constraint)
        - a positive point at the box center (foreground sample)
        - negative points at the centers of every OTHER Gemini box
          (so SAM-2 knows "the lemon is NOT part of the plate")
       This is the key trick that prevents the plate-mask from absorbing
       a lemon sitting inside it.
    3. Pick the highest-scoring mask from multimask_output=True.

    Returns (masks_list, labels_list, gemini_boxes_list, (width, height)).
    """
    from gemini_manip import get_manipulable_boxes

    print("  [Gemini] querying manipulable objects...")
    boxes = get_manipulable_boxes(image_path)
    if not boxes:
        print("  [Gemini] no manipulable objects found")
        return [], [], [], (0, 0)
    print(f"  [Gemini] found {len(boxes)} manipulable objects:")
    for b in boxes:
        x0, y0, x1, y1 = b["box_px"]
        print(f"    - {b['label']:20s}  px=({x0},{y0})-({x1},{y1})")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("  ERROR: SAM-2 not installed. pip install segment-anything-2")
        return [], []

    sam2_checkpoint = os.environ.get(
        "SAM2_CHECKPOINT", "checkpoints/sam2_hiera_large.pt"
    )
    sam2_config = os.environ.get("SAM2_CONFIG", "sam2_hiera_l.yaml")

    sam2 = build_sam2(sam2_config, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # Downscale large images so SAM-2 fits in GPU memory; scale boxes to match.
    # Predictor mode is much lighter than the auto generator, so we can afford
    # a higher resolution → sharper masks.
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        image_small = cv2.resize(image_rgb, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
        print(f"  Resized {w_orig}x{h_orig} → {new_w}x{new_h} for SAM-2")
    else:
        scale = 1.0
        image_small = image_rgb

    predictor.set_image(image_small)

    # Pre-compute box centers (in resized coordinates) for negative-point prompting
    centers = []
    for b in boxes:
        x0, y0, x1, y1 = b["box_px"]
        cx = (x0 + x1) / 2.0 * scale
        cy = (y0 + y1) / 2.0 * scale
        centers.append((cx, cy))

    masks = []
    labels = []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b["box_px"]
        box_small = np.array([
            x0 * scale, y0 * scale, x1 * scale, y1 * scale
        ], dtype=np.float32)

        # Positive prompt: this object's own center
        # Negative prompts: every other object's center
        own_cx, own_cy = centers[i]
        point_coords = [[own_cx, own_cy]]
        point_labels = [1]  # 1 = foreground
        for j, (cx, cy) in enumerate(centers):
            if j == i:
                continue
            # Only add as negative if the other center actually falls inside
            # this object's box (otherwise it's an irrelevant hint)
            if box_small[0] <= cx <= box_small[2] and box_small[1] <= cy <= box_small[3]:
                point_coords.append([cx, cy])
                point_labels.append(0)  # 0 = background

        m, scores, _ = predictor.predict(
            box=box_small,
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            multimask_output=True,
        )
        # Pick the best-scoring of the 3 candidate masks
        best_idx = int(np.argmax(scores))
        mask_small = m[best_idx].astype(np.uint8)

        if scale < 1.0:
            mask_full = cv2.resize(
                mask_small, (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_full = mask_small

        # Clip the mask to the Gemini box + 15% padding so SAM-2 can't
        # over-extend onto the table / neighbors (backstop beyond neg-points).
        x0, y0, x1, y1 = b["box_px"]
        pad_x = int((x1 - x0) * 0.15)
        pad_y = int((y1 - y0) * 0.15)
        cx0 = max(0, x0 - pad_x); cy0 = max(0, y0 - pad_y)
        cx1 = min(w_orig, x1 + pad_x); cy1 = min(h_orig, y1 + pad_y)
        clipped = np.zeros_like(mask_full)
        clipped[cy0:cy1, cx0:cx1] = mask_full[cy0:cy1, cx0:cx1]
        mask_full = clipped

        masks.append((mask_full * 255).astype(np.uint8))
        labels.append(b["label"])
        n_neg = sum(1 for l in point_labels if l == 0)
        print(
            f"    [SAM-2] {b['label']:20s} score={float(scores[best_idx]):.3f} "
            f"neg_pts={n_neg}"
        )

    print(f"  [Gemini+SAM-2] returned {len(masks)} masks")
    return masks, labels, boxes, (w_orig, h_orig)


def auto_segment_fallback(image_path, min_area_ratio=0.005,
                          max_area_ratio=0.6, max_objects=15):
    """
    Simple fallback segmentation using GrabCut + contours.
    Not as good as SAM-2 but requires no extra dependencies.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    total_pixels = h * w

    # Convert to grayscale, apply adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Edge detection + morphology to find object regions
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    masks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / total_pixels
        if min_area_ratio <= area_ratio <= max_area_ratio:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            masks.append(mask)

    masks.sort(key=lambda m: m.sum(), reverse=True)
    masks = masks[:max_objects]
    print(f"  Fallback segmentation found {len(masks)} objects")
    return masks


def load_premade_masks(mask_dir):
    """Load pre-made numbered mask PNGs from a directory."""
    masks = []
    i = 0
    while True:
        mask_path = os.path.join(mask_dir, f"{i}.png")
        if not os.path.exists(mask_path):
            break
        mask = np.array(Image.open(mask_path).convert("L"))
        masks.append(mask)
        i += 1
    return masks


# ──────────────────────────────────────────────────────────────────────
# Stage 2: Reconstruct + Export
# ──────────────────────────────────────────────────────────────────────

def save_object_outputs(output, object_dir, skip_mesh=False):
    """Save per-object reconstruction outputs."""
    os.makedirs(object_dir, exist_ok=True)

    # Gaussian splat (always save — lightweight)
    if "gs" in output:
        output["gs"].save_ply(os.path.join(object_dir, "gaussian.ply"))

    # Mesh via trimesh (glb key confirmed as Trimesh object)
    if not skip_mesh and output.get("glb") is not None:
        glb = output["glb"]
        glb.export(os.path.join(object_dir, "mesh.glb"))
        glb.export(os.path.join(object_dir, "mesh.obj"))
        # Save per-vertex colors separately (OBJ doesn't support them natively).
        # real2sim_franka.py loads these for Isaac Lab display.
        if hasattr(glb.visual, 'vertex_colors') and glb.visual.vertex_colors is not None:
            vc = np.array(glb.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            np.save(os.path.join(object_dir, "vertex_colors.npy"), vc)
            print(f"    ✓ Vertex colors:  {object_dir}/vertex_colors.npy ({len(vc)} verts)")

    # Pose metadata
    metadata = {}
    for key in ["rotation", "translation", "scale"]:
        if key in output:
            val = output[key]
            if hasattr(val, 'cpu'):
                metadata[key] = val.cpu().numpy().tolist()
            else:
                metadata[key] = val
    if "6drotation_normalized" in output:
        val = output["6drotation_normalized"]
        if hasattr(val, 'cpu'):
            metadata["rotation_6d"] = val.cpu().numpy().tolist()

    with open(os.path.join(object_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_scene_layout(all_outputs, output_dir, labels=None,
                       gemini_boxes=None, image_size=None):
    """Save complete scene layout with all objects' poses (and labels if any).

    If gemini_boxes/image_size are provided, persist the 2D bounding boxes and
    image dimensions so downstream tools (e.g. Isaac Lab loader) can use the
    pixel-space layout as the source of truth for object placement.
    """
    objects = []
    for i, output in enumerate(all_outputs):
        obj = {"id": i}
        for key in ["rotation", "translation", "scale"]:
            if key in output:
                val = output[key]
                if hasattr(val, 'cpu'):
                    obj[key] = val.cpu().numpy().tolist()
        obj["gaussian_path"] = f"object_{i}/gaussian.ply"
        obj["mesh_path"] = f"object_{i}/mesh.obj"
        if labels and i < len(labels):
            obj["label"] = labels[i]
        if gemini_boxes and i < len(gemini_boxes):
            obj["box_px"] = list(gemini_boxes[i]["box_px"])
        objects.append(obj)

    scene = {"num_objects": len(objects), "objects": objects}
    if image_size is not None:
        scene["image_size_px"] = list(image_size)  # (width, height)
    layout_path = os.path.join(output_dir, "scene_layout.json")
    with open(layout_path, "w") as f:
        json.dump(scene, f, indent=2)
    return layout_path


def save_mask_visualization(image_path, masks, output_dir):
    """Save a visualization of all masks overlaid on the image."""
    image = cv2.imread(image_path)
    overlay = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    ]

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        colored = np.zeros_like(image)
        colored[:] = color
        mask_bool = mask > 127 if mask.max() > 1 else mask > 0.5
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool], 0.5, colored[mask_bool], 0.5, 0
        )
        # Label
        ys, xs = np.where(mask_bool)
        if len(ys) > 0:
            cy, cx = int(ys.mean()), int(xs.mean())
            cv2.putText(overlay, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Save individual masks too
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(os.path.join(masks_dir, f"{i}.png"), mask)

    cv2.imwrite(os.path.join(output_dir, "segmentation_overlay.png"), overlay)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  SAM 3D Autonomous Pipeline")
    print(f"  Image: {args.image_path}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Stage 1: Get masks ──
    print("[Stage 1] Object segmentation...")
    gemini_labels = None
    gemini_boxes = None
    image_size = None
    if args.mask_dir:
        masks = load_premade_masks(args.mask_dir)
        print(f"  Loaded {len(masks)} pre-made masks from {args.mask_dir}")
    elif args.use_gemini:
        masks, gemini_labels, gemini_boxes, image_size = gemini_guided_segment_sam2(
            args.image_path
        )
    else:
        masks = auto_segment_sam2(
            args.image_path,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            max_objects=args.max_objects
        )

    if not masks:
        print("ERROR: No object masks found or generated.")
        print("  Options:")
        print("  1. Provide masks with --mask_dir /path/to/masks/")
        print("  2. Install SAM-2: pip install segment-anything-2")
        print("  3. Create masks manually (black/white PNGs: 0.png, 1.png, ...)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    save_mask_visualization(args.image_path, masks, args.output_dir)
    print(f"  Mask visualization saved to {args.output_dir}/segmentation_overlay.png")

    # ── Stage 2: Load SAM 3D model ──
    print(f"\n[Stage 2] Loading SAM 3D model (tag={args.tag})...")
    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    inference_model = Inference(config_path, compile=False)

    # ── Stage 3: Reconstruct each object ──
    print(f"\n[Stage 3] Reconstructing {len(masks)} objects...")
    image = load_image(args.image_path)

    all_outputs = []
    for i, mask in enumerate(masks):
        print(f"\n  Object {i}/{len(masks)-1}...")

        # SAM 3D expects mask as PIL-compatible format
        # Ensure mask matches image dimensions
        try:
            output = inference_model(image, mask, seed=args.seed + i)
        except Exception as e:
            print(f"    [SKIP] Object {i} failed: {e}")
            continue
        all_outputs.append(output)

        # Save per-object outputs
        object_dir = os.path.join(args.output_dir, f"object_{i}")
        save_object_outputs(output, object_dir, skip_mesh=args.skip_mesh)

        # Report what was saved
        has_mesh = output.get("glb") is not None and not args.skip_mesh
        print(f"    ✓ Gaussian splat: {object_dir}/gaussian.ply")
        if has_mesh:
            print(f"    ✓ Mesh:           {object_dir}/mesh.obj")
        print(f"    ✓ Pose metadata:  {object_dir}/metadata.json")

        # Print pose summary
        if "translation" in output:
            t = output["translation"].cpu().numpy().flatten()
            print(f"    Position: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
        if "scale" in output:
            s = output["scale"].cpu().numpy().flatten()
            print(f"    Scale:    ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})")

    # ── Stage 4: Save scene layout ──
    print(f"\n[Stage 4] Saving scene layout...")
    layout_path = save_scene_layout(
        all_outputs,
        args.output_dir,
        labels=gemini_labels,
        gemini_boxes=gemini_boxes,
        image_size=image_size,
    )
    print(f"  ✓ {layout_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_outputs)} objects reconstructed")
    print(f"  Output directory: {args.output_dir}/")
    print(f"  ")
    print(f"  Files per object:")
    print(f"    object_N/gaussian.ply  — Gaussian splat")
    if not args.skip_mesh:
        print(f"    object_N/mesh.obj      — Triangle mesh")
    print(f"    object_N/metadata.json — Pose (R, t, s)")
    print(f"  ")
    print(f"  Scene file:")
    print(f"    scene_layout.json      — All object poses")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()