"""Gemini → SAM-2 → Hunyuan3D-2 pipeline.

Drop-in alternative to demo.py that uses Hunyuan3D-2 for the actual mesh
generation, producing watertight (closed) meshes — fixing the hollow-bowl
problem from SAM 3D.

Stage 1: Ask Gemini for manipulable-object bounding boxes.
Stage 2: Run SAM-2 (prompted with box + positive/negative points) to get a
         clean mask per object.
Stage 3: For each object, crop the image to its bbox + apply the mask as
         alpha, then call Hunyuan3D-2 to generate a watertight mesh.
Stage 4: Save scene_layout.json with the SAME schema as demo.py
         (label, box_px, image_size_px) so real2sim_franka.py works
         transparently with either pipeline.

Run:
    conda activate hunyuan3d
    export GEMINI_API_KEY='...'
    python hunyuan_demo.py \
        --image_path images/raw_images/IMG_3199_fixed.jpeg \
        --output_dir outputs/scene_hunyuan
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Stage 1+2: Gemini-prompted SAM-2 masks
# (cloned from demo.py so we don't have a hard dependency on that file)
# ---------------------------------------------------------------------------

def gemini_guided_segment_sam2(image_path: str, max_dim: int = 2560):
    """Returns (masks, labels, gemini_boxes, (W, H))."""
    from gemini_manip import get_manipulable_boxes

    print("[Stage 1] Asking Gemini for manipulable objects...")
    boxes = get_manipulable_boxes(image_path)
    if not boxes:
        print("  [Gemini] no manipulable objects found")
        return [], [], [], (0, 0)
    print(f"  [Gemini] found {len(boxes)} manipulable objects:")
    for b in boxes:
        x0, y0, x1, y1 = b["box_px"]
        print(f"    - {b['label']:20s}  px=({x0},{y0})-({x1},{y1})")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = os.environ.get(
        "SAM2_CHECKPOINT", "checkpoints/sam2_hiera_large.pt"
    )
    sam2_config = os.environ.get("SAM2_CONFIG", "sam2_hiera_l.yaml")

    print(f"\n[Stage 2] SAM-2 mask prediction (checkpoint={sam2_checkpoint})")
    sam2 = build_sam2(sam2_config, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

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

    centers = []
    for b in boxes:
        x0, y0, x1, y1 = b["box_px"]
        cx = (x0 + x1) / 2.0 * scale
        cy = (y0 + y1) / 2.0 * scale
        centers.append((cx, cy))

    masks = []
    labels = []
    best_scores = []
    kept_boxes = []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b["box_px"]
        box_small = np.array([
            x0 * scale, y0 * scale, x1 * scale, y1 * scale
        ], dtype=np.float32)

        own_cx, own_cy = centers[i]
        point_coords = [[own_cx, own_cy]]
        point_labels = [1]
        for j, (cx, cy) in enumerate(centers):
            if j == i:
                continue
            if box_small[0] <= cx <= box_small[2] and box_small[1] <= cy <= box_small[3]:
                point_coords.append([cx, cy])
                point_labels.append(0)

        m, scores, _ = predictor.predict(
            box=box_small,
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        mask_small = m[best_idx].astype(np.uint8)

        if scale < 1.0:
            mask_full = cv2.resize(
                mask_small, (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_full = mask_small
        # CLIP mask to Gemini bbox + 15% padding — prevents SAM-2 from
        # hallucinating masks that extend far beyond the actual object.
        mask_clipped = np.zeros_like(mask_full)
        pad_x = int((x1 - x0) * 0.15)
        pad_y = int((y1 - y0) * 0.15)
        clip_x0 = max(0, x0 - pad_x)
        clip_y0 = max(0, y0 - pad_y)
        clip_x1 = min(w_orig, x1 + pad_x)
        clip_y1 = min(h_orig, y1 + pad_y)
        mask_clipped[clip_y0:clip_y1, clip_x0:clip_x1] = mask_full[clip_y0:clip_y1, clip_x0:clip_x1]

        masks.append((mask_clipped * 255).astype(np.uint8))
        labels.append(b["label"])
        best_scores.append(best_score)
        kept_boxes.append(b)
        n_neg = sum(1 for l in point_labels if l == 0)
        print(
            f"    [SAM-2] {b['label']:20s} score={best_score:.3f} "
            f"neg_pts={n_neg}"
        )

    # Filter out low-confidence masks
    min_score = 0.3
    filtered_masks = []
    filtered_labels = []
    filtered_boxes = []
    n_dropped = 0
    for idx in range(len(masks)):
        if best_scores[idx] >= min_score:
            filtered_masks.append(masks[idx])
            filtered_labels.append(labels[idx])
            filtered_boxes.append(kept_boxes[idx])
        else:
            n_dropped += 1
            print(f"    [FILTER] dropping '{labels[idx]}' — score {best_scores[idx]:.3f} < {min_score}")
    if n_dropped:
        print(f"  [FILTER] dropped {n_dropped}/{len(masks)} low-confidence masks")

    print(f"  [Gemini+SAM-2] returned {len(filtered_masks)} masks")
    return filtered_masks, filtered_labels, filtered_boxes, (w_orig, h_orig)


# ---------------------------------------------------------------------------
# Stage 3: Hunyuan3D-2 image-to-3D
# ---------------------------------------------------------------------------

def crop_and_mask_object(image_rgb: np.ndarray, mask_u8: np.ndarray,
                         all_masks: list, my_index: int,
                         box_px: tuple[int, int, int, int],
                         pad_ratio: float = 0.08) -> tuple[Image.Image, bool]:
    """Return an RGBA PIL crop of one object with background made transparent.

    The crop hides occluders: if another object's mask covers part of THIS
    object's interior (e.g. a lemon sitting on a plate), we inpaint those
    pixels with surrounding texture and include them as foreground in alpha.
    Otherwise Hunyuan would see a hole and either reconstruct an indented
    shape or hallucinate the occluder back into our mesh.

    Hunyuan3D-2 expects RGBA where alpha encodes the foreground mask.
    Returns (rgba_image, had_occluders).
    """
    H, W = image_rgb.shape[:2]
    x0, y0, x1, y1 = box_px
    bw = x1 - x0
    bh = y1 - y0
    pad = int(round(max(bw, bh) * pad_ratio))
    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(W, x1 + pad)
    y1p = min(H, y1 + pad)

    rgb_crop = image_rgb[y0p:y1p, x0p:x1p].copy()
    mask_crop = mask_u8[y0p:y1p, x0p:x1p].copy()

    # Build occluder mask: pixels where ANY OTHER object's mask is set,
    # but this object's mask is NOT, within the bbox.
    occluder_mask = np.zeros_like(mask_crop)
    for j, other in enumerate(all_masks):
        if j == my_index:
            continue
        other_crop = other[y0p:y1p, x0p:x1p]
        # Pixel is an occluder if it's foreground for some other object AND
        # not foreground for us
        occluder_mask = np.where(
            (other_crop > 127) & (mask_crop <= 127),
            np.uint8(255),
            occluder_mask,
        )

    had_occluders = bool(occluder_mask.any())
    if had_occluders:
        # Dilate slightly so we capture mask edges that SAM-2 missed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occluder_dil = cv2.dilate(occluder_mask, kernel, iterations=2)
        # Inpaint the occluder area using surrounding pixels
        rgb_inpainted = cv2.inpaint(rgb_crop, occluder_dil, 5, cv2.INPAINT_TELEA)
        # Treat the inpainted area as our object's surface (extend the mask)
        full_mask = np.where(occluder_dil > 0, np.uint8(255), mask_crop)
    else:
        rgb_inpainted = rgb_crop
        full_mask = mask_crop

    rgba = np.zeros((rgb_inpainted.shape[0], rgb_inpainted.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb_inpainted
    rgba[..., 3] = full_mask
    return Image.fromarray(rgba, mode="RGBA"), had_occluders


def _is_tall_narrow(mesh) -> bool:
    """A cup/bottle/tube — one dimension dominates the other two."""
    ext = sorted(float(v) for v in mesh.extents)
    if ext[2] <= 0:
        return False
    aspect = ext[2] / max(ext[0], 1e-6)
    return aspect > 2.0 and ext[1] / max(ext[0], 1e-6) < 1.5


def _try_trimesh_fill(mesh, label, n_faces_in):
    import trimesh
    try:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        if mesh.is_watertight:
            print(f"    [REPAIR {label}] watertight after trimesh.fill_holes "
                  f"(faces {n_faces_in} -> {len(mesh.faces)})")
            return mesh, True
    except Exception as e:
        print(f"    [REPAIR {label}] trimesh.fill_holes failed: {e}")
    return mesh, False


def _try_pymeshlab_close(mesh, label, n_faces_in):
    import trimesh
    import tempfile
    import os as _os
    try:
        import pymeshlab as ml
        ms = ml.MeshSet()
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            mesh.export(tmp.name)
            tmp_path = tmp.name
        try:
            ms.load_new_mesh(tmp_path)
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes(maxholesize=200)
            ms.save_current_mesh(tmp_path)
            mesh2 = trimesh.load(tmp_path, force="mesh")
        finally:
            _os.unlink(tmp_path)
        if mesh2.is_watertight:
            print(f"    [REPAIR {label}] watertight after pymeshlab close_holes "
                  f"(faces {n_faces_in} -> {len(mesh2.faces)})")
            return mesh2, True
        return mesh2, False
    except Exception as e:
        print(f"    [REPAIR {label}] pymeshlab close_holes failed: {e}")
        return mesh, False


def _try_voxel_remesh(mesh, label, n_faces_in):
    try:
        diag = float(mesh.extents.max())
        pitch = max(diag / 200.0, 0.003)
        vox = mesh.voxelized(pitch=pitch).fill()
        mesh3 = vox.marching_cubes
        if mesh3.is_watertight:
            print(f"    [REPAIR {label}] watertight after voxel remesh "
                  f"(pitch={pitch:.4f}, faces {n_faces_in} -> {len(mesh3.faces)})")
        else:
            print(f"    [REPAIR {label}] voxel remesh produced non-watertight "
                  f"mesh ({len(mesh3.faces)} faces) — using anyway")
        return mesh3, mesh3.is_watertight
    except Exception as e:
        print(f"    [REPAIR {label}] voxel remesh failed: {e}")
        return mesh, False


def repair_mesh(mesh, label: str = ""):
    """Make mesh watertight and artifact-free via voxel remesh.

    ALWAYS uses voxel remesh — this removes Hunyuan's reconstruction artifacts
    (wings, extra geometry, holes) and produces clean solid shapes. The
    previous per-strategy escalation (fill_holes → close_holes → voxel) left
    artifacts intact when earlier steps "succeeded" on technically-watertight
    but visually-broken meshes.
    """
    n_faces_in = len(mesh.faces)
    print(f"    [REPAIR {label}] voxel remesh (all objects)")
    out, _ = _try_voxel_remesh(mesh, label, n_faces_in)
    return out

    out, ok = _try_trimesh_fill(mesh, label, n_faces_in)
    if ok:
        return out
    out, ok = _try_pymeshlab_close(out, label, n_faces_in)
    if ok:
        return out
    out, _ = _try_voxel_remesh(out, label, n_faces_in)
    return out


def auto_orient_mesh(mesh, label: str = ""):
    """Automatically orient a mesh for correct physics simulation.

    - TALL objects (cups, bottles): longest axis → Z, then flip 180° so the
      opening/top faces up (Hunyuan consistently generates these inverted).
    - FLAT objects (keyboards, plates): shortest axis → Z so they lay flat.

    Also centers XY at origin and puts the bottom at z=0.
    """
    import trimesh as _trimesh

    # Two-ratio shape test (robust to shallow dishes that the old single
    # min/max ratio misclassified as TALL → "funnel" bug):
    #   FLAT  = ONE thin axis, TWO large axes  → thinnest/middle is small
    #   TALL  = ONE long axis, TWO small axes  → middle/longest is small
    ext = mesh.extents
    order = np.argsort(ext)            # ascending: [thin, mid, long]
    a_thin, a_mid, a_long = (int(order[0]), int(order[1]), int(order[2]))
    e_thin, e_mid, e_long = ext[a_thin], ext[a_mid], ext[a_long]
    flat_ratio = e_thin / e_mid if e_mid > 0 else 1.0   # << 1 ⇒ flat
    tall_ratio = e_mid / e_long if e_long > 0 else 1.0   # << 1 ⇒ elongated

    # Label priors as tiebreakers for geometrically-ambiguous shapes.
    ll = (label or "").lower()
    FLAT_KW = ("plate", "dish", "bowl", "tray", "saucer", "mat", "book",
               "phone", "keyboard", "lid", "pad", "card", "laptop", "coaster")
    TALL_KW = ("cup", "mug", "bottle", "can ", "glass", "jar", "vase",
               "tube", "spray", "thermos", "flask")
    is_flat_label = any(k in ll for k in FLAT_KW)
    is_tall_label = any(k in ll for k in TALL_KW)

    if flat_ratio < 0.5 and not is_tall_label:
        target, mode = a_thin, "FLAT"      # lay the thin axis up
    elif tall_ratio < 0.6 and not is_flat_label:
        target, mode = a_long, "TALL"      # stand the long axis up
    elif is_flat_label:
        target, mode = a_thin, "FLAT"
    elif is_tall_label:
        target, mode = a_long, "TALL"
    else:
        target, mode = a_thin, "FLAT"      # ambiguous: safest is lying flat

    if target == 0:
        mesh.apply_transform(_trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif target == 1:
        mesh.apply_transform(_trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    # For tall objects, flip 180° — Hunyuan consistently generates them upside-down
    if mode == "TALL":
        mesh.apply_transform(_trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    # Center XY, put bottom at z=0
    center_xy = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], -mesh.bounds[0][2]])

    _trimesh.repair.fix_normals(mesh)
    print(f"    [ORIENT] {mode}: Z height={mesh.bounds[1][2]*100:.1f}cm")
    return mesh


def run_hunyuan_shape(pipeline, rgba_image: Image.Image, seed: int = 42):
    """Call Hunyuan3D-2 shape pipeline. Returns a trimesh.Trimesh."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipeline(image=rgba_image, generator=generator)
    if isinstance(out, list) and len(out) > 0:
        return out[0]
    return out


def prepare_mesh(mesh, physical_size_m: float, label: str = "", obj_id: int = 0):
    """Re-center, scale to physical size, and auto-orient a raw Hunyuan mesh.

    Hunyuan3D-2 outputs meshes in a canonical [-1, 1] box (max extent ~ 2.0).
    We rescale so the largest dimension equals the physical size, then orient
    for physics. Returns the prepared mesh (Z-up, bottom at z=0, centered XY).
    Split out of save_object so the QA loop can prepare candidates without
    exporting them.
    """
    extents = mesh.extents
    canon_max = float(extents.max())
    scale_factor = (physical_size_m / canon_max) if canon_max > 0 else 1.0
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    mesh.apply_scale(scale_factor)
    mesh = auto_orient_mesh(mesh, label=label or f"obj{obj_id}")
    mesh._r2s_scale_factor = scale_factor  # for logging
    return mesh


def save_object(mesh, output_dir: str, obj_id: int,
                physical_size_m: float, color_rgb: tuple = (0.5, 0.5, 0.5),
                label: str = "", prepared: bool = False) -> dict:
    """Persist a trimesh to outputs/scene_X/object_N/mesh.obj at real-world scale.

    If `prepared` is False the mesh is scaled+oriented via prepare_mesh first;
    pass prepared=True when the caller (QA loop) already prepared it.
    """
    import numpy as np

    if not prepared:
        mesh = prepare_mesh(mesh, physical_size_m, label=label, obj_id=obj_id)
    scale_factor = float(getattr(mesh, "_r2s_scale_factor", 1.0))

    obj_dir = os.path.join(output_dir, f"object_{obj_id}")
    os.makedirs(obj_dir, exist_ok=True)
    mesh_path = os.path.join(obj_dir, "mesh.obj")
    mesh.export(mesh_path)
    print(
        f"    ✓ Mesh: {mesh_path}  "
        f"(physical_size={physical_size_m*100:.1f}cm, "
        f"scale_factor={scale_factor:.3f})"
    )

    metadata = {
        "rotation": [[0.0, 0.0, 0.0, 1.0]],   # identity (x,y,z,w)
        "translation": [[0.0, 0.0, 0.0]],
        "scale": [[1.0, 1.0, 1.0]],            # already baked into mesh
        "physical_size_m": physical_size_m,
        "display_color": list(color_rgb),
    }
    with open(os.path.join(obj_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def sample_object_color(image_rgb: np.ndarray, mask_u8: np.ndarray) -> tuple:
    """Mean RGB color of the masked foreground, with saturation boost.

    Real-world objects photographed under office lighting often have very
    desaturated colors. We boost saturation so they're visually distinct
    in Isaac Lab's grey-dominated renderer.
    """
    fg = mask_u8 > 127
    if not fg.any():
        return (0.5, 0.5, 0.5)
    img = np.asarray(image_rgb)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] >= 4:
        img = img[:, :, :3]
    pixels = img[fg].astype(np.float32) / 255.0
    mean_rgb = pixels.reshape(-1, pixels.shape[-1]).mean(axis=0)
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    # Boost saturation: move each channel away from the mean (grey)
    mean = (r + g + b) / 3.0
    boost = 1.8
    r = min(1.0, max(0.0, mean + (r - mean) * boost))
    g = min(1.0, max(0.0, mean + (g - mean) * boost))
    b = min(1.0, max(0.0, mean + (b - mean) * boost))
    return (float(r), float(g), float(b))


def estimate_physical_size(box_px, image_size, table_width_m: float = 0.50):
    """Rough physical size from a Gemini bbox.

    Assumption: the photo's longest image dimension covers approximately
    `table_width_m` meters in the real world. Each object's physical size
    is proportional to its bbox's longest side.
    """
    img_w, img_h = image_size
    img_long = max(img_w, img_h)
    if img_long <= 0:
        return 0.05
    px_per_m = img_long / table_width_m
    x0, y0, x1, y1 = box_px
    bbox_long = max(x1 - x0, y1 - y0)
    return float(bbox_long / px_per_m)


def save_scene_layout(metadatas: list[dict], output_dir: str,
                      labels: list[str], gemini_boxes: list[dict],
                      image_size: tuple[int, int]) -> str:
    """Write scene_layout.json in the same format demo.py uses."""
    objects = []
    for i, meta in enumerate(metadatas):
        obj = {"id": i}
        # Copy fields explicitly so the schema matches SAM 3D's, with physical_size_m
        obj["rotation"] = meta["rotation"]
        obj["translation"] = meta["translation"]
        obj["scale"] = meta["scale"]
        obj["physical_size_m"] = meta.get("physical_size_m")
        obj["display_color"] = meta.get("display_color")
        obj["icp_pose"] = meta.get("icp_pose")
        obj["physics"] = meta.get("physics")
        obj["depth_info"] = meta.get("depth_info")
        obj["qa"] = meta.get("qa")
        obj["mesh_path"] = f"object_{i}/mesh.obj"
        if i < len(labels):
            obj["label"] = labels[i]
        if i < len(gemini_boxes):
            obj["box_px"] = list(gemini_boxes[i]["box_px"])
        objects.append(obj)

    scene = {
        "num_objects": len(objects),
        "objects": objects,
        "image_size_px": list(image_size),  # (W, H)
        "source": "hunyuan3d-2",
        "physical_scale_baked": True,  # mesh.obj is already at real-world meters
    }
    layout_path = os.path.join(output_dir, "scene_layout.json")
    with open(layout_path, "w") as f:
        json.dump(scene, f, indent=2)
    return layout_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hunyuan3D-2 + Gemini scene pipeline")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/scene_hunyuan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2",
                   help="HuggingFace model id or local path")
    p.add_argument("--skip_texture", action="store_true",
                   help="Skip texture pipeline (faster, but mesh has no color)")
    p.add_argument("--use_depth_mesh", action="store_true",
                   help="Build meshes directly from depth point clouds instead of "
                        "Hunyuan3D-2. Much faster, no artifacts, uses REAL geometry "
                        "from the depth sensor. Requires --depth_path.")
    p.add_argument("--depth_path", type=str, default=None,
                   help="Path to depth .npy (float32, meters) from RealSense. "
                        "When provided, uses real depth for metric-accurate sizing "
                        "instead of bbox estimation.")
    p.add_argument("--intrinsics_path", type=str, default=None,
                   help="Path to camera intrinsics .json (fx, fy, cx, cy)")
    # --- Object focus (reconstruct only central/reachable objects) ---
    p.add_argument("--extrinsics_path", type=str, default=None,
                   help="Path to extrinsics.json (TF camera->robot transforms). "
                        "Enables reach-based object filtering.")
    p.add_argument("--robot_base_frame", type=str, default="ur5e_base_link",
                   help="Which frame in extrinsics.json to measure reach against.")
    p.add_argument("--max_objects", type=int, default=4,
                   help="Cap how many objects to reconstruct (closest/most central).")
    p.add_argument("--edge_margin_frac", type=float, default=0.04,
                   help="Drop objects whose bbox is within this fraction of an edge.")
    p.add_argument("--max_depth_m", type=float, default=1.2,
                   help="Drop objects farther than this (camera-frame z, meters).")
    p.add_argument("--max_reach_m", type=float, default=0.85,
                   help="Drop objects beyond this planar reach from the robot base.")
    p.add_argument("--no_focus", dest="focus", action="store_false", default=True,
                   help="Disable object focus filtering (reconstruct everything).")
    # --- Render-and-compare QA loop ---
    p.add_argument("--qa", dest="qa", action="store_true", default=True,
                   help="Enable render-and-compare QA + refine loop (default on).")
    p.add_argument("--no_qa", dest="qa", action="store_false",
                   help="Disable the render-and-compare QA loop.")
    p.add_argument("--qa_iou_thresh", type=float, default=0.6,
                   help="Min silhouette IoU (azimuth-searched + centroid-aligned) "
                        "to accept a reconstruction. Good objects score ~0.69-0.81; "
                        "shape failures (e.g. funnels) stay well below.")
    p.add_argument("--qa_max_attempts", type=int, default=3,
                   help="Max Hunyuan attempts (seeds) per object before accepting best.")
    p.add_argument("--no_primitives", dest="primitives", action="store_false",
                   default=True,
                   help="Disable clean primitive fitting for simple shapes "
                        "(round fruit -> ellipsoid, plates/bowls -> dish). When "
                        "on (default), those categories use a parametric mesh "
                        "sized from depth instead of unreliable single-view Hunyuan.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hunyuan3D-2 + Gemini Pipeline")
    print(f"  Image:  {args.image_path}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load depth data if provided (RealSense RGB-D)
    depth_data = None
    intrinsics_data = None
    if args.depth_path and args.intrinsics_path:
        depth_data = np.load(args.depth_path)
        with open(args.intrinsics_path) as f:
            intrinsics_data = json.load(f)
        print(f"[DEPTH] Loaded depth map: {depth_data.shape}, "
              f"range [{depth_data[depth_data>0].min():.3f}m, {depth_data.max():.3f}m]")
        print(f"[DEPTH] Intrinsics: fx={intrinsics_data['fx']:.1f}, fy={intrinsics_data['fy']:.1f}")

    # Stages 1 + 2
    masks, labels, gemini_boxes, image_size = gemini_guided_segment_sam2(
        args.image_path
    )
    if not masks:
        print("ERROR: no masks produced; aborting.")
        sys.exit(1)

    # ---- Calibrated camera->robot extrinsic (used by focus + QA pose) ----
    T_base_cam = None
    if args.extrinsics_path and os.path.isfile(args.extrinsics_path):
        with open(args.extrinsics_path) as f:
            _ext = json.load(f)
        _tf = (_ext.get("transforms") or {}).get(args.robot_base_frame)
        if _tf and _tf.get("T_base_cam"):
            T_base_cam = _tf["T_base_cam"]
            print(f"[EXTRINSIC] loaded '{args.robot_base_frame} <- camera' from {args.extrinsics_path}")

    # ---- Object focus: keep only central / reachable objects ----
    if args.focus:
        # Per-object camera-frame positions (for far/reach filtering)
        positions_cam = [None] * len(masks)
        if depth_data is not None and intrinsics_data is not None:
            from depth_scale import compute_physical_sizes
            _dr = compute_physical_sizes(depth_data, intrinsics_data, masks, gemini_boxes)
            positions_cam = [(r.get("position_cam") if r else None) for r in _dr]
        from object_focus import select_target_objects
        keep = select_target_objects(
            gemini_boxes, positions_cam, image_size, T_base_cam,
            edge_margin_frac=args.edge_margin_frac, max_depth_m=args.max_depth_m,
            max_reach_m=args.max_reach_m, max_objects=args.max_objects,
        )
        if keep and len(keep) < len(masks):
            print(f"  [FOCUS] kept {len(keep)}/{len(masks)} objects: "
                  f"{[gemini_boxes[i]['label'] for i in keep]}")
            masks = [masks[i] for i in keep]
            labels = [labels[i] for i in keep]
            gemini_boxes = [gemini_boxes[i] for i in keep]
        elif not keep:
            print("  [FOCUS] WARNING: filter kept 0 objects; keeping all as fallback")

    # Save mask overlay for visual debugging
    image_bgr = cv2.imread(args.image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    palette = np.random.RandomState(0).randint(0, 255, (len(masks), 3), dtype=np.uint8)
    for i, m in enumerate(masks):
        color = palette[i].tolist()
        overlay[m > 127] = (
            0.5 * overlay[m > 127] + 0.5 * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "segmentation_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Stage 3: reconstruct 3D meshes
    shape_pipeline = None
    texture_pipeline = None
    if args.use_depth_mesh:
        print("\n[Stage 3] Using depth point cloud → mesh (no Hunyuan needed)")
    else:
        print("\n[Stage 3] Loading Hunyuan3D-2...")
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    if not args.skip_texture and not args.use_depth_mesh:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            print("[Stage 3] Loading texture pipeline...")
            texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(args.model_path)
        except Exception as e:
            print(f"  [WARN] Could not load texture pipeline: {e}")
            print("  Continuing without textures (geometry only)")

    metadatas = []
    n = len(masks)
    for i in range(n):
        label = labels[i] if i < len(labels) else f"object_{i}"
        print(f"\n  Object {i}/{n - 1} ({label})...")
        rgba, had_occluders = crop_and_mask_object(
            image_rgb, masks[i], masks, i, tuple(gemini_boxes[i]["box_px"])
        )
        if had_occluders:
            print(f"    [INPAINT] occluders removed from crop (e.g. lemon-on-plate)")
        # Save the input crop for debugging
        obj_dir = os.path.join(args.output_dir, f"object_{i}")
        os.makedirs(obj_dir, exist_ok=True)
        rgba.save(os.path.join(obj_dir, "input_rgba.png"))

        # ---- Compute physical size + camera-frame position FIRST ----
        # (independent of the mesh, so the QA loop can retry generation).
        depth_result = None
        if depth_data is not None and intrinsics_data is not None:
            from depth_scale import compute_physical_sizes
            depth_results = compute_physical_sizes(
                depth_data, intrinsics_data, [masks[i]], [gemini_boxes[i]]
            )
            if depth_results and depth_results[0].get("physical_size_m"):
                physical_size = depth_results[0]["physical_size_m"]
                depth_result = depth_results[0]
            else:
                physical_size = estimate_physical_size(
                    tuple(gemini_boxes[i]["box_px"]), image_size)
                print(f"    [BBOX] fallback size={physical_size*100:.1f}cm")
        else:
            physical_size = estimate_physical_size(
                tuple(gemini_boxes[i]["box_px"]), image_size)

        # Depth-centroid pose (rotation refined below from the extrinsic).
        icp_pose = None
        if depth_result and depth_result.get("position_cam"):
            pos = depth_result["position_cam"]
            icp_pose = {"position_cam": pos, "rotation_cam": [0.0, 0.0, 0.0, 1.0],
                        "scale": 1.0, "source": "depth_centroid"}
            print(f"    [POSE] depth centroid: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

        # ---- 3D reconstruction (+ render-and-compare QA retry loop) ----
        mesh = None
        qa = None
        prim_cat = None
        if args.primitives:
            from primitive_fit import category_for_label, build_primitive
            prim_cat = category_for_label(label)
        if prim_cat:
            # Clean parametric primitive sized from depth (round fruit ->
            # ellipsoid, plate/bowl -> open dish). Avoids the blocky-cube /
            # warped-sliver failures of single-view Hunyuan on simple shapes.
            dw = (depth_result or {}).get("physical_width_m") or physical_size
            dh = (depth_result or {}).get("physical_height_m") or physical_size
            if prim_cat == "ellipsoid":
                ext = (dw, dh, min(dw, dh))
            else:  # dish (plate / bowl)
                ext = (physical_size, physical_size, 0.0)
            mesh = build_primitive(prim_cat, ext, label)
            mesh._r2s_scale_factor = 1.0  # already metric + oriented
            print(f"    [PRIMITIVE] {label} -> {prim_cat} "
                  f"ext={[round(float(e),3) for e in ext]}")
        elif args.use_depth_mesh and depth_data is not None:
            from depth_mesh import build_mesh_from_depth
            raw = build_mesh_from_depth(depth_data, masks[i], intrinsics_data,
                                        image_rgb, poisson_depth=8)
            if raw is None:
                print("    [SKIP] depth mesh failed, insufficient depth data")
                continue
            mesh = prepare_mesh(raw, physical_size, label=label, obj_id=i)
        else:
            from render_compare import score_reconstruction, mesh_to_camera_pose, \
                rotation_to_quat_wxyz
            qa_on = args.qa and icp_pose is not None
            attempts = max(1, args.qa_max_attempts) if qa_on else 1
            best_mesh, best_qa, best_seed = None, None, None
            for attempt in range(attempts):
                seed = args.seed + i * 100 + attempt
                try:
                    raw = run_hunyuan_shape(shape_pipeline, rgba, seed=seed)
                    if texture_pipeline is not None:
                        raw = texture_pipeline(raw, image=rgba)
                except Exception as e:
                    print(f"    [SKIP] Hunyuan failed (seed {seed}): {e}")
                    continue
                raw = repair_mesh(raw, label=f"obj{i}")
                cand = prepare_mesh(raw, physical_size, label=label, obj_id=i)
                if qa_on:
                    s = score_reconstruction(
                        cand, masks[i], intrinsics_data,
                        icp_pose["position_cam"], T_base_cam=T_base_cam,
                        rgb_image=image_rgb)
                    print(f"    [QA] attempt {attempt} (seed {seed}): "
                          f"IoU={s['iou']:.3f} coverage={s['coverage']:.3f}")
                    if best_qa is None or s["iou"] > best_qa["iou"]:
                        best_mesh, best_qa, best_seed = cand, s, seed
                    if s["iou"] >= args.qa_iou_thresh:
                        break
                else:
                    best_mesh, best_seed = cand, seed
                    break
            if best_mesh is None:
                print("    [SKIP] no valid reconstruction produced")
                continue
            mesh, qa = best_mesh, best_qa
            if qa is not None:
                # Save an overlay for the chosen (best) reconstruction.
                qa = score_reconstruction(
                    mesh, masks[i], intrinsics_data, icp_pose["position_cam"],
                    T_base_cam=T_base_cam, rgb_image=image_rgb,
                    out_overlay=os.path.join(obj_dir, "qa_overlay.png"))
                qa["accepted"] = qa["iou"] >= args.qa_iou_thresh
                qa["seed"] = best_seed
                print(f"    [QA] best IoU={qa['iou']:.3f} "
                      f"{'ACCEPTED' if qa['accepted'] else 'BELOW THRESHOLD'} "
                      f"(thresh {args.qa_iou_thresh})")

        # ---- Anti-funnel backstop: a tabletop object resting flat cannot be
        # taller than its footprint. Clamp Z if depth says otherwise. ----
        if mesh is not None and depth_result:
            fw = depth_result.get("physical_width_m")
            fh = depth_result.get("physical_height_m")
            foot = max([v for v in (fw, fh) if v] or [0.0])
            zext = float(mesh.extents[2])
            cap = foot * 1.5
            if foot > 0 and zext > cap > 0:
                sz = cap / zext
                Tz = np.eye(4); Tz[2, 2] = sz
                mesh.apply_transform(Tz)
                mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])  # reseat bottom
                print(f"    [CLAMP] height {zext*100:.1f}cm > footprint cap "
                      f"{cap*100:.1f}cm → scaled Z x{sz:.2f} (anti-funnel)")

        # ---- Refine pose rotation from extrinsic + QA azimuth (for color proj) ----
        # Use the azimuth that best matched the silhouette so the image projects
        # onto the mesh in the correct orientation (recovers rim colors, etc.).
        if icp_pose is not None and not args.use_depth_mesh:
            from render_compare import (up_in_camera, _rot_about,
                                        _rotation_z_to, rotation_to_quat_wxyz)
            _up = up_in_camera(T_base_cam)
            _az = float((qa or {}).get("azimuth_deg", 0.0))
            _R = _rot_about(_up, np.radians(_az)) @ _rotation_z_to(_up)
            icp_pose["rotation_cam"] = rotation_to_quat_wxyz(_R)
            icp_pose["azimuth_deg"] = _az

        # ---- Physics properties from material classification ----
        from physics_properties import estimate_physics
        physics = estimate_physics(
            mesh, label, rgba,
            use_gemini=bool(os.environ.get("GEMINI_API_KEY")),
            physical_size_m=physical_size,
        )

        # ---- Vertex colors from image projection ----
        color = sample_object_color(image_rgb, masks[i])
        if icp_pose and depth_data is not None:
            from vertex_colors import bake_vertex_colors, bake_uniform_color
            try:
                bake_vertex_colors(
                    mesh, image_rgb, depth_data, masks[i], intrinsics_data,
                    position_cam=icp_pose["position_cam"],
                    rotation_cam=icp_pose["rotation_cam"],
                    scale=icp_pose.get("scale", 1.0),
                )
            except Exception as e:
                print(f"    [COLOR] vertex baking failed: {e}, using uniform color")
                bake_uniform_color(mesh, color)
        else:
            from vertex_colors import bake_uniform_color
            bake_uniform_color(mesh, color)

        # mesh is already scaled+oriented (prepare_mesh) → prepared=True
        meta = save_object(mesh, args.output_dir, i, physical_size, color,
                           label=label, prepared=True)
        if icp_pose:
            meta["icp_pose"] = icp_pose
        if depth_result:
            meta["depth_info"] = depth_result
        if qa is not None:
            meta["qa"] = qa
        meta["physics"] = physics
        metadatas.append(meta)

    print("\n[Stage 4] Saving scene layout...")
    layout = save_scene_layout(
        metadatas, args.output_dir, labels, gemini_boxes, image_size
    )
    print(f"  ✓ {layout}")

    print(f"\n{'='*60}")
    print(f"  DONE — {len(metadatas)} objects reconstructed")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
