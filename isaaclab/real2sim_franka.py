"""
Simify-style Real-to-Sim: Load SAM 3D scene + Franka Panda.

Loads all objects from a SAM 3D scene_layout.json as individual rigid bodies
(same pipeline as the working single-object throw_sam3d_object.py),
transforms them from camera frame to Isaac Lab world frame, and spawns
a Franka Panda arm for manipulation.

Usage:
    ./isaaclab.sh -p scripts/real2sim_franka.py \
        --scene_dir ~/sam-3d-objects/outputs/kidsroom_test

    # Single object only:
    ./isaaclab.sh -p scripts/real2sim_franka.py \
        --scene_dir ~/sam-3d-objects/outputs/kidsroom_test \
        --object_id 5
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simify-style Real-to-Sim with Franka Panda.")
parser.add_argument("--scene_dir", type=str, required=True)
parser.add_argument("--object_id", type=int, default=None, help="Load only one object (default: all)")
parser.add_argument("--demo", type=str, default="none",
                    choices=["none", "push", "pick", "cup_to_plate", "move_plate",
                             "lemon_to_cup", "curobo_pick"],
                    help="After objects settle: 'curobo_pick' uses cuRobo GPU motion "
                         "planner for collision-free pick-and-place (sim-to-real quality); "
                         "'pick' uses IK state machine; 'cup_to_plate' picks the cup and "
                         "places it on the plate; 'lemon_to_cup' picks lemon into cup; "
                         "'push' sweeps through cluster; 'none' just observes.")
parser.add_argument("--pick_label", type=str, default="cup",
                    help="Substring of the object label to pick (used by 'pick' demo)")
parser.add_argument("--place_label", type=str, default="plate",
                    help="Substring of the object label to place near (used by cup_to_plate)")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--snap_stacked", action="store_true", default=False,
                    help="Snap small objects onto larger ones when footprints overlap. "
                         "Off by default — depth-derived XY is trusted.")
parser.add_argument("--workspace_x", type=float, nargs=2, default=[0.25, 0.85],
                    help="World X bounds (forward) for reachable workspace on table.")
parser.add_argument("--workspace_y", type=float, nargs=2, default=[-0.40, 0.40],
                    help="World Y bounds (lateral) for reachable workspace on table.")
# Orientation defaults encode the current handheld-camera-to-robot transform.
# They were dialed in against a RealSense held in front of the Franka looking
# down at the table. Any future capture with the SAME camera/robot relative
# orientation will work with no flags. If the camera is relocated, override
# from the CLI or eventually replace with hand-eye calibration.
parser.add_argument("--scene_yaw_deg", type=float, default=180.0,
                    help="Rotate the whole cluster about its centroid by this "
                         "angle (degrees, CCW about world Z). Default 180 "
                         "matches handheld-in-front-of-Franka setup.")
parser.add_argument("--mirror_scene", action="store_true", default=True,
                    help="Mirror the cluster about the robot-forward axis. "
                         "Default on for handheld camera (mirror image vs. robot view).")
parser.add_argument("--no_mirror_scene", dest="mirror_scene", action="store_false",
                    help="Disable the default mirror.")
parser.add_argument("--yaw_offset_deg", type=float, default=-90.0,
                    help="Extra constant rotation added to every object's "
                         "orientation (degrees about world Z). Default -90 "
                         "matches handheld-in-front-of-Franka setup.")
parser.add_argument("--flip_objects", type=str, default="",
                    help="Comma-separated object IDs to flip left/right "
                         "(180° about the object's own long axis). Useful "
                         "when a knife's blade or phone's camera ended up "
                         "on the wrong side after PCA yaw canonicalization. "
                         "E.g. '2,3' flips objects 2 and 3.")
parser.add_argument("--capture_dir", type=str, default=None,
                    help="RealSense capture directory (with depth.npy + "
                         "intrinsics.json). When provided and the scene is "
                         "not yet postprocessed, depth alignment runs "
                         "automatically. Auto-discovers ~/sam-3d-objects/"
                         "captures/latest if omitted.")
parser.add_argument("--camera_yaw_deg", type=float, default=0.0,
                    help="Camera yaw relative to the robot (degrees). "
                         "0 = camera directly in front of robot facing it. "
                         "90 = camera to the robot's left. "
                         "-90 = camera to the robot's right. "
                         "This is the one DOF that cannot be derived from "
                         "the table plane alone.")
parser.add_argument("--table_center", type=float, nargs=2, default=[0.55, 0.0],
                    help="Where the camera's optical axis hits the table "
                         "in robot base frame [X, Y] in meters. "
                         "X = forward from robot base (typically 0.3-0.7). "
                         "Y = left(+)/right(-) of robot (typically near 0). "
                         "Adjust until objects match their real positions.")
parser.add_argument("--robot_base_frame", type=str, default="ur5e_base_link",
                    help="Which robot base frame in extrinsics.json to place "
                         "objects relative to (default ur5e_base_link: the arm "
                         "base, REP-103 X-forward/Y-left/Z-up).")
parser.add_argument("--no_extrinsics", dest="use_extrinsics", action="store_false",
                    default=True,
                    help="Ignore the calibrated camera->robot extrinsic in the "
                         "capture dir and fall back to the table-plane + "
                         "--camera_yaw_deg heuristic.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

scene_dir = os.path.expanduser(args_cli.scene_dir)
layout_path = os.path.join(scene_dir, "scene_layout.json")
if not os.path.exists(layout_path):
    print(f"Error: scene_layout.json not found at {layout_path}")
    sys.exit(1)

with open(layout_path) as f:
    scene_layout = json.load(f)

print(f"\nSAM 3D scene: {scene_dir} ({scene_layout['num_objects']} objects)")


# ---------------------------------------------------------------------------
# Inline depth alignment — runs automatically when scene is not postprocessed
# ---------------------------------------------------------------------------

def _find_capture_dir(explicit_path, sdir, layout):
    """Resolve the RealSense capture directory.

    Priority: --capture_dir flag > scene_layout["capture_dir"] > auto-discover latest.
    """
    if explicit_path:
        p = os.path.expanduser(explicit_path)
        if os.path.isdir(p):
            return p
    stored = layout.get("capture_dir")
    if stored and os.path.isdir(os.path.expanduser(stored)):
        return os.path.expanduser(stored)
    sam3d_root = os.path.expanduser("~/sam-3d-objects")
    latest = os.path.join(sam3d_root, "captures", "latest")
    if os.path.exists(latest):
        real = os.path.realpath(latest)
        if os.path.exists(os.path.join(real, "depth.npy")):
            return real
    return None


def _auto_postprocess(sdir, layout, capture_dir):
    """Run depth alignment + voxel remesh inline and update scene_layout.

    Replicates the core of postprocess_scene.py so the user never needs to
    run it as a separate step. Modifies mesh.obj files on disk and writes
    the updated scene_layout.json.
    """
    import json as _json
    import numpy as _np

    depth_path = os.path.join(capture_dir, "depth.npy")
    intrinsics_path = os.path.join(capture_dir, "intrinsics.json")
    if not os.path.exists(depth_path) or not os.path.exists(intrinsics_path):
        print(f"  [AUTO-PP] Missing depth.npy or intrinsics.json in {capture_dir}")
        return layout

    print(f"\n{'='*60}")
    print(f"[AUTO-PP] Scene not postprocessed — running depth alignment")
    print(f"  capture: {capture_dir}")
    print(f"{'='*60}")

    # Make sam-3d-objects importable (depth_alignment, table_plane live there)
    sam3d_root = os.path.expanduser("~/sam-3d-objects")
    if sam3d_root not in sys.path:
        sys.path.insert(0, sam3d_root)

    from depth_alignment import align_mesh_to_depth, compute_table_positions
    import trimesh
    from PIL import Image as PILImage

    depth = _np.load(depth_path)
    with open(intrinsics_path) as f:
        intrinsics = _json.load(f)

    layout["image_size_px"] = [intrinsics.get("width", depth.shape[1]),
                                intrinsics.get("height", depth.shape[0])]
    layout.setdefault("source", "sam3d")
    layout["capture_dir"] = capture_dir

    # --- Load per-object masks ---
    masks = []
    labels = []
    for obj in layout["objects"]:
        i = obj["id"]
        mp = os.path.join(sdir, "masks", f"{i}.png")
        if os.path.exists(mp):
            masks.append(_np.array(PILImage.open(mp).convert("L")))
        else:
            masks.append(_np.zeros(depth.shape[:2], dtype=_np.uint8))
        labels.append(obj.get("label", f"object_{i}"))

    # --- Fit table plane + compute per-object positions ---
    table_normal = None
    table_d_val = None
    try:
        table_positions, table_info = compute_table_positions(
            depth, masks, intrinsics, labels=labels
        )
        if table_info:
            table_normal = _np.array(table_info["normal"])
            table_d_val = table_info["d"]
            layout["table_plane"] = table_info
        for obj, tpos in zip(layout["objects"], table_positions):
            obj["table_position"] = tpos
    except Exception as e:
        print(f"  [AUTO-PP] Table plane fitting failed: {e}")

    # --- Per-object: voxel remesh + depth alignment ---
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    for idx, obj in enumerate(layout["objects"]):
        i = obj["id"]
        label = obj.get("label", f"object_{i}")
        mesh_path = os.path.join(sdir, f"object_{i}", "mesh.obj")
        if not os.path.exists(mesh_path):
            print(f"  object_{i} ({label}): SKIP (no mesh.obj)")
            continue

        print(f"\n  object_{i} ({label}):")
        m = trimesh.load(mesh_path, force="mesh")

        # Repair mesh normals (skip voxel remesh — it destroys vertex
        # colors and fine geometry from SAM 3D's texture baking).
        trimesh.repair.fix_normals(m)
        if not m.is_watertight:
            trimesh.repair.fix_winding(m)
        print(f"    [MESH] {len(m.faces)} faces, watertight={m.is_watertight}")

        # Depth alignment
        mask_arr = masks[idx] if idx < len(masks) else None
        if mask_arr is not None:
            result = align_mesh_to_depth(
                m, depth, mask_arr, intrinsics,
                table_normal=table_normal, table_d=table_d_val,
            )
            m = result["mesh"]
            obj["physical_size_m"] = result["physical_size_m"]
            obj["physical_extents"] = [round(e, 4) for e in result["physical_extents"]]
            obj["scale_method"] = result["method"]
            print(f"    [ALIGN] {result['method']}: {result['physical_size_m']*100:.1f}cm "
                  f"({result['physical_extents'][0]*100:.1f}x"
                  f"{result['physical_extents'][1]*100:.1f}x"
                  f"{result['physical_extents'][2]*100:.1f}cm) "
                  f"coverage={result['depth_coverage']:.0%}")

            # Camera-frame position from depth centroid
            binary = mask_arr > 127
            valid_mask = binary & (depth > 0.05) & (depth < 3.0)
            vs, us = _np.where(valid_mask)
            if len(vs) > 10:
                zs = depth[vs, us]
                med_z = float(_np.median(zs))
                rows = _np.any(binary, axis=1)
                cols = _np.any(binary, axis=0)
                if rows.any() and cols.any():
                    r0, r1 = _np.where(rows)[0][[0, -1]]
                    c0, c1 = _np.where(cols)[0][[0, -1]]
                    u_c = (c0 + c1) / 2.0
                    v_c = (r0 + r1) / 2.0
                    x_cam = float((u_c - cx) * med_z / fx)
                    y_cam = float((v_c - cy) * med_z / fy)
                    obj["position_cam"] = [round(x_cam, 4), round(y_cam, 4),
                                           round(med_z, 4)]

        # Sample display color from RGB (for objects without vertex colors)
        rgb_path = os.path.join(capture_dir, "rgb.png")
        mask_path = os.path.join(sdir, "masks", f"{i}.png")
        if os.path.exists(rgb_path) and mask_arr is not None:
            try:
                import cv2
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                fg = mask_arr > 127
                if fg.any():
                    pixels = rgb[fg].astype(float) / 255
                    r, g, b = pixels.mean(axis=0)
                    mean = (r + g + b) / 3
                    boost = 1.5
                    r = min(1, max(0, mean + (r - mean) * boost))
                    g = min(1, max(0, mean + (g - mean) * boost))
                    b = min(1, max(0, mean + (b - mean) * boost))
                    obj["display_color"] = [round(r, 3), round(g, 3), round(b, 3)]
            except ImportError:
                pass

        # Mesh is now metric — reset scale/rotation/translation to identity
        obj["scale"] = [[1, 1, 1]]
        obj["rotation"] = [[0, 0, 0, 1]]
        obj["translation"] = [[0, 0, 0]]

        # Save processed mesh
        m.export(mesh_path)

    layout["physical_scale_baked"] = True

    # Persist to disk so subsequent runs skip the postprocessing
    out_path = os.path.join(sdir, "scene_layout.json")

    class _NpEnc(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(out_path, "w") as f:
        _json.dump(layout, f, indent=2, cls=_NpEnc)

    print(f"\n[AUTO-PP] Done — wrote {out_path}")
    return layout


def _compute_table_positions(sdir, layout, capture_dir):
    """Compute table-plane positions + yaw from depth for scenes that
    already have baked meshes but lack orientation data (e.g. Hunyuan).

    Does NOT modify mesh files — only adds table_position to scene_layout.
    """
    import json as _json
    import numpy as _np

    depth_path = os.path.join(capture_dir, "depth.npy")
    intrinsics_path = os.path.join(capture_dir, "intrinsics.json")
    if not os.path.exists(depth_path) or not os.path.exists(intrinsics_path):
        return layout

    print(f"\n[TABLE-POS] Computing table positions + yaw from depth...")
    print(f"  capture: {capture_dir}")

    sam3d_root = os.path.expanduser("~/sam-3d-objects")
    if sam3d_root not in sys.path:
        sys.path.insert(0, sam3d_root)

    from depth_alignment import compute_table_positions
    from PIL import Image as PILImage

    depth = _np.load(depth_path)
    with open(intrinsics_path) as f:
        intrinsics = _json.load(f)

    masks = []
    labels = []
    for obj in layout["objects"]:
        i = obj["id"]
        mp = os.path.join(sdir, "masks", f"{i}.png")
        if os.path.exists(mp):
            masks.append(_np.array(PILImage.open(mp).convert("L")))
        else:
            masks.append(_np.zeros(depth.shape[:2], dtype=_np.uint8))
        labels.append(obj.get("label", f"object_{i}"))

    try:
        table_positions, table_info = compute_table_positions(
            depth, masks, intrinsics, labels=labels
        )
        if table_info:
            layout["table_plane"] = table_info
        for obj, tpos in zip(layout["objects"], table_positions):
            obj["table_position"] = tpos
        print(f"[TABLE-POS] Done — {len(table_positions)} objects positioned")
    except Exception as e:
        print(f"[TABLE-POS] Failed: {e}")
        return layout

    # Persist so subsequent runs skip this step
    out_path = os.path.join(sdir, "scene_layout.json")

    class _NpEnc(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(out_path, "w") as f:
        _json.dump(layout, f, indent=2, cls=_NpEnc)

    return layout


# --- Trigger auto-postprocessing if needed ---
if not scene_layout.get("physical_scale_baked"):
    _cap_dir = _find_capture_dir(args_cli.capture_dir, scene_dir, scene_layout)
    if _cap_dir:
        scene_layout = _auto_postprocess(scene_dir, scene_layout, _cap_dir)
    else:
        print("[WARN] Scene not postprocessed and no capture directory found.\n"
              "       Objects may not be to scale. Pass --capture_dir or place\n"
              "       captures in ~/sam-3d-objects/captures/latest/ to enable\n"
              "       automatic depth alignment.")

# --- Compute table positions if missing (e.g. Hunyuan scenes) ---
_has_table_pos = any(
    isinstance(o.get("table_position"), dict)
    for o in scene_layout.get("objects", [])
)
if not _has_table_pos:
    _cap_dir = _find_capture_dir(args_cli.capture_dir, scene_dir, scene_layout)
    if _cap_dir:
        scene_layout = _compute_table_positions(scene_dir, scene_layout, _cap_dir)

# --- Load calibrated camera->robot extrinsic (from TF) if available ---
if args_cli.use_extrinsics:
    _ext_cap_dir = _find_capture_dir(args_cli.capture_dir, scene_dir, scene_layout)
    _ext_path = os.path.join(_ext_cap_dir, "extrinsics.json") if _ext_cap_dir else None
    if _ext_path and os.path.isfile(_ext_path):
        try:
            with open(_ext_path) as _f:
                _ext = json.load(_f)
            _frame = args_cli.robot_base_frame
            _tf = (_ext.get("transforms") or {}).get(_frame)
            if _tf and _tf.get("T_base_cam"):
                scene_layout["camera_extrinsics"] = {
                    "frame": _frame, "T_base_cam": _tf["T_base_cam"]}
                print(f"[EXTRINSICS] Using calibrated '{_frame} <- camera' transform "
                      f"from {_ext_path}")
            else:
                print(f"[EXTRINSICS] {_ext_path} has no transform for frame "
                      f"'{_frame}' (available: {list((_ext.get('transforms') or {}).keys())})")
        except Exception as _e:
            print(f"[EXTRINSICS] failed to load {_ext_path}: {_e!r}")
    else:
        print("[EXTRINSICS] no extrinsics.json found; using table-plane heuristic")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports after app launch."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Table surface is at z=0 in world frame (ground pushed down 1.05m). Franka base sits at z=0.
# Table top center is at (0.5, 0, 0) with a 90 deg Z rotation (matching lift_env_cfg).
TABLE_POS = (0.5, 0.0, 0.0)
TABLE_ROT = (0.707, 0.0, 0.0, 0.707)  # 90 deg around Z
TABLE_SURFACE_Z = 0.0
# Legacy scale multiplier for raw (non-postprocessed) SAM 3D meshes.
# Only used as a fallback when physical_size_m is unavailable.
OBJECT_SCALE_MULT = 0.15

# Target rectangle on the table where object cluster gets projected when we
# have Gemini bounding boxes. Centered in front of Franka, leaving margin from
# the table edges. (x = forward from Franka, y = left/right.)
GEMINI_TARGET_X = (0.35, 0.70)
GEMINI_TARGET_Y = (-0.25, 0.25)
MAX_OBJECT_SIZE_M = 0.50  # Skip objects larger than 50cm (likely background/misdetection)

# Anchor-front layout: the largest object lands at this world position, and
# every other object fans out behind it (toward higher x).
ANCHOR_X = 0.36
ANCHOR_Y = 0.0


# ---------------------------------------------------------------------------
# Coordinate transform: camera frame -> robot base frame
# ---------------------------------------------------------------------------

def cam_to_world_position(tx, ty, tz):
    """Legacy fallback: naive -90° X rotation (no calibration)."""
    return (tx, -tz, -ty)


def compute_T_robot_cam(table_plane, table_center_robot=(0.5, 0.0, 0.0),
                        camera_yaw_deg=0.0):
    """Compute the 4×4 camera-to-robot transform from table plane geometry.

    The table plane provides 5 of 6 DOF (pitch, roll, height, XY position).
    The remaining DOF — camera yaw around the table normal — must be
    specified via camera_yaw_deg because it cannot be derived from geometry.

    Args:
        table_plane: dict with 'normal' (3,) and 'd' (float)
        table_center_robot: (x, y, z) of table center in robot frame
        camera_yaw_deg: camera yaw relative to robot (degrees).
            0 = camera directly in front of robot facing it.
            90 = camera to the robot's left.
            -90 = camera to the robot's right.

    Returns:
        T_robot_cam: 4×4 numpy array transforming camera-frame points
                     to robot-base-frame points
    """
    import numpy as _np

    n = _np.array(table_plane["normal"], dtype=float)
    n /= _np.linalg.norm(n)
    d = float(table_plane["d"])

    # 1. Robot Z axis in camera frame = table "up" = opposite of table normal
    robot_z_cam = -n

    # 2. Camera viewing direction projected onto the table plane
    cam_z = _np.array([0.0, 0.0, 1.0])
    cam_z_proj = cam_z - cam_z.dot(n) * n
    proj_norm = _np.linalg.norm(cam_z_proj)
    if proj_norm < 1e-6:
        cam_z_proj = _np.array([0.0, -1.0, 0.0])
        cam_z_proj = cam_z_proj - cam_z_proj.dot(n) * n
        proj_norm = _np.linalg.norm(cam_z_proj)
    cam_z_proj /= proj_norm

    # 3. Base robot X axis = opposite of camera viewing direction on table
    robot_x_cam = -cam_z_proj

    # 4. Apply camera yaw correction. This rotates robot_x_cam around
    #    the table normal by camera_yaw_deg, accounting for the camera's
    #    actual facing direction relative to the robot.
    if abs(camera_yaw_deg) > 0.01:
        yaw_rad = _np.radians(camera_yaw_deg)
        cos_y = _np.cos(yaw_rad)
        sin_y = _np.sin(yaw_rad)
        # Rodrigues rotation around robot_z_cam (= -normal)
        robot_x_cam_orig = robot_x_cam.copy()
        robot_x_cam = (cos_y * robot_x_cam_orig
                       + sin_y * _np.cross(robot_z_cam, robot_x_cam_orig)
                       + (1 - cos_y) * robot_z_cam.dot(robot_x_cam_orig) * robot_z_cam)

    # 5. Complete right-handed frame
    robot_y_cam = _np.cross(robot_z_cam, robot_x_cam)
    robot_y_cam /= _np.linalg.norm(robot_y_cam)
    robot_x_cam = _np.cross(robot_y_cam, robot_z_cam)
    robot_x_cam /= _np.linalg.norm(robot_x_cam)

    # 5. Rotation matrix: transforms camera-frame vectors to robot-frame vectors
    #    Each ROW is a robot axis expressed in camera frame
    R_robot_cam = _np.array([robot_x_cam, robot_y_cam, robot_z_cam])

    # 6. Translation: where the camera's optical axis hits the table in camera frame
    #    Ray from camera origin along (0,0,1), intersection with plane n·p + d = 0:
    #    t = -d / n[2]
    if abs(n[2]) > 1e-6:
        t_hit = -d / n[2]
        hit_cam = _np.array([0.0, 0.0, t_hit])
    else:
        # Camera nearly parallel to table — use plane origin
        hit_cam = -d * n

    # This point on the table corresponds to ~table_center in robot frame.
    # Camera origin in robot frame: p_robot = R @ p_cam + t
    # For hit_cam → table_center:  table_center = R @ hit_cam + t
    # So: t = table_center - R @ hit_cam
    tc = _np.array(table_center_robot, dtype=float)
    t_robot = tc - R_robot_cam @ hit_cam

    T = _np.eye(4)
    T[:3, :3] = R_robot_cam
    T[:3, 3] = t_robot
    return T


def cam_to_world_quaternion(qx, qy, qz, qw):
    """Convert camera-frame quaternion to Isaac Lab world frame.

    Apply the -90 deg X rotation to the quaternion.
    R_world = R_cam2world @ R_cam_object
    R_cam2world = Rx(-90) = quat(cos(-45), sin(-45), 0, 0) = (0.7071, -0.7071, 0, 0)
    """
    # Quaternion for -90 deg around X: (w, x, y, z) = (0.7071, -0.7071, 0, 0)
    cw, cx, cy, cz = 0.7071067811865476, -0.7071067811865476, 0.0, 0.0

    # Quaternion multiplication: q_cam2world * q_object
    # Input is (qx, qy, qz, qw) from SAM 3D
    # Need to output (w, x, y, z) for Isaac Lab
    ow = cw * qw - cx * qx - cy * qy - cz * qz
    ox = cw * qx + cx * qw + cy * qz - cz * qy
    oy = cw * qy - cx * qz + cy * qw + cz * qx
    oz = cw * qz + cx * qy - cy * qx + cz * qw

    return (ow, ox, oy, oz)


# ---------------------------------------------------------------------------
# Per-object USD preparation — semantic primitive pipeline
# ---------------------------------------------------------------------------
#
# We combine two signals:
#   - Gemini's label ("water bottle", "phone", "knife", ...) → shape class
#   - physical_extents from depth OBB                         → metric size
#
# The Gemini label tells us WHAT the object is; the depth tells us how BIG.
# For common shape classes we instantiate a clean geometric primitive
# (cylinder, thin box, disc, sphere, elongated box) sized from the depth
# measurement. Unknown labels fall back to the SAM3D mesh.
#
# Primitives are used for BOTH visual and collision:
#   - Cleaner-looking than single-view SAM3D meshes
#   - Correct dimensions from depth (no hallucinated back)
#   - Stable physics (flat bottoms, symmetric COM by construction)
# ---------------------------------------------------------------------------


_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("cylinder", ("bottle", "thermos", "flask", "tumbler", "can", "cup",
                  "mug", "glass", "jar", "vase", "tube", "roller", "tin")),
    ("thin_box", ("phone", "smartphone", "tablet", "ipad", "book",
                  "card", "remote", "laptop")),
    ("elongated_box", ("knife", "pen", "pencil", "marker", "spoon",
                       "fork", "screwdriver", "chopstick", "stylus",
                       "ruler", "wand", "brush")),
    ("disc", ("plate", "bowl", "dish", "saucer", "coaster", "frisbee")),
    ("box", ("box", "case", "carton", "airpods", "wallet", "pack", "block",
             "charger", "brick")),
    ("sphere", ("apple", "orange", "lemon", "ball", "tomato", "grape",
                "peach", "onion", "tennis")),
]


def category_from_label(label: str) -> str:
    """Map a free-text label ('water bottle', 'airpods case', ...) to one of
    the supported primitive categories: cylinder / thin_box / elongated_box /
    disc / box / sphere / mesh. 'mesh' = use SAM3D mesh as-is."""
    lab = (label or "").lower()
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(kw in lab for kw in kws):
            return cat
    return "mesh"


def prepare_obj_usd_with_obb(obj_path: str, extents, mass: float,
                             out_usd_path: str,
                             display_color=None,
                             category: str = "mesh",
                             label: str = "",
                             visual_scale: float = 1.0) -> str:
    """Build a USD for an object.

    category: semantic class that controls which geometric primitive is used
              for visual+collision: cylinder, thin_box, elongated_box, disc,
              box, sphere, or mesh (SAM3D fallback).
    extents : (ex, ey, ez) in metres, measured by depth OBB, used to size
              whichever primitive is chosen.
    Root:     RigidBodyAPI + MassAPI on /Object.
    """
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt, UsdShade
    try:
        from pxr import PhysxSchema
        has_physx = True
    except ImportError:
        has_physx = False

    if os.path.exists(out_usd_path):
        os.remove(out_usd_path)
    stage = Usd.Stage.CreateNew(out_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/Object", "Xform")
    stage.SetDefaultPrim(root)

    # Rigid body + mass on root.
    UsdPhysics.RigidBodyAPI.Apply(root)
    if has_physx:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        # Cap velocities so a spawn-time collider overlap can't FLING the object
        # off the table. MaxDepenetrationVelocity 5.0 -> 0.5 m/s = gentle push-out;
        # cap max linear velocity and add light damping so objects settle in place.
        physx_rb.GetMaxLinearVelocityAttr().Set(2.0)
        physx_rb.GetMaxAngularVelocityAttr().Set(10.0)
        physx_rb.GetMaxDepenetrationVelocityAttr().Set(0.5)
        physx_rb.GetLinearDampingAttr().Set(0.2)
        physx_rb.GetAngularDampingAttr().Set(0.2)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.GetMassAttr().Set(float(mass))

    # Physics material: high friction + zero restitution so objects grip the
    # table and settle without sliding/drifting (like the real world), instead
    # of skating around on default-friction contacts.
    mat = UsdShade.Material.Define(stage, "/Object/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(1.0)
    pm.CreateDynamicFrictionAttr().Set(0.9)
    pm.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(root)
    UsdShade.MaterialBindingAPI(root).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics")

    ex, ey, ez = (float(extents[0]), float(extents[1]), float(extents[2]))

    # Visual = detailed SAM3D mesh with per-vertex colors from RGB projection
    # (see rgb_projection.py). We load vertex_colors.npy if it exists next to
    # the .obj; falls back to the scalar display_color primvar otherwise.
    m = trimesh.load(obj_path, force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=float)
    if visual_scale != 1.0:
        verts = verts * visual_scale
    faces = np.asarray(m.faces, dtype=int)

    mesh_prim = stage.DefinePrim("/Object/visual", "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    mn = verts.min(axis=0).astype(float)
    mx = verts.max(axis=0).astype(float)
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*mn.tolist()), Gf.Vec3f(*mx.tolist())])
    )

    # Per-vertex display colors from RGB projection, falling back to a
    # constant scalar display_color if the projection file isn't present.
    vcol_path = os.path.join(os.path.dirname(obj_path), "vertex_colors.npy")
    if os.path.exists(vcol_path):
        vcol = np.load(vcol_path)
        if len(vcol) == len(verts):
            pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.vertex,
            )
            pv.Set(Vt.Vec3fArray(
                [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in vcol]
            ))
        else:
            print(f"  [WARN] vertex_colors.npy len={len(vcol)} != n_verts={len(verts)};"
                  " falling back to display_color")
            vcol = None
    else:
        vcol = None
    if vcol is None:
        col = (tuple(display_color) if (display_color and len(display_color) >= 3)
               else (0.6, 0.6, 0.6))
        pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        )
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))]))

    # Collision = convex decomposition on the visual mesh.
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_col.GetApproximationAttr().Set("convexDecomposition")

    # Stability base: for meshes with sparse bottoms (e.g. Hunyuan
    # double-head artifacts), add a thin invisible cylinder at z=0 to
    # ensure the object sits flat on the table. Standard practice for
    # imperfect reconstructed meshes in physics simulation.
    bottom_verts = np.sum(np.abs(verts[:, 2] - verts[:, 2].min()) < 0.003)
    bottom_ratio = bottom_verts / max(len(verts), 1)
    if bottom_ratio < 0.03 and ez > 0.03:
        # Sparse bottom — add a flat collision disc
        base_radius = max(ex, ey) / 2.0
        base_prim = stage.DefinePrim("/Object/base_collision", "Cylinder")
        base_xf = UsdGeom.Xformable(base_prim)
        base_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.002))
        base_xf.AddScaleOp().Set(Gf.Vec3f(float(base_radius), float(base_radius), 0.002))
        UsdPhysics.CollisionAPI.Apply(base_prim)
        UsdGeom.Imageable(base_prim).MakeInvisible()

    stage.Save()
    return out_usd_path


def prepare_object_usd(src_usd_path: str, scale: float) -> str:
    """Prepare a SAM 3D mesh.usd with vertex colors + rigid body physics.

    This is the same pipeline that makes single objects look good.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    try:
        from pxr import PhysxSchema
        has_physx = True
    except ImportError:
        has_physx = False

    output_dir = os.path.join(os.path.dirname(src_usd_path), "isaaclab_r2s")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rigid_mesh.usd")
    if os.path.exists(out_path):
        os.remove(out_path)

    src_stage = Usd.Stage.Open(src_usd_path)
    dst_stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst_stage, 1.0)

    s = scale
    root_prim = dst_stage.DefinePrim("/Object", "Xform")
    dst_stage.SetDefaultPrim(root_prim)

    # Rigid body
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    if has_physx:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_rb.GetMaxLinearVelocityAttr().Set(1000.0)
        physx_rb.GetMaxAngularVelocityAttr().Set(1000.0)
        physx_rb.GetMaxDepenetrationVelocityAttr().Set(5.0)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(0.2)

    # Geometry: scale only, NO rotation (we handle frame transform via init_state)
    geom_prim = dst_stage.DefinePrim("/Object/geometry", "Xform")
    geom_xform = UsdGeom.Xformable(geom_prim)
    geom_xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))

    # Copy mesh
    src_mesh_prim = src_stage.GetPrimAtPath("/World/mesh/mesh")
    src_mesh = UsdGeom.Mesh(src_mesh_prim)
    dst_mesh_prim = dst_stage.DefinePrim("/Object/geometry/mesh", "Mesh")
    dst_mesh = UsdGeom.Mesh(dst_mesh_prim)

    dst_mesh.GetPointsAttr().Set(src_mesh.GetPointsAttr().Get())
    dst_mesh.GetFaceVertexCountsAttr().Set(src_mesh.GetFaceVertexCountsAttr().Get())
    dst_mesh.GetFaceVertexIndicesAttr().Set(src_mesh.GetFaceVertexIndicesAttr().Get())
    if src_mesh.GetNormalsAttr().Get():
        dst_mesh.GetNormalsAttr().Set(src_mesh.GetNormalsAttr().Get())
        dst_mesh.SetNormalsInterpolation(src_mesh.GetNormalsInterpolation())

    # Copy vertex colors
    src_pvapi = UsdGeom.PrimvarsAPI(src_mesh_prim)
    dst_pvapi = UsdGeom.PrimvarsAPI(dst_mesh_prim)
    for src_pv in src_pvapi.GetPrimvars():
        name = src_pv.GetPrimvarName()
        dst_pv = dst_pvapi.CreatePrimvar(name, src_pv.GetTypeName(), src_pv.GetInterpolation())
        val = src_pv.Get()
        if val is not None:
            dst_pv.Set(val)

    # Collision
    UsdPhysics.CollisionAPI.Apply(dst_mesh_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(dst_mesh_prim)
    mesh_col.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    dst_stage.Save()
    return out_path


# ---------------------------------------------------------------------------
# SAM 3D over-segmentation: automatic deduplication via AABB containment.
#
# SAM-2 frequently produces multiple masks for a single physical object
# (e.g. plate rim + plate body, cup body + cup handle). Each mask becomes
# its own reconstructed mesh, giving 11 "objects" where the photo had 4.
#
# Fix: load each mesh's canonical AABB, transform to world space using the
# per-object translation + scale, then greedily drop any object whose
# world-frame XY footprint is mostly contained inside a larger surviving
# object. No manual prompting required.
# ---------------------------------------------------------------------------

def load_obj_aabb(obj_path: str):
    """Return canonical-frame (min_xyz, max_xyz) by streaming .obj vertices."""
    vmin = [float("inf")] * 3
    vmax = [float("-inf")] * 3
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    p = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
                for i in range(3):
                    if p[i] < vmin[i]:
                        vmin[i] = p[i]
                    if p[i] > vmax[i]:
                        vmax[i] = p[i]
    return tuple(vmin), tuple(vmax)


def world_xy_footprint(pos_world, canonical_min, canonical_max, scale_mult):
    """Compute (xmin, ymin, xmax, ymax) of an object's XY footprint in world frame.

    We apply scale to the canonical AABB extent, then center it at the object's
    world position. This is an approximation (ignores per-object rotation, which
    we zero out anyway) but plenty good enough to detect sub-part masks.
    """
    ex = (canonical_max[0] - canonical_min[0]) * scale_mult
    ey = (canonical_max[1] - canonical_min[1]) * scale_mult
    # Also consider Z extent so round objects (cups) that have narrow XY
    # footprints but tall profiles don't look microscopic.
    ez = (canonical_max[2] - canonical_min[2]) * scale_mult
    # Use the max of X/Y/Z to pick an effective XY radius (handles cups reconstructed
    # from any canonical orientation).
    r = 0.5 * max(ex, ey, ez)
    cx, cy, _ = pos_world
    return (cx - r, cy - r, cx + r, cy + r)


def canonical_shape_signature(canonical_min, canonical_max):
    """Return (min_dim, mid_dim, max_dim, aspect, volume) from canonical AABB."""
    ex = canonical_max[0] - canonical_min[0]
    ey = canonical_max[1] - canonical_min[1]
    ez = canonical_max[2] - canonical_min[2]
    dims = sorted([ex, ey, ez])
    min_d, mid_d, max_d = dims
    aspect = min_d / max_d if max_d > 0 else 0.0
    volume = ex * ey * ez
    return min_d, mid_d, max_d, aspect, volume


def is_manipulable(canonical_min, canonical_max, scale_mult,
                   min_size_m=0.02, max_size_m=0.40, min_aspect=0.05):
    """Geometry-only manipulability check (GPT-4o-free alternative).

    A real manipulable object must satisfy:
      1. Non-degenerate 3D volume (not a paper-thin planar fragment)
      2. Size within a tabletop-manipulable range (not microscopic or furniture-scale)

    Returns (bool, reason_string).
    """
    min_d, mid_d, max_d, aspect, _ = canonical_shape_signature(
        canonical_min, canonical_max
    )
    if max_d <= 0:
        return False, "empty mesh"
    if aspect < min_aspect:
        return False, f"planar fragment (aspect={aspect:.3f})"
    max_world = max_d * scale_mult
    if max_world < min_size_m:
        return False, f"too small ({max_world*100:.1f}cm)"
    if max_world > max_size_m:
        return False, f"too large ({max_world*100:.1f}cm)"
    return True, ""


def _shapes_compatible(a, b, ratio_tol=2.5):
    """Two objects have 'similar shape' if their aspect ratios are within tol.

    Used during dedupe: a plate (aspect≈0.13) and a lemon (aspect≈0.58) inside
    the plate have overlapping XY footprints, but very different aspect ratios.
    We keep both when shapes diverge strongly — that's the lemon-in-plate case.
    """
    if a <= 0 or b <= 0:
        return True
    r = max(a, b) / min(a, b)
    return r <= ratio_tol


def dedupe_by_containment(staged_objs, contain_thresh=0.55):
    """Drop objects whose XY footprint is mostly inside another surviving footprint
    AND whose canonical aspect ratio is similar (so they're likely the same
    physical object split by SAM-2, not a lemon sitting inside a plate).

    Greedy: sort by footprint area descending (biggest first), then for each
    candidate check whether >=contain_thresh of its area lies inside an already
    kept object with compatible shape. If so, it's a sub-part — drop it.
    """
    items = []
    for s in staged_objs:
        xmin, ymin, xmax, ymax = s["footprint"]
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        items.append((area, s))
    items.sort(key=lambda t: -t[0])

    kept = []
    dropped = []
    for area, s in items:
        if area <= 0:
            dropped.append((s["id"], "degenerate footprint"))
            continue
        merged_into = None
        for k in kept:
            kxmin, kymin, kxmax, kymax = k["footprint"]
            sxmin, symin, sxmax, symax = s["footprint"]
            ov_w = max(0.0, min(sxmax, kxmax) - max(sxmin, kxmin))
            ov_h = max(0.0, min(symax, kymax) - max(symin, kymin))
            overlap = ov_w * ov_h
            if overlap / area >= contain_thresh and _shapes_compatible(
                s["aspect"], k["aspect"]
            ):
                merged_into = k["id"]
                break
        if merged_into is None:
            kept.append(s)
        else:
            dropped.append((s["id"], f"sub-part of object_{merged_into}"))

    if dropped:
        print(f"  [DEDUPE] dropped {len(dropped)} over-segmented masks:")
        for oid, reason in dropped:
            print(f"    - object_{oid}: {reason}")
    print(f"  [DEDUPE] kept {len(kept)} / {len(staged_objs)} objects")
    return kept


def apply_manipulability_filter(staged_objs, scale_mult):
    """Drop planar/microscopic/oversized objects. Logs what goes and why."""
    kept = []
    dropped = []
    for s in staged_objs:
        ok, reason = is_manipulable(s["canon_min"], s["canon_max"], scale_mult)
        if ok:
            kept.append(s)
        else:
            dropped.append((s["id"], reason))
    if dropped:
        print(f"  [MANIP] dropped {len(dropped)} non-manipulable objects:")
        for oid, reason in dropped:
            print(f"    - object_{oid}: {reason}")
    print(f"  [MANIP] kept {len(kept)} / {len(staged_objs)} objects")
    return kept


def cap_to_top_k(staged_objs, k):
    """Optional hard cap: keep top-k by canonical volume."""
    if k is None or len(staged_objs) <= k:
        return staged_objs
    ranked = sorted(staged_objs, key=lambda s: -s["volume"])
    kept = ranked[:k]
    dropped = ranked[k:]
    print(f"  [CAP]   hard-capping to top-{k} by volume, dropped {len(dropped)}:")
    for s in dropped:
        print(f"    - object_{s['id']}: volume={s['volume']:.3f}")
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    # Top-down view matching the wrist camera perspective. Objects should appear
    # in the same relative positions as in the RealSense capture.
    sim.set_camera_view(eye=[0.50, 0.0, 1.5], target=[0.50, 0.0, 0.0])

    # Ground (pushed down so the table surface lands at z=0)
    sim_utils.GroundPlaneCfg().func(
        "/World/GroundPlane", sim_utils.GroundPlaneCfg(), translation=(0.0, 0.0, -1.05)
    )
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    sim_utils.create_prim("/World/envs/env_0", "Xform")

    # --- Table (SeattleLabTable — same one used by Isaac-Lift-Cube-Franka env) ---
    table_cfg = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    table_cfg.func(
        "/World/envs/env_0/Table", table_cfg, translation=TABLE_POS, orientation=TABLE_ROT
    )

    # --- Franka Panda (sits on table surface at z=0) ---
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    robot = Articulation(robot_cfg)

    # --- Load SAM 3D objects ---
    scene_path = Path(scene_dir)
    objects_to_load = scene_layout["objects"]
    if args_cli.object_id is not None:
        objects_to_load = [o for o in objects_to_load if o["id"] == args_cli.object_id]
        if not objects_to_load:
            print(f"Error: object_id {args_cli.object_id} not found")
            return

    # Pass 1: gather metadata + canonical AABBs (no USD conversion yet, so
    # we don't waste work converting sub-part masks that will be dropped).
    staged = []
    for obj in objects_to_load:
        obj_id = obj["id"]
        obj_dir = scene_path / f"object_{obj_id}"

        usd_path = str(obj_dir / "mesh.usd")
        obj_path = str(obj_dir / "mesh.obj")
        converted_usd = str(obj_dir / "mesh_converted.usd")

        if not os.path.exists(obj_path) and not os.path.exists(usd_path) and not os.path.exists(converted_usd):
            print(f"  [SKIP] object_{obj_id}: no mesh.usd or mesh.obj found")
            continue

        tx, ty, tz = obj["translation"][0]
        sx = obj["scale"][0][0]
        wx, wy, wz = cam_to_world_position(tx, ty, tz)

        # Compute canonical AABB from the .obj for containment filtering.
        if os.path.exists(obj_path):
            canon_min, canon_max = load_obj_aabb(obj_path)
        else:
            # Fallback: rough unit-cube; won't dedupe well but won't crash either
            canon_min, canon_max = (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)

        _, _, _, aspect, volume = canonical_shape_signature(canon_min, canon_max)

        footprint = world_xy_footprint(
            (wx, wy, wz), canon_min, canon_max, sx * OBJECT_SCALE_MULT
        )

        # Skip oversized objects (likely misdetected background like cables)
        phys_size = float(obj.get("physical_size_m") or 0)
        if phys_size > MAX_OBJECT_SIZE_M:
            print(f"  [SKIP] object_{obj_id} ({obj.get('label','')}): "
                  f"too large ({phys_size*100:.0f}cm > {MAX_OBJECT_SIZE_M*100:.0f}cm)")
            continue

        staged.append({
            "id": obj_id,
            "label": obj.get("label", f"object_{obj_id}"),
            "box_px": obj.get("box_px"),
            "physical_size_m": obj.get("physical_size_m"),  # set by Hunyuan
            "physical_extents": obj.get("physical_extents"),  # [x,y,z] meters from depth OBB
            "display_color": obj.get("display_color"),       # set by Hunyuan
            "obj_dir": obj_dir,
            "obj_path": obj_path,
            "usd_path": usd_path,
            "converted_usd": converted_usd,
            "pos_raw": (wx, wy, wz),
            "scale": sx,
            "footprint": footprint,
            "canon_min": canon_min,
            "canon_max": canon_max,
            "aspect": aspect,
            "volume": volume,
        })

    # Safety-net dedupe: if SAM 3D was run with --use_gemini OR Hunyuan, every
    # input mask is already a real manipulable object (Gemini filtered them).
    # For Hunyuan we MUST skip dedupe — meshes are re-centered to origin so the
    # canonical AABB footprints overlap and dedupe wrongly drops 3 of 5 objects.
    if scene_layout.get("physical_scale_baked"):
        print(f"  [DEDUPE] skipped (source = {scene_layout.get('source', '?')}, "
              f"Gemini already guarantees uniqueness)")
    else:
        staged = dedupe_by_containment(staged, contain_thresh=0.55)

    # Detect mesh source early (needed for visual scale in Pass 2).
    physical_scale_baked = scene_layout.get("physical_scale_baked", False)
    mesh_source = scene_layout.get("source", "sam3d")
    print(f"  [SCENE] mesh source = {mesh_source}, physical_scale_baked = {physical_scale_baked}")

    # Pass 2: resolve/convert USD only for surviving objects.
    for s in staged:
        obj_id = s["id"]
        obj_dir = s["obj_dir"]
        obj_path = s["obj_path"]
        usd_path = s["usd_path"]
        converted_usd = s["converted_usd"]
        sx = s["scale"]
        _converted = False

        # Universal path: if a .obj exists, build a USD with the detailed
        # visual mesh + an OBB primitive collider sized to physical_extents.
        # This replaces MeshConverter's convex hull/decomposition which is
        # sensitive to marching-cubes base noise for tall objects.
        if os.path.exists(obj_path):
            obj_data = next((o for o in scene_layout["objects"] if o["id"] == obj_id), {})
            extents = obj_data.get("physical_extents") or [0.05, 0.05, 0.05]
            # Density-based mass: uniform default 200 kg/m^3 (plastic-ish),
            # scaled by OBB volume. Capped to [0.05, 1.5] kg for stability.
            vol = float(extents[0] * extents[1] * extents[2])
            obj_mass = max(0.05, min(1.5, vol * 200.0))
            display_color = obj_data.get("display_color")
            label = obj_data.get("label", "")
            category = category_from_label(label)
            # Compute visual scale: for baked meshes the vertices are already
            # metric (scale=1); for raw SAM 3D meshes, derive from depth.
            if physical_scale_baked:
                vis_scale = 1.0
            else:
                phys = float(obj_data.get("physical_size_m") or 0)
                canon_ext = max(s["canon_max"][i] - s["canon_min"][i] for i in range(3))
                if phys > 0 and canon_ext > 1e-6:
                    vis_scale = phys / canon_ext
                else:
                    vis_scale = sx * OBJECT_SCALE_MULT
                    print(f"  [WARN] object_{obj_id}: no physical_size_m — "
                          f"using legacy scale {vis_scale:.4f}")
            out_usd = str(obj_dir / "mesh_obb.usd")
            print(f"  Building object_{obj_id}/mesh_obb.usd "
                  f"(label='{label}', category={category}, "
                  f"OBB={extents[0]*100:.1f}×{extents[1]*100:.1f}×{extents[2]*100:.1f}cm, "
                  f"mass={obj_mass*1000:.0f}g, vis_scale={vis_scale:.4f})")
            usd_path = prepare_obj_usd_with_obb(
                obj_path=obj_path,
                extents=extents,
                mass=obj_mass,
                out_usd_path=out_usd,
                display_color=display_color,
                category=category,
                label=label,
                visual_scale=vis_scale,
            )
            _converted = True
        elif os.path.exists(converted_usd):
            usd_path = converted_usd
            _converted = True

        if _converted:
            s["usd"] = usd_path
        else:
            s["usd"] = prepare_object_usd(usd_path, sx)
        s["converted"] = _converted

    # ---- Decide where each object lands on the table ----
    #
    # Priority:
    #   1. Depth 3D positions + T_robot_cam (proper geometric transform)
    #   2. Table-plane projected positions (legacy path for SAM 3D scenes)
    #   3. Gemini bbox projection (approximate, no depth)

    # Check if we have ICP poses (from depth pipeline)
    have_icp = any(
        isinstance(o.get("icp_pose"), dict) and o["icp_pose"].get("position_cam")
        for o in scene_layout.get("objects", [])
        if o["id"] in {s["id"] for s in staged}
    )
    # Check if we have depth info (position_cam from depth_scale)
    have_depth = any(
        isinstance(o.get("depth_info"), dict) and o["depth_info"].get("position_cam")
        for o in scene_layout.get("objects", [])
        if o["id"] in {s["id"] for s in staged}
    )

    # Use camera-frame 3D positions when available (from depth + mask centroid).
    # These preserve the REAL metric spatial relationships between objects.
    # Fall back to Gemini bbox projection when no depth data.
    have_cam_positions = any(
        isinstance(next((o for o in scene_layout.get("objects", []) if o["id"] == s["id"]), {}).get("position_cam"), list)
        for s in staged
    )
    image_size_px = scene_layout.get("image_size_px")
    have_gemini = bool(image_size_px) and all(s.get("box_px") for s in staged)

    # Check for table-plane projected positions (most accurate).
    # Require at least one non-zero position — all-zeros means the
    # computation failed silently (e.g. no masks for Hunyuan scenes).
    have_table_positions = False
    for s in staged:
        obj_data = next((o for o in scene_layout.get("objects", []) if o["id"] == s["id"]), {})
        tp = obj_data.get("table_position")
        if isinstance(tp, dict) and (tp.get("table_x", 0) != 0 or tp.get("table_y", 0) != 0):
            have_table_positions = True
            break

    # Check for depth_info camera-frame positions (Hunyuan pipeline)
    have_depth_positions = any(
        isinstance(next((o for o in scene_layout.get("objects", []) if o["id"] == s["id"]), {})
                   .get("depth_info", {}).get("position_cam"), list)
        for s in staged
    )

    if not have_depth_positions and have_table_positions:
        # Table-plane positions (legacy path for SAM 3D scenes).
        #
        # Mapping:
        #   table_x (axis_u from camera X) = left/right in image → world -Y
        #   table_y (axis_v = cross(normal, axis_u)) = near/far in image → world X
        #
        # We center the object cluster at (0.50, 0.00) on the table.
        print("  [LAYOUT] Table-plane projected positions (RANSAC, metric)")

        # First compute the centroid of all table positions to center the cluster
        all_tx = [next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                  .get("table_position", {}).get("table_x", 0) for s in staged]
        all_ty = [next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                  .get("table_position", {}).get("table_y", 0) for s in staged]
        cx = sum(all_tx) / max(len(all_tx), 1)
        cy = sum(all_ty) / max(len(all_ty), 1)

        # Target workspace centered on the table in front of Franka.
        ws_x_lo, ws_x_hi = args_cli.workspace_x
        ws_y_lo, ws_y_hi = args_cli.workspace_y
        ws_center_x = 0.5 * (ws_x_lo + ws_x_hi)
        ws_center_y = 0.5 * (ws_y_lo + ws_y_hi)
        max_dx = 0.5 * (ws_x_hi - ws_x_lo)
        max_dy = 0.5 * (ws_y_hi - ws_y_lo)

        # Compute the real cluster's half-extent (in meters, table frame).
        # table_y maps to world X (forward/back), table_x maps to world -Y.
        world_dx = [(tpos_y - cy) for tpos_y in all_ty]
        world_dy = [-(tpos_x - cx) for tpos_x in all_tx]
        ext_x = max((abs(d) for d in world_dx), default=0.0)
        ext_y = max((abs(d) for d in world_dy), default=0.0)

        # Proportional rescale-to-fit: shrink uniformly (same factor on both
        # axes) only if the cluster exceeds the workspace, so relative spacing
        # and aspect ratio are preserved. Never upscale.
        scale_xy = 1.0
        if ext_x > max_dx and max_dx > 0:
            scale_xy = min(scale_xy, max_dx / ext_x)
        if ext_y > max_dy and max_dy > 0:
            scale_xy = min(scale_xy, max_dy / ext_y)
        if scale_xy < 1.0:
            print(f"    [RESCALE] cluster extent ({ext_x*100:.1f},{ext_y*100:.1f})cm "
                  f"exceeds workspace; shrinking by {scale_xy:.3f} to fit")
        else:
            print(f"    [RESCALE] cluster fits workspace — preserving real spacing 1:1")

        yaw_scene = math.radians(args_cli.scene_yaw_deg)
        cos_y = math.cos(yaw_scene)
        sin_y = math.sin(yaw_scene)
        if args_cli.scene_yaw_deg:
            print(f"    [SCENE_YAW] rotating cluster by {args_cli.scene_yaw_deg:+.0f}°")
        if args_cli.mirror_scene:
            print("    [MIRROR] mirroring cluster about robot-forward axis")

        for s in staged:
            obj_data = next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
            tpos = obj_data.get("table_position", {})
            tx = tpos.get("table_x", 0) - cx  # center relative to cluster
            ty = tpos.get("table_y", 0) - cy
            # Table coords → world coords, with uniform rescale to fit workspace.
            wx0 = ty * scale_xy  # table_y (forward/back) → world X
            wy0 = -tx * scale_xy  # table_x (left/right) → world -Y
            if args_cli.mirror_scene:
                wy0 = -wy0
            # Rotate (wx0, wy0) about cluster centroid by scene_yaw.
            wx_r = cos_y * wx0 - sin_y * wy0
            wy_r = sin_y * wx0 + cos_y * wy0
            wx = ws_center_x + wx_r
            wy = ws_center_y + wy_r
            s["pos_world"] = (wx, wy)
            print(f"    {s['label']:20s} table({tx*100:+6.1f},{ty*100:+6.1f})cm → world({wx:.3f},{wy:.3f})")

    elif have_depth_positions:
        _calib = scene_layout.get("camera_extrinsics")
        tp = scene_layout.get("table_plane")
        if _calib and _calib.get("T_base_cam"):
            # CALIBRATED path: place objects at their true positions in the
            # robot base frame using the TF hand-eye extrinsic. No heuristic
            # rotation, mirror, or auto-centering — this is metrically correct.
            T_be = np.array(_calib["T_base_cam"], dtype=float)
            print(f"  [LAYOUT] Using CALIBRATED extrinsic ('{_calib['frame']}' <- camera)")
            for s in staged:
                obj_data = next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                pos_cam = (obj_data.get("depth_info", {}).get("position_cam")
                           or obj_data.get("icp_pose", {}).get("position_cam")
                           or obj_data.get("position_cam"))
                if pos_cam:
                    p = T_be @ np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0])
                    s["pos_world"] = (float(p[0]), float(p[1]))
                    s["calib_z"] = float(p[2])  # real height (for stacking)
                    print(f"    {s['label']:20s} cam({pos_cam[0]:+.3f},{pos_cam[1]:+.3f},"
                          f"{pos_cam[2]:.3f}) -> robot({p[0]:.3f},{p[1]:.3f},{p[2]:+.3f})")
                else:
                    s["pos_world"] = (args_cli.table_center[0], args_cli.table_center[1])
            # Keep the cluster ON the table: recenter it on the workspace and
            # shrink-to-fit if it overruns the bounds, PRESERVING the real
            # relative arrangement (so nothing lands off the table edge).
            pts = [s["pos_world"] for s in staged if s.get("pos_world")]
            if pts:
                ccx = sum(p[0] for p in pts) / len(pts)
                ccy = sum(p[1] for p in pts) / len(pts)
                ws_cx = 0.5 * (args_cli.workspace_x[0] + args_cli.workspace_x[1])
                ws_cy = 0.5 * (args_cli.workspace_y[0] + args_cli.workspace_y[1])
                half_x = 0.5 * (args_cli.workspace_x[1] - args_cli.workspace_x[0]) - 0.06
                half_y = 0.5 * (args_cli.workspace_y[1] - args_cli.workspace_y[0]) - 0.06
                ex = max((abs(p[0] - ccx) for p in pts), default=0.0)
                ey = max((abs(p[1] - ccy) for p in pts), default=0.0)
                sfit = min(1.0,
                           (half_x / ex) if ex > 1e-6 else 1.0,
                           (half_y / ey) if ey > 1e-6 else 1.0)
                for s in staged:
                    if s.get("pos_world"):
                        px, py = s["pos_world"]
                        s["pos_world"] = (ws_cx + (px - ccx) * sfit,
                                          ws_cy + (py - ccy) * sfit)
                print(f"    [LAYOUT] centered cluster on table (fit scale={sfit:.2f})")
        elif tp and tp.get("normal"):
            T_robot_cam = compute_T_robot_cam(tp, table_center_robot=(
                args_cli.table_center[0],
                args_cli.table_center[1],
                TABLE_SURFACE_Z,
            ), camera_yaw_deg=args_cli.camera_yaw_deg)
            print("  [LAYOUT] Camera-to-robot transform (from table plane geometry)")
            print(f"    T_robot_cam rotation:\n"
                  f"      {T_robot_cam[0,:3]}\n"
                  f"      {T_robot_cam[1,:3]}\n"
                  f"      {T_robot_cam[2,:3]}")
            print(f"    T_robot_cam translation: {T_robot_cam[:3,3]}")

            # Use T_robot_cam ROTATION for correct relative arrangement
            # (orientation + mirror), then auto-center on workspace.
            R = T_robot_cam[:3, :3].copy()
            # Mirror correction: camera faces the robot, so left/right
            # are swapped. Negate the Y row of the rotation.
            R[1, :] = -R[1, :]

            # Transform all objects (rotation only, no translation)
            raw_positions = []
            for s in staged:
                obj_data = next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                pos_cam = (obj_data.get("depth_info", {}).get("position_cam")
                           or obj_data.get("icp_pose", {}).get("position_cam")
                           or obj_data.get("position_cam"))
                if pos_cam:
                    p_robot = R @ np.array(pos_cam)
                    raw_positions.append((float(p_robot[0]), float(p_robot[1])))
                else:
                    raw_positions.append((0.0, 0.0))

            # Auto-center: shift cluster centroid to workspace center
            ws_cx = 0.5 * (args_cli.workspace_x[0] + args_cli.workspace_x[1])
            ws_cy = 0.5 * (args_cli.workspace_y[0] + args_cli.workspace_y[1])
            cx = sum(p[0] for p in raw_positions) / max(len(raw_positions), 1)
            cy = sum(p[1] for p in raw_positions) / max(len(raw_positions), 1)

            for s, (rx, ry) in zip(staged, raw_positions):
                wx = ws_cx + (rx - cx)
                wy = ws_cy + (ry - cy)
                s["pos_world"] = (wx, wy)
                obj_data = next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                pos_cam = (obj_data.get("depth_info", {}).get("position_cam")
                           or obj_data.get("icp_pose", {}).get("position_cam")
                           or [0, 0, 0])
                print(f"    {s['label']:20s} cam({pos_cam[0]:+.3f},{pos_cam[1]:+.3f},{pos_cam[2]:.3f}) "
                      f"→ robot({wx:.3f},{wy:.3f})")
        else:
            # No table plane — fall back to raw cam_to_world with centering
            print("  [LAYOUT] Depth-centroid positions (no table plane, legacy fallback)")
            ws_center_x = 0.5 * (args_cli.workspace_x[0] + args_cli.workspace_x[1])
            ws_center_y = 0.5 * (args_cli.workspace_y[0] + args_cli.workspace_y[1])
            raw = []
            for s in staged:
                obj_data = next((o for o in scene_layout["objects"] if o["id"] == s["id"]), {})
                pos_cam = (obj_data.get("depth_info", {}).get("position_cam")
                           or obj_data.get("position_cam")
                           or [0, 0, 0.5])
                wx, wy, _ = cam_to_world_position(*pos_cam)
                raw.append((wx, wy))
            cx = sum(p[0] for p in raw) / max(len(raw), 1)
            cy = sum(p[1] for p in raw) / max(len(raw), 1)
            for s, (rwx, rwy) in zip(staged, raw):
                s["pos_world"] = (ws_center_x + rwx - cx, ws_center_y + rwy - cy)
                print(f"    {s['label']:20s} → world({s['pos_world'][0]:.3f},{s['pos_world'][1]:.3f})")

    else:
        if have_gemini:
            img_w, img_h = image_size_px
            u_centers = []
            v_centers = []
            for s in staged:
                x0, y0, x1, y1 = s["box_px"]
                u_centers.append((x0 + x1) / 2.0 / img_w)
                v_centers.append((y0 + y1) / 2.0 / img_h)
            u_min, u_max = min(u_centers), max(u_centers)
            v_min, v_max = min(v_centers), max(v_centers)
            u_span = max(u_max - u_min, 1e-6)
            v_span = max(v_max - v_min, 1e-6)
            tx_lo, tx_hi = GEMINI_TARGET_X
            ty_lo, ty_hi = GEMINI_TARGET_Y
            for s, u, v in zip(staged, u_centers, v_centers):
                u_n = (u - u_min) / u_span
                v_n = (v - v_min) / v_span
                wx = tx_lo + (1.0 - v_n) * (tx_hi - tx_lo)
                wy = ty_hi - u_n * (ty_hi - ty_lo)
                s["pos_world"] = (wx, wy)
            print(f"  [LAYOUT] Gemini bbox projection (no depth available)")
        else:
            for s in staged:
                s["pos_world"] = (0.50, 0.00)
            print("  [LAYOUT] No layout data — using default positions")

    # Pass 3: spawn each object with re-centered pose + real-world scale.
    # NOTE: we intentionally drop SAM 3D's per-object rotation. It is unreliable
    # (each mesh is reconstructed in its own canonical frame and the output
    # rotation often has wrong roll/pitch), which leaves plates tilted and cups
    # lying on their side. Spawning upright and letting physics settle for a
    # few frames gives a much more faithful tabletop arrangement.
    #
    # Spawn-Z: meshes from depth_alignment.py are bottom-at-z=0, so we place
    # them 1mm above the table surface. No freefall — XY from depth is trusted,
    # and the drop-and-bounce that used to shift positions during settle is
    # eliminated. Fallback for non-baked meshes uses a small conservative hop.
    spawn_z_by_id = {}
    # DEFAULT: spawn each object's bottom exactly on the table (no freefall) so
    # it stays put instead of dropping, bouncing, and scattering. Baked meshes
    # have their bottom at z=0; +1mm avoids interpenetration.
    for s in staged:
        spawn_z_by_id[s["id"]] = TABLE_SURFACE_Z + (
            0.001 if physical_scale_baked else 0.05)
    # Elevate ONLY genuinely stacked objects (small object whose footprint sits
    # inside a larger one AND whose real height is above it) so they rest ON the
    # support (e.g. lemon on plate) rather than falling through it.
    for s in staged:
        cz = s.get("calib_z")
        if cz is None or not s.get("pos_world"):
            continue
        sx, sy = s["pos_world"]
        s_size = float(s.get("physical_size_m") or 0.0)
        for big in staged:
            if big is s or not big.get("pos_world"):
                continue
            b_size = float(big.get("physical_size_m") or 0.0)
            bcz = big.get("calib_z")
            if b_size <= s_size or bcz is None:
                continue
            bx, by = big["pos_world"]
            dist = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5
            if dist < b_size * 0.5 and cz > bcz + 0.005:
                spawn_z_by_id[s["id"]] = TABLE_SURFACE_Z + max(0.0, cz - bcz) + 0.005
                print(f"  [STACK] {s['label']} rests on {big['label']} "
                      f"-> z={spawn_z_by_id[s['id']]:.3f}")
                break

    # Opt-in "stack-snap" hack for cases like lemon-on-plate where depth
    # says they're nearly co-located and we want the small object exactly
    # centered on the larger. Off by default — depth XY is trusted.
    if args_cli.snap_stacked:
        for i, s in enumerate(staged):
            s_size = float(s.get("physical_size_m") or 0.0)
            sx, sy = s["pos_world"]
            for j, big in enumerate(staged):
                if i == j:
                    continue
                b_size = float(big.get("physical_size_m") or 0.0)
                if b_size <= s_size:
                    continue  # only snap to LARGER objects
                bx, by = big["pos_world"]
                dist = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5
                if dist < b_size * 0.5:
                    s["pos_world"] = (bx, by)
                    print(f"  [SNAP] {s['label']} snapped onto {big['label']} center")

    sam_objects = []
    for s in staged:
        obj_id = s["id"]
        wx, wy = s["pos_world"]
        sx = s["scale"]

        wz = spawn_z_by_id.get(obj_id, TABLE_SURFACE_Z + 0.10)

        # Metric scale is baked into mesh vertices during USD preparation
        # (either by postprocess_scene.py or by visual_scale in
        # prepare_obj_usd_with_obb). No external scale needed.
        usd_scale = None
        displayed_size = float(s.get("physical_size_m") or 0.0)

        # Object yaw on the table. Three sources (priority order):
        #   1. yaw_on_table from depth alignment PCA (most accurate)
        #   2. bbox direction mapped through T_robot_cam (for Hunyuan scenes)
        #   3. Identity (circular/unknown objects)
        obj_data = next((o for o in scene_layout["objects"] if o["id"] == obj_id), {})
        tpos = obj_data.get("table_position", {}) or {}
        yaw_on_table = tpos.get("yaw_on_table")
        yaw_conf = tpos.get("yaw_confidence", 0.0)
        flip_ids = {int(x) for x in args_cli.flip_objects.split(",") if x.strip().isdigit()}

        if yaw_on_table is not None and yaw_conf >= 1.3:
            # Source 1: depth PCA yaw — transform through T_robot_cam
            tp = scene_layout.get("table_plane")
            if tp and tp.get("normal"):
                # The yaw is defined in table-plane (axis_u, axis_v) coords.
                # Map the yaw direction vector through T_robot_cam to get robot yaw.
                n_tbl = np.array(tp["normal"], dtype=float)
                n_tbl /= np.linalg.norm(n_tbl)
                cam_x = np.array([1.0, 0.0, 0.0])
                axis_u = cam_x - cam_x.dot(n_tbl) * n_tbl
                au_norm = np.linalg.norm(axis_u)
                if au_norm < 1e-6:
                    cam_x = np.array([0.0, 1.0, 0.0])
                    axis_u = cam_x - cam_x.dot(n_tbl) * n_tbl
                    au_norm = np.linalg.norm(axis_u)
                axis_u /= au_norm
                axis_v = np.cross(n_tbl, axis_u)
                axis_v /= np.linalg.norm(axis_v)
                # Yaw direction in camera frame
                yaw_dir_cam = (math.cos(yaw_on_table) * axis_u
                               + math.sin(yaw_on_table) * axis_v)
                # Transform to robot frame (rotation only)
                _T = compute_T_robot_cam(tp, table_center_robot=(0.55, 0.0, 0.0),
                                        camera_yaw_deg=args_cli.camera_yaw_deg)
                yaw_dir_robot = _T[:3, :3] @ yaw_dir_cam
                world_yaw = math.atan2(yaw_dir_robot[1], yaw_dir_robot[0])
            else:
                world_yaw = yaw_on_table
            qw = math.cos(world_yaw / 2.0)
            qz = math.sin(world_yaw / 2.0)
            rot_quat = [qw, 0.0, 0.0, qz]

        elif obj_data.get("box_px") and 'T_robot_cam' in dir():
            # Source 2: derive yaw from bbox shape + T_robot_cam.
            # The bbox long axis in image pixels → direction in camera frame
            # → direction in robot frame via T_robot_cam.
            box = obj_data["box_px"]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 1.5:
                # Elongated object — compute the target direction in robot frame
                if bh > bw:
                    long_axis_cam = np.array([0.0, 1.0, 0.0])
                else:
                    long_axis_cam = np.array([1.0, 0.0, 0.0])
                tp = scene_layout.get("table_plane")
                if tp:
                    n_tbl = np.array(tp["normal"], dtype=float)
                    n_tbl /= np.linalg.norm(n_tbl)
                    long_on_table = long_axis_cam - long_axis_cam.dot(n_tbl) * n_tbl
                    ln = np.linalg.norm(long_on_table)
                    if ln > 1e-6:
                        long_on_table /= ln
                        long_robot = T_robot_cam[:3, :3] @ long_on_table
                        target_yaw = math.atan2(long_robot[1], long_robot[0])

                        # Account for the mesh's own long axis direction.
                        # The rotation must align the mesh's long axis (X or Y)
                        # with the target direction, not just rotate around Z.
                        mesh_ext_xy = [
                            s["canon_max"][0] - s["canon_min"][0],
                            s["canon_max"][1] - s["canon_min"][1],
                        ]
                        if mesh_ext_xy[1] > mesh_ext_xy[0]:
                            # Mesh long axis is Y → subtract 90° so Y aligns with target
                            world_yaw = target_yaw - math.pi / 2.0
                        else:
                            # Mesh long axis is X → use target directly
                            world_yaw = target_yaw

                        qw = math.cos(world_yaw / 2.0)
                        qz = math.sin(world_yaw / 2.0)
                        rot_quat = [qw, 0.0, 0.0, qz]
                    else:
                        rot_quat = [1.0, 0.0, 0.0, 0.0]
                else:
                    rot_quat = [1.0, 0.0, 0.0, 0.0]
            else:
                rot_quat = [1.0, 0.0, 0.0, 0.0]
        else:
            rot_quat = [1.0, 0.0, 0.0, 0.0]

        if obj_id in flip_ids:
            # 180° flip about the object's long axis
            if rot_quat[0] != 1.0:
                world_yaw = 2.0 * math.acos(max(-1, min(1, rot_quat[0])))
                ax = math.cos(world_yaw)
                ay = math.sin(world_yaw)
                q_flip = (0.0, ax, ay, 0.0)
                w1, x1, y1, z1 = rot_quat
                w2, x2, y2, z2 = q_flip
                rot_quat = [
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ]
            else:
                rot_quat = [0.0, 1.0, 0.0, 0.0]

        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_0/Object_{obj_id}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[wx, wy, wz],
                rot=rot_quat,
            ),
            spawn=UsdFileCfg(
                usd_path=s["usd"],
                scale=usd_scale,
            ),
        )
        sam_obj = RigidObject(obj_cfg)
        sam_objects.append((obj_id, sam_obj))
        print(
            f"  object_{obj_id} ({s['label']}): pos=({wx:.3f}, {wy:.3f}, {wz:.3f}) "
            f"size~{displayed_size:.3f}m"
        )

    print(f"\n[INFO] Loaded {len(sam_objects)} objects + Franka Panda")

    # Apply per-object display colors if Hunyuan provided them. We walk every
    # mesh prim under the object and set the displayColor primvar.
    try:
        from pxr import Gf, Usd, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for s in staged:
            color = s.get("display_color")
            if not color:
                continue
            prim_path = f"/World/envs/env_0/Object_{s['id']}"
            root = stage.GetPrimAtPath(prim_path)
            if not root.IsValid():
                continue
            r, g, b = color
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    pv = mesh.GetDisplayColorPrimvar()
                    pv.Set([Gf.Vec3f(r, g, b)])
                    pv.SetInterpolation(UsdGeom.Tokens.constant)
        print("  [COLOR] applied display colors to spawned meshes")
    except Exception as e:
        print(f"  [COLOR] failed to apply display colors: {e}")

    # Reset
    sim.reset()
    robot.reset()
    for _, obj in sam_objects:
        obj.reset()

    print(f"\n{'='*60}")
    print(f"Simify Real-to-Sim: {len(sam_objects)} objects + Franka")
    print(f"Objects are manipulable rigid bodies")
    print(f"Demo mode: {args_cli.demo}")
    print(f"{'='*60}\n")

    dt = sim.get_physics_dt()
    device = args_cli.device

    # ---- IK + state machine setup ----
    if args_cli.demo != "none":
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim import SphereCfg, PreviewSurfaceCfg
        from isaaclab.utils.math import subtract_frame_transforms

        arm_joint_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
        finger_joint_ids, _ = robot.find_joints("panda_finger_joint.*", preserve_order=True)
        hand_body_ids, _ = robot.find_bodies("panda_hand")
        hand_body_id = hand_body_ids[0]
        ee_jacobi_idx = hand_body_id - 1  # fixed base

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

        # ---- Debug markers: red=pick, green=place, yellow=current IK target ----
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/demo_targets",
            markers={
                "pick": SphereCfg(
                    radius=0.025,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "place": SphereCfg(
                    radius=0.025,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "ik_target": SphereCfg(
                    radius=0.018,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        debug_markers = VisualizationMarkers(marker_cfg)
        # Indices: 0=pick, 1=place, 2=ik_target
        marker_translations = torch.tensor(
            [[0.0, 0.0, -10.0]] * 3, device=device, dtype=torch.float32
        )
        marker_indices = torch.tensor([0, 1, 2], device=device, dtype=torch.long)

        # EE points down — qw, qx, qy, qz
        DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)
        # IK targets the panda_hand frame which sits ~10cm above the fingertip
        # contact point. We add this offset so the fingertips actually reach the
        # commanded world Z (otherwise descend(z=0.02) puts fingers at z=-0.08
        # which is inside the table — IK clamps and the arm stays high).
        EE_FINGERTIP_OFFSET = 0.105

        # Synonym map so user-friendly labels still find Gemini's actual wording
        # ("pink tray" matches "plate", etc.)
        LABEL_SYNONYMS = {
            "plate": ["plate", "tray", "dish", "saucer"],
            "cup":   ["cup", "mug", "glass", "tumbler"],
            "bowl":  ["bowl", "dish"],
            "fruit": ["fruit", "lemon", "apple", "orange", "ball", "tomato"],
            "tube":  ["tube", "cylinder", "stick", "rod", "bottle"],
        }

        def find_obj_by_label(query):
            """Find an (id, RigidObject, label) tuple matching query (with synonyms)."""
            q = query.lower()
            candidates = LABEL_SYNONYMS.get(q, [q])
            for s, (oid, rigid) in zip(staged, sam_objects):
                lbl = s["label"].lower()
                if any(c in lbl for c in candidates):
                    return oid, rigid, s["label"]
            return None

        def world_pos_of(rigid):
            """Live world-frame position of a RigidObject (after physics settle)."""
            p = rigid.data.root_pos_w[0]
            return float(p[0].item()), float(p[1].item()), float(p[2].item())

    # Action plan dispatched from --demo. Each entry is a sequential pick→place
    # operation: ((pick_label, place_label_or_None, place_xyz_override_or_None))
    if args_cli.demo == "pick":
        plan = [(args_cli.pick_label, None, (0.55, 0.30, 0.05))]
    elif args_cli.demo == "cup_to_plate":
        plan = [(args_cli.pick_label, args_cli.place_label, None)]
    elif args_cli.demo == "move_plate":
        plan = [(args_cli.place_label, None, (0.55, -0.20, 0.05))]
    elif args_cli.demo == "lemon_to_cup":
        plan = [("fruit", "cup", None)]
    elif args_cli.demo == "curobo_pick":
        # cuRobo multi-action plan: use --pick_label and --place_label from CLI
        curobo_plan_actions = [
            (args_cli.pick_label, args_cli.place_label, None),
        ]
        plan = []  # IK plan stays empty; cuRobo handles everything
    elif args_cli.demo == "push":
        plan = []
    else:
        plan = []

    # =====================================================================
    # cuRobo collision-free motion planning path
    # =====================================================================
    if args_cli.demo == "curobo_pick":
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose as CuPose
        from curobo.types.state import JointState as CuJointState
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        print("\n[cuRobo] Setting up GPU motion planner...")
        tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

        # World model: table as a cuboid collision obstacle
        # Table surface is at z=0, but we lower the collision boundary by 2cm
        # so the gripper fingers can reach objects sitting flat on the surface
        # (like a plate at z=0.007). Without this offset, cuRobo's finger
        # collision spheres clip the table and planning fails for flat objects.
        TABLE_COLLISION_OFFSET = 0.02
        world_config = WorldConfig(
            cuboid=[Cuboid(
                name="table",
                dims=[1.2, 1.2, 1.05],
                pose=[0.5, 0.0, -0.525 - TABLE_COLLISION_OFFSET, 1, 0, 0, 0],
            )]
        )

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            world_model=world_config,
            tensor_args=tensor_args,
            num_trajopt_seeds=8,
            num_graph_seeds=8,
            interpolation_dt=0.02,
            use_cuda_graph=False,  # avoid goal-type mismatch after warmup
        )
        motion_gen = MotionGen(motion_gen_config)
        print("[cuRobo] Warming up (compiling CUDA kernels)...")
        motion_gen.warmup(warmup_js_trajopt=False)
        print("[cuRobo] Ready.")

        JOINT_NAMES = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ]
        HOME_JS = torch.tensor(
            [[0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]], device="cuda"
        )
        # Gripper-down quaternion (wxyz for cuRobo)
        DOWN_QUAT_CU = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device="cuda")
        # Side-grasp quaternion: gripper approaches along +Y, fingers open vertically
        # (for flat/wide objects like plates and bowls that exceed gripper opening)
        SIDE_QUAT_CU = torch.tensor([[0.7071, -0.7071, 0.0, 0.0]], device="cuda")
        EE_OFFSET_Z = 0.105  # hand frame → fingertip offset
        GRIPPER_MAX_OPENING = 0.08  # Panda gripper max finger opening (m)

        plan_config = MotionGenPlanConfig(
            enable_graph=True, max_attempts=30, enable_finetune_trajopt=True,
            num_trajopt_seeds=12, num_graph_seeds=12,
        )

        def curobo_plan(current_js_tensor, goal_pos, goal_quat=None):
            """Plan a collision-free trajectory with cuRobo."""
            if goal_quat is None or (isinstance(goal_quat, torch.Tensor) and goal_quat.shape == DOWN_QUAT_CU.shape):
                pass  # use as-is
            if goal_quat is None:
                goal_quat = DOWN_QUAT_CU
            cu_js = CuJointState(
                position=current_js_tensor,
                velocity=torch.zeros_like(current_js_tensor),
                acceleration=torch.zeros_like(current_js_tensor),
                joint_names=JOINT_NAMES,
            )
            goal = CuPose(
                position=goal_pos.unsqueeze(0).cuda() if goal_pos.dim() == 1 else goal_pos.cuda(),
                quaternion=goal_quat.cuda(),
            )
            result = motion_gen.plan_single(cu_js, goal, plan_config)
            if result.success.item():
                traj = result.get_interpolated_plan()
                return traj.position  # (T, 7)
            else:
                print(f"    [cuRobo] planning FAILED for goal {goal_pos.tolist()}")
                return None

        # Let objects settle
        print("[cuRobo] Settling objects (3 seconds)...")
        settle_steps = int(3.0 / dt)
        for _ in range(settle_steps):
            sim.step()
            robot.update(dt)
            for _, obj in sam_objects:
                obj.update(dt)

        current_js = robot.data.joint_pos[:, arm_joint_ids].clone()

        for action_idx, (pick_lbl, place_lbl, place_override) in enumerate(curobo_plan_actions):
            print(f"\n{'='*50}")
            print(f"[cuRobo] Action {action_idx + 1}/{len(curobo_plan_actions)}")

            # Re-query positions from live physics (objects may have moved)
            pick_match = find_obj_by_label(pick_lbl)
            if pick_match is None:
                print(f"  No object matches '{pick_lbl}', skipping")
                continue
            _, pick_rigid, pick_label_str = pick_match
            px, py, pz = world_pos_of(pick_rigid)

            if place_override is not None:
                plx, ply, plz = place_override
                place_label_str = "drop point"
            elif place_lbl:
                place_match = find_obj_by_label(place_lbl)
                if place_match:
                    _, place_rigid, place_label_str = place_match
                    plx, ply, plz = world_pos_of(place_rigid)
                    # Get the target object's physical size to compute its top surface
                    place_obj_size = 0.0
                    for s in staged:
                        if s["label"].lower() == place_label_str.lower():
                            place_obj_size = float(s.get("physical_size_m") or 0.0)
                            break
                    # Place just above the target's top surface (center_z + half_height)
                    plz += place_obj_size * 0.5 + 0.01  # 1cm clearance above top
                else:
                    plx, ply, plz = 0.55, 0.15, 0.05
                    place_label_str = "fallback drop"
            else:
                plx, ply, plz = 0.55, 0.15, 0.05
                place_label_str = "drop point"

            # Determine grasp strategy: if object is wider than gripper, offset
            # to the edge where it's narrower (grippable from top-down).
            obj_phys_size = 0.0
            for s in staged:
                if s["label"].lower() == pick_label_str.lower():
                    obj_phys_size = float(s.get("physical_size_m") or 0.0)
                    break
            # Offset to edge for wide objects so gripper closes on narrow rim
            # For objects wider than the gripper, offset to grip the RIM on one
            # side. Offset = 40% of diameter puts the gripper center at 80% of
            # the radius — the rim falls between the fingers.
            # For objects wider than the gripper, offset along Y (left/right) to
            # grip the RIM. The Panda fingers close along Y, so shifting in Y
            # puts the rim between the fingers (one inside, one outside the bowl).
            edge_offset_y = obj_phys_size * 0.40 if obj_phys_size > GRIPPER_MAX_OPENING else 0.0
            gpx = px  # no X offset
            gpy = py + edge_offset_y  # Y offset to grip the rim from the side

            # The hand frame is ~10.5cm above the fingertips. cuRobo's trajectory
            # typically undershoots the Z target by ~2-3cm (PD tracking error).
            # For rim grasps on wide objects, descend extra low so fingers wrap
            # UNDER the rim (not just press against the side).
            UNDERSHOOT_COMP = 0.03
            rim_extra = 0.02 if obj_phys_size > GRIPPER_MAX_OPENING else 0.0
            grasp_z = max(pz - UNDERSHOOT_COMP - rim_extra, 0.005) + EE_OFFSET_Z

            print(f"  Pick: '{pick_label_str}' at ({px:.3f}, {py:.3f}, {pz:.3f})"
                  f"{'  [rim grasp Y+%.1fcm]' % (edge_offset_y*100) if edge_offset_y > 0 else ''}")
            print(f"  Place: '{place_label_str}' at ({plx:.3f}, {ply:.3f}, {plz:.3f})")

            segments = [
                ("approach_above", torch.tensor([gpx, gpy, 0.30 + EE_OFFSET_Z]), 0.04, DOWN_QUAT_CU),
                ("descend",        torch.tensor([gpx, gpy, grasp_z]), 0.04, DOWN_QUAT_CU),
                ("close_gripper",  None, 0.0, DOWN_QUAT_CU),
                ("lift",           torch.tensor([gpx, gpy, 0.35 + EE_OFFSET_Z]), 0.0, DOWN_QUAT_CU),
                ("move_above",     torch.tensor([plx, ply, 0.35 + EE_OFFSET_Z]), 0.0, DOWN_QUAT_CU),
                ("lower",          torch.tensor([plx, ply, max(plz - UNDERSHOOT_COMP, 0.005) + EE_OFFSET_Z]), 0.0, DOWN_QUAT_CU),
                ("open_gripper",   None, 0.04, DOWN_QUAT_CU),
                ("retreat",        torch.tensor([plx, ply, 0.30 + EE_OFFSET_Z]), 0.04, DOWN_QUAT_CU),
            ]

            for seg_name, goal_pos, gripper_val, seg_quat in segments:
                print(f"\n  [cuRobo] segment: {seg_name}")

                # How many physics steps per cuRobo waypoint (match interpolation_dt)
                INTERP_DT = 0.02
                steps_per_wp = max(1, int(round(INTERP_DT / dt)))

                def step_sim():
                    robot.write_data_to_sim()
                    sim.step()
                    robot.update(dt)
                    for _, obj in sam_objects:
                        obj.update(dt)

                def settle(seconds=0.5):
                    """Hold position for a short time to let physics stabilize."""
                    for _ in range(int(seconds / dt)):
                        step_sim()

                if goal_pos is not None:
                    traj = curobo_plan(current_js, goal_pos, goal_quat=seg_quat)
                    if traj is not None:
                        print(f"    planned {traj.shape[0]} waypoints "
                              f"({steps_per_wp} physics steps each)")
                        # Execute trajectory at correct speed
                        finger_target = torch.tensor(
                            [[gripper_val, gripper_val]], device=device
                        )
                        for t in range(traj.shape[0]):
                            robot.set_joint_position_target(
                                traj[t:t+1], joint_ids=arm_joint_ids
                            )
                            robot.set_joint_position_target(
                                finger_target, joint_ids=finger_joint_ids
                            )
                            for _ in range(steps_per_wp):
                                step_sim()
                        current_js = traj[-1:].clone()
                        # Let the arm settle before next segment
                        settle(0.3)
                    else:
                        print(f"    SKIPPING (plan failed)")
                else:
                    # Gripper-only action — actively hold arm position while
                    # closing/opening the gripper. Without this, the arm drifts
                    # and the fingers lose contact with the object.
                    finger_target = torch.tensor(
                        [[gripper_val, gripper_val]], device=device
                    )
                    hold_steps = int(2.5 / dt)
                    for step_i in range(hold_steps):
                        robot.set_joint_position_target(
                            current_js, joint_ids=arm_joint_ids
                        )
                        robot.set_joint_position_target(
                            finger_target, joint_ids=finger_joint_ids
                        )
                        step_sim()
                    # Diagnostics: finger positions + actual EE world position
                    actual_fingers = robot.data.joint_pos[:, finger_joint_ids]
                    ee_pos_w = robot.data.body_pose_w[:, hand_body_id, :3]
                    fingertip_z = ee_pos_w[0, 2].item() - EE_OFFSET_Z
                    print(f"    gripper {'closed' if gripper_val < 0.02 else 'opened'}"
                          f"  target={gripper_val:.3f}"
                          f"  actual=[{actual_fingers[0,0]:.4f}, {actual_fingers[0,1]:.4f}]"
                          f"  gap={actual_fingers.sum().item()*2*100:.1f}mm"
                          f"  hand_z={ee_pos_w[0,2].item():.4f}"
                          f"  fingertip_z≈{fingertip_z:.4f}")

        print(f"\n[cuRobo] All {len(curobo_plan_actions)} actions complete! Holding position...")

        # Hold indefinitely so user can observe
        while simulation_app.is_running():
            sim.step()
            robot.update(dt)
            for _, obj in sam_objects:
                obj.update(dt)
        return  # skip the IK state machine below

    # State machine variables (IK path — used by all demos except curobo_pick)
    state = "settle" if args_cli.demo != "none" else "idle"
    state_start_step = 0
    SETTLE_SECONDS = 2.0

    # Per-action runtime state (filled in after settle)
    action_idx = 0
    pick_x = pick_y = pick_z = 0.0
    place_x = place_y = place_z = 0.0
    pick_label_str = ""
    place_label_str = ""
    target_xyz = None
    gripper_pos = 0.04

    def begin_action(idx):
        """Resolve targets for plan[idx] from live physics. Returns False if invalid."""
        nonlocal pick_x, pick_y, pick_z, place_x, place_y, place_z
        nonlocal pick_label_str, place_label_str
        if idx >= len(plan):
            return False
        pick_lbl, place_lbl, place_override = plan[idx]
        pick_match = find_obj_by_label(pick_lbl)
        if pick_match is None:
            print(f"  [DEMO] action {idx}: no object matches pick label '{pick_lbl}'")
            return False
        _, pick_rigid, pick_label_str = pick_match
        pick_x, pick_y, pick_z = world_pos_of(pick_rigid)
        # Lift slightly above object center
        if place_override is not None:
            place_x, place_y, place_z = place_override
            place_label_str = "drop point"
        else:
            place_match = find_obj_by_label(place_lbl)
            if place_match is None:
                print(f"  [DEMO] action {idx}: no object matches place label '{place_lbl}', using fallback drop point")
                place_x, place_y, place_z = 0.55, 0.30, 0.05
                place_label_str = "fallback drop"
            else:
                _, place_rigid, place_label_str = place_match
                px, py, pz = world_pos_of(place_rigid)
                # Place ABOVE the place-object, with some clearance so we don't ram it
                place_x, place_y, place_z = px, py, pz + 0.06
        print(
            f"\n  [DEMO] action {idx + 1}/{len(plan)}: "
            f"pick '{pick_label_str}' at ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f}) "
            f"-> place near '{place_label_str}' at ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})"
        )
        # Update pick (red) and place (green) markers
        marker_translations[0, 0] = pick_x
        marker_translations[0, 1] = pick_y
        marker_translations[0, 2] = pick_z
        marker_translations[1, 0] = place_x
        marker_translations[1, 1] = place_y
        marker_translations[1, 2] = place_z
        return True

    # Sweep params for push demo (z values are HAND-frame, fingertip ≈ z - 0.105)
    push_targets = [
        (0.5, -0.25, 0.30),  # hover left
        (0.5, -0.25, 0.13),  # descend left (fingertip ≈ 2.5cm above table)
        (0.5,  0.25, 0.13),  # sweep right at low height
        (0.5,  0.25, 0.40),  # retreat
    ]
    push_idx = 0

    step = 0
    while simulation_app.is_running():
        if args_cli.demo != "none":
            elapsed = (step - state_start_step) * dt

            # ----- State machine transitions -----
            if state == "settle":
                target_xyz = None
                gripper_pos = 0.04
                if elapsed >= SETTLE_SECONDS:
                    if args_cli.demo == "push":
                        state = "push_step"
                        state_start_step = step
                    else:
                        if begin_action(action_idx):
                            state = "approach_top"
                            state_start_step = step
                            diff_ik.reset()
                        else:
                            state = "hold"
                            state_start_step = step
                    print(f"  [DEMO step {step}] settle -> {state}")

            elif state == "approach_top":
                # Hover well above the object so we don't crash through it
                target_xyz = (pick_x, pick_y, 0.30 + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.04
                if elapsed >= 2.5:
                    descend_z = max(pick_z, 0.005) + EE_FINGERTIP_OFFSET
                    state, state_start_step = "descend", step
                    print(f"  [DEMO step {step}] -> descend  hand_target=({pick_x:.3f}, {pick_y:.3f}, {descend_z:.3f})  fingertip_z≈{descend_z - EE_FINGERTIP_OFFSET:.3f}")
                    diff_ik.reset()

            elif state == "descend":
                # Hand z = object center z + EE offset so fingertips wrap the object
                target_xyz = (pick_x, pick_y, max(pick_z, 0.005) + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.04
                if elapsed >= 2.0:
                    state, state_start_step = "close_gripper", step
                    print(f"  [DEMO step {step}] -> close_gripper")
                    diff_ik.reset()

            elif state == "close_gripper":
                target_xyz = (pick_x, pick_y, max(pick_z, 0.005) + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.0
                if elapsed >= 1.0:
                    state, state_start_step = "lift", step
                    print(f"  [DEMO step {step}] -> lift")
                    diff_ik.reset()

            elif state == "lift":
                target_xyz = (pick_x, pick_y, 0.35 + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.0
                if elapsed >= 2.0:
                    state, state_start_step = "move", step
                    print(f"  [DEMO step {step}] -> move")
                    diff_ik.reset()

            elif state == "move":
                target_xyz = (place_x, place_y, 0.35 + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.0
                if elapsed >= 3.0:
                    state, state_start_step = "lower", step
                    lower_z = max(place_z + 0.02, 0.005) + EE_FINGERTIP_OFFSET
                    print(f"  [DEMO step {step}] -> lower  hand_target=({place_x:.3f}, {place_y:.3f}, {lower_z:.3f})")
                    diff_ik.reset()

            elif state == "lower":
                target_xyz = (place_x, place_y, max(place_z + 0.02, 0.005) + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.0
                if elapsed >= 2.0:
                    state, state_start_step = "open_gripper", step
                    print(f"  [DEMO step {step}] -> open_gripper")
                    diff_ik.reset()

            elif state == "open_gripper":
                target_xyz = (place_x, place_y, max(place_z + 0.02, 0.005) + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.04
                if elapsed >= 1.0:
                    state, state_start_step = "retreat", step
                    print(f"  [DEMO step {step}] -> retreat")
                    diff_ik.reset()

            elif state == "retreat":
                target_xyz = (place_x, place_y, 0.30 + EE_FINGERTIP_OFFSET)
                gripper_pos = 0.04
                if elapsed >= 2.0:
                    action_idx += 1
                    if begin_action(action_idx):
                        state, state_start_step = "approach_top", step
                        diff_ik.reset()
                    else:
                        state, state_start_step = "hold", step
                        print(f"  [DEMO step {step}] -> hold (plan complete)")

            elif state == "push_step":
                target_xyz = push_targets[push_idx]
                gripper_pos = 0.0
                if elapsed >= 2.5:
                    push_idx += 1
                    if push_idx >= len(push_targets):
                        state, state_start_step = "hold", step
                        print(f"  [DEMO step {step}] push complete -> hold")
                    else:
                        state_start_step = step
                        diff_ik.reset()

            elif state == "hold":
                # Just keep the last commanded gripper state, no IK
                pass

            # ----- IK execution (when we have a target) -----
            if target_xyz is not None:
                # Push current IK target into yellow marker for visual debug
                marker_translations[2, 0] = target_xyz[0]
                marker_translations[2, 1] = target_xyz[1]
                marker_translations[2, 2] = target_xyz[2]

                ik_command = torch.tensor(
                    [[target_xyz[0], target_xyz[1], target_xyz[2], *DOWN_QUAT]],
                    device=device, dtype=torch.float32,
                )
                diff_ik.set_command(ik_command)

                jacobian = robot.root_physx_view.get_jacobians()[
                    :, ee_jacobi_idx, :, arm_joint_ids
                ]
                ee_pose_w = robot.data.body_pose_w[:, hand_body_id]
                root_pose_w = robot.data.root_pose_w
                joint_pos_arm = robot.data.joint_pos[:, arm_joint_ids]

                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
                )
                joint_target = diff_ik.compute(
                    ee_pos_b, ee_quat_b, jacobian, joint_pos_arm
                )
                robot.set_joint_position_target(joint_target, joint_ids=arm_joint_ids)

            # Gripper command every step
            finger_target = torch.tensor([[gripper_pos, gripper_pos]],
                                         device=device, dtype=torch.float32)
            robot.set_joint_position_target(finger_target, joint_ids=finger_joint_ids)
            robot.write_data_to_sim()

            # Update debug markers (red=pick, green=place, yellow=current IK target)
            debug_markers.visualize(
                translations=marker_translations,
                marker_indices=marker_indices,
            )

        sim.step()
        robot.update(dt)
        for _, obj in sam_objects:
            obj.update(dt)
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
