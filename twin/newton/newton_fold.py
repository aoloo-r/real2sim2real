"""Newton twin — robot FOLDS the captured cloth.

A Franka arm (Featherstone) folds the real captured fabric (VBD cloth, textured with the
capture photo) by pinching a corner, lifting, carrying it diagonally over the cloth, and
releasing — VBD particle self-contact handles the stacked layers, intersection-free.

Built on the shipped cloth_franka example (coupled Featherstone+VBD, diff-IK velocity
control, cm-scale sim / m-scale viz). Two adaptations:
  * cloth = our measured, TEXTURED add_cloth_grid (not the shirt USD); rendered via log_mesh
    with the fabric image (show_triangles=False so log_state doesn't double-draw it).
  * GRASP = pin the corner particle(s) to the gripper tip while closed (a flat-grid corner
    pinch by friction alone is tuning-fragile; the pin is load-bearing, the arm does the
    motion). Release restores the particles to free.

Run (newton-spike env):
  .../python twin/newton/newton_fold.py --scene_dir <out> [--capture_dir <cap>] --viewer gl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.viewer import ViewerGL, ViewerNull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sam3d_scene  # noqa: E402  faithful SAM3D loader (real meshes/colours/poses)

CLOTH_KW = ("fabric", "cloth", "towel", "shirt", "blanket", "napkin", "garment", "sheet", "scarf", "rag")
BOX_KW = ("box", "bag", "container", "basket", "crate", "bin", "carton", "tray")


def masked_grid(dim_x, dim_y, cell, mask, box_px, img):
    """Build a flat cloth grid (local frame, centred) but keep only the triangles whose vertices fall
    inside the object's segmentation mask -> the cloth takes the real SILHOUETTE (e.g. a shirt shape).
    Returns (vertices Nx3, triangle indices flat, uvs Nx2) mapped to the object's pixel box."""
    nx, ny = dim_x + 1, dim_y + 1
    bx0, by0, bx1, by1 = box_px; W, H = img
    verts = np.zeros((nx * ny, 3), np.float32)
    uvs = np.zeros((nx * ny, 2), np.float32)
    inside = np.ones(nx * ny, bool)
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            verts[k] = [(i - dim_x / 2.0) * cell, (j - dim_y / 2.0) * cell, 0.0]
            u, v = i / dim_x, j / dim_y
            px = bx0 + u * (bx1 - bx0); py = by0 + (1.0 - v) * (by1 - by0)   # match texture V-flip
            uvs[k] = [px / W, py / H]
            if mask is not None:
                inside[k] = mask[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))] > 127
    tris = []
    for j in range(dim_y):
        for i in range(dim_x):
            a = j * nx + i; b = a + 1; c = a + nx; d = c + 1
            for tri in ((a, b, c), (b, d, c)):
                if inside[tri[0]] and inside[tri[1]] and inside[tri[2]]:
                    tris.append(tri)
    tris = np.array(tris, np.int32)
    used = np.unique(tris)
    remap = -np.ones(nx * ny, np.int64); remap[used] = np.arange(len(used))
    return verts[used], remap[tris].reshape(-1).astype(np.int32), uvs[used]
DOWN_Q = (0.9239, -0.3827, 0.0, 0.0)   # gripper pointing down-ish (qw,qx,qy,qz), from the demo
OPEN, CLOSE = 0.8, 0.1


# ----------------------------- warp kernels (from the example + pin) -----------------------------
@wp.kernel
def scale_positions(src: wp.array(dtype=wp.vec3), scale: float, dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array(dtype=wp.transform), scale: float, dst: wp.array(dtype=wp.transform)):
    i = wp.tid()
    p = wp.transform_get_translation(src[i]); q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(body_q: wp.array(dtype=wp.transform), offset: wp.transform, body_id: int,
                     bodies_per_world: int, target: wp.transform,
                     ee_delta: wp.array(dtype=wp.spatial_vector)):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf); pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf); rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2],
                                           ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def compute_tip(body_q: wp.array(dtype=wp.transform), ee_id: int, ee_off: wp.vec3,
                out: wp.array(dtype=wp.vec3)):
    out[0] = wp.transform_point(body_q[ee_id], ee_off)


@wp.kernel
def pin_to_ee(body_q: wp.array(dtype=wp.transform), ee_id: int, ee_off: wp.vec3,
              pin_idx: wp.array(dtype=wp.int32), pin_off: wp.array(dtype=wp.vec3),
              particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    tip = wp.transform_point(body_q[ee_id], ee_off)
    pidx = pin_idx[i]
    particle_q[pidx] = tip + pin_off[i]
    particle_qd[pidx] = wp.vec3(0.0, 0.0, 0.0)


def faithful_objects(scene_dir, capture_dir, franka_x, franka_y, table_top, scene_yaw=0.0):
    """Import the SAM3D models AS-IS: real per-object mesh (Gemini-selected, SAM3D-built), calibrated
    to its depth-measured size (only the overall scale is corrected — geometry untouched), then the
    whole scene is rigidly translated so the cloth sits in the robot's fold zone on the table.
    Returns cloth dict {verts(cm), faces, color, bbox} and box dict (or None)."""
    sc = sam3d_scene.load_scene(scene_dir, capture_dir)
    objs = sc["objects"]
    cloth = next((o for o in objs if any(k in o["label"].lower() for k in CLOTH_KW)), None)
    box = next((o for o in objs if any(k in o["label"].lower() for k in BOX_KW)), None)
    if cloth is None:
        return None, None

    def calibrate(o):
        v = o["verts"].astype(np.float64).copy()               # base-frame metres
        c = v.mean(0)
        fp = (v[:, :2].max(0) - v[:, :2].min(0)).max()          # footprint max dim (m)
        meas = max([m for m in o["measured_m"] if m > 0] or [fp])
        s = meas / fp if fp > 1e-6 else 1.0                     # scale mesh to measured size
        v = (c + s * (v - c)) * 100.0                           # -> cm, scaled about centroid
        return v

    cv = calibrate(cloth)
    parts = {"cloth": cv}
    if box is not None:
        parts["box"] = calibrate(box)
    # FAITHFUL placement: keep each object at its REAL reconstructed position relative to the robot
    # (ur5e_base_link frame), with the Franka standing in for the base at (franka_x, franka_y). The
    # real robot-relative layout (shirt in front, box offset in +y) is preserved; only z is snapped
    # to the table to remove reconstruction depth-noise.
    for p in parts.values():
        p[:, 0] += franka_x; p[:, 1] += franka_y
        p[:, 2] += table_top - p[:, 2].min()
    # align the SHIRT orientation to reality: rotate ONLY the shirt about its own centre (flips its
    # top to face away from the robot) — the box stays at its real reconstructed position
    if abs(scene_yaw) > 1e-6:
        th = np.radians(scene_yaw); ct, st = np.cos(th), np.sin(th)
        R = np.array([[ct, -st], [st, ct]]); c = cv[:, :2].mean(0)
        cv[:, :2] = (cv[:, :2] - c) @ R.T + c

    cbb = [cv[:, 0].min(), cv[:, 0].max(), cv[:, 1].min(), cv[:, 1].max()]
    cloth_out = {"verts": cv.astype(np.float32), "faces": cloth["faces"].reshape(-1, 3),
                 "color": cloth["color"], "bbox": cbb, "label": cloth["label"],
                 "vcolors": cloth.get("vcolors")}
    box_out = None
    if box is not None:
        box_out = {"verts": parts["box"].astype(np.float32), "faces": box["faces"].reshape(-1, 3),
                   "color": box["color"], "label": box["label"], "vcolors": box.get("vcolors")}
    return cloth_out, box_out


def load_cloth_spec(scene_dir):
    import json
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    objs = layout.get("objects", [])
    idx = next((i for i, o in enumerate(objs) if any(k in str(o.get("label", "")).lower() for k in CLOTH_KW)), 0)
    o = objs[idx]
    di = o.get("depth_info") or {}
    w = float(di.get("physical_width_m") or o.get("physical_size_m") or 0.35)
    h = float(di.get("physical_height_m") or w)
    img = layout.get("image_size_px", [640, 480])
    box = o.get("box_px") or [0, 0, img[0], img[1]]
    return w, h, str(o.get("label", "cloth")), box, img, idx


def load_box_object(scene_dir):
    """Find the container object (box/bag/...) -> (mesh_path, mean_color, idx) or None."""
    import json
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    objs = layout.get("objects", [])
    idx = next((i for i, o in enumerate(objs) if any(k in str(o.get("label", "")).lower() for k in BOX_KW)), None)
    if idx is None:
        return None
    o = objs[idx]
    mp = os.path.join(scene_dir, o.get("mesh_path") or f"object_{idx}/mesh.obj")
    vc_path = os.path.join(scene_dir, f"object_{idx}/vertex_colors.npy")
    if os.path.exists(vc_path):                       # true reconstructed colour (more accurate)
        col = list(np.load(vc_path).reshape(-1, 3).mean(0)[:3])
    else:
        col = o.get("display_color") or [0.59, 0.57, 0.54]
    return mp, [float(c) for c in col[:3]], idx


class Fold:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_substeps = 10
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viz_scale = 0.01
        self.pins = args.pins
        self.fold_mode = args.fold_mode
        self.folds = args.folds
        self.next_fold = 0
        self.arm = args.arm
        self.base_yaw_deg = args.base_yaw
        self.place_in_box = args.place_in_box
        self.box = [float(v) for v in args.box.split(",")]   # cx,cy,w,d,h (cm)

        # cloth footprint + texture
        self.scene_dir = args.scene_dir
        w, h, label, box, img, cloth_idx = load_cloth_spec(args.scene_dir)
        w = min(w, args.cloth_max); h = min(h, args.cloth_max)   # cap for Franka reach
        cap = args.capture_dir or args.scene_dir.replace("/outputs/", "/captures/")
        self.rgb_path = os.path.join(cap, "rgb.png")
        self.use_tex = (not args.no_texture) and (not args.faithful) and os.path.exists(self.rgb_path)
        # shirt SILHOUETTE: cut the cloth grid to the object's real segmentation mask
        self.sil_mask = None
        mask_path = os.path.join(args.scene_dir + "_sam3d_raw", "masks", f"{cloth_idx}.png")
        if (not args.no_silhouette) and os.path.exists(mask_path):
            from PIL import Image
            self.sil_mask = np.array(Image.open(mask_path).convert("L"))
        S = 100.0
        cell = 2.0                                   # cm
        self.dim_x = max(2, round(w * S / cell)); self.dim_y = max(2, round(h * S / cell))
        cloth_w = self.dim_x * cell; cloth_h = self.dim_y * cell
        # cloth centred on the table in front of the robot
        self.cloth_cx, self.cloth_cy, self.cloth_top = 0.0, -50.0, 21.0
        self.cell = cell
        xh, yh = cloth_w / 2.0, cloth_h / 2.0
        self.init_bbox = [self.cloth_cx - xh, self.cloth_cx + xh, self.cloth_cy - yh, self.cloth_cy + yh]
        # FAITHFUL: import the real SAM3D meshes as-is (overrides grid/silhouette + bbox)
        self.faithful = args.faithful
        self.fcloth = self.fbox = None
        if self.faithful:
            self.fcloth, self.fbox = faithful_objects(args.scene_dir, cap,
                                                      -50.0, -50.0, self.cloth_top, args.scene_yaw)
            self.init_bbox = list(self.fcloth["bbox"])
            print(f"[FOLD] FAITHFUL import: cloth '{self.fcloth['label']}' "
                  f"{len(self.fcloth['verts'])} verts, box "
                  f"{'yes' if self.fbox is not None else 'none'}", flush=True)
        print(f"[FOLD] '{label}' {w*100:.0f}x{h*100:.0f}cm -> grid {self.dim_x}x{self.dim_y}, "
              f"mode={self.fold_mode} folds={self.folds} faithful={self.faithful}", flush=True)

        self.scene = ModelBuilder(gravity=-981.0)

        # robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count

        # table — sized to actually cover the real object layout (else objects placed at their real
        # positions hang off the small default table). Extend toward the robot base too.
        self.table_hz = 10.0
        if self.faithful and self.fcloth is not None:
            allv = self.fcloth["verts"][:, :2]
            if self.fbox is not None:
                allv = np.vstack([allv, self.fbox["verts"][:, :2]])
            # COMPACT table under the objects only (do NOT extend back to the robot base)
            xmin, xmax = allv[:, 0].min(), allv[:, 0].max()
            ymin, ymax = allv[:, 1].min(), allv[:, 1].max()
            M = 12.0                                    # margin
            self.table_pos_cm = wp.vec3((xmin + xmax) / 2, (ymin + ymax) / 2, 10.0)
            self.table_hx = (xmax - xmin) / 2 + M; self.table_hy = (ymax - ymin) / 2 + M
        else:
            self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0); self.table_hx = self.table_hy = 40.0
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(-1, wp.transform(self.table_pos_cm, wp.quat_identity()),
                                 hx=self.table_hx, hy=self.table_hy, hz=self.table_hz)
        # pedestal / mobile base under the robot (floor -> robot base) so it isn't floating/buried
        rbx, rby, rbz = self.robot_base
        self.ped_pos = wp.vec3(rbx, rby, rbz / 2.0); self.ped_h = (14.0, 14.0, rbz / 2.0)
        self.ped_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(-1, wp.transform(self.ped_pos, wp.quat_identity()),
                                 hx=self.ped_h[0], hy=self.ped_h[1], hz=self.ped_h[2])

        # open-top container (4 walls on the table top) at the captured box's real dimensions
        self.box_shape_idx = []
        self.box_walls = []
        if self.place_in_box:
            bcx, bcy, bw, bd, bh = self.box
            t = 1.0; ztop = self.table_pos_cm[2] + self.table_hz   # table top z (=20)
            self.box_walls = [
                (bcx + bw / 2, bcy, ztop + bh / 2, t / 2, bd / 2 + t, bh / 2),   # +x wall
                (bcx - bw / 2, bcy, ztop + bh / 2, t / 2, bd / 2 + t, bh / 2),   # -x wall
                (bcx, bcy + bd / 2, ztop + bh / 2, bw / 2 + t, t / 2, bh / 2),   # +y wall
                (bcx, bcy - bd / 2, ztop + bh / 2, bw / 2 + t, t / 2, bh / 2),   # -y wall
            ]
            for (cx, cy, cz, hx, hy, hz) in self.box_walls:
                self.box_shape_idx.append(self.scene.shape_count)
                self.scene.add_shape_box(-1, wp.transform((cx, cy, cz), wp.quat_identity()),
                                         hx=hx, hy=hy, hz=hz)

        # cloth (centred), resting just above the table top
        self.sil_uvs = None
        if self.faithful:
            # the REAL SAM3D shirt mesh, imported as-is, IS the deformable
            fv = self.fcloth["verts"]; ff = self.fcloth["faces"]
            self.scene.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*v) for v in fv], indices=ff.reshape(-1).tolist(), density=0.02,
                tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6, edge_ke=5.0, edge_kd=1.0e-2,
                particle_radius=0.5)
            print(f"[FOLD] cloth = real SAM3D mesh: {len(fv)} verts, {len(ff)} tris", flush=True)
        elif self.sil_mask is not None:
            # SILHOUETTE: cut the grid to the real shirt shape via the segmentation mask
            verts, tris, uvs = masked_grid(self.dim_x, self.dim_y, cell, self.sil_mask, box, img)
            self.sil_uvs = uvs
            self.scene.add_cloth_mesh(
                pos=wp.vec3(self.cloth_cx, self.cloth_cy, self.cloth_top),
                rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*v) for v in verts], indices=tris.tolist(), density=0.02,
                tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6, edge_ke=5.0, edge_kd=1.0e-2,
                particle_radius=0.8)
            print(f"[FOLD] cloth SILHOUETTE: {len(verts)} verts, {len(tris)//3} tris (from {label} mask)", flush=True)
        else:
            self.scene.add_cloth_grid(
                pos=wp.vec3(self.cloth_cx - 0.5 * cloth_w, self.cloth_cy - 0.5 * cloth_h, self.cloth_top),
                rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.dim_x, dim_y=self.dim_y, cell_x=cell, cell_y=cell, mass=0.1,
                tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6, edge_ke=5.0, edge_kd=1.0e-2,
                particle_radius=0.8)
        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # hide table from auto shape rendering (GL bakes prim dims, ignores scale) -> draw manually
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        flags[self.ped_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        for si in self.box_shape_idx:                 # walls drawn manually at m-scale too
            flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # contact material
        self.model.soft_contact_ke = 1e4; self.model.soft_contact_kd = 1e-2; self.model.soft_contact_mu = 0.25
        ke = self.model.shape_material_ke.numpy(); kd = self.model.shape_material_kd.numpy(); mu = self.model.shape_material_mu.numpy()
        ke[...] = 5e4; kd[...] = 1e-3; mu[...] = 1.5
        self.model.shape_material_ke = wp.array(ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.8)
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.model.edge_rest_angle.zero_()
        self.cloth_solver = SolverVBD(
            self.model, iterations=8, integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5, particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16, particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1)

        # ---- grasp / pin state ----
        self.orig_inv_mass = self.model.particle_inv_mass.numpy().copy()
        self.grasp_active = False
        self.prev_grip = OPEN
        self.pin_idx = None; self.pin_off = None
        self.tip_buf = wp.zeros(1, dtype=wp.vec3)
        self.ee_off_vec = wp.vec3(0.0, 0.0, 22.0)

        # ---- texture / render set-up ----
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False          # we draw the cloth ourselves (textured)
        self.viewer.show_particles = False
        self.tri_flat = wp.array(self.model.tri_indices.numpy().reshape(-1).astype(np.int32), dtype=wp.int32)
        # FAITHFUL: bake the real SAM3D per-vertex colours onto the deforming cloth (unwelded per frame)
        self.cloth_bake = None
        if self.faithful and self.fcloth.get("vcolors") is not None:
            faces = self.model.tri_indices.numpy().reshape(-1, 3)
            vcol = self.fcloth["vcolors"]
            _, rf, uv, tex = sam3d_scene.bake_colors(np.zeros((len(vcol), 3), np.float32), faces, vcol)
            self.cloth_bake = {"faceidx": faces.reshape(-1), "rfaces": wp.array(rf, dtype=wp.int32),
                               "uvs": wp.array(uv, dtype=wp.vec2), "tex": tex}
            print(f"[FOLD] cloth colours = real SAM3D per-vertex (baked)", flush=True)
        if self.sil_uvs is not None:                 # silhouette: UVs precomputed per kept vertex
            self.uvs = wp.array(self.sil_uvs.astype(np.float32), dtype=wp.vec2)
        else:
            pq0 = self.state_0.particle_q.numpy()
            nx = (pq0[:, 0] - pq0[:, 0].min()) / max(1e-6, np.ptp(pq0[:, 0]))
            ny = (pq0[:, 1] - pq0[:, 1].min()) / max(1e-6, np.ptp(pq0[:, 1]))
            bx0, by0, bx1, by1 = box
            uu = (bx0 + nx * (bx1 - bx0)) / img[0]; vv = (by0 + (1.0 - ny) * (by1 - by0)) / img[1]
            self.uvs = wp.array(np.stack([uu, vv], 1).astype(np.float32), dtype=wp.vec2)
        self.tex_img = None
        if self.use_tex:
            from newton._src.utils.texture import load_texture
            self.tex_img = load_texture(self.rgb_path)
        # table viz (meters)
        self.table_viz_xform = wp.array([wp.transform(
            (float(self.table_pos_cm[0]) * self.viz_scale, float(self.table_pos_cm[1]) * self.viz_scale,
             float(self.table_pos_cm[2]) * self.viz_scale), wp.quat_identity())], dtype=wp.transform)
        self.table_viz_scale = (self.table_hx * self.viz_scale, self.table_hy * self.viz_scale, self.table_hz * self.viz_scale)
        self.table_viz_color = wp.array([wp.vec3(0.55, 0.55, 0.58)], dtype=wp.vec3)
        # pedestal viz (meters)
        self.ped_viz_xform = wp.array([wp.transform(
            (float(self.ped_pos[0]) * self.viz_scale, float(self.ped_pos[1]) * self.viz_scale,
             float(self.ped_pos[2]) * self.viz_scale), wp.quat_identity())], dtype=wp.transform)
        self.ped_viz_scale = tuple(h * self.viz_scale for h in self.ped_h)
        self.ped_viz_color = wp.array([wp.vec3(0.3, 0.3, 0.33)], dtype=wp.vec3)
        # container viz: render the REAL reconstructed box mesh (true shape + colour), top opened.
        # Physics still uses the invisible open walls above; this is appearance only.
        self.box_mesh_viz = None
        self.box_walls_viz = []
        self.cloth_solid_color = self.fcloth["color"] if self.faithful else (0.3, 0.28, 0.32)
        if self.faithful and self.fbox is not None:
            # the real SAM3D box mesh, as-is (cm -> m), with its real per-vertex colours baked
            if self.fbox["vcolors"] is not None:
                rv, rf, uv, tex = sam3d_scene.bake_colors(
                    self.fbox["verts"], self.fbox["faces"], self.fbox["vcolors"])
                self.box_mesh_viz = ("baked", wp.array(rv * self.viz_scale, dtype=wp.vec3),
                                     wp.array(rf, dtype=wp.int32), wp.array(uv, dtype=wp.vec2), tex)
            else:
                bv = self.fbox["verts"].astype(np.float32) * self.viz_scale
                self.box_mesh_viz = ("solid", wp.array(bv, dtype=wp.vec3),
                                     wp.array(self.fbox["faces"].reshape(-1).astype(np.int32), dtype=wp.int32),
                                     tuple(self.fbox["color"]))
            print(f"[FOLD] container = real SAM3D box mesh (baked colours)", flush=True)
        boxobj = load_box_object(self.scene_dir) if (self.place_in_box and not self.faithful) else None
        if boxobj is not None:
            import trimesh
            mp, bcol, _ = boxobj
            bm = trimesh.load(mp, force="mesh")
            v = np.asarray(bm.vertices, np.float32); f = np.asarray(bm.faces, np.int64)
            v[:, :2] -= 0.5 * (v[:, :2].min(0) + v[:, :2].max(0))   # centre footprint at origin
            v[:, 2] -= v[:, 2].min()                                # base at z=0
            ztop = v[:, 2].max()
            keep = v[f].mean(1)[:, 2] < 0.82 * ztop                 # drop the top cap -> open box
            f = f[keep]
            bcx, bcy = self.box[0], self.box[1]
            vp = v.copy()
            vp[:, 0] += bcx / 100.0; vp[:, 1] += bcy / 100.0        # place (m); base sits on table top
            vp[:, 2] += (self.table_pos_cm[2] + self.table_hz) / 100.0
            self.box_mesh_viz = (
                "solid", wp.array(vp, dtype=wp.vec3),
                wp.array(f.reshape(-1).astype(np.int32), dtype=wp.int32),
                tuple(bcol))
            print(f"[FOLD] container: real mesh {len(vp)} verts, colour {np.round(bcol,2)}", flush=True)
        else:
            for (cx, cy, cz, hx, hy, hz) in self.box_walls:        # fallback: flat walls
                self.box_walls_viz.append((
                    (hx * self.viz_scale, hy * self.viz_scale, hz * self.viz_scale),
                    wp.array([wp.transform((cx * self.viz_scale, cy * self.viz_scale, cz * self.viz_scale),
                                           wp.quat_identity())], dtype=wp.transform),
                    wp.array([wp.vec3(0.59, 0.57, 0.54)], dtype=wp.vec3)))

        # meter-scale shape data for the robot (the example's two-path swap)
        self.sim_shape_transform = self.model.shape_transform; self.sim_shape_scale = self.model.shape_scale
        xf = self.model.shape_transform.numpy().copy(); xf[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xf, dtype=wp.transform, device=self.model.device)
        sc = self.model.shape_scale.numpy().copy(); sc *= self.viz_scale
        self.viz_shape_scale = wp.array(sc, dtype=wp.vec3, device=self.model.device)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy(); xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                scc = shapes.scales.numpy(); scc *= self.viz_scale
                shapes.scales = wp.array(scc, dtype=wp.vec3, device=shapes.device)

        self.viz_state = self.model.state()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    # ----------------------------- robot setup + keyframes -----------------------------
    def create_articulation(self, builder):
        # robot base: OPPOSITE side of the objects (user: robot should be on the far side, facing across
        # the table), ~5cm ABOVE the table top, facing the objects
        rq = wp.quat_identity()
        if self.faithful and self.fcloth is not None:
            ov = self.fcloth["verts"][:, :2]
            if self.fbox is not None:
                ov = np.vstack([ov, self.fbox["verts"][:, :2]])
            oc = ov.mean(0)
            rxy = 2.0 * oc - np.array([-50.0, -50.0])           # reflect default base through the objects
            face = np.arctan2(oc[1] - rxy[1], oc[0] - rxy[0]) + np.radians(self.base_yaw_deg)
            self.robot_base = (float(rxy[0]), float(rxy[1]), 25.0)
            rq = wp.quat(0.0, 0.0, float(np.sin(face / 2)), float(np.cos(face / 2)))
        else:
            self.robot_base = (-50.0, -50.0, 25.0)
        if self.arm == "ur5e":
            from ur5e_gripper import add_ur5e_gripper
            w3, dof0, n_arm, pinch = add_ur5e_gripper(builder, wp.transform(self.robot_base, rq), scale=100.0)
            self.ur5e_w3 = w3; self.n_arm = n_arm; self.pinch_local = pinch
            print(f"[FOLD] arm = UR5e + Robotiq 2F-85 ({n_arm} arm dof, {builder.joint_dof_count - dof0 - n_arm} gripper dof)", flush=True)
        else:
            asset_path = newton.utils.download_asset("franka_emika_panda")
            builder.add_urdf(str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
                             xform=wp.transform(self.robot_base, rq),
                             floating=False, scale=100, enable_self_collisions=False,
                             collapse_fixed_joints=True, force_show_colliders=False)
            builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
            self.n_arm = 7

        q = DOWN_Q
        if self.fold_mode == "corner":
            cx, cy = 19.0, -35.0      # pick corner (near-right)
            fx, fy = -19.0, -65.0     # fold target (far-left corner)
            self.robot_key_poses = np.array([
                [3.0,  cx, cy, 42.0, *q, OPEN],    # approach above corner
                [2.0,  cx, cy, 21.5, *q, OPEN],    # descend to corner
                [1.5,  cx, cy, 21.5, *q, CLOSE],   # CLOSE -> grasp (pin)
                [2.5,  cx, cy, 42.0, *q, CLOSE],   # lift
                [4.0,  fx, fy, 42.0, *q, CLOSE],   # carry diagonally over the cloth
                [2.0,  fx, fy, 25.0, *q, CLOSE],   # lower onto far corner
                [1.5,  fx, fy, 25.0, *q, OPEN],    # OPEN -> release
                [2.0,  fx, fy, 44.0, *q, OPEN],    # retreat up
                [2.5, -45.0, -50.0, 44.0, *q, OPEN],  # clear away
            ], dtype=np.float32)
            self.fold_specs = []
        else:
            # multi / half: N alternating half-folds (y, x, y, x, ...) down to a small piece.
            # Each fold grasps the cloth's CURRENT max edge along the axis and brings it onto the
            # opposite edge (fold line = mid). bbox shrinks toward the (xmin,ymin) corner; layers double.
            nf = 1 if self.fold_mode == "half" else self.folds
            bbox = list(self.init_bbox)               # [x0, x1, y0, y1]
            poses, specs = [], []
            ZHI, ZLIFT, ZAPEX = 42.0, 39.0, 41.0
            for i in range(nf):
                axis = "y" if i % 2 == 0 else "x"
                x0, x1, y0, y1 = bbox
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                zg = 21.5 + 1.3 * i                   # grasp/land rise as the stack thickens
                zland = 23.0 + 1.3 * i
                if axis == "y":
                    g = (cx, y1); t = (cx, y0); mid = (cx, (y0 + y1) / 2.0)
                    bbox[3] = (y0 + y1) / 2.0
                else:
                    g = (x1, cy); t = (x0, cy); mid = ((x0 + x1) / 2.0, cy)
                    bbox[1] = (x0 + x1) / 2.0
                poses += [
                    [1.5, g[0], g[1], ZHI,   *q, OPEN],    # approach above grasp edge
                    [1.2, g[0], g[1], zg,    *q, OPEN],    # descend
                    [1.0, g[0], g[1], zg,    *q, CLOSE],   # grasp edge (pin band)
                    [1.5, g[0], g[1], ZLIFT, *q, CLOSE],   # peel/lift
                    [1.5, mid[0], mid[1], ZAPEX, *q, CLOSE],  # arc over the fold line
                    [1.5, t[0], t[1], ZLIFT, *q, CLOSE],   # carry over the far edge
                    [1.2, t[0], t[1], zland, *q, CLOSE],   # lay down
                    [1.0, t[0], t[1], zland, *q, OPEN],    # release
                    [1.2, t[0], t[1], ZHI,   *q, OPEN],    # retreat up
                ]
                specs.append({"axis": axis})
            if self.place_in_box:
                # pick the folded bundle (centre of the final bbox) and place it in the container
                bx, by = (bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0
                bcx, bcy, bw, bd, bh = self.box
                box_top = 20.0 + bh                   # table top (z=20) + wall height
                zgb = 22.0 + 1.3 * nf                 # grasp height ~ top of the folded stack
                poses += [
                    [2.0, bx, by, ZHI, *q, OPEN],          # approach above bundle
                    [1.5, bx, by, zgb, *q, OPEN],          # descend onto bundle
                    [1.2, bx, by, zgb, *q, CLOSE],         # grasp bundle
                    [2.0, bx, by, ZHI, *q, CLOSE],         # lift
                    [2.5, bcx, bcy, ZHI, *q, CLOSE],       # carry over the box
                    [2.0, bcx, bcy, box_top + 4.0, *q, CLOSE],  # lower to the rim
                    [1.2, bcx, bcy, box_top + 4.0, *q, OPEN],   # release -> drops in
                    [2.0, bcx, bcy, ZHI, *q, OPEN],        # retreat
                ]
                specs.append({"axis": "bundle"})
            poses += [[1.5, 0.0, -38.0, ZHI, *q, OPEN]]   # clear away (reachable retreat)
            self.robot_key_poses = np.array(poses, dtype=np.float32)
            self.fold_specs = specs
        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.target = self.targets[0]
        if self.arm == "ur5e":
            self.endeffector_id = self.ur5e_w3               # wrist_3
            self.endeffector_offset = wp.transform(list(self.pinch_local), wp.quat_identity())
        else:
            self.endeffector_id = builder.body_count - 3
            # control the FINGERTIP (not a virtual point below the hand) so the fingers contact the cloth
            self.endeffector_offset = wp.transform([0.0, 0.0, 11.0], wp.quat_identity())

    def set_up_control(self):
        self.control = self.model.control()
        out_dim = 6
        self.Jacobian_one_hots = [wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
                                  for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                             body_com: wp.array(dtype=wp.vec3), body_out: wp.array(dtype=float)):
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            X_wb = body_q[ee_id]
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd); v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]; body_out[1] = v_tip[1]; body_out[2] = v_tip[2]
            body_out[3] = omega[0]; body_out[4] = omega[1]; body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * self.model.joint_dof_count, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def compute_body_jacobian(self, joint_q, joint_qd):
        joint_q.requires_grad = True; joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(self.compute_body_out_kernel, 1,
                      inputs=[self.temp_state_for_jacobian.body_q, self.temp_state_for_jacobian.body_qd,
                              self.model.body_com],
                      outputs=[self.body_out])
        for i in range(6):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim:(i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def load_exec_plan(self, path):
        """Load a cuRobo joint trajectory and flatten it to a time-indexed arm-joint path."""
        import json
        traj = json.load(open(path))
        ts, qs, gs = [], [], []
        t = 0.0
        for s in traj["segments"]:
            q = s["q"]; dur = float(s["dur"]); grip = 0.1 if s["gripper"] == "closed" else 0.8
            n = len(q)
            for j, qj in enumerate(q):
                ts.append(t + dur * (j / (n - 1) if n > 1 else 0.0)); qs.append(qj); gs.append(grip)
            t += dur
        self.exec_t = np.array(ts); self.exec_q = np.array(qs, np.float64); self.exec_g = np.array(gs)
        self.exec_total = t; self.exec_mode = True; self.cur_grip = 0.8
        print(f"[FOLD] EXEC cuRobo trajectory: {len(ts)} waypoints over {t:.1f}s", flush=True)

    def generate_control_joint_qd(self, state_in):
        if getattr(self, "exec_mode", False):          # execute the cuRobo joint trajectory
            i = int(np.clip(np.searchsorted(self.exec_t, self.sim_time), 0, len(self.exec_t) - 1))
            qt = self.exec_q[i]; self.cur_grip = float(self.exec_g[i])
            q = state_in.joint_q.numpy()
            dq = np.zeros(self.model.joint_dof_count)
            dq[:7] = qt - q[:7]                         # drive arm joints toward the planned config
            dq[-2] = self.cur_grip * 4.0 - q[-2]; dq[-1] = self.cur_grip * 4.0 - q[-1]
            self.target_joint_qd.assign(dq)
            return
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_(); return
        interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        self.target = self.targets[interval]
        wp.launch(compute_ee_delta, dim=1,
                  inputs=[state_in.body_q, self.endeffector_offset, self.endeffector_id,
                          self.bodies_per_world, wp.transform(*self.target[:7])],
                  outputs=[self.ee_delta])
        self.compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        q = state_in.joint_q.numpy()
        if self.arm == "ur5e":
            # restrict IK to the 6 arm joints; HOLD the gripper joints at their home (frozen open)
            na = self.n_arm
            # POSITION-ONLY IK: bring the pinch point to the target (orientation free — DOWN_Q is the
            # Franka's convention, not the UR5e's; the pin grasps the pinch point regardless)
            dq_arm = np.linalg.pinv(J[:3, :na]) @ delta_target[:3] * 3.0   # gain for faster convergence
            delta_q = np.zeros(self.model.joint_dof_count)   # gripper dofs -> 0 velocity (held open)
            delta_q[:na] = dq_arm
            self.target_joint_qd.assign(delta_q)
            return
        J_inv = np.linalg.pinv(J)
        N = np.eye(J.shape[1], dtype=np.float32) - J_inv @ J
        q_des = q.copy(); q_des[1:] = self.initial_pose[1:]
        delta_q = J_inv @ delta_target + N @ (1.0 * (q_des - q))
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]
        self.target_joint_qd.assign(delta_q)

    def export_plan(self, path):
        """Write the fold's geometric keyframe fingertip poses + obstacles in the FRANKA-BASE frame
        (metres) for cuRobo to plan collision-free joint trajectories through."""
        import json
        base = np.array([-50.0, -50.0, 0.0])          # Franka base in world (cm)
        kfs = []
        for row in self.robot_key_poses:
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            kfs.append({"dur": float(row[0]),
                        "fingertip_m": [(x - base[0]) / 100.0, (y - base[1]) / 100.0, (z - base[2]) / 100.0],
                        "gripper": "open" if float(row[8]) > 0.4 else "closed"})
        obstacles = [{"name": "table", "center_m": [0.5, 0.0, 0.1], "dims_m": [0.8, 0.8, 0.2]}]
        if self.fbox is not None:
            bv = self.fbox["verts"].astype(np.float64)
            c = ((bv.max(0) + bv.min(0)) / 2.0 - base) / 100.0
            d = (bv.max(0) - bv.min(0)) / 100.0
            obstacles.append({"name": "box", "center_m": c.tolist(), "dims_m": d.tolist()})
        json.dump({"keyframes": kfs, "obstacles": obstacles,
                   "down_quat_wxyz": [0.0, 1.0, 0.0, 0.0], "ee_offset_z": 0.105,
                   "home_js": [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]},
                  open(path, "w"), indent=2)
        print(f"[FOLD] exported {len(kfs)} keyframes + {len(obstacles)} obstacles -> {path}", flush=True)

    # ----------------------------- grasp (pin) -----------------------------
    def update_grasp(self):
        grip = float(self.cur_grip) if getattr(self, "exec_mode", False) else float(self.target[-1])
        if self.prev_grip > 0.4 and grip <= 0.4:        # close -> grasp
            self._grasp()
        elif self.prev_grip <= 0.4 and grip > 0.4:      # open -> release
            self._release()
        self.prev_grip = grip

    def _grasp(self):
        bq = self.state_0.body_q.numpy()
        hand = bq[self.endeffector_id, :3]
        q6 = bq[self.endeffector_id, 3:7]                 # (x,y,z,w)

        def qrot(q, v):                                   # rotate v by quaternion q (x,y,z,w)
            x, y, z, w = q; t = 2.0 * np.cross([x, y, z], v)
            return np.asarray(v) + w * t + np.cross([x, y, z], t)
        if self.arm == "ur5e":
            # gripper pinch point = fixed offset in the wrist_3 frame (2F-85 has no simple finger bodies)
            self.ee_off_vec = wp.vec3(*self.pinch_local)
            pinch = hand + qrot(q6, np.array(list(self.pinch_local)))
        else:
            # pinch = midpoint of the two fingertip bodies, extended to the pad tips
            fmid = 0.5 * (bq[-2, :3] + bq[-1, :3])        # bodies -2,-1 are the two fingers
            adir = fmid - hand; n = np.linalg.norm(adir)
            adir = adir / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
            pinch = fmid + 3.0 * adir                     # ~pad tip, just past the finger bodies
            qinv = np.array([-q6[0], -q6[1], -q6[2], q6[3]])
            self.ee_off_vec = wp.vec3(*qrot(qinv, pinch - hand).astype(float))  # pinch in hand-local frame

        pq = self.state_0.particle_q.numpy()
        if self.fold_mode == "corner":
            idx = np.argsort(np.linalg.norm(pq - pinch, axis=1))[:self.pins]
            what = f"corner ({self.pins} particles)"
        else:
            axis = self.fold_specs[min(self.next_fold, len(self.fold_specs) - 1)]["axis"]
            self.next_fold += 1
            if axis == "bundle":
                # grab the whole folded bundle: particles within a radius of the fingertip (xy)
                d2 = np.linalg.norm(pq[:, :2] - pinch[:2], axis=1)
                idx = np.where(d2 < 9.0)[0]
                if len(idx) < 4:
                    idx = np.argsort(d2)[:24]
                what = f"bundle ({len(idx)} particles)"
            else:
                col = 1 if axis == "y" else 0
                idx = np.where(pq[:, col] >= pq[:, col].max() - 3.0)[0]
                what = f"fold {self.next_fold} {axis}-edge ({len(idx)} particles)"
        self.pin_idx = wp.array(idx.astype(np.int32), dtype=wp.int32)
        self.pin_off = wp.array((pq[idx] - pinch).astype(np.float32), dtype=wp.vec3)  # hold at fingertips
        im = self.orig_inv_mass.copy(); im[idx] = 0.0
        self.model.particle_inv_mass.assign(im)
        self.grasp_active = True
        ft_gap = float(pinch[2] - pq[idx, 2].mean())
        print(f"[FOLD] GRASP {what} at t={self.sim_time:.1f}s  fingertip z={pinch[2]:.1f} "
              f"cloth z={pq[idx,2].mean():.1f} (gap {ft_gap:+.1f}cm)", flush=True)

    def _release(self):
        self.model.particle_inv_mass.assign(self.orig_inv_mass)
        self.grasp_active = False
        print(f"[FOLD] RELEASE at t={self.sim_time:.1f}s", flush=True)

    # ----------------------------- sim -----------------------------
    def step(self):
        self.generate_control_joint_qd(self.state_0)
        self.update_grasp()
        self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # robot
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = pc
            self.model.gravity.assign(self.gravity_earth)
            # pin the grasped corner to the NEW ee pose (state_1.body_q), into the cloth input (state_0)
            if self.grasp_active and self.pin_idx is not None:
                wp.launch(pin_to_ee, dim=len(self.pin_idx),
                          inputs=[self.state_1.body_q, self.endeffector_id, self.ee_off_vec,
                                  self.pin_idx, self.pin_off],
                          outputs=[self.state_0.particle_q, self.state_0.particle_qd])
            # cloth
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return
        wp.launch(scale_positions, dim=self.model.particle_count,
                  inputs=[self.state_0.particle_q, self.viz_scale], outputs=[self.viz_state.particle_q])
        if self.model.body_count > 0:
            wp.launch(scale_body_transforms, dim=self.model.body_count,
                      inputs=[self.state_0.body_q, self.viz_scale], outputs=[self.viz_state.body_q])
        self.model.shape_transform = self.viz_shape_transform; self.model.shape_scale = self.viz_shape_scale
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)       # robot only (triangles hidden)
        self.viewer.log_shapes("/table", newton.GeoType.BOX, self.table_viz_scale,
                               self.table_viz_xform, self.table_viz_color)
        self.viewer.log_shapes("/pedestal", newton.GeoType.BOX, self.ped_viz_scale,
                               self.ped_viz_xform, self.ped_viz_color)
        for i, (sc, xf, col) in enumerate(self.box_walls_viz):
            self.viewer.log_shapes(f"/box_wall{i}", newton.GeoType.BOX, sc, xf, col)
        if self.box_mesh_viz is not None:
            if self.box_mesh_viz[0] == "baked":
                _, bpts, bidx, buv, btex = self.box_mesh_viz
                self.viewer.log_mesh("container", bpts, bidx, uvs=buv, texture=btex,
                                     color=(1.0, 1.0, 1.0), backface_culling=False)
                ob = getattr(self.viewer, "objects", {}).get("container")
                if ob is not None:
                    r, m, c, _ = ob.material; ob.material = (r, m, c, 1.0)
            else:
                _, bpts, bidx, bcol = self.box_mesh_viz
                self.viewer.log_mesh("container", bpts, bidx, color=bcol, backface_culling=False)
        # textured cloth (meter scale)
        if self.use_tex:
            self.viewer.log_mesh("cloth", self.viz_state.particle_q, self.tri_flat, uvs=self.uvs,
                                 texture=self.tex_img, color=(1.0, 1.0, 1.0), backface_culling=False)
            o = getattr(self.viewer, "objects", {}).get("cloth")
            if o is not None:
                r, m, c, _ = o.material; o.material = (r, m, c, 1.0)
        elif self.cloth_bake is not None:
            # real SAM3D per-vertex colours on the deforming shirt: unweld current particle positions
            pq = self.viz_state.particle_q.numpy()
            rv = pq[self.cloth_bake["faceidx"]]        # (n_tris*3, 3) following the fold
            self.viewer.log_mesh("cloth", wp.array(rv, dtype=wp.vec3), self.cloth_bake["rfaces"],
                                 uvs=self.cloth_bake["uvs"], texture=self.cloth_bake["tex"],
                                 color=(1.0, 1.0, 1.0), backface_culling=False)
            o = getattr(self.viewer, "objects", {}).get("cloth")
            if o is not None:
                r, m, c, _ = o.material; o.material = (r, m, c, 1.0)
        else:
            self.viewer.log_mesh("cloth", self.viz_state.particle_q, self.tri_flat,
                                 color=self.cloth_solid_color, backface_culling=False)
        self.viewer.end_frame()
        self.model.shape_transform = self.sim_shape_transform; self.model.shape_scale = self.sim_shape_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", default=None)
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--fold_mode", default="multi", choices=["multi", "half", "corner"])
    ap.add_argument("--folds", type=int, default=3, help="number of alternating half-folds (multi mode)")
    ap.add_argument("--pins", type=int, default=2, help="corner particles pinned (corner mode only)")
    ap.add_argument("--place_in_box", action="store_true", help="after folding, pick the bundle and place it in the box")
    ap.add_argument("--box", default="-33,-40,27,29,15", help="container cx,cy,width,depth,wall_height (cm)")
    ap.add_argument("--cloth_max", type=float, default=0.38, help="cap cloth dimension (m) for reachability")
    ap.add_argument("--no_silhouette", action="store_true", help="use a plain rectangle, not the real shirt shape")
    ap.add_argument("--faithful", action="store_true", help="import the real SAM3D meshes as-is (cloth+box)")
    ap.add_argument("--arm", default="franka", choices=["franka", "ur5e"], help="robot arm (ur5e = real UR5e+Robotiq)")
    ap.add_argument("--base_yaw", type=float, default=0.0, help="extra base-facing yaw offset (deg) for the arm")
    ap.add_argument("--scene_yaw", type=float, default=0.0, help="rotate layout about the shirt (deg); 0 = as reconstructed (no flip)")
    ap.add_argument("--export_plan", default=None, help="write keyframe poses+obstacles (base frame) for cuRobo, then exit")
    ap.add_argument("--exec_plan", default=None, help="execute a cuRobo joint trajectory (from curobo_fold_plan.py)")
    ap.add_argument("--no_texture", action="store_true")
    ap.add_argument("--hold", type=float, default=1800.0)
    ap.add_argument("--screenshot", default=None, help="PNG glob: save frames at fold milestones")
    ap.add_argument("--cam", default="0.25,-0.92,0.52,-36,120", help="x,y,z,pitch,yaw")
    args = ap.parse_args()

    wp.init()
    if args.viewer == "gl":
        viewer = ViewerGL()
    else:
        viewer = ViewerNull(num_frames=10 ** 9)

    ex = Fold(viewer, args)
    if args.export_plan:                             # stage 1: export poses for cuRobo, then exit
        ex.export_plan(args.export_plan)
        return
    if args.exec_plan:                               # stage 3: execute the cuRobo trajectory
        ex.load_exec_plan(args.exec_plan)
    if args.viewer == "gl":                          # AFTER set_model() (which auto-frames the scene)
        cx, cy, cz, cp, cyaw = (float(v) for v in args.cam.split(","))
        viewer.set_camera(wp.vec3(cx, cy, cz), cp, cyaw)
    sched_end = ex.exec_total if getattr(ex, "exec_mode", False) else ex.robot_key_poses_time[-1]
    n_frames = int(sched_end * ex.fps) + 30
    shots = {}
    if args.screenshot:
        base = args.screenshot.replace(".png", "")
        # a shot just after every release (grip OPEN following CLOSE), plus the final result
        grips = ex.robot_key_poses[:, -1]
        rels = [i for i in range(1, len(grips)) if grips[i] > 0.4 >= grips[i - 1]]
        for k, i in enumerate(rels):
            shots[int(ex.robot_key_poses_time[i] * ex.fps)] = f"{base}_step{k+1}.png"
        shots[n_frames - 1] = f"{base}_done.png"

    print(f"[FOLD] running {n_frames} frames ({ex.robot_key_poses_time[-1]:.0f}s schedule)...", flush=True)
    t0 = time.time()
    for f in range(n_frames):
        ex.step(); ex.render()
        if f in shots and args.viewer == "gl":
            from PIL import Image
            Image.fromarray(viewer.get_frame().numpy()).save(shots[f])
            print(f"[FOLD] shot -> {shots[f]}", flush=True)
        if args.viewer == "gl" and not viewer.is_running():
            break
    zmin = float(ex.state_0.particle_q.numpy()[:, 2].min())
    print(f"[FOLD] done in {time.time()-t0:.0f}s; lowest particle z={zmin:.2f}cm "
          f"({'NaN!' if not np.isfinite(zmin) else 'ok'})", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[FOLD] holding GUI {args.hold:.0f}s", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            ex.render()


if __name__ == "__main__":
    main()
