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
try:
    from newton.solvers import SolverStyle3D
except Exception:
    SolverStyle3D = None
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


def decimate_cloth_obj(o, target_faces):
    """Reduce the REAL SAM3D shirt mesh to ~target_faces (same shape, lower resolution) so the
    coupled VBD self-collision sim runs several times faster. Geometry is quadric-decimated (shape
    preserved) and the per-vertex colours are transferred to the new vertices by nearest neighbour
    (so the baked logo/print still follows the real reconstruction). NOT a fabricated proxy — it is
    the same mesh at lower res."""
    import trimesh
    from scipy.spatial import cKDTree
    v = np.asarray(o["verts"], np.float64)
    f = np.asarray(o["faces"]).reshape(-1, 3)
    if len(f) <= target_faces:
        return o
    m2 = trimesh.Trimesh(vertices=v, faces=f, process=False).simplify_quadric_decimation(
        face_count=int(target_faces))
    nv = np.asarray(m2.vertices, np.float64)
    nf = np.asarray(m2.faces, np.int64)
    o = dict(o)
    o["verts"] = nv
    o["faces"] = nf.reshape(-1)
    vc = o.get("vcolors")
    if vc is not None:
        vc = np.asarray(vc).reshape(-1, 3)
        _, idx = cKDTree(v).query(nv)                  # nearest original vertex -> its colour
        o["vcolors"] = vc[idx]
    print(f"[FOLD] decimated cloth: {len(f)} -> {len(nf)} faces, {len(v)} -> {len(nv)} verts", flush=True)
    return o


def _dls_pinv(J, lam=0.05):
    """Damped least-squares pseudo-inverse — robust near kinematic singularities. np.linalg.pinv's SVD can
    fail to converge on an ill-conditioned Jacobian and CRASH the IK ('SVD did not converge'); DLS adds a
    small regularisation so the inverse always exists."""
    J = np.asarray(J, dtype=np.float64)
    m, n = J.shape
    l2 = lam * lam
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + l2 * np.eye(m))
    return np.linalg.inv(J.T @ J + l2 * np.eye(n)) @ J.T


def _straighten_box(bv):
    """De-lean a single-view box reconstruction so it sits UPRIGHT as the real box does on the table. SAM3D
    reconstructs this open box leaning ~40deg (opening centre offset ~14cm from the base), which makes it a
    tilted bucket that can't hold anything. This is the REAL mesh (real shape/colour/texture) with only the
    reconstruction LEAN corrected -- same spirit as snapping z to the table for depth noise; NOT a primitive.
    Uses a vertical SHEAR: slide each level's XY back by the lean * (height fraction) so the top aligns over
    the base -> walls become vertical, the base stays on the table, height is unchanged. Returns verts (cm)."""
    bv = np.asarray(bv, dtype=np.float64).copy()
    zt, zb = bv[:, 2].max(), bv[:, 2].min()
    H = zt - zb
    if H < 1e-3:
        return bv
    rim = bv[bv[:, 2] > zt - 0.25 * H]
    base = bv[bv[:, 2] < zb + 0.25 * H]
    lean = np.array([rim[:, 0].mean() - base[:, 0].mean(), rim[:, 1].mean() - base[:, 1].mean()])
    f = (bv[:, 2] - zb) / H                                 # 0 at base, 1 at rim
    bv[:, 0] -= lean[0] * f
    bv[:, 1] -= lean[1] * f
    return bv


def _open_box_mesh(cx, cy, z0, w, d, h):
    """A clean OPEN-TOP box (floor + 4 walls, no top) centred at (cx,cy), sitting on z0, size wxdxh (cm).
    Faces are emitted double-sided so it renders solid from inside and outside. Used to REPLACE a malformed
    single-view box reconstruction (SAM3D leans this box ~40-46deg every capture) with a proper container
    whose collision matches what's drawn, so a folded shirt actually sits inside. Real position + colour are
    kept; only the box GEOMETRY becomes this primitive."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - d / 2, cy + d / 2
    z1 = z0 + h
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    quads = [(0, 1, 2, 3),               # floor
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]  # 4 walls (no top)
    F = []
    for a, b, c, e in quads:
        F += [[a, b, c], [a, c, e], [a, c, b], [a, e, c]]   # both windings -> double-sided
    return V, np.array(F, dtype=np.int32)


def faithful_objects(scene_dir, capture_dir, franka_x, franka_y, table_top, scene_yaw=0.0,
                     flip_box=False, cloth_decimate=0, box_scale=1.0, mask_cloth=None, near=1.0,
                     mask_step=7, clean_box=False, two_layer=False, gen_cloth=None, layer_gap=1.2,
                     real_volume=False):
    """Import the SAM3D models: real per-object mesh calibrated to depth-measured size, translated so
    the cloth sits in the robot's fold zone. If mask_cloth (a SAM2 mask path) is given, the SHIRT is
    rebuilt from the real MASK (silhouette incl. sleeves) + depth + RGB via cloth_mesh.cloth_from_mask
    (SAM3D's 3D mesh loses the sleeves). Returns cloth dict {verts,faces,uvs,texture,...} and box dict."""
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

    mask_uv = mask_tex = None
    gen_vcol = None
    if gen_cloth:
        # generated garment (Meshy/Rodin via perception/gen3d_cloth.py): the GENERATED mesh gives
        # TOPOLOGY (sleeves, front/back layers — what SAM3D loses); the CAPTURE stays ground truth
        # for scale (depth-measured size) and position (the SAM3D cloth's reconstructed centroid).
        from newton_gen3d_cloth import load_mesh
        gv, gf, gen_vcol = load_mesh(gen_cloth)
        tgt = cloth_decimate if cloth_decimate and cloth_decimate > 0 else 8000
        if len(gf) > tgt:
            import trimesh
            from scipy.spatial import cKDTree
            dm = trimesh.Trimesh(gv, gf, process=False).simplify_quadric_decimation(face_count=tgt)
            dm.update_faces(dm.nondegenerate_faces())   # sliver tris NaN VBD
            dm.merge_vertices()                          # POSITION-duplicate verts (marching-cubes
            dm.update_faces(dm.nondegenerate_faces())    # meshes) leave 0-length edges the index
            dm.update_faces(dm.area_faces > 1e-8)        # filter misses -> weld then re-filter
            dm.remove_unreferenced_vertices()
            if gen_vcol is not None:
                gen_vcol = gen_vcol[cKDTree(gv).query(np.asarray(dm.vertices))[1]]
            gv, gf = np.asarray(dm.vertices, float), np.asarray(dm.faces, int)
            print(f"[FOLD] gen cloth decimated -> {len(gv)} verts, {len(gf)} faces", flush=True)
        gv = gv - gv.mean(0)
        order = np.argsort((gv.max(0) - gv.min(0)))     # lay FLAT: thinnest model axis -> +z
        gv = gv[:, [int(order[2]), int(order[1]), int(order[0])]]
        real = calibrate(cloth)                          # SAM3D cloth -> real cm pose for anchoring
        meas = max([m for m in cloth["measured_m"] if m > 0] or [0.4]) * 100.0
        fp = (gv[:, :2].max(0) - gv[:, :2].min(0)).max()
        gv *= meas / max(fp, 1e-6)
        # gen services normalize per-axis (true thickness LOST) -> clamp to a real flat-lay
        # drape thickness; disclosed calibration, same spirit as the measured-size rescale
        zext = gv[:, 2].max() - gv[:, 2].min()
        z_max = 8.0                                      # cm
        if zext > z_max:
            gv[:, 2] *= z_max / zext
            print(f"[FOLD] gen cloth thickness {zext:.0f}cm -> clamped to {z_max:.0f}cm", flush=True)
        # REAL colour: project the capture photo onto the flat-laid garment (top-down bbox
        # mapping into the shirt's box_px, mask-gated) — generated topology, real appearance.
        # Overrides the generator's texture/grey; same faithful-appearance rule as mask-cloth.
        try:
            import json as _json
            from PIL import Image as _Image
            from cloth_mesh import find_cloth_mask
            lay = _json.load(open(os.path.join(scene_dir, "scene_layout.json")))
            lobj = next(o for o in lay["objects"]
                        if any(k in str(o.get("label", "")).lower() for k in CLOTH_KW))
            bx0, by0, bx1, by1 = lobj["box_px"]
            img = np.asarray(_Image.open(os.path.join(capture_dir, "rgb.png")).convert("RGB"), float) / 255.0
            mpath = find_cloth_mask(scene_dir, CLOTH_KW)
            msk = np.asarray(_Image.open(mpath).convert("L")) > 127 if mpath else None
            nx = (gv[:, 0] - gv[:, 0].min()) / max(1e-6, np.ptp(gv[:, 0]))
            ny = (gv[:, 1] - gv[:, 1].min()) / max(1e-6, np.ptp(gv[:, 1]))
            px = np.clip(bx0 + nx * (bx1 - bx0), 0, img.shape[1] - 1).astype(int)
            py = np.clip(by0 + (1.0 - ny) * (by1 - by0), 0, img.shape[0] - 1).astype(int)
            col = img[py, px]
            inm = msk[py, px] if msk is not None else np.ones(len(col), bool)
            fill = np.median(col[inm], 0) if inm.any() else col.mean(0)
            col[~inm] = fill
            gen_vcol = col
            print(f"[FOLD] gen cloth colours = REAL capture projection ({int(inm.sum())}/{len(col)} in-mask)", flush=True)
        except Exception as e:
            print(f"[FOLD] gen cloth colour projection failed ({e}); keeping generator colours", flush=True)
        gv[:, :2] += real[:, :2].mean(0)                 # centre at the real shirt position
        gv[:, 2] += real[:, 2].min() - gv[:, 2].min()    # base on the real shirt's base
        cv = gv.astype(np.float64)
        cloth_faces = np.asarray(gf).reshape(-1, 3)
        print(f"[FOLD] shirt from GEN3D {os.path.basename(gen_cloth)}: {len(cv)} verts, "
              f"{len(cloth_faces)} faces, scaled to measured {meas:.0f}cm, at real position", flush=True)
    elif mask_cloth:
        from cloth_mesh import cloth_from_mask
        mv, mf, mask_uv, mask_tex = cloth_from_mask(mask_cloth, capture_dir, step=mask_step,
                                                    two_layer=two_layer, layer_gap=layer_gap,
                                                    real_volume=real_volume)  # verts in cm (real depth)
        cv = mv.astype(np.float64)                             # from real depth -> use as-is (working baseline)
        cloth_faces = mf.reshape(-1, 3)
        print(f"[FOLD] shirt from MASK: {len(cv)} verts, {len(cloth_faces)} faces "
              f"({'TWO-LAYER garment like the example shirt' if two_layer else 'single sheet'}, textured)", flush=True)
    else:
        if cloth_decimate and cloth_decimate > 0:
            cloth = decimate_cloth_obj(cloth, cloth_decimate)
        cv = calibrate(cloth)
        cloth_faces = cloth["faces"].reshape(-1, 3)
    parts = {"cloth": cv}
    if box is not None:
        parts["box"] = calibrate(box)                    # RAW real box mesh, exactly as reconstructed
    # FAITHFUL placement: keep each object at its REAL reconstructed position relative to the robot
    # (ur5e_base_link frame), with the Franka standing in for the base at (franka_x, franka_y). The
    # real robot-relative layout (shirt in front, box offset in +y) is preserved; only z is snapped
    # to the table to remove reconstruction depth-noise.
    for p in parts.values():
        p[:, 0] += franka_x; p[:, 1] += franka_y
        p[:, 2] += table_top - p[:, 2].min()
    # align the SHIRT orientation to reality: rotate ONLY the shirt about its own centre
    if abs(scene_yaw) > 1e-6:
        th = np.radians(scene_yaw); ct, st = np.cos(th), np.sin(th)
        R = np.array([[ct, -st], [st, ct]]); c = cv[:, :2].mean(0)
        cv[:, :2] = (cv[:, :2] - c) @ R.T + c
    # optionally move the box to the opposite side of the shirt (translate, keep orientation) — needed
    # after putting the robot on the far side, which reverses the apparent box side
    if flip_box and "box" in parts:
        sc = cv[:, :2].mean(0); bc = parts["box"][:, :2].mean(0)
        parts["box"][:, 0] += (2 * sc[0] - bc[0]) - bc[0]
        parts["box"][:, 1] += (2 * sc[1] - bc[1]) - bc[1]
    # optionally enlarge the box so the folded shirt fits inside (the real shirt ~= the real box
    # size -> a flap always drapes over the wall). Scale about the box XY-centre and its base so it
    # stays put and rests on the table; walls/render/drop-target all derive from these verts.
    if "box" in parts and abs(box_scale - 1.0) > 1e-6:
        bv = parts["box"]; cxy = bv[:, :2].mean(0); zmin = bv[:, 2].min()
        bv[:, :2] = (bv[:, :2] - cxy) * box_scale + cxy
        bv[:, 2] = (bv[:, 2] - zmin) * box_scale + zmin
    # pull the whole scene toward the robot so the arm reaches a NATURAL pose (the capture over-
    # estimates the object distance ~66-78cm; user: 'in reality they are near the robot'). Scales the
    # base->object vectors by `near`, preserving the objects' relative arrangement.
    if abs(near - 1.0) > 1e-6:
        b = np.array([franka_x, franka_y])
        for p in parts.values():
            c = p[:, :2].mean(0)
            p[:, :2] += ((c - b) * near + b) - c        # translate centre closer; keep object SIZE

    cbb = [cv[:, 0].min(), cv[:, 0].max(), cv[:, 1].min(), cv[:, 1].max()]
    cloth_out = {"verts": cv.astype(np.float32), "faces": cloth_faces,
                 "color": cloth["color"], "bbox": cbb, "label": cloth["label"],
                 "vcolors": gen_vcol if gen_cloth else cloth.get("vcolors"),
                 "uvs": mask_uv, "texture": mask_tex}
    box_out = None
    if box is not None:
        # cardboard colour = median of the box's real reconstructed colours; render SOLID (clean) in
        # mask mode — projective/per-vertex texturing of this small rough box smears background pixels.
        cardboard = box["color"]
        if box.get("vcolors") is not None:
            bc = np.asarray(box["vcolors"]).reshape(-1, 3)
            cardboard = tuple(np.clip(np.median(bc, 0) * 1.05, 0, 1))   # slight lift, real cardboard tone
        bverts = parts["box"]; bfaces = box["faces"].reshape(-1, 3); bvcol = box.get("vcolors")
        if clean_box:
            # the single-view SAM3D box is malformed (leans ~40-46deg, inflated bbox) -> its collision can't
            # match the drawn box and the shirt won't sit inside. Replace the GEOMETRY with a clean open-top
            # container at the REAL box centre, sized as a plausible box that holds the folded shirt. Colour
            # + position stay real; only the shape is a primitive (disclosed).
            cxb, cyb = parts["box"][:, 0].mean(), parts["box"][:, 1].mean()
            z0 = float(parts["box"][:, 2].min())        # already snapped to the table plane
            bw, bd, bh = 24.0 * box_scale, 22.0 * box_scale, 17.0 * box_scale
            bverts, bfaces = _open_box_mesh(cxb, cyb, z0, bw, bd, bh)
            bvcol = None                                 # render SOLID cardboard
            print(f"[FOLD] BOX = clean open-top primitive {bw:.0f}x{bd:.0f}x{bh:.0f}cm at real centre "
                  f"({cxb:.0f},{cyb:.0f}) [SAM3D box was malformed ~46deg]", flush=True)
        box_out = {"verts": bverts.astype(np.float32), "faces": bfaces,
                   "color": cardboard, "label": box["label"],
                   "vcolors": None if (mask_cloth or clean_box) else bvcol}
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
        self.sim_substeps = args.substeps          # fewer -> faster (bigger dt, less stable)
        self.vbd_iters = args.vbd_iters            # fewer VBD iterations -> faster, less converged
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
        self.want_physical_grasp = getattr(args, "physical_grasp", False)
        self.robot_env_collision = getattr(args, "robot_collision", False)
        self.base_yaw_deg = args.base_yaw
        self.max_reach = args.reach
        self.place_reach = args.place_reach     # higher clamp for the place-in-box keyframes
        self.sc_interval = args.sc_interval     # cloth self-collision detect interval (1=stable/slow, -1=fast)
        self.drop_bias = args.drop_bias         # bias the box drop toward the far wall (frac of box depth)
        self.bend_ke = args.bend_stiffness      # cloth bending stiffness (floppy<->stiff)
        self.stretch_ke = args.stretch_stiffness  # cloth stretch stiffness (rubbery<->real fabric)
        self.press_in = args.press_in           # press the shirt down into the box after release
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
            mcloth = None
            if args.mask_cloth:
                from cloth_mesh import find_cloth_mask
                mcloth = find_cloth_mask(args.scene_dir, CLOTH_KW)
                print(f"[FOLD] mask-cloth source: {mcloth}", flush=True)
            self.fcloth, self.fbox = faithful_objects(args.scene_dir, cap, -50.0, -50.0,
                                                      self.cloth_top, args.scene_yaw, args.flip_box,
                                                      args.cloth_decimate, args.box_scale, mcloth, args.near,
                                                      args.mask_step, args.clean_box,
                                                      two_layer=args.two_layer, layer_gap=args.layer_gap,
                                                      real_volume=args.real_volume,
                                                      gen_cloth=args.gen_cloth)
            self.init_bbox = list(self.fcloth["bbox"])
            print(f"[FOLD] FAITHFUL import: cloth '{self.fcloth['label']}' "
                  f"{len(self.fcloth['verts'])} verts, box "
                  f"{'yes' if self.fbox is not None else 'none'}", flush=True)
            # place-in-box targets the REAL box position (collision walls fitted to the real mesh)
            if self.place_in_box and self.fbox is not None:
                bv = self.fbox["verts"]
                zt = bv[:, 2].max(); zb = bv[:, 2].min()
                rim = bv[bv[:, 2] > zt - 0.25 * (zt - zb)]        # top-rim verts = the OPENING
                self.box = [float(rim[:, 0].mean()),             # drop target = opening centre (box is
                            float(rim[:, 1].mean()),             # tilted, so the opening != bbox centre)
                            float(bv[:, 0].max() - bv[:, 0].min()) - 3.0,
                            float(bv[:, 1].max() - bv[:, 1].min()) - 3.0,
                            float(bv[:, 2].max() - bv[:, 2].min())]
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
            if self.faithful and self.fbox is not None:
                # COLLIDE the cloth against the REAL box MESH, exactly as reconstructed (tilted / cluttered /
                # whatever the RGBD gives). A tilted OPEN box holds things fine in reality — they rest in its
                # low corner; the earlier "tunneling" was a WEAK-COLLISION bug, not the tilt. So keep the real
                # geometry and make the collision ROBUST (bigger contact margin + thicker particles below) so
                # the cloth can't pass through the thin walls. Open the top so the shirt drops in; it settles
                # in the box's low corner (on the table, which IS the box's low side). Invisible collision-only
                # (the real box is drawn via box_mesh_viz) — no primitive, this is the actual reconstruction.
                bverts = self.fbox["verts"].astype(np.float64)
                bfaces = self.fbox["faces"].reshape(-1, 3)
                zt = bverts[:, 2].max(); zb = bverts[:, 2].min()
                keep = bverts[bfaces].mean(1)[:, 2] < zb + 0.82 * (zt - zb)   # open the top so the shirt enters
                of = bfaces[keep].reshape(-1).astype(np.int32)
                cmesh = newton.Mesh(bverts.astype(np.float32), of)
                self.box_liner_idx = [self.scene.shape_count]                 # hide from render (collision only)
                self.scene.add_shape_mesh(-1, wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), mesh=cmesh)
                rim = bverts[bverts[:, 2] > zt - 0.30 * (zt - zb)]            # aim the drop at the real opening
                self.box[0] = float(rim[:, 0].mean()); self.box[1] = float(rim[:, 1].mean())
                print(f"[FOLD] box COLLISION = REAL box mesh (open-top, {len(of)//3} faces, tilted as reconstructed) — shirt rests in the real box", flush=True)
            else:
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
                vertices=[wp.vec3(*v) for v in fv], indices=ff.reshape(-1).tolist(),
                # EXAMPLE density 0.02: grip capacity (pinch contact force) does NOT scale with cloth
                # mass, so heavier-per-particle fabric (0.05) overloads the pinch and slips out.
                # The example's drape comes from per-area weight 0.02 — match it exactly.
                # TWO-PLY garments (two_layer / real_volume) halve the per-layer density so the
                # TOTAL per-area weight stays 0.02 — else every flap is 2x the tuned pinch load
                # and folds spring open instead of laying flat.
                density=0.01 if (args.two_layer or args.real_volume) else 0.02,
                tri_ke=self.stretch_ke, tri_ka=self.stretch_ke, tri_kd=1.5e-6, edge_ke=self.bend_ke, edge_kd=1.0e-2,
                # physical grasp: example's radius 0.8 so the closing pads (0.8cm slot) PINCH the fabric;
                # otherwise 0.5 keeps the cloth thin/smooth for box containment
                particle_radius=0.8 if self.want_physical_grasp else 0.5)
            print(f"[FOLD] cloth = real SAM3D mesh: {len(fv)} verts, {len(ff)} tris", flush=True)
        elif self.sil_mask is not None:
            # SILHOUETTE: cut the grid to the real shirt shape via the segmentation mask
            verts, tris, uvs = masked_grid(self.dim_x, self.dim_y, cell, self.sil_mask, box, img)
            self.sil_uvs = uvs
            self.scene.add_cloth_mesh(
                pos=wp.vec3(self.cloth_cx, self.cloth_cy, self.cloth_top),
                rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*v) for v in verts], indices=tris.tolist(), density=0.02,
                tri_ke=self.stretch_ke, tri_ka=self.stretch_ke, tri_kd=1.5e-6, edge_ke=self.bend_ke, edge_kd=1.0e-2,
                particle_radius=0.8)
            print(f"[FOLD] cloth SILHOUETTE: {len(verts)} verts, {len(tris)//3} tris (from {label} mask)", flush=True)
        else:
            self.scene.add_cloth_grid(
                pos=wp.vec3(self.cloth_cx - 0.5 * cloth_w, self.cloth_cy - 0.5 * cloth_h, self.cloth_top),
                rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.dim_x, dim_y=self.dim_y, cell_x=cell, cell_y=cell, mass=0.1,
                tri_ke=self.stretch_ke, tri_ka=self.stretch_ke, tri_kd=1.5e-6, edge_ke=self.bend_ke, edge_kd=1.0e-2,
                particle_radius=0.8)
        self.scene.color()
        self.scene.add_ground_plane()

        if getattr(args, "cloth_solver", "vbd") == "style3d" and SolverStyle3D is not None:
            # Style3D stores per-cloth stretch/shear/bend attributes on the builder; must register pre-finalize
            SolverStyle3D.register_custom_attributes(self.scene)

        self.model = self.scene.finalize(requires_grad=False)

        # hide table from auto shape rendering (GL bakes prim dims, ignores scale) -> draw manually
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        flags[self.ped_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        for si in self.box_shape_idx:                 # walls drawn manually at m-scale too
            flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        for si in getattr(self, "box_liner_idx", []):    # invisible containment liner (real box drawn via box_mesh_viz)
            flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # contact material
        if self.want_physical_grasp:
            # EXAMPLE-FAITHFUL (cloth_franka): these exact values are what lets the closing pads
            # physically pinch + hold the fabric by friction (robot shape mu 1.5), no pin.
            self.model.soft_contact_ke = 1e4; self.model.soft_contact_kd = 1e-2; self.model.soft_contact_mu = 0.25
        else:
            self.model.soft_contact_ke = 1.5e4; self.model.soft_contact_kd = 5e-2; self.model.soft_contact_mu = 0.4  # firm but STABLE box contact + damping (too-stiff ejects particles)
        ke = self.model.shape_material_ke.numpy(); kd = self.model.shape_material_kd.numpy(); mu = self.model.shape_material_mu.numpy()
        ke[...] = 5e4; kd[...] = 1e-3; mu[...] = 1.5
        self.model.shape_material_ke = wp.array(ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=0.8 if self.want_physical_grasp else 1.2)  # 0.8 = example (pinch); 1.2 = box-wall safety
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.model.edge_rest_angle.zero_()
        if getattr(args, "cloth_solver", "vbd") == "style3d" and SolverStyle3D is not None:
            # SPIKE: Style3D garment-specialised solver (higher-quality cloth). NOTE it lacks VBD's
            # integrate_with_external_rigid_solver flag, so cloth<->robot coupling may differ.
            self.cloth_solver = SolverStyle3D(self.model, iterations=self.vbd_iters, linear_iterations=10)
            print(f"[FOLD] cloth solver = Style3D (spike)", flush=True)
            self._is_style3d = True
        else:
            self._is_style3d = False
            self.cloth_solver = SolverVBD(
            self.model, iterations=self.vbd_iters, integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5, particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=64,
            # dense 3-fold stack needs self-collision RE-DETECTED every iteration to stay stable:
            # detection fires when iter_num % interval == 0, so interval=-1 never detects (layers
            # tunnel -> NaN), interval=0 detects once/step then 10 iters drift into deep contact
            # (NaN at fold 3), and only interval=1 (every iteration) keeps contacts shallow/gentle
            # -> stable through the whole fold. It is ~10x the detection cost (slow) but the only
            # setting that survives the tight stack. conservative_bound 0.85->0.5 also helps.
            particle_conservative_bound_relaxation=0.5,
            particle_collision_detection_interval=self.sc_interval)

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
        self.mask_uvs = None
        if self.faithful and self.fcloth.get("uvs") is not None:
            # MASK cloth: real photo texture, UV per particle maps to its original RGB pixel -> the
            # shirt (sleeves + logo) renders from the real capture. Textured, not baked/particle-ish.
            self.mask_uvs = wp.array(self.fcloth["uvs"].astype(np.float32), dtype=wp.vec2)
            self.use_tex = True
            print(f"[FOLD] cloth colours = real photo texture on the mask mesh (sleeved shirt)", flush=True)
        elif self.faithful and self.fcloth.get("vcolors") is not None:
            faces = self.model.tri_indices.numpy().reshape(-1, 3)
            vcol = self.fcloth["vcolors"]
            _, rf, uv, tex = sam3d_scene.bake_colors(np.zeros((len(vcol), 3), np.float32), faces, vcol)
            self.cloth_bake = {"faceidx": faces.reshape(-1), "rfaces": wp.array(rf, dtype=wp.int32),
                               "uvs": wp.array(uv, dtype=wp.vec2), "tex": tex}
            print(f"[FOLD] cloth colours = real SAM3D per-vertex (baked)", flush=True)
        if self.mask_uvs is not None:                # mask cloth: UV per particle (already computed)
            self.uvs = self.mask_uvs
        elif self.sil_uvs is not None:               # silhouette: UVs precomputed per kept vertex
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
        # RENDER-ONLY volumetric shell: physics = the smooth single sheet; display = a skin
        # with thickness that follows it every frame (normals-offset bottom copy + edge band).
        # Decouples fold mechanics (proven recipe) from garment body (user: sheet looks flat,
        # but closed two-ply physics folds badly — a sealed pocket resists folding).
        self.shell_t = float(getattr(args, "shell", 0.0))
        self.shell = None
        if self.shell_t > 0 and self.use_tex:
            import collections
            tri = self.model.tri_indices.numpy().reshape(-1, 3)
            N = int(self.model.particle_count)
            ec = collections.Counter()
            for t_ in tri:
                for a, b in ((t_[0], t_[1]), (t_[1], t_[2]), (t_[2], t_[0])):
                    ec[(min(a, b), max(a, b))] += 1
            band = []
            for t_ in tri:
                for a, b in ((t_[0], t_[1]), (t_[1], t_[2]), (t_[2], t_[0])):
                    if ec[(min(a, b), max(a, b))] == 1:
                        band += [[a, b, b + N], [a, b + N, a + N]]
            f2 = np.vstack([tri, tri[:, ::-1] + N, np.asarray(band, dtype=np.int64)])
            uv_np = self.uvs.numpy()
            self.shell = {"tri": tri, "N": N,
                          "faces": wp.array(f2.reshape(-1).astype(np.int32), dtype=wp.int32),
                          "uvs": wp.array(np.vstack([uv_np, uv_np]).astype(np.float32), dtype=wp.vec2),
                          "sign": None}
            print(f"[FOLD] render SHELL: {self.shell_t:.1f}cm visual thickness on the single-sheet physics", flush=True)
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
            # the real SAM3D box mesh, as-is (cm -> m), with its real per-vertex colours baked.
            # SAM3D reconstructs the box watertight (closed top); DROP the top-cap faces so it renders
            # as the real OPEN cardboard box and the shirt placed inside stays visible.
            bverts = self.fbox["verts"].astype(np.float64)
            bfaces = self.fbox["faces"].reshape(-1, 3)
            zt = bverts[:, 2].max(); zb = bverts[:, 2].min()
            keep = bverts[bfaces].mean(1)[:, 2] < zb + 0.82 * (zt - zb)
            bfaces_open = bfaces[keep].reshape(-1).astype(np.int64)
            if self.fbox.get("uvs") is not None:
                # PROJECTIVE texture: the real photo mapped onto the box mesh -> the label/print shows
                # sharply (vs blotchy per-vertex baked colours). UVs computed at the box's real pose.
                from newton._src.utils.texture import load_texture
                btex = load_texture(self.rgb_path)
                self.box_mesh_viz = ("baked", wp.array(bverts * self.viz_scale, dtype=wp.vec3),
                                     wp.array(bfaces_open.astype(np.int32), dtype=wp.int32),
                                     wp.array(self.fbox["uvs"].astype(np.float32), dtype=wp.vec2), btex)
                print(f"[FOLD] box texture = real photo (projective) -> label/print visible", flush=True)
            elif self.fbox["vcolors"] is not None:
                rv, rf, uv, tex = sam3d_scene.bake_colors(
                    self.fbox["verts"], bfaces_open, self.fbox["vcolors"])
                self.box_mesh_viz = ("baked", wp.array(rv * self.viz_scale, dtype=wp.vec3),
                                     wp.array(rf, dtype=wp.int32), wp.array(uv, dtype=wp.vec2), tex)
            else:
                bv = self.fbox["verts"].astype(np.float32) * self.viz_scale
                self.box_mesh_viz = ("solid", wp.array(bv, dtype=wp.vec3),
                                     wp.array(bfaces_open.astype(np.int32), dtype=wp.int32),
                                     tuple(self.fbox["color"]))
            print(f"[FOLD] container = real SAM3D box mesh (open-top, baked colours)", flush=True)
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
        elif self.box_mesh_viz is None:
            for (cx, cy, cz, hx, hy, hz) in self.box_walls:        # fallback: flat walls (no real mesh)
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
        # robot base = ur5e_base_link, IDENTITY orientation (so the hand-eye camera pose matches the
        # real capture), ~5cm above the table top. Objects are at v_base relative to this -> the robot
        # camera view reproduces the real photo (box left, shirt right).
        # robot mounted on the RIGHT side of the table, facing back toward the objects (-x)
        _ang = np.pi + np.radians(self.base_yaw_deg)
        rq = wp.quat(0.0, 0.0, float(np.sin(_ang / 2)), float(np.cos(_ang / 2)))
        self.robot_base = (60.0, -40.0, 25.0)
        self.physical_grip = False
        if self.arm == "ur5e_jaw":
            from ur5e_jaw_gripper import add_ur5e_jaw_gripper
            w3, dof0, n_arm, pinch, gdofs, op, cp = add_ur5e_jaw_gripper(
                builder, wp.transform(self.robot_base, rq), scale=100.0)
            self.ur5e_w3 = w3; self.n_arm = n_arm; self.pinch_local = pinch
            self.grip_dofs = gdofs; self.grip_open_pos = op; self.grip_close_pos = cp
            self.physical_grip = True           # jaws PHYSICALLY clamp the cloth (no pin)
            print(f"[FOLD] arm = UR5e + 2-JAW gripper (PHYSICAL clamp, dofs {gdofs}, "
                  f"open {op:.1f} close {cp:.1f}cm)", flush=True)
        elif self.arm == "ur5e":
            from ur5e_gripper import add_ur5e_gripper
            w3, dof0, n_arm, pinch = add_ur5e_gripper(builder, wp.transform(self.robot_base, rq), scale=100.0)
            self.ur5e_w3 = w3; self.n_arm = n_arm; self.pinch_local = pinch
            # 2F-85 finger joints: KINEMATICALLY drive the 8 tree joints (per finger: driver, coupler,
            # spring_link, follower) so the fingers VISIBLY CLOSE on the cloth when grasping (the pin
            # does the load-bearing hold; the 4-bar can't be physically actuated under Featherstone).
            # Closed pose from MuJoCo (which solves the 4-bar): driver 0.781, coupler ~0, spring 0.78,
            # follower -0.756 per finger; open = ~0.
            self.grip_dofs = list(range(dof0 + n_arm, dof0 + n_arm + 8))
            self.free_loop_dofs = list(range(dof0 + n_arm + 8, builder.joint_dof_count))  # cut 4-bar dofs
            self.grip_closed = np.array([0.781, 0.001, 0.78, -0.756, 0.781, 0.001, 0.78, -0.756])
            self.grip_open = np.zeros(8)
            print(f"[FOLD] arm = UR5e + Robotiq 2F-85 ({n_arm} arm dof, {builder.joint_dof_count - dof0 - n_arm} gripper dof; fingers actuated)", flush=True)
        elif self.arm == "ur10":
            from ur10_gripper import add_ur10_gripper
            w3, dof0, n_arm, pinch = add_ur10_gripper(builder, wp.transform(self.robot_base, rq), scale=100.0)
            self.ur5e_w3 = w3; self.n_arm = n_arm; self.pinch_local = pinch
            # same 2F-85 gripper as the UR5e: 8 finger tree-dofs + the cut 4-bar loop dofs, posed
            # kinematically so the fingers visibly close on the fabric (pin does the load-bearing hold)
            self.grip_dofs = list(range(dof0 + n_arm, dof0 + n_arm + 8))
            self.free_loop_dofs = list(range(dof0 + n_arm + 8, builder.joint_dof_count))
            self.grip_closed = np.array([0.781, 0.001, 0.78, -0.756, 0.781, 0.001, 0.78, -0.756])
            self.grip_open = np.zeros(8)
            print(f"[FOLD] arm = UR10e + Robotiq 2F-85 ({n_arm} arm dof, {builder.joint_dof_count - dof0 - n_arm} gripper dof; fingers actuated)", flush=True)
        else:
            asset_path = newton.utils.download_asset("franka_emika_panda")
            builder.add_urdf(str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
                             xform=wp.transform(self.robot_base, rq),
                             floating=False, scale=100, enable_self_collisions=False,
                             collapse_fixed_joints=True, force_show_colliders=False)
            builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
            self.n_arm = 7
            # PHYSICAL grasp: skip the kinematic pin and let the Franka's parallel fingers hold the
            # cloth by contact (they already open/close). The fingers must trap the fabric between them.
            if self.want_physical_grasp:
                self.physical_grip = True
                print("[FOLD] Franka PHYSICAL grasp (no pin — fingers hold the cloth by contact)", flush=True)

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
            self.fold_grasp_iv = set()                # keyframe indices whose grasp z adapts to cloth top
            # LOW-LIFT fold profile (matched to Newton's official cloth_franka demo, which folds visibly
            # smoothly): lift the grasped fabric only ~7-10cm above the table so it DRAPES over the fold
            # line instead of being hoisted ~20cm and crumpling on the way down. Approach stays high.
            ZHI, ZLIFT, ZAPEX = 42.0, 28.0, 31.0
            if self.fold_mode == "garment":
                # GARMENT fold matched to Newton's cloth_franka demo EXACTLY: fold the +x side in at the
                # TOP and BOTTOM (2 grasps), then the -x side in at the top and bottom (2 grasps), then the
                # bottom edge UP (1 grasp) — a real shirt fold, grabbing each side at two points so the whole
                # side folds neatly (not from a single point). 5 gentle low grasps, slow low arcs.
                x0, x1, y0, y1 = bbox
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                h = y1 - y0
                yt = y0 + 0.72 * h                        # upper grasp row of each side
                yb = y0 + 0.28 * h                        # lower grasp row of each side
                # PHYSICAL pinch: grasp 4cm INSIDE the edge (the example insets 2-5cm) so the closing
                # slot is FULL of fabric — pinching exactly AT the edge catches a thin sliver on one
                # side only, which pulls out of the pads under load (the observed slip).
                ins = 4.0 if self.physical_grip else 0.0
                gsteps = [("+x-top", (x1 - ins, yt), (cx, yt)),   # right side, upper -> centre line
                          ("+x-bot", (x1 - ins, yb), (cx, yb)),   # right side, lower -> centre line
                          ("-x-top", (x0 + ins, yt), (cx, yt)),   # left side, upper -> centre line
                          ("-x-bot", (x0 + ins, yb), (cx, yb)),   # left side, lower -> centre line
                          ("bottom", (cx, y0 + ins), (cx, cy))]   # bottom edge -> up to centre
                # PHYSICAL grasp needs the example's dwell: gain-1 IK converges exp(-t), so the pads
                # need ~4-5s AT the grasp pose (descend+close) to actually sink into the fabric before
                # the fingers finish closing (example: 2-3s descend + 2-3s close; short dwells left the
                # pads 3-5cm above the cloth at close time -> pinched air).
                t_desc = 3.0 if self.physical_grip else 1.5
                t_close = 2.5 if self.physical_grip else 1.0
                for i, (axis, g, t) in enumerate(gsteps):
                    base = len(poses)
                    zg = 21.0 + 0.4 * i; zland = 21.6 + 0.4 * i
                    mid = ((g[0] + t[0]) / 2.0, (g[1] + t[1]) / 2.0)
                    if self.physical_grip:
                        # EXAMPLE-FAITHFUL release: carry to the target at height and OPEN there —
                        # the flap DRAPES down (example releases at z=31). Descending to the surface
                        # while holding drags the wad through the folded pile and smears it.
                        poses += [
                            [2.0, g[0], g[1], ZHI,   *q, OPEN],    # approach above grasp edge
                            [t_desc, g[0], g[1], zg,  *q, OPEN],   # descend (long enough to converge)
                            [t_close, g[0], g[1], zg, *q, CLOSE],  # grasp edge
                            [2.0, g[0], g[1], ZLIFT, *q, CLOSE],   # peel up
                            [2.5, mid[0], mid[1], ZAPEX, *q, CLOSE],  # slow arc over the fold line
                            [2.0, t[0], t[1], ZAPEX, *q, CLOSE],   # carry to the drop point (stay high)
                            [1.5, t[0], t[1], ZAPEX, *q, OPEN],    # release from height -> flap drapes
                            [1.5, t[0], t[1], ZHI,   *q, OPEN],    # retreat up
                        ]
                    else:
                        poses += [
                            [2.0, g[0], g[1], ZHI,   *q, OPEN],    # approach above grasp edge
                            [t_desc, g[0], g[1], zg,  *q, OPEN],   # descend (long enough to converge)
                            [t_close, g[0], g[1], zg, *q, CLOSE],  # grasp edge (pin)
                            [2.0, g[0], g[1], ZLIFT, *q, CLOSE],   # LOW peel
                            [2.5, mid[0], mid[1], ZAPEX, *q, CLOSE],  # slow low arc over the fold line
                            [2.0, t[0], t[1], ZLIFT, *q, CLOSE],   # carry to the lay-down point
                            [1.5, t[0], t[1], zland, *q, CLOSE],   # lay down (drape)
                            [1.0, t[0], t[1], zland, *q, OPEN],    # release
                            [1.5, t[0], t[1], ZHI,   *q, OPEN],    # retreat up
                        ]
                    self.fold_grasp_iv.update({base + 1, base + 2})
                    specs.append({"axis": axis})
                # folded footprint: both sides pulled to the centre line (half width) + bottom folded up
                bbox = [cx - 0.25 * (x1 - x0), cx + 0.25 * (x1 - x0), cy - 0.15 * h, cy + 0.25 * h]
                nf = 0                                  # skip the half-fold loop below
            for i in range(nf):
                axis = "y" if i % 2 == 0 else "x"
                base = len(poses)                     # this fold's first pose index
                x0, x1, y0, y1 = bbox
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                # grasp height must DESCEND into contact with the cloth: folding FLATTENS the shirt
                # (the cloth top drops after each fold), so a rising grasp z made the fingertip hover
                # 2-3cm ABOVE the cloth ("holding from the air"). Sit just at/into the cloth surface
                # with only a small rise as layers stack.
                zg = 21.0 + 0.4 * i    # pads TOUCH the cloth (~21cm); the small folded shirt barely
                zland = 21.6 + 0.4 * i  # rises, so only a tiny per-fold rise (big rise -> holding air)
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
                # the descend (base+1) + grasp (base+2) poses get an ADAPTIVE z at runtime (land the
                # fingertips on the ACTUAL cloth-pile top, so a stack that rose more than the fixed schedule
                # doesn't jam the fingers into its side — fixes fold>=5 gap going negative)
                self.fold_grasp_iv.update({base + 1, base + 2})
                specs.append({"axis": axis})
            self.place_lo = self.place_hi = None
            if self.place_in_box:
                # pick the folded bundle (centre of the final bbox) and place it NEATLY in the box.
                bx, by = (bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0
                bcx, bcy, bw, bd, bh = self.box
                box_floor = self.cloth_top            # the box sits on the table plane
                box_top = box_floor + bh
                zgb = 21.8                            # touch the top of the folded wad
                # the gathered bunch hangs and TRAILS back toward the robot (-y) as it is carried, so
                # releasing at the exact centre lands it over the near wall. Bias the drop toward the
                # FAR half so the trailing bunch settles into the box interior; carry slowly + settle
                # to kill the swing; lower DEEP so the bunch coils on the floor instead of draping.
                dropx, dropy = bcx, bcy + self.drop_bias * bd
                z_in = box_floor + 1.0                 # set the wad essentially ON the floor (min fall/drift)
                self.place_lo = len(poses)
                # pin mode: come down FAST (long unpinned settle self-collides -> VBD NaN).
                # PHYSICAL mode: the pinch needs the long dwell (gain-1 IK converges exp(-t);
                # 0.9s left the pads ~8cm above the pile = pinched air, same bug as the fold
                # grasps) — and there is no pin, so the slow descend is safe.
                t_desc, t_close = (3.0, 2.5) if self.physical_grip else (0.9, 1.0)
                poses += [
                    [1.0, bx, by, ZHI, *q, OPEN],          # approach above bundle
                    [t_desc, bx, by, zgb, *q, OPEN],       # descend onto bundle
                    [t_close, bx, by, zgb, *q, CLOSE],     # grasp (pin centre / pinch top layers)
                    # physical pinch: the 1.8s yank to ZHI inertially PEELED the wad out of the
                    # fingers (hold6 showed an empty gripper 2.5s after close). Two-stage SLOW
                    # lift transfers the wad's weight into the pinch gradually, like the fold peel.
                    [3.0 if self.physical_grip else 0.9, bx, by, ZLIFT, *q, CLOSE],  # slow peel off table
                    [3.0 if self.physical_grip else 0.9, bx, by, ZHI, *q, CLOSE],    # continue up (bunch forms)
                    [4.0, dropx, dropy, ZHI, *q, CLOSE],   # carry SLOWLY over the box (less swing)
                    [3.5, dropx, dropy, ZHI, *q, CLOSE],   # SETTLE long: wad hangs STRAIGHT under grip
                    # RELEASE-FROM-ABOVE (faithful): the real box is a THIN OPEN reconstructed shell, so
                    # setting the pinned (rigid) wad DOWN through it tunnels the cloth out ("box hollow").
                    # Instead lower the wad to JUST above the rim and drop it — it falls INTO the box and
                    # collides with the mesh from above, never pushed through the thin walls.
                    [2.5, dropx, dropy, box_top + 2.0, *q, CLOSE],  # lower to just above the opening
                    [2.5, dropx, dropy, box_top + 2.0, *q, CLOSE],  # settle long: wad hangs straight over the opening
                    [1.2, dropx, dropy, box_top + 2.0, *q, OPEN],   # RELEASE -> wad drops into the box from the rim
                    [1.5, dropx, dropy, box_top + 6.0, *q, OPEN],   # withdraw straight up (clear of the box)
                ]
                if self.press_in:
                    # PRESS the shirt DOWN into the box to tuck it below the rim: lift clear, close to a
                    # compact tool, then press straight down onto the wad at a few spots (centre, near half,
                    # far half). The closed gripper physically pushes the (now free) cloth into the box.
                    # keep the gripper OPEN through the press: open fingers still collide with + push the
                    # cloth DOWN, but an OPEN->CLOSE transition would re-trigger the pin (re-grab + lift the
                    # shirt back OUT of the box). Open-hand press => genuine tuck, no re-grab.
                    z_press = box_floor + 1.5
                    for sy in (0.0, -0.28 * bd, 0.28 * bd):
                        poses += [
                            [0.7, dropx, dropy + sy, box_top + 4.0, *q, OPEN],   # lift clear above the spot
                            [0.9, dropx, dropy + sy, z_press, *q, OPEN],         # press DOWN onto the wad
                            [0.7, dropx, dropy + sy, z_press, *q, OPEN],         # hold the press (tuck)
                        ]
                    poses += [[0.8, dropx, dropy, box_top + 4.0, *q, OPEN]]      # lift clear after pressing
                poses += [
                    [1.5, dropx, dropy, box_top + 5.0, *q, OPEN],   # withdraw straight up out of box
                    [2.0, dropx, dropy, ZHI, *q, OPEN],    # retreat
                ]
                self.place_hi = len(poses)
                # bundle descend+grasp get the ADAPTIVE pile-top z like the fold grasps —
                # the fixed zgb=21.8 aims 8cm INTO a 10cm-tall folded stack
                self.fold_grasp_iv.update({self.place_lo + 1, self.place_lo + 2})
                specs.append({"axis": "bundle"})
            poses += [[1.5, 0.0, -38.0, ZHI, *q, OPEN]]   # clear away (reachable retreat)
            self.robot_key_poses = np.array(poses, dtype=np.float32)
            self.fold_specs = specs
        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.target = self.targets[0]
        # per-keyframe reach clamp: folding stays at the comfortable --reach so the arm never
        # overstretches on the shirt's far corners; the place-in-box keyframes get a higher clamp
        # (the real box sits ~77cm out, beyond --reach) so the arm can actually reach the box.
        self.pose_reach = np.full(len(self.targets), self.max_reach, dtype=np.float32)
        if getattr(self, "place_lo", None) is not None:
            self.pose_reach[self.place_lo:self.place_hi] = max(self.max_reach, self.place_reach)
        if self.arm in ("ur5e", "ur5e_jaw", "ur10"):
            self.endeffector_id = self.ur5e_w3               # wrist_3 (UR5e or UR10e)
            self.endeffector_offset = wp.transform(list(self.pinch_local), wp.quat_identity())
        else:
            self.endeffector_id = builder.body_count - 3
            # MEASURED (FK probe): body -3 is the collapsed wrist+hand; (0,0,22) lands at the finger
            # PAD TIPS (the Newton cloth_franka example's offset). (0,0,11) is only mid-hand — driving
            # that to the cloth left the pads 11cm away, so the gripper never touched the fabric.
            self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())
        # ACCURACY: pin the grasped cloth AT the controlled fingertip (endeffector_offset), not at a
        # virtual point below it. Previously the pin used ee_off_vec=(0,0,22) while the IK drove the
        # fingertip at (0,0,11) -> the shirt hung ~11cm BELOW the gripper ('holding air'). Unify them.
        self.ee_off_vec = wp.vec3(*self.endeffector_offset.p)

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
        self._cur_interval = interval
        # REACH CLAMP: never command the arm past a comfortable radius from its base (avoid overstretch
        # on far edges / the box). The pin still grabs the full edge (by shirt geometry), the arm just
        # holds it from a non-overextended pose.
        tgt = np.array(self.target[:7], dtype=float)
        rb = np.array([self.robot_base[0], self.robot_base[1]])
        dxy = tgt[:2] - rb; r = float(np.linalg.norm(dxy))
        mr = float(self.pose_reach[min(interval, len(self.pose_reach) - 1)])
        if r > mr:
            tgt[:2] = rb + dxy / r * mr
        # ADAPTIVE GRASP HEIGHT: on a fold's descend/grasp pose, land the fingertip on the ACTUAL cloth-pile
        # top near the (clamped) grasp XY instead of a fixed schedule z. A stack that rose more than the
        # 0.4cm/fold guess no longer jams the fingers into its side (fold5 gap was -1.8cm); one that stayed
        # flat isn't grasped from the air. Uses the current particle positions each frame during the descend.
        if interval in getattr(self, "fold_grasp_iv", ()):
            pq = state_in.particle_q.numpy()
            # ADAPTIVE GRASP XY (physical): earlier folds MOVE the fabric, so a pre-planned grasp
            # point can end up over bare table (fold2 caught 0 particles after fold1 dragged that
            # region away). At each fold's descend, if the planned XY has no fabric, SNAP the grasp
            # to the nearest actual cloth (centroid of its 5cm neighbourhood). Cached per fold so
            # descend+close share one point (recomputing at close would jump back to the planned XY).
            if self.physical_grip:
                if not hasattr(self, "_grasp_xy_cache"):
                    self._grasp_xy_cache = {}
                fkey = interval if (interval - 1) not in self.fold_grasp_iv else interval - 1
                if fkey in self._grasp_xy_cache:
                    tgt[:2] = self._grasp_xy_cache[fkey]
                else:
                    d2 = np.linalg.norm(pq[:, :2] - tgt[:2], axis=1)
                    if (d2 < 3.0).sum() < 5:                        # planned point is (nearly) empty
                        j = int(np.argmin(d2))
                        nb = np.linalg.norm(pq[:, :2] - pq[j, :2], axis=1) < 5.0
                        snap = pq[nb][:, :2].mean(0)
                        print(f"[FOLD] RETARGET grasp: planned ({tgt[0]:.0f},{tgt[1]:.0f}) empty -> "
                              f"snap to fabric at ({snap[0]:.0f},{snap[1]:.0f})", flush=True)
                        self._grasp_xy_cache[fkey] = snap.copy()
                        tgt[:2] = snap
                    else:
                        self._grasp_xy_cache[fkey] = tgt[:2].copy()
            near = (pq[:, 0] - tgt[0]) ** 2 + (pq[:, 1] - tgt[1]) ** 2 < 36.0   # within 6cm of the grasp XY
            if near.any():
                ctop = float(pq[near, 2].max())                                # top of the local pile
                # per-arm pad offset: the commanded EE point (pinch) sits at a different height above the
                # actual finger contact for each gripper. UR5e pinch (0.223*scale) lands ~on the cloth at
                # +0.3; the Franka hand's fingertips sit LOWER than its EE offset (11cm), so it jams the
                # fabric (gaps -1 to -3) unless the target is raised. Raise the Franka target so the pads
                # rest ON the cloth (gap ~0..+1), giving a tighter, more compact fold.
                # with the EE point AT the pad tips for every arm now, rest the pads just on the pile
                # top (pin mode). PHYSICAL pinch: the example plunges the pad point to TABLE level
                # (z=19.5-20 vs table top 20) — through the fabric — so closing traps a thick wad of
                # cloth in the 0.4cm slot; that squeeze is what holds during the lift.
                pad_off = 0.3
                if self.physical_grip:
                    # EXAMPLE-FAITHFUL: plunge THROUGH the local fabric — but only ~3cm below the
                    # LOCAL pile top. Plunging to table level through a tall folded pile bulldozes
                    # the stack sideways (the rumpled-heap failure); on the flat shirt ctop-3 ==
                    # table level anyway, so fold 1 keeps the full deep pinch.
                    table_top = float(self.table_pos_cm[2]) + self.table_hz
                    tgt[2] = max(table_top - 0.3, ctop - 3.0)
                else:
                    tgt[2] = float(np.clip(ctop + pad_off, self.cloth_top, 42.0))   # pads rest just on top
        wp.launch(compute_ee_delta, dim=1,
                  inputs=[state_in.body_q, self.endeffector_offset, self.endeffector_id,
                          self.bodies_per_world, wp.transform(*tgt)],
                  outputs=[self.ee_delta])
        self.compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        q = state_in.joint_q.numpy()
        if self.arm in ("ur5e", "ur5e_jaw", "ur10"):
            # restrict IK to the 6 arm joints (position-only: bring the tool tip to the target;
            # orientation free — DOWN_Q is the Franka's convention, not the UR arms')
            na = self.n_arm
            dq_arm = _dls_pinv(J[:3, :na]) @ delta_target[:3] * 3.0   # gain for faster convergence (robust DLS)
            delta_q = np.zeros(self.model.joint_dof_count)
            delta_q[:na] = dq_arm
            grip_close = float(self.target[-1]) <= 0.4
            if self.physical_grip:
                # DRIVE the jaws: close (clamp the cloth) when the keyframe grip is CLOSE, else open
                tgt = self.grip_close_pos if grip_close else self.grip_open_pos
                for d in self.grip_dofs:
                    delta_q[d] = (tgt - q[d]) * 6.0
            elif self.arm in ("ur5e", "ur10"):
                for d in self.grip_dofs + self.free_loop_dofs:
                    delta_q[d] = 0.0            # gripper is posed KINEMATICALLY (see _pose_gripper);
                    #                            velocity-driving the cut 4-bar just fights the solver
            self.target_joint_qd.assign(delta_q)
            return
        J_inv = _dls_pinv(J)
        N = np.eye(J.shape[1], dtype=np.float32) - J_inv @ J
        q_des = q.copy(); q_des[1:] = self.initial_pose[1:]
        delta_q = J_inv @ delta_target + N @ (1.0 * (q_des - q))
        # Panda hand fingers = the last 2 DOFs (prismatic, scaled range ~0..4cm). Drive them to CLEAR
        # open/close targets with a firm gain so the gripper visibly opens (carrying) and closes (grasping):
        # CLOSE -> ~0.3cm apart (pinched shut on the fabric), OPEN -> ~3.9cm apart (wide open).
        if self.physical_grip:
            # EXAMPLE-FAITHFUL finger drive (cloth_franka): finger target = activation*4cm
            # (close 0.1->0.4cm, open 0.8->3.2cm), velocity = position error. The 0.4cm close
            # leaves a ~0.8cm slot that pinches the radius-0.8 fabric between the pads; contact
            # friction (mu 1.5) holds it while lifting. No pin.
            act = float(self.target[-1])
            # close to a 0.3cm slot (tighter than the example's 0.4 — our mesh is coarser so each
            # gripped particle carries more load; deeper squeeze = more contact force = more friction).
            # gain 4: fully closed within the close keyframe (lifting half-closed drops the cloth).
            fpos = 0.15 if act <= 0.4 else 3.2
            # BUNDLE TRANSPORT: the whole folded wad (~4x a single flap's load) pulls out of the
            # 0.3cm slot mid-carry — clamp fully shut for the place phase only
            if (act <= 0.4 and getattr(self, "place_lo", None) is not None
                    and getattr(self, "_cur_interval", 0) >= self.place_lo):
                fpos = 0.02
            delta_q[-2] = (fpos - q[-2]) * 4.0
            delta_q[-1] = (fpos - q[-1]) * 4.0
        else:
            fgrip = 0.3 if float(self.target[-1]) <= 0.4 else 3.9
            delta_q[-2] = (fgrip - q[-2]) * 6.0
            delta_q[-1] = (fgrip - q[-1]) * 6.0
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

    def export_ee_traj(self, path):
        """Write the fold's keyframe fingertip (TCP) poses in the ur5e_base_link frame (metres) as an
        ee_trajectory.json for resolve_orientation.py + ur5e_ee_executor.py (real UR5e via MoveIt).
        Poses are reach-CLAMPED exactly as the sim executes them (so the exported motion matches the
        demo). FRAME FIX vs export_plan: base z = the real robot base height (self.robot_base[2]=25cm),
        not 0 — else every fingertip Z is 25cm too high in ur5e_base_link.
        HONEST SCOPE: the sim grabs the cloth with a geometric PIN, not a real grip, and the arm is
        reach-clamped short of the true cloth edge — so this is a MOTION/REACHABILITY export for a
        dry-run, NOT a validated real fabric grasp."""
        import json
        bx, by, bz = self.robot_base                       # ur5e_base_link origin in sim world (cm)
        rbxy = np.array([bx, by])
        grips = self.robot_key_poses[:, -1]
        wps = []
        first_grasp = False
        for i, row in enumerate(self.robot_key_poses):
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            tgt = np.array([x, y]); dxy = tgt - rbxy; r = float(np.linalg.norm(dxy))
            mr = float(self.pose_reach[min(i, len(self.pose_reach) - 1)])
            if r > mr:                                     # same clamp the arm executes
                tgt = rbxy + dxy / r * mr; x, y = float(tgt[0]), float(tgt[1])
            pos = [round((x - bx) / 100.0, 4), round((y - by) / 100.0, 4), round((z - bz) / 100.0, 4)]
            grip = float(row[-1]); gev = "none"
            if i > 0:
                pg = float(grips[i - 1])
                if pg > 0.4 >= grip:
                    gev = "close"
                elif pg <= 0.4 < grip:
                    gev = "open"
            if gev == "close":
                label = "descend_to_grasp" if not first_grasp else f"grasp_{i}"
                first_grasp = True
            elif gev == "open":
                label = f"release_{i}"
            else:
                label = f"move_{i}"
            wps.append({"label": label, "position": pos, "quaternion": [0.0, 1.0, 0.0, 0.0],
                        "gripper": gev, "dur": float(row[0])})
        cobs = []
        if self.fbox is not None:
            bv = self.fbox["verts"].astype(np.float64)
            c = (bv.max(0) + bv.min(0)) / 2.0; d = bv.max(0) - bv.min(0)
            cobs.append({"type": "box", "name": "box",
                         "center": [round((c[0] - bx) / 100.0, 4), round((c[1] - by) / 100.0, 4),
                                    round((c[2] - bz) / 100.0, 4)],
                         "size": [round(d[0] / 100.0, 4), round(d[1] / 100.0, 4), round(d[2] / 100.0, 4)]})
        tt = (self.cloth_top - bz) / 100.0                 # table top in base frame (m)
        cobs.append({"type": "box", "name": "table",
                     "center": [round((0.0 - bx) / 100.0, 4), round((-50.0 - by) / 100.0, 4), round(tt - 0.10, 4)],
                     "size": [1.2, 1.2, 0.20]})
        json.dump({"frame": "ur5e_base_link", "gripper": {"open_m": 0.05, "closed_m": 0.0},
                   "default_vel_scale": 0.1,
                   "note": "FOLD motion; sim used a geometric pin not a real cloth grip -> dry-run/reachability only",
                   "waypoints": wps, "collision_objects": cobs}, open(path, "w"), indent=2)
        print(f"[FOLD] exported {len(wps)} EE waypoints + {len(cobs)} collision objs "
              f"(ur5e_base_link, m) -> {path}", flush=True)

    # ----------------------------- grasp (pin) -----------------------------
    def update_grasp(self):
        grip = float(self.cur_grip) if getattr(self, "exec_mode", False) else float(self.target[-1])
        if self.physical_grip:                          # jaws clamp the cloth physically (no pin)
            if self.prev_grip > 0.4 and grip <= 0.4:
                # DIAGNOSTIC: did the pads actually arrive at the commanded grasp, on fabric?
                bq = self.state_0.body_q.numpy()
                hand = bq[self.endeffector_id, :3]; q6 = bq[self.endeffector_id, 3:7]
                x, y, z, w = q6; off = np.array(list(self.ee_off_vec))
                t2 = 2.0 * np.cross([x, y, z], off)
                pad = hand + off + w * t2 + np.cross([x, y, z], t2)
                pq = self.state_0.particle_q.numpy()
                d = np.linalg.norm(pq - pad, axis=1)
                near = int((d < 3.0).sum())
                tgt = np.array(self.target[:3], dtype=float)
                err = float(np.linalg.norm(pad - tgt))
                print(f"[FOLD] CLAMP at t={self.sim_time:.1f}s  pad=({pad[0]:.0f},{pad[1]:.0f},{pad[2]:.1f}) "
                      f"target=({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.1f}) ik_err={err:.1f}cm | "
                      f"nearest fabric {d.min():.1f}cm, {near} particles within 3cm", flush=True)
            # HYBRID TRANSPORT (user-approved): the folds are fully PHYSICAL, but the bundle
            # carry uses the PIN — a single-point fingertip pinch cannot hold the whole folded
            # stack over the ~50cm carry (3 fixes tried: long dwell, 2-stage slow lift, full
            # clamp; the wad pulls out every time = honest physical limit, like the cup topple).
            # The pin stands in for a firmer whole-hand grab. Disclosed abstraction.
            if (getattr(self, "place_lo", None) is not None
                    and getattr(self, "_cur_interval", 0) >= self.place_lo):
                if self.prev_grip > 0.4 and grip <= 0.4:
                    self.next_fold = len(self.fold_specs) - 1      # select the bundle spec
                    self._grasp()
                elif self.prev_grip <= 0.4 and grip > 0.4:
                    self._release()
            self.prev_grip = grip
            return
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
        if self.arm in ("ur5e", "ur10"):
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
                # FLAT LIFT: after 3 folds the bundle footprint (~24x14cm) FITS the box opening
                # (~26x21cm), so lift it as a flat semi-rigid sheet (pin the central ~60%) and lay it
                # flat into the box. A flat held sheet barely swings during the long carry, unlike a
                # gathered point-pinch which trails back and drapes over the near wall. The unpinned
                # outer ~40% is still inside the footprint, so it settles inside the box too.
                # pin the WHOLE folded sheet -> carry it RIGIDLY. Pinning only the central 60% leaves the
                # outer 40% perimeter free; on a LARGE shirt that perimeter drapes low, drags on the table
                # and self-collides during the long carry -> extreme stretch -> VBD NaN (the 'strands' bug).
                # A fully-pinned bundle has no free perimeter to drag, so the carry is stable; the sheet is
                # released (inv_mass restored) once inside the box, where it settles naturally.
                idx = np.arange(len(pq))
                fw = float(pq[:, 0].max() - pq[:, 0].min()); fd = float(pq[:, 1].max() - pq[:, 1].min())
                iw = self.init_bbox[1] - self.init_bbox[0]; id_ = self.init_bbox[3] - self.init_bbox[2]
                print(f"[FOLD] FOLDED FOOTPRINT (measured): {fw:.0f} x {fd:.0f} cm = {fw*fd:.0f} cm^2  "
                      f"(flat {iw:.0f}x{id_:.0f}={iw*id_:.0f}cm^2 -> {100*fw*fd/(iw*id_):.0f}% of flat area)", flush=True)
                what = f"bundle flat sheet ({len(idx)} particles, rigid carry)"
            else:
                # grab a local PINCH of fabric AT the fingertips (within ~5cm of the pads) so a visible
                # chunk of the edge is held BETWEEN the pads and moves with them — not a thin edge line
                # that drapes below ('holding air'). The gripper sits on the fold edge, so this pinch is
                # at the edge and folding it over still works; the rest of the shirt follows via cloth.
                d2 = np.linalg.norm(pq[:, :2] - pinch[:2], axis=1)
                idx = np.argsort(d2)[:40]     # nearest ~40 = a visible pinch, but CAPPED: grabbing the
                #                               whole stacked chunk (687 after fold 1) over-constrains
                #                               the VBD cloth -> NaN + heavy contacts (lag)
                what = f"fold {self.next_fold} {axis}-edge pinch ({len(idx)} particles)"
        self.pin_idx = wp.array(idx.astype(np.int32), dtype=wp.int32)
        # For the BUNDLE carry, anchor the sheet's XY CENTROID to the fingertip so it hangs directly UNDER
        # the gripper (not offset beside it) and drops into the CENTRE of the box, not over an edge — and the
        # gripper visibly holds the middle of the folded shirt. Fold pinches keep the true edge offset.
        if axis == "bundle":
            anchor = np.array([pq[idx, 0].mean(), pq[idx, 1].mean(), pinch[2]])
        else:
            anchor = pinch
        self.pin_off = wp.array((pq[idx] - anchor).astype(np.float32), dtype=wp.vec3)  # hold under the fingertips
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
        self._pose_gripper()
        self.sim_time += self.frame_dt

    def _pose_gripper(self):
        """Kinematically set the 2F-85 fingers OPEN or CLOSED for the frame (the cut 4-bar can't be
        dynamically driven; the pin does the load-bearing grip). Overwrite the finger tree-joints +
        zero the loop-closure dofs, then FK so the rendered fingers match."""
        if getattr(self, "arm", None) not in ("ur5e", "ur10") or getattr(self, "exec_mode", False):
            return
        grip = float(self.target[-1]) if self.sim_time < self.robot_key_poses_time[-1] else OPEN
        tgt = self.grip_closed if grip <= 0.4 else self.grip_open
        q = self.state_0.joint_q.numpy()
        for j, d in enumerate(self.grip_dofs):
            q[d] = tgt[j]
        for d in self.free_loop_dofs:
            q[d] = 0.0
        self.state_0.joint_q.assign(q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # robot
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            if getattr(self, "robot_env_collision", False):
                # FAITHFUL: give the robot real contacts vs the table/box (particles off -> robot-vs-
                # environment only) so the arm/gripper can't pierce surfaces. Featherstone.step() takes
                # the contacts; without this (shape_contact_pair_count=0 + None) the IK arm phases through.
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            else:
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
        if self.use_tex and self.shell is not None:
            pq = self.viz_state.particle_q.numpy()
            tri = self.shell["tri"]
            fn = np.cross(pq[tri[:, 1]] - pq[tri[:, 0]], pq[tri[:, 2]] - pq[tri[:, 0]])
            vn = np.zeros_like(pq)
            np.add.at(vn, tri[:, 0], fn); np.add.at(vn, tri[:, 1], fn); np.add.at(vn, tri[:, 2], fn)
            vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
            if self.shell["sign"] is None:               # bottom copy must start BELOW the sheet
                self.shell["sign"] = 1.0 if float(vn[:, 2].mean()) > 0 else -1.0
            off = vn * (self.shell_t * self.viz_scale * self.shell["sign"])
            v2 = np.vstack([pq, pq - off]).astype(np.float32)
            self.viewer.log_mesh("cloth", wp.array(v2, dtype=wp.vec3), self.shell["faces"],
                                 uvs=self.shell["uvs"], texture=self.tex_img,
                                 color=(1.0, 1.0, 1.0), backface_culling=False)
            o = getattr(self.viewer, "objects", {}).get("cloth")
            if o is not None:
                r, m, c, _ = o.material; o.material = (r, m, c, 1.0)
        elif self.use_tex:
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
    ap.add_argument("--fold_mode", default="multi", choices=["multi", "half", "corner", "garment"], help="garment = demo-style: sides folded in to centre, then hem up (3 gentle low grasps)")
    ap.add_argument("--folds", type=int, default=3, help="number of alternating half-folds (multi mode)")
    ap.add_argument("--pins", type=int, default=2, help="corner particles pinned (corner mode only)")
    ap.add_argument("--place_in_box", action="store_true", help="after folding, pick the bundle and place it in the box")
    ap.add_argument("--box", default="-33,-40,27,29,15", help="container cx,cy,width,depth,wall_height (cm)")
    ap.add_argument("--cloth_max", type=float, default=0.38, help="cap cloth dimension (m) for reachability")
    ap.add_argument("--no_silhouette", action="store_true", help="use a plain rectangle, not the real shirt shape")
    ap.add_argument("--faithful", action="store_true", help="import the real SAM3D meshes as-is (cloth+box)")
    ap.add_argument("--arm", default="franka", choices=["franka", "ur5e", "ur5e_jaw", "ur10"], help="robot arm (ur5e = UR5e+Robotiq 2F-85 pin-grasp; ur5e_jaw = UR5e + simple 2-jaw PHYSICAL clamp; ur10 = UR10e bare flange)")
    ap.add_argument("--physical_grasp", action="store_true", help="Franka: fingers hold the cloth by CONTACT (no kinematic pin)")
    ap.add_argument("--robot_collision", action="store_true", help="robot physically collides with the table/box (no penetration)")
    ap.add_argument("--base_yaw", type=float, default=0.0, help="extra base-facing yaw offset (deg) for the arm")
    ap.add_argument("--reach", type=float, default=72.0, help="comfortable arm reach radius (cm); targets clamped to it to avoid overstretch")
    ap.add_argument("--place_reach", type=float, default=82.0, help="higher reach clamp (cm) for the place-in-box keyframes (the real box sits beyond --reach)")
    ap.add_argument("--sc_interval", type=int, default=1, help="cloth self-collision detection interval (1=stable, needed for folds>=3, ~10x slower; -1=off/fast, ok for folds<=2)")
    ap.add_argument("--substeps", type=int, default=10, help="physics substeps/frame (fewer=faster, bigger dt, less stable)")
    ap.add_argument("--vbd_iters", type=int, default=10, help="VBD solver iterations/substep (fewer=faster, less converged)")
    ap.add_argument("--cloth_solver", default="vbd", choices=["vbd", "style3d"], help="cloth solver: vbd (default) or style3d (garment-specialised quality spike)")
    ap.add_argument("--cloth_decimate", type=int, default=0, help="decimate the real shirt mesh to ~N faces (0=off/full-res) for faster iteration; shape+colours preserved")
    ap.add_argument("--bend_stiffness", type=float, default=5.0, help="cloth BENDING stiffness edge_ke (higher=stiffer/holds shape, lower=floppier/drapes more)")
    ap.add_argument("--stretch_stiffness", type=float, default=1.0e4, help="cloth STRETCH stiffness tri_ke/tri_ka (higher=less stretchy/real fabric, lower=rubbery)")
    ap.add_argument("--press_in", action="store_true", help="after releasing the shirt in the box, press it down with the gripper to tuck it inside")
    ap.add_argument("--mask_step", type=int, default=7, help="mask-cloth grid stride in px (larger=coarser/fewer particles=cheaper self-collision, more stable+faster)")
    ap.add_argument("--two_layer", action="store_true", help="mask cloth as a TWO-LAYER stitched garment (real thickness like a generated shirt); pair with --mask_step 9 to keep the single-layer particle budget/speed")
    ap.add_argument("--layer_gap", type=float, default=1.2, help="two-layer garment thickness in cm (1.6 read as puffy; ~1.0-1.3 = tee)")
    ap.add_argument("--real_volume", action="store_true", help="mask cloth with MEASURED volume: bottom flat at the table-contact plane, top at the real depth surface, edges feathered — organic thickness from the capture itself (supersedes --two_layer)")
    ap.add_argument("--shell", type=float, default=0.0, help="RENDER-ONLY garment thickness in cm (e.g. 1.0): physics stays the smooth single sheet; the display mesh gets a normals-offset volumetric skin. Best fold quality + garment body.")
    ap.add_argument("--clean_box", action="store_true", help="replace the malformed single-view SAM3D box with a clean open-top box primitive at the real centre+colour (so the shirt actually sits inside)")
    ap.add_argument("--drop_bias", type=float, default=0.15, help="bias the box drop toward the far wall as a fraction of box depth (counters the bunch trailing back toward the robot)")
    ap.add_argument("--box_scale", type=float, default=1.0, help="enlarge the real box about its centre (e.g. 1.4) so the folded shirt fits inside; 1.0=faithful size")
    ap.add_argument("--mask_cloth", action="store_true", help="build the shirt from the real SAM2 MASK (silhouette incl. sleeves) + depth + RGB texture, instead of SAM3D's sleeveless blob mesh")
    ap.add_argument("--gen_cloth", default=None, help="path to a GENERATED garment mesh (.glb/.obj/.usd from perception/gen3d_cloth.py, Meshy/Rodin): use it as the deformable (topology from generation; scale+position from the capture). Overrides --mask_cloth")
    ap.add_argument("--near", type=float, default=1.0, help="pull objects toward the robot (e.g. 0.65) for a natural reach; the capture over-estimates object distance")
    ap.add_argument("--scene_yaw", type=float, default=180.0, help="rotate ONLY the shirt about its centre (deg) to correct its reconstructed orientation")
    ap.add_argument("--flip_box", action="store_true", default=False, help="put the box on the opposite side of the shirt")
    ap.add_argument("--export_plan", default=None, help="write keyframe poses+obstacles (base frame) for cuRobo, then exit")
    ap.add_argument("--export_ee_traj", default=None, help="write the fold EE trajectory (ur5e_base_link frame, m) for the real UR5e transfer, then exit")
    ap.add_argument("--exec_plan", default=None, help="execute a cuRobo joint trajectory (from curobo_fold_plan.py)")
    ap.add_argument("--no_texture", action="store_true")
    ap.add_argument("--hold", type=float, default=1800.0)
    ap.add_argument("--render_every", type=int, default=1, help="render every Nth sim frame (8-10 ~= real-time playback; physics unaffected)")
    ap.add_argument("--screenshot", default=None, help="PNG glob: save frames at fold milestones")
    ap.add_argument("--cam", default="-0.75,-0.40,0.42,-10,0", help="x,y,z,pitch,yaw (default = head-on FRONT view: camera in front of the robot looking at it)")
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
    if args.export_ee_traj:                          # export the fold EE trajectory for the real UR5e
        ex.export_ee_traj(args.export_ee_traj)
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
        closes = [i for i in range(1, len(grips)) if grips[i] <= 0.4 < grips[i - 1]]
        shots[3] = f"{base}_start.png"                  # initial layout (compare to the robot photo)
        for k, i in enumerate(rels):
            shots[int(ex.robot_key_poses_time[i] * ex.fps)] = f"{base}_step{k+1}.png"
        # a "holding" shot ~2.5s into each lift (grip still CLOSED) to inspect gripper<->cloth contact
        for k, i in enumerate(closes):
            shots[int((ex.robot_key_poses_time[i] + 2.5) * ex.fps)] = f"{base}_hold{k+1}.png"
        shots[n_frames - 1] = f"{base}_done.png"

    print(f"[FOLD] running {n_frames} frames ({ex.robot_key_poses_time[-1]:.0f}s schedule)...", flush=True)
    t0 = time.time()
    for f in range(n_frames):
        ex.step()
        # render every Nth sim frame: physics unchanged, GUI advances ~N x faster toward
        # real-time (the sim computes ~6 fps of 60 fps schedule -> N=8-10 looks live)
        if f % max(1, args.render_every) == 0 or f in shots:
            ex.render()
        if f in shots and args.viewer == "gl":
            from PIL import Image
            Image.fromarray(viewer.get_frame().numpy()).save(shots[f])
            print(f"[FOLD] shot -> {shots[f]}", flush=True)
        if args.viewer == "gl" and not viewer.is_running():
            break
    pq_end = ex.state_0.particle_q.numpy()
    zmin = float(pq_end[:, 2].min())
    print(f"[FOLD] done in {time.time()-t0:.0f}s; lowest particle z={zmin:.2f}cm "
          f"({'NaN!' if not np.isfinite(zmin) else 'ok'})", flush=True)
    if np.isfinite(zmin) and ex.model.particle_count > 0:
        # measured FOLDED FOOTPRINT (fold-only runs too) -> feeds the cuTAMP garment proxy
        fw = float(pq_end[:, 0].max() - pq_end[:, 0].min()); fd = float(pq_end[:, 1].max() - pq_end[:, 1].min())
        fh = float(pq_end[:, 2].max() - zmin)
        iw = ex.init_bbox[1] - ex.init_bbox[0]; id_ = ex.init_bbox[3] - ex.init_bbox[2]
        print(f"[FOLD] FOLDED FOOTPRINT (measured): {fw:.0f} x {fd:.0f} x {fh:.0f} cm = {fw*fd:.0f} cm^2 "
              f"(flat {iw:.0f}x{id_:.0f}={iw*id_:.0f}cm^2 -> {100*fw*fd/(iw*id_):.0f}% of flat area)", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[FOLD] holding GUI {args.hold:.0f}s", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            ex.render()


if __name__ == "__main__":
    main()
