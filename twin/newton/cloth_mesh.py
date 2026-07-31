"""Build a realistic, DEFORMABLE cloth MESH from the real capture — mask (shape, incl. sleeves)
+ depth (3D placement on the table) + RGB (texture).

WHY: SAM3D's single-view 3D mesh reconstructs a flat garment as a rounded, SLEEVELESS blob (the
sleeves are smoothed away). But the SAM2 MASK captures the true t-shirt silhouette WITH sleeves, and
the depth+RGB give the real placement + appearance. So for a flat garment this is a MORE faithful
reconstruction than SAM3D's mesh — still 100% real sensor data (mask/depth/photo), no fabricated
proxy. Result: a connected triangle mesh (VBD-ready) shaped like the real shirt, textured with the
real photo (logo + fabric), lying flat on the table at its real pose.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
from scipy import ndimage


def _largest(m):
    lbl, n = ndimage.label(m)
    if n > 1:
        s = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1)); m = (lbl == np.argmax(s) + 1)
    return m


def refine_mask_by_color(mask, rgb):
    """SAM2 often UNDER-segments a garment (grabs the crumpled body, drops the sleeves -> a sleeveless
    blob). Recover the full silhouette by growing the mask to include NEIGHBOURING pixels that match the
    garment's OWN median colour (sampled inside the mask), then smooth the ragged boundary. General: keys
    off the object's own colour, not a hardcoded hue. Returns a clean bool mask."""
    rgb = rgb.astype(np.float32)
    core = ndimage.binary_erosion(mask, iterations=4)
    core = core if core.sum() > 50 else mask
    med = np.median(rgb[core], axis=0)
    dist = np.linalg.norm(rgb - med[None, None, :], axis=2)
    # robust threshold from the 85th-pct in-mask distance (ignores logo/print outliers), clamped
    thr = float(np.clip(np.percentile(dist[core], 85) * 1.4, 35.0, 70.0))
    near = ndimage.binary_dilation(mask, iterations=35)            # only ADD fabric adjacent to the body
    grow = ((dist < thr) & near) | mask
    grow = ndimage.binary_closing(ndimage.binary_opening(grow, iterations=1), iterations=3)
    grow = ndimage.binary_fill_holes(_largest(grow))
    grow = ndimage.binary_fill_holes(ndimage.gaussian_filter(grow.astype(np.float32), sigma=2.0) > 0.5)
    return grow


def _tri_min_angle(P):
    """min interior angle (deg) per triangle, P shape [T,3,2]."""
    def ang(u, w):
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        return np.degrees(np.arccos(np.clip((u * w).sum(1), -1, 1)))
    A = ang(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    B = ang(P[:, 0] - P[:, 1], P[:, 2] - P[:, 1])
    return np.stack([A, B, 180 - A - B], 1).min(1)


def _resample_contour(m, step):
    """Trace the mask's outer contour (ordered) and resample it at UNIFORM arc-length spacing ~step.
    Even spacing is what keeps the boundary triangulation sliver-free AND smooth (no notches)."""
    from contourpy import contour_generator
    lines = contour_generator(z=m.astype(np.float64)).lines(0.5)        # ordered (x,y) polylines
    if not lines:
        return np.zeros((0, 2))
    outer = max(lines, key=len).astype(float)                          # longest = outer boundary
    # smooth the polyline (periodic moving average) to remove small zigzags -> clean silhouette
    if len(outer) > 8:
        closed = np.allclose(outer[0], outer[-1])
        p = outer[:-1] if closed else outer
        w = 5
        k = np.ones(w) / w
        sx = np.convolve(np.r_[p[-w:, 0], p[:, 0], p[:w, 0]], k, "same")[w:-w]
        sy = np.convolve(np.r_[p[-w:, 1], p[:, 1], p[:w, 1]], k, "same")[w:-w]
        outer = np.stack([sx, sy], 1)
        outer = np.vstack([outer, outer[:1]])                          # re-close
    seg = np.sqrt((np.diff(outer, axis=0) ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    n = max(12, int(round(L / step)))
    s = np.linspace(0.0, L, n, endpoint=False)
    return np.stack([np.interp(s, cum, outer[:, 0]), np.interp(s, cum, outer[:, 1])], 1)


def _dense_contour(m):
    """Ordered, dense outer-boundary polyline (x,y) of the mask, for projecting points onto the edge."""
    from contourpy import contour_generator
    lines = contour_generator(z=m.astype(np.float64)).lines(0.5)
    return max(lines, key=len).astype(float) if lines else np.zeros((0, 2))


def _conforming_triangulation(m, step, min_angle=28.0):
    """Triangulate the mask into a WELL-CONDITIONED, boundary-conforming cloth sheet using a CONSTRAINED
    Delaunay refiner (Shewchuk's Triangle). The traced+smoothed silhouette is passed as ENFORCED boundary
    segments, so the mesh edge is EXACTLY the smooth contour (no sawtooth, no notches); the 'q' quality flag
    inserts interior Steiner points until every triangle's min angle >= min_angle (no slivers -> VBD stable);
    the 'a' area flag sets the resolution (~step-sized triangles). Returns (verts_px[N,2] int, faces[M,3])."""
    import triangle as tr
    bnd = _resample_contour(m, step)                                    # ordered, smoothed contour (~step spacing)
    if len(bnd) > 1 and np.allclose(bnd[0], bnd[-1]):
        bnd = bnd[:-1]                                                  # open ring (triangle closes via segments)
    n = len(bnd)
    if n < 4:
        return np.zeros((0, 2), int), np.zeros((0, 3), int)
    segs = np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)          # closed boundary loop = the constraint
    max_area = 0.9 * step * step                                        # ~step-edge triangles (boundary stays smooth)
    out = tr.triangulate({"vertices": bnd, "segments": segs}, f"pq{min_angle:g}a{max_area:g}")
    V = out["vertices"]; F = out["triangles"]
    return np.rint(V).astype(int), F.astype(int)


def _largest_component(faces, npts):
    """Return only the faces in the largest vertex-connected component."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    if len(faces) == 0:
        return faces
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(npts, npts)).tocsr()
    ncomp, lab = csg.connected_components(A + A.T, directed=False)
    if ncomp == 1:
        return faces
    big = np.argmax(np.bincount(lab))
    return faces[lab[faces[:, 0]] == big]


def _double_layer(base, faces, uvs, gap=1.6, bot_z=None):
    """Build a TWO-LAYER garment from the single mask sheet — like the Newton example's
    unisex_shirt (front + back panels stitched at the silhouette): a real tee lying flat IS
    two layers of fabric joined at the boundary. top = the photo-textured sheet at the real
    depth; bottom = a copy `gap` cm below; a side band stitches the silhouette ring closed.
    Doubles the fabric (mass, drape) and gives the gripper a thick pinchable bunch.
    Returns (verts[2N,3], faces[M',3], uvs[2N,2])."""
    N = len(base)
    bot = base.copy()
    if bot_z is not None:
        bot[:, 2] = bot_z                                # per-vertex bottom (measured-volume mode)
    else:
        bot[:, 2] -= gap
    verts2 = np.vstack([base, bot])
    uvs2 = np.vstack([uvs, uvs])
    bot_faces = faces[:, ::-1] + N                       # reversed winding, offset indices
    # boundary ring = edges used by exactly one triangle
    import collections
    ec = collections.Counter()
    for t in faces:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(min(a, b), max(a, b))] += 1
    band = []
    for t in faces:                                       # keep the triangle's edge ORIENTATION for winding
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if ec[(min(a, b), max(a, b))] == 1:
                band.append([a, b, b + N]); band.append([a, b + N, a + N])
    all_faces = np.vstack([faces, bot_faces, np.array(band, dtype=faces.dtype)])
    return verts2, all_faces, uvs2


def cloth_from_mask(mask_path, cap_dir, step=7, base_frame="ur5e_base_link", refine=True,
                    two_layer=False, layer_gap=1.6, real_volume=False):
    """Return (verts_cm[N,3] in base frame, faces[M,3], uvs[N,2], rgb_texture[H,W,3] uint8).
    verts lie flat on the table at the real pose; uvs map each vertex to its real photo pixel.
    refine=True recovers SAM2-dropped sleeves via colour-grow + smooths the ragged boundary."""
    mask = np.array(Image.open(mask_path).convert("L")) > 128
    rgb = np.array(Image.open(cap_dir + "/rgb.png").convert("RGB"))
    depth = np.load(cap_dir + "/depth.npy").astype(np.float64)          # metres
    intr = json.load(open(cap_dir + "/intrinsics.json"))
    extr = json.load(open(cap_dir + "/extrinsics.json"))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    if refine:
        m = refine_mask_by_color(mask, rgb)                            # sleeves back + smooth silhouette
    else:
        m = ndimage.binary_fill_holes(_largest(ndimage.binary_opening(mask, iterations=2)))

    dsh = depth[m & (depth > 0)]
    dmed = float(np.median(dsh)) if len(dsh) else 1.0
    dfill = depth.copy(); dfill[dfill <= 0] = dmed
    dfill = ndimage.median_filter(dfill, size=9)                        # smooth -> clean flat sheet
    H, W = m.shape

    # BOUNDARY-CONFORMING mesh (clean edges, no grid "comb teeth"): triangulate interior GRID points
    # together with points sampled ALONG the smooth mask contour, so the silhouette follows the real
    # boundary instead of snapping to the coarse lattice.
    verts_px, faces = _conforming_triangulation(m, step)

    # back-project each vert with its real (smoothed) depth -> camera frame -> base frame (cm)
    u = verts_px[:, 0].astype(float); v = verts_px[:, 1].astype(float)
    Z = dfill[verts_px[:, 1], verts_px[:, 0]]
    cam = np.stack([(u - cx) / fx * Z, (v - cy) / fy * Z, Z], 1)
    T = np.array(extr["transforms"][base_frame]["T_base_cam"])
    base = (cam @ T[:3, :3].T + T[:3, 3]) * 100.0                       # -> cm
    uvs = np.stack([u / W, 1.0 - v / H], 1)                            # V-flip for the viewer
    if real_volume:
        # MEASURED-VOLUME garment: the capture already measured the shirt's true volume —
        # top surface = real depth (crumple), bottom = the table-contact plane, thickness
        # FEATHERS to ~0 at the silhouette (45deg-ish taper via the mask distance transform).
        # Organic like a generated garment, but every millimetre of it is real data — unlike
        # the uniform-gap extrusion, which reads as an inflated cutout.
        dt_px = ndimage.distance_transform_edt(m)[verts_px[:, 1], verts_px[:, 0]]
        cm_per_px = dmed / fx * 100.0
        dt_cm = dt_px * cm_per_px
        z_floor = float(np.percentile(base[:, 2], 3.0))    # table-contact plane of the garment
        bot_z = np.maximum.reduce([
            np.full(len(base), z_floor),                   # never below the table contact
            base[:, 2] - dt_cm * 1.2,                      # edge feather (taper toward silhouette)
        ])
        bot_z = np.minimum(bot_z, base[:, 2] - 0.4)        # keep >=4mm gap: thinner z-FIGHTS
        #                                                    (dark flicker rim) + degenerate stitch
        #                                                    faces NaN VBD
        # EDGE TEXTURE BLEED: silhouette vertices sample photo pixels half on the dark table ->
        # a baked black rim. Snap near-boundary vertices' UV source to the nearest pixel of the
        # ERODED mask so every sample is pure fabric.
        inner = ndimage.binary_erosion(m, iterations=4)
        _, (iy, ix) = ndimage.distance_transform_edt(~inner, return_indices=True)
        nearb = dt_px < 5.0
        su = ix[verts_px[nearb, 1], verts_px[nearb, 0]].astype(float)
        sv = iy[verts_px[nearb, 1], verts_px[nearb, 0]].astype(float)
        uvs[nearb, 0] = su / W
        uvs[nearb, 1] = 1.0 - sv / H
        base, faces, uvs = _double_layer(base, faces.astype(np.int64), uvs, bot_z=bot_z)
    elif two_layer:
        base, faces, uvs = _double_layer(base, faces.astype(np.int64), uvs, gap=layer_gap)
    return base.astype(np.float32), faces.astype(np.int32), uvs.astype(np.float32), rgb.astype(np.uint8)


def project_uv(verts_base_cm, cap_dir, base_frame="ur5e_base_link"):
    """Projective UVs: map each base-frame vertex (cm) back into the real RGB via the camera, so the
    box renders with the ACTUAL photo (label/print sharp), not blotchy per-vertex baked colours.
    Returns uvs[N,2] in [0,1] (V-flipped for the viewer). Verts behind/off-image clamp to the border."""
    intr = json.load(open(cap_dir + "/intrinsics.json"))
    extr = json.load(open(cap_dir + "/extrinsics.json"))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    T = np.array(extr["transforms"][base_frame]["T_base_cam"])          # base <- camera
    R, t = T[:3, :3], T[:3, 3]
    cam = (np.asarray(verts_base_cm, float) / 100.0 - t) @ R            # base(m) -> camera frame
    Z = np.clip(cam[:, 2], 1e-3, None)
    from PIL import Image
    import os
    rgb = Image.open(os.path.join(cap_dir, "rgb.png"))
    W, H = rgb.size
    u = (fx * cam[:, 0] / Z + cx) / W
    v = (fy * cam[:, 1] / Z + cy) / H
    return np.stack([np.clip(u, 0, 1), 1.0 - np.clip(v, 0, 1)], 1).astype(np.float32)


def box_vertex_colors(verts_base_cm, faces, cap_dir, base_frame="ur5e_base_link"):
    """Per-vertex colours for the box: sample the REAL photo for CAMERA-FACING vertices (so the
    label/print shows), and a clean cardboard colour for back/occluded vertices (so projective
    smearing of background pixels is avoided). Returns colors[N,3] float in [0,1]."""
    import os
    from PIL import Image
    import trimesh
    intr = json.load(open(cap_dir + "/intrinsics.json"))
    extr = json.load(open(cap_dir + "/extrinsics.json"))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    T = np.array(extr["transforms"][base_frame]["T_base_cam"])
    R, t = T[:3, :3], T[:3, 3]
    v = np.asarray(verts_base_cm, float)
    f = np.asarray(faces).reshape(-1, 3)
    vn = trimesh.Trimesh(vertices=v, faces=f, process=False).vertex_normals
    cam_pos = t * 100.0                                         # camera origin in base frame (cm)
    view = cam_pos - v; view /= (np.linalg.norm(view, axis=1, keepdims=True) + 1e-9)
    facing = (vn * view).sum(1) > 0.1                          # vertex faces the camera
    cam = (v / 100.0 - t) @ R
    Z = np.clip(cam[:, 2], 1e-3, None)
    rgb = np.array(Image.open(os.path.join(cap_dir, "rgb.png")).convert("RGB"))
    H, W = rgb.shape[:2]
    u = np.clip((fx * cam[:, 0] / Z + cx).astype(int), 0, W - 1)
    vv = np.clip((fy * cam[:, 1] / Z + cy).astype(int), 0, H - 1)
    samp = rgb[vv, u] / 255.0
    cardboard = np.median(samp[facing], 0) if facing.any() else np.array([0.62, 0.47, 0.35])
    cols = np.where(facing[:, None], samp, cardboard[None, :])
    return cols.astype(np.float32)


def find_cloth_mask(scene_dir, cloth_kw, cloth_idx=None):
    """Locate the SHIRT's SAM2 mask (outputs/<scene>_sam3d_raw/masks/<i>.png). Returns path or None.

    The mask files are numbered by the object `id` in scene_layout.json (mask <id>.png <-> object id),
    so DON'T blindly grab 0.png — that may be the box. Map the cloth by LABEL: find the scene_layout
    object whose label matches a cloth keyword and use its id as the mask index."""
    import glob, json, os
    raw = scene_dir.rstrip("/") + "_sam3d_raw"
    if not os.path.isdir(os.path.join(raw, "masks")):
        cand = glob.glob(scene_dir.rstrip("/") + "*_sam3d_raw")
        raw = cand[0] if cand else raw
    md = os.path.join(raw, "masks")
    if cloth_idx is not None and os.path.exists(os.path.join(md, f"{cloth_idx}.png")):
        return os.path.join(md, f"{cloth_idx}.png")
    # match by label via scene_layout.json (id -> mask index)
    kws = [k.lower() for k in ([cloth_kw] if isinstance(cloth_kw, str) else list(cloth_kw or []))]
    kws += ["shirt", "t-shirt", "tshirt", "cloth", "garment", "fabric", "towel", "tee"]
    layout = os.path.join(scene_dir.rstrip("/"), "scene_layout.json")
    if os.path.exists(layout):
        try:
            objs = json.load(open(layout)).get("objects", [])
            for o in objs:
                lbl = str(o.get("label", "")).lower()
                if any(k in lbl for k in kws):
                    i = o.get("id", o.get("index"))
                    p = os.path.join(md, f"{i}.png")
                    if os.path.exists(p):
                        return p
        except Exception:
            pass
    # last resort: whichever mask is present (prefer a non-0 if only one object is cloth-like)
    return os.path.join(md, "0.png") if os.path.exists(os.path.join(md, "0.png")) else None
