"""Shared grasp + place + EE-trajectory geometry — the SINGLE SOURCE OF TRUTH.

Pure numpy/trimesh (NO Isaac dependency), so it is imported by BOTH:
  - the sim twin   (IsaacLab/scripts/real2sim_franka.py  --export_ee_traj)
  - the TAMP motion compiler (sam-3d-objects/tamp_to_ee.py)

This was extracted verbatim from real2sim_franka.export_ee_trajectory (+ its 4
table-plane / OBB / AABB helpers) so the standalone compiler produces trajectories
IDENTICAL to the known-good twin export. `args_cli.*` became an explicit GraspCfg.

compute_pick_place_trajectory(...) returns the EE-trajectory dict (frame, waypoints,
collision_objects, ...) in the ur5e_base_link frame; it does not write any file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class GraspCfg:
    """Grasp/place tuning — defaults MATCH real2sim_franka.py's argparse defaults."""
    grip_max: float = 0.05          # max graspable width for a center grasp (else rim)
    grip_up: float = 0.030          # min grasp height above table
    rim_frac: float = 0.70          # rim-grasp height as fraction of object height
    thin_ratio: float = 2.8         # long/short ratio above which an object is "thin"
    grasp_z_offset: float = 0.0     # manual grasp-height nudge
    lift_clearance: float = 0.18    # travel/lift height above table (approach,lift,move,retreat)
    cal_dx: float = 0.0             # residual calibration translation (base frame, m)
    cal_dy: float = 0.0
    cal_dz: float = 0.0
    no_auto_level: bool = False     # disable table auto-level


# ============================================================================
# Helpers (verbatim from real2sim_franka.py lines 497-664; pure numpy/PIL).
# ============================================================================
def _fit_table_plane_base(capture_dir, T_base_cam, seed=0):
    """RANSAC-fit the dominant (table) plane from the depth image and return it in
    BASE frame: dict with normal·X + d = 0 and a table_z(x,y) closure."""
    import numpy as _np, json as _json, os as _os
    dpath = _os.path.join(capture_dir or "", "depth.npy")
    ipath = _os.path.join(capture_dir or "", "intrinsics.json")
    if not (_os.path.exists(dpath) and _os.path.exists(ipath) and T_base_cam):
        return None
    depth = _np.load(dpath).astype(float)
    intr = _json.load(open(ipath))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ys, xs = _np.where((depth > 0.45) & (depth < 1.25))
    if len(xs) < 500:
        ys, xs = _np.where((depth > 0.2) & (depth < 3.0))   # fallback
    if len(xs) < 500:
        return None
    idx = _np.linspace(0, len(xs) - 1, num=min(8000, len(xs))).astype(int)
    xs, ys = xs[idx], ys[idx]
    z = depth[ys, xs]
    pts_cam = _np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    T = _np.asarray(T_base_cam, float)
    pts = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
    g = _np.random.default_rng(seed)
    N = len(pts)
    best_inl, best = 0, None
    for _ in range(500):
        s = pts[g.integers(0, N, 3)]
        n = _np.cross(s[1] - s[0], s[2] - s[0])
        nn = _np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[2]) < 0.7:                       # keep near-horizontal planes only
            continue
        d = -n.dot(s[0])
        inl = int((_np.abs(pts.dot(n) + d) < 0.01).sum())
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    if best is None:
        return None
    n, d = best
    if n[2] < 0:
        n, d = -n, -d
    inl_mask = _np.abs(pts.dot(n) + d) < 0.01
    centroid = pts[inl_mask].mean(axis=0)
    return {"normal": n, "d": float(d), "inliers": best_inl, "n_pts": N,
            "centroid": centroid,
            "table_z": (lambda x, y: float(-(n[0] * x + n[1] * y + d) / n[2]))}


def _align_rotation(a, b):
    """3x3 rotation R with R @ a = b (a, b need not be unit)."""
    import numpy as _np
    a = _np.asarray(a, float); a = a / _np.linalg.norm(a)
    b = _np.asarray(b, float); b = b / _np.linalg.norm(b)
    v = _np.cross(a, b); c = float(a.dot(b)); s = float(_np.linalg.norm(v))
    if s < 1e-8:
        return _np.eye(3) if c > 0 else -_np.eye(3)
    vx = _np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return _np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _object_obb_level(scene_dir, box_px, depth, intr, T_base_cam, R, cam):
    """PCA footprint of the pick object in the level base frame:
    (cx, cy, phi_long, long_ext, short_ext) or None."""
    import numpy as _np, glob as _glob, os as _os
    try:
        from PIL import Image
    except Exception:
        return None
    mdir = scene_dir.rstrip("/") + "_sam3d_raw/masks"
    if box_px is None or not _os.path.isdir(mdir):
        return None
    bx0, by0, bx1, by1 = box_px
    barea = max(1, (bx1 - bx0) * (by1 - by0))
    best, best_iou = None, 0.0
    for f in _glob.glob(mdir + "/*.png"):
        m = _np.array(Image.open(f).convert("L")) > 127
        ys, xs = _np.where(m)
        if len(xs) < 20:
            continue
        mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max(), ys.max()
        ix0, iy0, ix1, iy1 = max(bx0, mx0), max(by0, my0), min(bx1, mx1), min(by1, my1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        iou = inter / max(1, barea + (mx1 - mx0) * (my1 - my0) - inter)
        if iou > best_iou:
            best_iou, best = iou, m
    if best is None or best_iou < 0.2:
        return None
    fx, fy, cxi, cyi = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ys, xs = _np.where(best)
    z = depth[ys, xs]
    sel = (z > 0.2) & (z < 3.0)
    if sel.sum() < 30:
        return None
    xs, ys, z = xs[sel], ys[sel], z[sel]
    pc = _np.stack([(xs - cxi) * z / fx, (ys - cyi) * z / fy, z], axis=1)
    T = _np.asarray(T_base_cam, float)
    pb = (T[:3, :3] @ pc.T).T + T[:3, 3]
    pl = (R @ (pb - cam).T).T + cam                      # auto-level
    xy = pl[:, :2]
    c = _np.median(xy, axis=0)                           # robust centre
    d = xy - c
    evals, evecs = _np.linalg.eigh(d.T @ d / len(d))     # ascending eigenvalues
    long_v = evecs[:, 1]
    phi = float(_np.arctan2(long_v[1], long_v[0]))
    proj = d @ evecs
    short_ext = float(_np.percentile(proj[:, 0], 98) - _np.percentile(proj[:, 0], 2))
    long_ext = float(_np.percentile(proj[:, 1], 98) - _np.percentile(proj[:, 1], 2))
    if long_ext > 0.45:                                  # implausibly long -> bad mask
        return None
    return float(c[0]), float(c[1]), phi, long_ext, short_ext


def _object_aabb_level(scene_dir, box_px, depth, intr, T_base_cam, R, cam):
    """Leveled axis-aligned bbox of an object: (xmin,xmax,ymin,ymax,zmin,zmax) or None."""
    import numpy as _np, glob as _glob, os as _os
    try:
        from PIL import Image
    except Exception:
        return None
    mdir = scene_dir.rstrip("/") + "_sam3d_raw/masks"
    if box_px is None or not _os.path.isdir(mdir):
        return None
    bx0, by0, bx1, by1 = box_px
    barea = max(1, (bx1 - bx0) * (by1 - by0))
    best, best_iou = None, 0.0
    for f in _glob.glob(mdir + "/*.png"):
        m = _np.array(Image.open(f).convert("L")) > 127
        ys, xs = _np.where(m)
        if len(xs) < 20:
            continue
        mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max(), ys.max()
        ix0, iy0, ix1, iy1 = max(bx0, mx0), max(by0, my0), min(bx1, mx1), min(by1, my1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        iou = inter / max(1, barea + (mx1 - mx0) * (my1 - my0) - inter)
        if iou > best_iou:
            best_iou, best = iou, m
    if best is None or best_iou < 0.2:
        return None
    fx, fy, cxi, cyi = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ys, xs = _np.where(best)
    z = depth[ys, xs]
    sel = (z > 0.2) & (z < 3.0)
    if sel.sum() < 30:
        return None
    xs, ys, z = xs[sel], ys[sel], z[sel]
    pc = _np.stack([(xs - cxi) * z / fx, (ys - cyi) * z / fy, z], axis=1)
    T = _np.asarray(T_base_cam, float)
    pb = (T[:3, :3] @ pc.T).T + T[:3, 3]
    pl = (R @ (pb - cam).T).T + cam                      # auto-level
    lo = _np.percentile(pl, 2, axis=0)                   # robust to depth speckle
    hi = _np.percentile(pl, 98, axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
            float(lo[2]), float(hi[2]))


# ============================================================================
# Main entry — grasp + place + waypoints + collision (returns the EE-traj dict).
# Extracted from real2sim_franka.export_ee_trajectory; relation in/on/beside added.
# With relation="on" + the same inputs it reproduces the twin export byte-for-byte.
# ============================================================================
def compute_pick_place_trajectory(layout, T_base_cam, scene_dir, capture_dir,
                                  pick_label, place_on=None, relation="on",
                                  place_dx=0.0, place_dy=0.0, cfg=None,
                                  verbose=True):
    _np = np
    cfg = cfg or GraspCfg()

    def _log(*a):
        if verbose:
            print(*a)

    objs = layout.get("objects", []) or []
    pick = next((o for o in objs
                 if pick_label.lower() in (o.get("label", "").lower())), None)
    if pick is None and objs:
        pick = max(objs, key=lambda o: (o.get("physical_size_m") or 0.0))
    if pick is None:
        _log("[EE-GEOM] no objects in scene; nothing to compute")
        return None
    di = pick.get("depth_info") or {}
    pc = di.get("position_cam") or (pick.get("icp_pose") or {}).get("position_cam")
    if not pc or not T_base_cam:
        _log("[EE-GEOM] need depth position_cam + camera extrinsic; skipped")
        return None
    P = _np.asarray(T_base_cam, float) @ _np.array([pc[0], pc[1], pc[2], 1.0])
    X, Y, Z = float(P[0]), float(P[1]), float(P[2])
    size = float(pick.get("physical_size_m") or 0.0)
    width = float(di.get("physical_width_m") or size)

    R = _np.eye(3)
    cam = _np.asarray(T_base_cam, float)[:3, 3]
    plane = _fit_table_plane_base(capture_dir, T_base_cam)
    if plane is not None and not cfg.no_auto_level:
        n_obs = plane["normal"]
        tilt_deg = float(_np.degrees(_np.arccos(min(1.0, abs(n_obs[2])))))
        R = _align_rotation(n_obs, _np.array([0.0, 0.0, 1.0]))
        Xc, Yc, Zc = (R @ (_np.array([X, Y, Z]) - cam) + cam).tolist()
        Z_table = float((R @ (plane["centroid"] - cam) + cam)[2])
        _log(f"[EE-GEOM] auto-level: corrected camera tilt {tilt_deg:.2f} deg; "
             f"object ({X:.3f},{Y:.3f},{Z:.3f})->({Xc:.3f},{Yc:.3f},{Zc:.3f}), "
             f"level table z={Z_table:+.3f}m ({plane['inliers']}/{plane['n_pts']} inliers)")
        X, Y, Z = Xc, Yc, Zc
    elif plane is not None:
        Z_table = plane["table_z"](X, Y)
        _log(f"[EE-GEOM] table plane (no auto-level): z_table@object={Z_table:+.3f}m")
    else:
        Z_table = Z - 0.5 * size
        _log(f"[EE-GEOM] table plane fit failed; heuristic z_table={Z_table:+.3f}m")

    X += cfg.cal_dx; Y += cfg.cal_dy; Z += cfg.cal_dz; Z_table += cfg.cal_dz

    height_est = max(2.0 * (Z - Z_table), 0.02)
    DOWNq = [0.0, 1.0, 0.0, 0.0]
    gq = DOWNq

    obb = None
    try:
        import numpy as _np2, json as _json2, os as _os2
        _dp = _os2.path.join(capture_dir or "", "depth.npy")
        _ip = _os2.path.join(capture_dir or "", "intrinsics.json")
        if _os2.path.exists(_dp) and _os2.path.exists(_ip):
            _depth = _np2.load(_dp).astype(float)
            _intr = _json2.load(open(_ip))
            obb = _object_obb_level(scene_dir, pick.get("box_px"),
                                    _depth, _intr, T_base_cam, R, cam)
    except Exception as _e:
        _log(f"[EE-GEOM] OBB skipped: {_e!r}")

    is_thin = (obb is not None and obb[4] <= cfg.grip_max
               and obb[3] / max(obb[4], 1e-3) >= cfg.thin_ratio)
    if is_thin:
        strat = "thin/cross-axis"
        ocx, ocy, phi, long_ext, short_ext = obb
        gx, gy = ocx + cfg.cal_dx, ocy + cfg.cal_dy
        grasp_z = Z_table + max(cfg.grip_up, 0.0)
        theta = phi + _np.pi / 2.0
        gq = [0.0, float(_np.cos(theta / 2.0)), float(_np.sin(theta / 2.0)), 0.0]
        _log(f"[EE-GEOM] THIN object: long={long_ext:.3f}m short={short_ext:.3f}m "
             f"yaw={_np.degrees(theta):.0f}deg (close across short axis)")
    else:
        gwidth = obb[4] if obb is not None else width
        if gwidth <= cfg.grip_max:
            strat = "center"
            gx, gy = X, Y
            grasp_z = max(Z, Z_table + max(cfg.grip_up, 0.0))
        else:
            strat = "side/rim"
            r_rim = 0.5 * (obb[3] if obb is not None else width)
            mesh_h = height_est
            try:
                import trimesh as _tm
                _mp = os.path.join(scene_dir, "object_%d" % pick.get("id"), "mesh.obj")
                _m = _tm.load(_mp, force="mesh", process=False)
                _e = _m.bounds[1] - _m.bounds[0]
                r_rim = 0.5 * float(max(_e[0], _e[1]))
                mesh_h = float(_e[2])
                _log(f"[EE-GEOM] rim grasp from mesh: footprint r={r_rim:.3f}m "
                     f"height={mesh_h:.3f}m")
            except Exception as _e2:
                _log(f"[EE-GEOM] mesh dims load failed ({_e2!r}); using depth OBB")
            gx, gy = X - r_rim, Y
            rim_angle = float(_np.arctan2(gy - Y, gx - X))
            ya = rim_angle - _np.pi / 2.0
            gq = [0.0, float(_np.cos(ya / 2.0)), float(_np.sin(ya / 2.0)), 0.0]
            _log(f"[EE-GEOM] rim straddle: TCP at near wall, fingers close radially "
                 f"(yaw {_np.degrees(ya):.0f}deg)")
            grasp_z = Z_table + cfg.rim_frac * mesh_h
    grasp_z = max(grasp_z + cfg.grasp_z_offset, Z_table + 0.010)

    # PLACE: resolve a target object.
    place_obj = None
    pdx, pdy = place_dx, place_dy
    place_lower_z = grasp_z
    if place_on:
        place_obj = next((o for o in objs
                          if place_on.lower() in (o.get("label", "").lower())
                          and o is not pick), None)
        if place_obj is not None:
            d3 = place_obj.get("depth_info") or {}
            p3 = d3.get("position_cam") or (place_obj.get("icp_pose") or {}).get("position_cam")
            P3 = _np.asarray(T_base_cam, float) @ _np.array([p3[0], p3[1], p3[2], 1.0])
            Pp = (R @ (P3[:3] - cam) + cam) if (plane is not None and not cfg.no_auto_level) else P3[:3]
            ppx, ppy = float(Pp[0]) + cfg.cal_dx, float(Pp[1]) + cfg.cal_dy
            place_h = float(place_obj.get("physical_size_m") or 0.03)
            plate_r = 0.5 * float(place_obj.get("physical_size_m") or 0.15)
            try:
                import trimesh as _tm
                _m = _tm.load(os.path.join(scene_dir, "object_%d" % place_obj.get("id"), "mesh.obj"),
                              force="mesh", process=False)
                _pe = _m.bounds[1] - _m.bounds[0]
                place_h = float(_pe[2])
                plate_r = 0.5 * float(max(_pe[0], _pe[1]))
            except Exception:
                pass
            cup_r = 0.05
            try:
                _cm = _tm.load(os.path.join(scene_dir, "object_%d" % pick.get("id"), "mesh.obj"),
                               force="mesh", process=False)
                _cce = _cm.bounds[1] - _cm.bounds[0]
                cup_r = 0.5 * float(max(_cce[0], _cce[1]))
            except Exception:
                pass
            # AVOID STACKING onto an existing occupant of the target.
            on_plate = []
            for _o in objs:
                if _o is pick or _o is place_obj:
                    continue
                _do = _o.get("depth_info") or {}
                _po = _do.get("position_cam") or (_o.get("icp_pose") or {}).get("position_cam")
                if not _po:
                    continue
                _Po = _np.asarray(T_base_cam, float) @ _np.array([_po[0], _po[1], _po[2], 1.0])
                _Pbo = (R @ (_Po[:3] - cam) + cam) if (plane is not None and not cfg.no_auto_level) else _Po[:3]
                _ox, _oy = float(_Pbo[0]) + cfg.cal_dx, float(_Pbo[1]) + cfg.cal_dy
                if ((_ox - ppx) ** 2 + (_oy - ppy) ** 2) ** 0.5 < plate_r:
                    on_plate.append((_ox, _oy, 0.5 * float(_o.get("physical_size_m") or 0.05)))
            if on_plate:
                _ox, _oy, _or = min(on_plate, key=lambda t: (t[0] - ppx) ** 2 + (t[1] - ppy) ** 2)
                _vx, _vy = ppx - _ox, ppy - _oy
                _vn = (_vx * _vx + _vy * _vy) ** 0.5
                if _vn < 1e-3:
                    _vx, _vy, _vn = 1.0, 0.0, 1.0
                _clr = _or + cup_r + 0.01
                _nx, _ny = _ox + _vx / _vn * _clr, _oy + _vy / _vn * _clr
                _dx, _dy = _nx - ppx, _ny - ppy
                _dn = (_dx * _dx + _dy * _dy) ** 0.5
                _lim = max(0.0, plate_r - cup_r - 0.005)
                if _dn > _lim and _dn > 1e-6:
                    _nx, _ny = ppx + _dx / _dn * _lim, ppy + _dy / _dn * _lim
                _log(f"[EE-GEOM] target occupied (obj@{_ox:.3f},{_oy:.3f} r={_or:.3f}); "
                     f"placing CLEAR at ({_nx:.3f},{_ny:.3f}) instead of centre ({ppx:.3f},{ppy:.3f})")
                ppx, ppy = _nx, _ny
            # relation: "beside" offsets laterally off the target by its radius.
            if relation == "beside":
                ppx = ppx + (plate_r + cup_r + 0.02)
            else:
                # "in"/"on": clamp placement to stay WELL INSIDE the target footprint
                # so a round object isn't set near the rim where it rolls off. Pull the
                # release point to within (target_r - object_r) of the target centre.
                _tcx, _tcy = float(Pp[0]) + cfg.cal_dx, float(Pp[1]) + cfg.cal_dy
                _dx, _dy = ppx - _tcx, ppy - _tcy
                _dn = (_dx * _dx + _dy * _dy) ** 0.5
                _safe = max(0.0, plate_r - cup_r - 0.01)
                if _dn > _safe and _dn > 1e-6:
                    ppx, ppy = _tcx + _dx / _dn * _safe, _tcy + _dy / _dn * _safe
            pdx, pdy = ppx - X, ppy - Y
            if relation == "in":
                # set down GENTLY just above the container's top surface (minimal drop
                # -> minimal bounce/roll), centred. (A disc-shaped bowl can't truly
                # contain, so "in" rests on it low, like a gentle "on".)
                place_lower_z = Z_table + place_h + 0.02
            elif relation == "beside":
                place_lower_z = grasp_z                   # set on the table beside
            else:                                         # "on" (unchanged; cup works)
                place_lower_z = grasp_z + place_h + 0.015
            _log(f"[EE-GEOM] place {relation} '{place_obj.get('label')}' at "
                 f"({ppx:.3f},{ppy:.3f}); lower_z={place_lower_z:.3f}")
        else:
            _log(f"[EE-GEOM] place target '{place_on}' not found; pick only")

    AP_Z = Z_table + cfg.lift_clearance
    HOV_Z = grasp_z + 0.05
    _log(f"[EE-GEOM] strategy={strat} width={width:.3f}m grip_max={cfg.grip_max:.3f}m "
         f"height_est={height_est:.3f}m -> grasp=({gx:.3f},{gy:.3f},{grasp_z:.3f})")
    wps = [
        {"label": "open_pregrasp",  "position": None,             "quaternion": None, "gripper": "open"},
        {"label": "approach_above", "position": [gx, gy, AP_Z],   "quaternion": gq, "gripper": "none"},
        {"label": "hover_object",   "position": [gx, gy, HOV_Z],  "quaternion": gq, "gripper": "none"},
        {"label": "descend_to_grasp", "position": [gx, gy, grasp_z], "quaternion": gq, "gripper": "none"},
        {"label": "close_gripper",  "position": None,             "quaternion": None, "gripper": "close"},
        {"label": "lift",           "position": [gx, gy, AP_Z],   "quaternion": gq, "gripper": "none"},
    ]
    if place_on or place_dx or place_dy:
        wps += [
            {"label": "move_above_place", "position": [gx + pdx, gy + pdy, AP_Z], "quaternion": gq, "gripper": "none"},
            {"label": "lower_to_place",   "position": [gx + pdx, gy + pdy, place_lower_z], "quaternion": gq, "gripper": "none"},
            {"label": "open_gripper",     "position": None, "quaternion": None, "gripper": "open"},
            {"label": "retreat",          "position": [gx + pdx, gy + pdy, AP_Z], "quaternion": gq, "gripper": "none"},
        ]
        _log(f"[EE-GEOM] place at ({gx+pdx:.3f},{gy+pdy:.3f},{place_lower_z:.3f}) "
             f"(offset dX={pdx:+.2f} dY={pdy:+.2f})")
    else:
        wps += [
            {"label": "retreat",       "position": [gx, gy, AP_Z], "quaternion": gq, "gripper": "none"},
            {"label": "open_gripper",  "position": None,           "quaternion": None, "gripper": "open"},
        ]

    coll = [{"name": "table", "type": "box",
             "center": [round(gx, 3), round(gy, 3), round(Z_table - 0.04, 3)],
             "size": [1.2, 1.2, 0.04]}]
    _cdepth = _cintr = None
    try:
        import numpy as _np3, json as _json3, os as _os3
        _dp3 = _os3.path.join(capture_dir or "", "depth.npy")
        _ip3 = _os3.path.join(capture_dir or "", "intrinsics.json")
        if _os3.path.exists(_dp3) and _os3.path.exists(_ip3):
            _cdepth = _np3.load(_dp3).astype(float)
            _cintr = _json3.load(open(_ip3))
    except Exception:
        _cdepth = _cintr = None
    for o in objs:
        if o is pick or o is place_obj:
            continue
        d2 = o.get("depth_info") or {}
        p2 = d2.get("position_cam") or (o.get("icp_pose") or {}).get("position_cam")
        if not p2:
            continue
        P2 = _np.asarray(T_base_cam, float) @ _np.array([p2[0], p2[1], p2[2], 1.0])
        Pb = (R @ (P2[:3] - cam) + cam) if (plane is not None and not cfg.no_auto_level) else P2[:3]
        sz = float(o.get("physical_size_m") or 0.05)
        aabb = None
        if _cdepth is not None:
            try:
                aabb = _object_aabb_level(scene_dir, o.get("box_px"),
                                          _cdepth, _cintr, T_base_cam, R, cam)
            except Exception:
                aabb = None
        if aabb is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = aabb
            sx = max(0.02, xmax - xmin); sy = max(0.02, ymax - ymin)
            sz_box = max(0.02, zmax - Z_table)
            ccx = 0.5 * (xmin + xmax) + cfg.cal_dx
            ccy = 0.5 * (ymin + ymax) + cfg.cal_dy
            ccz = Z_table + 0.5 * sz_box
            coll.append({"name": "obj_%s" % o.get("id"), "type": "box",
                         "center": [round(ccx, 3), round(ccy, 3), round(ccz, 3)],
                         "size": [round(sx, 3), round(sy, 3), round(sz_box, 3)]})
        else:
            coll.append({"name": "obj_%s" % o.get("id"), "type": "box",
                         "center": [round(float(Pb[0]) + cfg.cal_dx, 3),
                                    round(float(Pb[1]) + cfg.cal_dy, 3),
                                    round(Z_table + 0.5 * sz, 3)],
                         "size": [round(sz, 3), round(sz, 3), round(sz, 3)]})
    _log(f"[EE-GEOM] collision geometry: table + {len(coll)-1} object box(es)")

    out = {"frame": (layout.get("camera_extrinsics") or {}).get("frame", "ur5e_base_link"),
           "tcp_orientation": "gripper down (wxyz 0,1,0,0)",
           "gripper": {"open_m": 0.0, "closed_m": 0.025},
           "default_vel_scale": 0.1,
           "pick_label": pick.get("label"),
           "grasp_strategy": strat,
           "table_z_at_object": round(Z_table, 4),
           "pick_position_base": [round(gx, 4), round(gy, 4), round(grasp_z, 4)],
           "object_center_base": [round(X, 4), round(Y, 4), round(Z, 4)],
           "collision_objects": coll,
           "waypoints": wps}
    return out
