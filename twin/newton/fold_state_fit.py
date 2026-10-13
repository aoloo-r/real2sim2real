"""Initialise the twin from a FOLDED garment: fit the fold state of the tuned shirt to one RGBD.

The cloth is the tuned unisex_shirt.usd - same topology, same material constants, same rest
shape (optionally scaled to a FLAT capture of this garment). The capture only says WHERE that
cloth is and WHAT STATE it is in. Nothing in the capture is turned into geometry: a folded
state is produced by SIMULATING fold primitives from the flat shirt (the same FoldActions the
planner and the twin execute), so both layers, the crease and the layer contact exist and are
at rest. The capture is matched by searching over

    fold sequence  (flat, one primitive, or two)   x   SE(2) placement on the table

and scoring each settled candidate against the capture through the capture's own camera:
silhouette IoU (mask vs projected cloth) first, top-surface height residual second.

    fold_state_fit.py --ts <capture> --out <workdir>          # writes observed.npy (+.robot.json)
    fold_state_fit.py --ts <capture> --probe                  # observed area fraction only
    fold_state_fit.py --ts <capture> --selftest               # synthetic fold -> recover it

Outputs go where the observe stage's do, so fold_tamp (--start) and the twin (TWIN_LOAD_STATE)
continue unchanged. Coordinates are the twin's world frame (cm); the base-frame mapping is the
twin's own (_place_world), inverted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CAPTURES = "/home/aoloo/sam-3d-objects/captures"
OUTPUTS = "/home/aoloo/sam-3d-objects/outputs"

# The asset's flat footprint (cm) in its own axes: x = width, y = length.
ASSET_WH = (55.6, 59.2)
SIM_GROUND_CM = 0.8            # particle radius: a resting sheet's z in fold_env

# Rough area a primitive leaves, as a fraction of flat - only to skip hopeless candidates.
EXPECT_AREA = {"half": 0.5, "panel": 0.67, "sleeve": 0.85, "flat": 1.0}


# ----------------------------------------------------------------------------- frames
class Frames:
    """Capture camera + the twin's world<->robot-base mapping for one capture."""

    def __init__(self, ts: str):
        from twin_stockfold import _place_world, _FWD, _LEFT, _UP, _CLOSER, _CLOSER_LEFT, _TABLE_TOP_Z
        self.ts = ts
        pw = _place_world(ts)
        # the cloth is anchored to the REAL base (world cm); the arm alone is stepped in
        self.base_w = np.asarray(pw["base_true"], float)
        self.base_true = self.base_w.copy()
        self.closer = float(_CLOSER)
        self.A = np.stack([_FWD, _LEFT, _UP], 1)               # world = base_w + A @ (f,l,u)
        self.table_z = float(_TABLE_TOP_Z)
        cap = f"{CAPTURES}/{ts}"
        intr = json.load(open(f"{cap}/intrinsics.json"))
        extr = json.load(open(f"{cap}/extrinsics.json"))
        self.fx, self.fy, self.cx, self.cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        self.W, self.H = int(intr["width"]), int(intr["height"])
        T = np.array(extr["transforms"]["ur5e_base_link"]["T_base_cam"], float)
        self.R, self.t = T[:3, :3], T[:3, 3]

    def world_to_cam(self, P):
        b = ((np.asarray(P, float) - self.base_w) @ self.A) / 100.0      # base frame, m
        return (b - self.t) @ self.R                                       # camera frame, m

    def world_to_pix(self, P):
        c = self.world_to_cam(P)
        z = np.clip(c[:, 2], 1e-3, None)
        return (self.fx * c[:, 0] / z + self.cx, self.fy * c[:, 1] / z + self.cy, c[:, 2])

    def pix_to_world(self, u, v, Z):
        cam = np.stack([(u - self.cx) / self.fx * Z, (v - self.cy) / self.fy * Z, Z], 1)
        b = (cam @ self.R.T + self.t) * 100.0                              # base frame, cm
        return self.base_w[None, :] + b @ self.A.T


# ----------------------------------------------------------------------------- observation
def _grid_area_cm2(xy, cell=1.0):
    """Footprint area of a point cloud on the table: occupied 1cm cells, gaps closed."""
    if len(xy) == 0:
        return 0.0
    lo = xy.min(0) - 2 * cell
    ij = np.floor((xy - lo) / cell).astype(int)
    g = np.zeros(ij.max(0) + 3, bool)
    g[ij[:, 0], ij[:, 1]] = True
    g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=2))
    return float(g.sum()) * cell * cell


def observe(ts: str, fr: Frames, refine: bool = True) -> dict:
    """The capture as a matching target: mask, per-pixel height above the table, footprint."""
    from cloth_mesh import refine_mask_by_color, _largest, garment_mask_path
    cap = f"{CAPTURES}/{ts}"
    mask = np.array(Image.open(garment_mask_path(ts)).convert("L")) > 128   # TWIN_GARMENT-aware
    rgb = np.array(Image.open(f"{cap}/rgb.png").convert("RGB"))
    depth = np.load(f"{cap}/depth.npy").astype(np.float64)
    m = (refine_mask_by_color(mask, rgb) if refine else
         ndimage.binary_fill_holes(_largest(ndimage.binary_opening(mask, iterations=2))))
    valid = depth > 0
    dmed = float(np.median(depth[m & valid])) if (m & valid).any() else 1.0
    dfill = depth.copy()
    dfill[~valid] = dmed
    dfill = ndimage.median_filter(dfill, size=5)
    dfill = np.clip(dfill, dmed - 0.18, dmed + 0.18)
    vv, uu = np.nonzero(m)
    Pw = fr.pix_to_world(uu.astype(float), vv.astype(float), dfill[vv, uu])
    # THE TABLE AS THIS CAMERA SEES IT. The garment's top surface is planar to ~3mm but, on
    # the 2026-08-31 capture, tilted 8.5deg in the twin's world (extrinsic rotation error, or a
    # depth scale error - the twin never noticed, it only uses the centroid). A level sim sheet
    # cannot match the perspective silhouette of a tilted plane; it "fits" 6-8cm too high.
    # So candidates are scored lying IN the observed plane, and heights are measured along its
    # normal. The state handed to the twin is the level one.
    pc, pn = _fit_plane(Pw)
    h = np.full(m.shape, np.nan)
    h[vv, uu] = (Pw - pc) @ pn
    return {"mask": m, "height": h, "xy": Pw[:, :2], "plane_c": pc, "plane_n": pn,
            "plane_tilt_deg": float(np.degrees(np.arccos(np.clip(pn[2], -1, 1)))),
            "centroid": Pw[:, :2].mean(0), "area_cm2": _grid_area_cm2(Pw[:, :2]),
            "rgb": rgb, "dmed": dmed}


def _fit_plane(P):
    """Least-squares plane through points (twice, dropping outliers): centre, unit normal (+z)."""
    P = np.asarray(P, float)
    keep = np.ones(len(P), bool)
    for _ in range(2):
        c = P[keep].mean(0)
        _, _, vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = vt[2] if vt[2][2] > 0 else -vt[2]
        d = (P - c) @ n
        keep = np.abs(d) < max(3.0 * np.std(d[keep]), 0.5)
    return c, n


def _tilt_matrix(n):
    """Rotation taking +z onto the unit normal n."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    sn, cs = np.linalg.norm(v), float(np.dot(z, n))
    if sn < 1e-9:
        return np.eye(3)
    k = v / sn
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sn * K + (1.0 - cs) * (K @ K)


def to_plane(Q, obs, z_bias: float = 0.0):
    """Lay the level state Q into the observed plane: tilt about the observed centroid, then
    slide along the normal so the cloth's top-surface median sits on the plane (+ z_bias)."""
    Rt = obs.get("_Rt")
    if Rt is None:
        Rt = obs["_Rt"] = _tilt_matrix(obs["plane_n"])
    piv = np.array([obs["centroid"][0], obs["centroid"][1], float(np.median(Q[:, 2]))])
    T = (Q - piv) @ Rt.T + piv
    d = (T - obs["plane_c"]) @ obs["plane_n"]
    top = np.percentile(d, 75)                       # the layer the camera sees
    return T + obs["plane_n"][None, :] * (z_bias - top)


# ----------------------------------------------------------------------------- rendering
def _closing_radius(fr: Frames, z_near: float, res: int) -> int:
    """Particle spacing (~0.7cm) in pixels at the cloth's NEAREST depth. The hem is closest
    to the camera, so its particles are the most spread out; a radius sized for the median
    depth left the near half of every silhouette full of holes (visible in the overlay)."""
    px = fr.fx * 0.007 / max(z_near, 0.2) / res
    return int(np.clip(np.ceil(1.0 * px), 1, 8))


def sim_silhouette(P, fr: Frames, dmed: float = 0.0, res: int = 2):
    """Boolean image (H/res, W/res): the cloth's silhouette seen by the capture camera.
    Fast particle splat + closing; used inside the pose optimiser."""
    u, v, z = fr.world_to_pix(P)
    Hs, Ws = fr.H // res, fr.W // res
    ui = np.round(u / res).astype(int)
    vi = np.round(v / res).astype(int)
    ok = (z > 0.05) & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
    img = np.zeros((Hs, Ws), bool)
    img[vi[ok], ui[ok]] = True
    r = _closing_radius(fr, float(np.percentile(z[ok], 5)) if ok.any() else 0.8, res)
    st = np.ones((2 * r + 1, 2 * r + 1), bool)
    img = ndimage.binary_closing(img, structure=st, border_value=0)
    return ndimage.binary_fill_holes(img)


def sim_silhouette_exact(P, tris, fr: Frames, res: int = 1):
    """Exact silhouette: the union of the cloth's projected triangles. Slower (PIL polygon
    fill per triangle), so it scores the FINAL pose of each candidate and draws the overlay."""
    u, v, z = fr.world_to_pix(P)
    Hs, Ws = fr.H // res, fr.W // res
    img = Image.new("1", (Ws, Hs), 0)
    d = ImageDraw.Draw(img)
    uu, vv = u / res, v / res
    front = z > 0.05
    for a, b, c in tris:
        if front[a] and front[b] and front[c]:
            d.polygon([(uu[a], vv[a]), (uu[b], vv[b]), (uu[c], vv[c])], fill=1)
    return np.array(img, dtype=bool)


def sim_height(P, fr: Frames, dmed: float, res: int = 2, plane=None):
    """Top-surface height per pixel (cm along the observed plane normal, or z above the sim
    ground when no plane is given), NaN where no cloth."""
    hp = ((P - plane[0]) @ plane[1]) if plane is not None else (P[:, 2] - SIM_GROUND_CM)
    u, v, z = fr.world_to_pix(P)
    Hs, Ws = fr.H // res, fr.W // res
    ui = np.round(u / res).astype(int)
    vi = np.round(v / res).astype(int)
    ok = (z > 0.05) & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
    top = np.full((Hs, Ws), -np.inf)
    np.maximum.at(top, (vi[ok], ui[ok]), hp[ok])
    r = _closing_radius(fr, float(np.percentile(z[ok], 5)) if ok.any() else 0.8, res)
    top = ndimage.grey_dilation(top, size=(2 * r + 1, 2 * r + 1))
    top[~np.isfinite(top)] = np.nan
    return top


def iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni else 0.0


# ----------------------------------------------------------------------------- pose fit
def fold_stack_for(seq, pose, c, z_table):
    """The fit's fold records for `seq`, moved by the fitted SE(2) pose into the twin's world:
    the stack the twin pops to unfold a garment it did not fold itself."""
    recs = simulate_candidates.records.get(seq, [])
    out = []
    for r in recs:
        r2 = dict(r)
        for k in ("grab", "drop", "sleeve_home"):
            if k in r:
                p = se2(np.array([[r[k][0], r[k][1], 0.0]]), pose["dx"], pose["dy"],
                        np.radians(pose["theta_deg"]), c)[0]
                r2[k] = [float(p[0]), float(p[1])]
        # a fold line stays the mirror line between grab and drop along the fold axis
        r2["line"] = 0.5 * (r2["grab"][r["axis"]] + r2["drop"][r["axis"]])
        out.append(r2)
    return out


def se2(P, dx, dy, th, c):
    """Rigid move on the table: rotate about z through c, then translate."""
    Q = np.array(P, float)
    ct, st = np.cos(th), np.sin(th)
    x, y = Q[:, 0] - c[0], Q[:, 1] - c[1]
    Q[:, 0] = ct * x - st * y + c[0] + dx
    Q[:, 1] = st * x + ct * y + c[1] + dy
    return Q


def _pca_axis(xy):
    d = xy - xy.mean(0)
    _, sv, vt = np.linalg.svd(d, full_matrices=False)
    return vt[0], float(sv[0] / max(sv[1], 1e-9))


def fit_pose(q, obs, fr: Frames, tris=None, res: int = 2, maxfev: int = 90):
    """Best SE(2) placement of the settled candidate `q` against the observation. The
    optimiser runs on the fast splat silhouette; the reported IoU is the exact one."""
    m_ds = obs["mask"][::res, ::res]
    # REGISTER HEIGHT FIRST. fold_env's ground is z=0, the capture's table is wherever the
    # twin's mapping put it (~18-20cm). Through an oblique camera a 20cm height error moves
    # the silhouette 15cm+ across the table, which no SE(2) fit can undo: the flat capture
    # scored IoU 0.52 at a bogus (+15,-15)cm/-71deg pose until the cloth was lifted onto
    # the observed table level.
    q = np.array(q, float)
    c = q[:, :2].mean(0)
    t0 = obs["centroid"] - c
    ax_s, an_s = _pca_axis(q[:, :2])
    ax_o, an_o = _pca_axis(obs["xy"])
    th0 = float(np.arctan2(ax_o[1], ax_o[0]) - np.arctan2(ax_s[1], ax_s[0]))
    inits = [th0, th0 + np.pi]
    if an_s < 1.2 or an_o < 1.2:                    # near-square: the axis is not informative
        inits += [th0 + np.pi / 2, th0 - np.pi / 2]

    def cost(x):
        Q = to_plane(se2(q, x[0], x[1], x[2], c), obs)
        return 1.0 - iou(sim_silhouette(Q, fr, obs["dmed"], res), m_ds)

    best = None
    for th in inits:
        x0 = np.array([t0[0], t0[1], th])
        simplex = np.array([x0, x0 + [4.0, 0, 0], x0 + [0, 4.0, 0], x0 + [0, 0, 0.2]])
        r = optimize.minimize(cost, x0, method="Nelder-Mead",
                              options={"initial_simplex": simplex, "maxfev": maxfev,
                                       "xatol": 0.3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    Q_level = se2(q, x[0], x[1], x[2], c)
    Q = to_plane(Q_level, obs)
    sil = sim_silhouette(Q, fr, obs["dmed"], res)
    iou_exact = iou(sim_silhouette_exact(Q, tris, fr, res), m_ds) if tris is not None else None
    hs = sim_height(Q, fr, obs["dmed"], res, plane=(obs["plane_c"], obs["plane_n"]))
    ho = obs["height"][::res, ::res]
    both = sil & m_ds & np.isfinite(hs) & np.isfinite(ho)
    rmse = float(np.sqrt(np.mean((hs[both] - ho[both]) ** 2))) if both.sum() > 20 else float("nan")
    return {"dx": float(x[0]), "dy": float(x[1]), "theta_deg": float(np.degrees(x[2])),
            "centroid": [float(c[0]), float(c[1])],
            "iou": float(iou_exact if iou_exact is not None else 1.0 - best.fun),
            "iou_splat": 1.0 - float(best.fun), "height_rmse_cm": rmse,
            "state": Q_level, "state_in_plane": Q}


# ----------------------------------------------------------------------------- candidates
def candidate_sequences(depth: int):
    from fold_env import FoldAction
    from fold_search import ACTIONS
    acts = {k: ACTIONS[k] for k in ("half_hem_up", "half_collar_dn", "half_right_in",
                                    "panel_left_in", "panel_right_in",
                                    "sleeve_l_in", "sleeve_r_in")}
    # The planner never needs "fold the LEFT half over" (its plans are canonical), but a
    # person setting up a shirt does it as often as the right - the STATE has to be fittable.
    acts["half_left_in"] = FoldAction(axis=0, frac=0.5, side=-1)
    # a hand taking one sleeve across to the far side of the shirt (2026-09-03 capture 124234)
    acts["sleeve_l_across"] = FoldAction(sleeve="left", travel="across")
    acts["sleeve_r_across"] = FoldAction(sleeve="right", travel="across")
    seqs = [()]
    names = list(acts)
    if depth >= 1:
        seqs += [(n,) for n in names]
    if depth >= 2:
        seqs += [(a, b) for a in names for b in names]
    return acts, seqs


def expected_area(seq) -> float:
    f = 1.0
    for n in seq:
        f *= EXPECT_AREA["half" if n.startswith("half") else
                         "panel" if n.startswith("panel") else "sleeve"]
    return f


def simulate_candidates(env, flat_q, acts, seqs, log=print):
    """Settled canonical (world-0 frame) state per sequence, run in batches of env.n.
    The folds each sequence applied are kept in simulate_candidates.records[seq] (what was
    grabbed where and where it landed) so the fitted state can later be UNFOLDED."""
    from fold_env import FoldAction
    if not hasattr(simulate_candidates, "records"):
        simulate_candidates.records = {}
    out = {}
    per = env.per_world
    for b0 in range(0, len(seqs), env.n):
        batch = seqs[b0:b0 + env.n]
        tiles = []
        for w in range(env.n):
            t = flat_q.copy()
            t[:, 0] += env.world_xy[w][0]
            t[:, 1] += env.world_xy[w][1]
            tiles.append(t)
        env.reset(q=np.concatenate(tiles, 0), randomize=False, settle_s=0.5)
        t0 = time.time()
        for step in range(max(len(s) for s in batch) if batch else 0):
            a = []
            for w in range(env.n):
                if w < len(batch) and step < len(batch[w]):
                    a.append(acts[batch[w][step]])
                else:
                    a.append(FoldAction(side=0))
            env.step(a)
        qs = env.particles()
        for w, s in enumerate(batch):
            q = qs[w].copy()
            q[:, 0] -= env.world_xy[w][0]
            q[:, 1] -= env.world_xy[w][1]
            out[s] = q if np.isfinite(q).all() else None
            simulate_candidates.records[s] = list(env.fold_records[w])
        log(f"[statefit] simulated {min(b0 + env.n, len(seqs))}/{len(seqs)} candidates "
            f"({time.time() - t0:.0f}s): " + ", ".join(" > ".join(s) or "flat" for s in batch))
    # VBD occasionally blows up (NaN) on a fold that runs clean the next time - sleeve_r_in
    # diverged in four consecutive runs here while the identical action validated fine in
    # fold_tamp. A diverged candidate cannot be scored, and if it is the TRUE state the fit is
    # wrong; give each one a second chance in its own batch before writing it off.
    bad = [s for s in seqs if out.get(s) is None]
    if bad and not getattr(simulate_candidates, "_retrying", False):
        simulate_candidates._retrying = True
        try:
            log(f"[statefit] retrying {len(bad)} diverged candidate(s): "
                + ", ".join(" > ".join(s) for s in bad))
            out.update(simulate_candidates(env, flat_q, acts, bad, log=log))
        finally:
            simulate_candidates._retrying = False
    return out


# ----------------------------------------------------------------------------- scale
def flat_scale_from_capture(flat_ts: str) -> tuple[float, float]:
    """Per-axis rest-shape scale (asset x=width, y=length) from a FLAT capture of the garment."""
    from cloth_mesh import cloth_from_mask, garment_mask_path
    fr = Frames(flat_ts)
    mv, _, _, _ = cloth_from_mask(garment_mask_path(flat_ts), f"{CAPTURES}/{flat_ts}", step=4)
    xy = (fr.base_w[None, :] + mv @ fr.A.T)[:, :2]
    ax, _ = _pca_axis(xy)
    P = np.stack([ax, [-ax[1], ax[0]]])
    pr = (xy - xy.mean(0)) @ P.T
    ext = np.percentile(pr, 98, 0) - np.percentile(pr, 2, 0)
    long_, short = float(max(ext)), float(min(ext))
    return (short / ASSET_WH[0], long_ / ASSET_WH[1])


# ----------------------------------------------------------------------------- recording
def attach_recorder(env, path: str, every: int = 4):
    """Record every world's garment as a time-sampled USD (Isaac Sim plays it) - the same
    recording fold_tamp makes of its validation grid, so the fold that produces the fitted
    state can be WATCHED in physics rather than inferred from a score."""
    from pxr import Gf, Usd, UsdGeom, UsdLux
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetTimeCodesPerSecond(env.FPS / every)
    stage.SetStartTimeCode(0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    ta = env.model.tri_indices.numpy().reshape(-1, 3)
    tpw = len(ta) // env.n
    tris0 = ta[:tpw] % env.per_world
    attrs = []
    for i in range(env.n):
        m = UsdGeom.Mesh.Define(stage, f"/World/env_{i}/garment")
        m.CreateFaceVertexIndicesAttr(tris0.flatten().tolist())
        m.CreateFaceVertexCountsAttr([3] * tpw)
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDisplayColorAttr([Gf.Vec3f(0.83, 0.90, 0.28)])
        attrs.append(m.GetPointsAttr())
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(400.0)
    q0 = env.particles().reshape(-1, 3)
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(floor)
    lo, hi = q0.min(0), q0.max(0)
    xf.AddTranslateOp().Set(Gf.Vec3d(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
                                     float(lo[2]) - 5.5))
    xf.AddScaleOp().Set(Gf.Vec3f(float(hi[0] - lo[0]) + 160.0, float(hi[1] - lo[1]) + 160.0, 10.0))
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])
    ctr = {"f": 0, "t": 0}

    def cb(e):
        ctr["f"] += 1
        if ctr["f"] % every:
            return
        qs = e.state_0.particle_q.numpy()
        tc = Usd.TimeCode(ctr["t"])
        for i in range(e.n):
            qq = qs[i * e.per_world:(i + 1) * e.per_world]
            if np.isfinite(qq).all():
                attrs[i].Set([Gf.Vec3f(*map(float, v)) for v in qq], tc)
        ctr["t"] += 1
        if ctr["t"] % 150 == 0:
            stage.SetEndTimeCode(ctr["t"] - 1)
            stage.GetRootLayer().Save()

    def save():
        stage.SetEndTimeCode(max(ctr["t"] - 1, 1))
        stage.GetRootLayer().Save()
        print(f"[statefit] recorded {ctr['t']} frames of {env.n} worlds -> {path} "
              f"(open in Isaac Sim, press PLAY)", flush=True)

    env.frame_cb = cb
    import atexit
    atexit.register(save)
    return save


# ----------------------------------------------------------------------------- outputs
def write_topview(path, Q, tris, cell: float = 0.25):
    """Top-down render of a state: triangles painted back-to-front, shaded by height, so the
    layer order (which sleeve lies on top) is visible - the silhouette cannot show it."""
    Q = np.asarray(Q, float)
    lo = Q[:, :2].min(0) - 3.0
    hi = Q[:, :2].max(0) + 3.0
    W, H = int((hi[0] - lo[0]) / cell), int((hi[1] - lo[1]) / cell)
    img = Image.new("RGB", (W, H), (235, 235, 235))
    d = ImageDraw.Draw(img)
    z = Q[:, 2] - Q[:, 2].min()
    zmax = max(float(np.percentile(z, 99)), 1e-3)
    order = np.argsort(z[tris].mean(1))                     # low first, top layer last
    for t in tris[order]:
        h = float(np.clip(z[t].mean() / zmax, 0, 1))
        col = (int(120 + 120 * h), int(150 + 90 * h), int(40 + 40 * h))
        pts = [((Q[i, 0] - lo[0]) / cell, H - (Q[i, 1] - lo[1]) / cell) for i in t]
        d.polygon(pts, fill=col, outline=(int(col[0] * 0.8), int(col[1] * 0.8), int(col[2] * 0.8)))
    d.text((6, 6), "top view, brighter = higher layer; +x right, +y up", fill=(0, 0, 0))
    img.save(path)


def write_overlay(path, obs, fr: Frames, Q, tris):
    img = Image.fromarray(obs["rgb"]).convert("RGB")
    d = ImageDraw.Draw(img)
    def _edges(mask):
        e = mask & ~ndimage.binary_erosion(mask)
        v, u = np.nonzero(e)
        return list(zip(u.tolist(), v.tolist()))
    for p in _edges(obs["mask"]):
        d.point(p, fill=(40, 255, 40))
    for p in _edges(sim_silhouette_exact(Q, tris, fr, res=1)):
        d.point(p, fill=(255, 40, 40))
    d.text((8, 8), "green = capture mask   red = fitted sim cloth", fill=(255, 255, 255))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True, help="capture id showing the (folded) garment")
    ap.add_argument("--out", default="", help="work dir: observed.npy, .robot.json, report, overlay")
    ap.add_argument("--flat-ts", default="", help="a FLAT capture of the same garment -> rest-shape scale")
    ap.add_argument("--garment", default="",
                    help="which garment of a multi-garment capture: object id or label substring "
                         "(sets TWIN_GARMENT for every stage this script calls into)")
    ap.add_argument("--scale", default="", help="explicit rest-shape scale sx,sy (asset axes)")
    ap.add_argument("--depth", type=int, default=1, help="max primitives in a candidate (0-2)")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--res", type=int, default=2, help="image downsample for scoring")
    ap.add_argument("--all", action="store_true", help="no area pruning of candidates")
    ap.add_argument("--no-refine", action="store_true", help="use the raw SAM mask")
    ap.add_argument("--probe", action="store_true", help="print observed footprint only, no sim")
    ap.add_argument("--selftest", action="store_true",
                    help="fold + move the sim shirt, synthesise its observation, recover it")
    ap.add_argument("--selftest-seq", default="half_right_in")
    ap.add_argument("--selftest-pose", default="6,-4,20", help="dx,dy(cm),theta(deg)")
    ap.add_argument("--seq", default="",
                    help="the fold sequence that was actually performed, e.g. "
                         "panel_right_in,sleeve_r_in: skip the search and fit only the placement. "
                         "Use when the silhouette cannot tell two orders apart (a sleeve folded "
                         "before vs after a panel has the same outline, different top layer)")
    ap.add_argument("--rec-usd", default="",
                    help="record the candidate folds (all worlds) as a time-sampled USD for "
                         "Isaac Sim playback - watch the fold that produced the fitted state")
    a = ap.parse_args()
    if a.garment:
        os.environ["TWIN_GARMENT"] = a.garment      # before twin_stockfold is imported (Frames)

    if a.scale:
        scale = tuple(float(v) for v in a.scale.split(","))[:2]
    elif a.flat_ts:
        scale = flat_scale_from_capture(a.flat_ts)
        print(f"[statefit] rest-shape scale from flat capture {a.flat_ts}: "
              f"({scale[0]:.3f}, {scale[1]:.3f})", flush=True)
    else:
        scale = (1.0, 1.0)
    flat_area = ASSET_WH[0] * ASSET_WH[1] * scale[0] * scale[1] * 0.93   # shirt is not its bbox

    fr = Frames(a.ts)
    obs = None
    if not a.selftest:
        obs = observe(a.ts, fr, refine=not a.no_refine)
        frac = obs["area_cm2"] / flat_area
        print(f"[statefit] capture {a.ts}: footprint {obs['area_cm2']:.0f}cm2 = {frac:.2f} of "
              f"flat ({flat_area:.0f}cm2), surface plane tilt {obs['plane_tilt_deg']:.1f}deg "
              f"(z {obs['plane_c'][2]:.1f}cm at centre), "
              f"centroid ({obs['centroid'][0]:.0f},{obs['centroid'][1]:.0f})", flush=True)
        if a.probe:
            print(f"AREA_FRAC={frac:.3f}")
            return

    from fold_env import FoldEnv
    env = FoldEnv(n_worlds=a.worlds, spacing_cm=130.0, cloth="shirt", scale_xy=scale)
    env.reset()
    env.measure_flat()
    if a.rec_usd:
        attach_recorder(env, a.rec_usd)
        print(f"[statefit] recording candidate folds to {a.rec_usd}", flush=True)
    flat_q = env.particles()[0].copy()
    flat_area = _grid_area_cm2(flat_q[:, :2])           # measured, same estimator as the capture
    _ta = env.model.tri_indices.numpy().reshape(-1, 3)
    tris = _ta[:len(_ta) // env.n] % env.per_world       # world-0 topology, local ids
    acts, seqs = candidate_sequences(a.depth)
    if a.seq:
        forced = tuple(x for x in a.seq.split(",") if x)
        bad = [x for x in forced if x not in acts]
        if bad:
            print(f"[statefit] unknown primitive(s) {bad}; known: {sorted(acts)}", flush=True)
            return
        seqs = [forced]
        a.all = True
        print(f"[statefit] sequence given: {' > '.join(forced) or 'flat'} - fitting placement only",
              flush=True)

    truth = None
    if a.selftest:
        # ground truth: one known sequence, one known placement, observed through the capture's
        # camera with the SAME renderers - a test of the search, not of the camera model
        seq = tuple(s for s in a.selftest_seq.split(",") if s)
        gt = simulate_candidates(env, flat_q, acts, [seq])[seq]
        if gt is None:
            print("[statefit] SELFTEST ABORT: the truth fold diverged twice", flush=True)
            return
        dx, dy, th = (float(v) for v in a.selftest_pose.split(","))
        c = gt[:, :2].mean(0)
        G = se2(gt, dx, dy, np.radians(th), c)
        G[:, 2] += 18.0 - float(G[:, 2].min())            # a table 18cm up, like a capture
        _pc = np.array([0.0, 0.0, float(np.percentile(G[:, 2], 75))])
        _pn = np.array([0.0, 0.0, 1.0])
        mask = sim_silhouette_exact(G, tris, fr, res=1)
        hgt = sim_height(G, fr, 0.8, res=1, plane=(_pc, _pn))
        obs = {"mask": mask, "height": hgt, "xy": G[:, :2], "dmed": 0.8,
               "plane_c": _pc, "plane_n": _pn, "plane_tilt_deg": 0.0,
               "centroid": G[:, :2].mean(0), "area_cm2": _grid_area_cm2(G[:, :2]),
               "rgb": np.zeros((fr.H, fr.W, 3), np.uint8)}
        truth = {"seq": seq, "dx": dx, "dy": dy, "theta_deg": th}
        print(f"[statefit] SELFTEST truth: {' > '.join(seq) or 'flat'} moved "
              f"({dx:+.0f},{dy:+.0f})cm, {th:+.0f}deg", flush=True)

    frac = obs["area_cm2"] / flat_area
    if not a.all:
        keep = [s for s in seqs if 0.55 * expected_area(s) <= frac <= 1.6 * expected_area(s)]
        print(f"[statefit] area fraction {frac:.2f}: {len(keep)}/{len(seqs)} candidates "
              f"plausible", flush=True)
        seqs = keep or seqs
    states = simulate_candidates(env, flat_q, acts, seqs)

    results = []
    t0 = time.time()
    for s in seqs:
        q = states.get(s)
        if q is None:
            results.append({"seq": s, "iou": 0.0, "note": "diverged"})
            print(f"[statefit]   {(' > '.join(s) or 'flat'):34s} DIVERGED (NaN) - not scored",
                  flush=True)
            continue
        r = fit_pose(q, obs, fr, tris=tris, res=a.res)
        r["seq"] = s
        r["sym"] = env.symbolic_state(0, q=r["state"])["label"]
        results.append(r)
        print(f"[statefit]   {(' > '.join(s) or 'flat'):34s} IoU {r['iou']:.3f}  "
              f"height rmse {r['height_rmse_cm']:.1f}cm  pose ({r['dx']:+.0f},{r['dy']:+.0f})cm "
              f"{r['theta_deg']:+.0f}deg  -> {r['sym']}", flush=True)
    print(f"[statefit] pose fits took {time.time() - t0:.0f}s", flush=True)

    # RANK: silhouette first; near-ties (within 0.03 IoU) go to the better height match. A
    # symmetric shirt folded left-over-right and right-over-left has the same silhouette; the
    # height term is what can tell them apart, and when it cannot, either is a valid start.
    good = [r for r in results if "state" in r]
    if not good:
        print("[statefit] every candidate diverged", flush=True)
        return
    top = max(r["iou"] for r in good)
    tied = [r for r in good if r["iou"] >= top - 0.03]
    tied.sort(key=lambda r: (np.nan_to_num(r["height_rmse_cm"], nan=99.0), -r["iou"]))
    best = tied[0]
    print(f"[statefit] BEST: {' > '.join(best['seq']) or 'flat'}  IoU {best['iou']:.3f}  "
          f"height rmse {best['height_rmse_cm']:.1f}cm  state {best['sym']}", flush=True)
    if best["iou"] < 0.75:
        print("[statefit] WARNING: best IoU < 0.75 - the state family does not explain this "
              "capture (deeper --depth, --all, or a fold the primitives cannot make)", flush=True)

    if truth is not None:
        ok = best["seq"] == truth["seq"]
        dpos = np.hypot(best["dx"] - truth["dx"], best["dy"] - truth["dy"])
        dth = (best["theta_deg"] - truth["theta_deg"] + 180.0) % 360.0 - 180.0
        print(f"[statefit] SELFTEST {'PASS' if ok and dpos < 2.0 and abs(dth) < 4.0 else 'FAIL'}: "
              f"sequence {'matches' if ok else 'WRONG'}, pose error {dpos:.1f}cm / {dth:+.1f}deg",
              flush=True)

    # settle the placed state once (a rigid move on a flat table is already at rest; this only
    # lets the solver re-establish contacts) and hand it over at the twin's table height
    tiles = []
    for w in range(env.n):
        t = best["state"].copy()
        t[:, 2] -= float(t[:, 2].min()) - 0.5              # fold_env ground is z=0
        t[:, 0] += env.world_xy[w][0]
        t[:, 1] += env.world_xy[w][1]
        tiles.append(t)
    # When recording, hold the fitted state for a few seconds at the end so the playback loop
    # lingers on the state that matches the capture, not just on the search that found it.
    env.reset(q=np.concatenate(tiles, 0), randomize=False, settle_s=4.0 if a.rec_usd else 1.0)
    final = env.particles()[0].copy()
    final[:, 2] += (fr.table_z + 0.5) - float(final[:, 2].min())
    sym = env.symbolic_state(0, q=final)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        np.save(f"{a.out}/observed.npy", final.astype(np.float32))
        with open(f"{a.out}/observed.npy.robot.json", "w") as fh:
            json.dump({"base_cm": [float(v) for v in fr.base_true],
                       # what the TWIN can execute (its 78cm waypoint clamp + the 10cm base
                       # step-in), never more than the hardware limit - see twin_stockfold dump
                       "reach_cm": min(float(os.environ.get("TWIN_REACH_LIMIT_M", "1.02")) * 100.0,
                                       78.0 + fr.closer),
                       "fwd": [0.0, -1.0, 0.0],
                       "closer_cm": fr.closer,
                       "fit_scale": list(scale)}, fh, indent=2)
        rep = {"capture": a.ts, "scale": list(scale), "area_frac": frac,
               "plane_tilt_deg": obs["plane_tilt_deg"],
               "best": {k: v for k, v in best.items() if not k.startswith("state")},
               "symbolic": sym,
               "candidates": [{k: v for k, v in r.items() if not k.startswith("state")}
                              for r in results]}
        for r in rep["candidates"] + [rep["best"]]:
            r["seq"] = list(r["seq"])
        with open(f"{a.out}/fit_report.json", "w") as fh:
            json.dump(rep, fh, indent=2, default=float)
        write_overlay(f"{a.out}/fit_overlay.png", obs, fr, best["state_in_plane"], tris)
        write_topview(f"{a.out}/fit_topview.png", final, tris)
        stack = fold_stack_for(best["seq"], best, np.array(best["centroid"]), fr.table_z)
        with open(f"{a.out}/fold_stack.json", "w") as fh:
            json.dump({"capture": a.ts, "sequence": list(best["seq"]), "stack": stack}, fh,
                      indent=1)
        print(f"[statefit] fold stack ({len(stack)} fold(s)) -> {a.out}/fold_stack.json - "
              f"the twin can UNFOLD from it (TWIN_UNFOLD_STACK)", flush=True)
        print(f"[statefit] observed state -> {a.out}/observed.npy ({sym['label']}, "
              f"{len(final)} particles); robot frame, report and overlay written", flush=True)


if __name__ == "__main__":
    main()
