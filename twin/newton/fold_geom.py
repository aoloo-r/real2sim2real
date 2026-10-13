"""Cheap GEOMETRIC model of the fold operators, and predicates MEASURED on geometry.

The symbolic planner's effects model (fold_tamp.apply_op) is bookkeeping: it believes a half
fold halves the area and a sleeve fold tucks a sleeve, and it never sees layers. The 2026-09-10
ablation showed that once a garment has layers those beliefs go stale: on folded starts the
skeleton planner's precision fell to chance and every blind plan that passed was one the
bookkept preconditions forbid.

This module replaces bookkeeping with geometry that costs microseconds: an operator mirrors the
moving part of the particle cloud across its fold line and stacks it on top; predicates (area,
sleeves, layers, thickness, free edges) are then re-measured on the result. It is still an
abstraction - no drape, no sag, no landing short - but it is an abstraction of the SAME
quantities physics is scored on, so its preconditions and goal test are about the cloth as it
is, not as the operator names suggest.
"""

from __future__ import annotations

import numpy as np

LAYER_CM = 0.8            # one cloth layer's thickness in the stacked model
# Free-edge precondition threshold on the top-layer share of the moving edge band. The asset is
# a TWO-layer garment (front + back panels), so every edge is a seam where half the particles are
# the lower panel: a share test cannot tell a seam from a crease. Disabled (0.0) until the test
# is reformulated; the sleeve preconditions (overhang, tip not buried) remain.
FREE_EDGE_MIN_SHARE = 0.0


class GeomModel:
    def __init__(self, env):
        """`env` is a FoldEnv: supplies the flat reference, landmark indices and sleeve sets.
        Everything is measured in the GARMENT's frame (from its body particles), not the
        world's - a rotated shirt must read the same as an aligned one."""
        self.env = env
        self.flat_area = float(env.FLAT_CM[0] * env.FLAT_CM[1])
        self.cell = 2.0
        # density baseline from the SETTLED flat cloth (world 0 of the current env state is
        # assumed flat right after env.reset(); the caller constructs GeomModel then)
        flat = env.particles()[0]
        self.per_layer = self._cell_q95(flat)
        self.flat_ext = self.frame_extents(flat)

    # ---- garment frame ------------------------------------------------------------
    def frame(self, q):
        """(centre, u, v): u = the body's long axis pointing collar -> hem, v = lateral (left ->
        right of the garment, by the sleeve tips). From the body particles' PCA, so it follows
        the garment through any rotation."""
        env = self.env
        body = q[env.body_idx, :2]
        c = body.mean(0)
        _, _, vt = np.linalg.svd(body - c, full_matrices=False)
        u = vt[0] / np.linalg.norm(vt[0])
        lm = q[env.landmark_idx, :2]                      # sleeve_l, sleeve_r, collar, hem_l, hem_r
        hem = 0.5 * (lm[3] + lm[4])
        if np.dot(hem - lm[2], u) < 0:
            u = -u
        v = np.array([-u[1], u[0]])
        if np.dot(lm[1] - lm[0], v) < 0:
            v = -v
        return c, u, v

    def to_frame(self, q, fr=None):
        c, u, v = fr or self.frame(q)
        d = q[:, :2] - c
        return np.stack([d @ v, d @ u, q[:, 2]], 1)      # x = lateral, y = long, z

    def from_frame(self, p, fr):
        c, u, v = fr
        xy = c + p[:, 0:1] * v[None, :] + p[:, 1:2] * u[None, :]
        return np.concatenate([xy, p[:, 2:3]], 1)

    def frame_extents(self, q):
        p = self.to_frame(q)
        lo, hi = np.percentile(p[:, :2], 3.0, axis=0), np.percentile(p[:, :2], 97.0, axis=0)
        return hi - lo

    # ---- predicates ------------------------------------------------------------
    def _cell_q95(self, q):
        ij = np.floor(q[:, :2] / self.cell).astype(int)
        _, counts = np.unique(ij, axis=0, return_counts=True)
        return float(np.percentile(counts, 95))

    def layers(self, q):
        """Stacked layers, from particle density per 2cm cell against the settled flat cloth."""
        return self._cell_q95(q) / max(self.per_layer, 1e-6)

    def overhang(self, q, fr=None):
        """How far each sleeve tip sticks out past the BODY, laterally, in the garment frame."""
        env = self.env
        p = self.to_frame(q, fr)
        body = p[env.body_idx, 0]
        lo, hi = np.percentile(body, 3.0), np.percentile(body, 97.0)
        tl, tr = p[env.sleeve_tip_idx["left"], 0], p[env.sleeve_tip_idx["right"], 0]
        return (max(0.0, float(lo - tl)), max(0.0, float(tr - hi)))

    def predicates(self, q):
        fr = self.frame(q)
        p = self.to_frame(q, fr)
        lo, hi = np.percentile(p, 3.0, axis=0), np.percentile(p, 97.0, axis=0)
        ext = (hi - lo)[:2]
        area = float(ext[0] * ext[1]) / self.flat_area
        oh = self.overhang(q, fr)
        small, large = sorted(float(v) for v in ext)
        return {
            "area_frac": area,
            "size": ("spread" if area > 0.80 else "half" if area > 0.45 else
                     "quarter" if area > 0.22 else "packed"),
            "sleeves": ("both_in" if oh[0] <= 2.0 and oh[1] <= 2.0 else
                        "left_in" if oh[0] <= 2.0 else "right_in" if oh[1] <= 2.0 else "both_out"),
            "overhang": oh,
            "long_axis": ("square" if abs(ext[0] - ext[1]) < 0.15 * max(ext) else "x" if ext[0] > ext[1] else "y"),
            "aspect": large / max(small, 1e-6),
            "layers": self.layers(q),
            "thickness_cm": float(hi[2] - lo[2]),
            "ext": (float(ext[0]), float(ext[1])),
        }

    # ---- operators -------------------------------------------------------------
    @staticmethod
    def _local_top_share(p, band):
        """Share of `band` particles lying on the top layer, judged against the highest
        particle within 3cm of each (local, so wrinkles and a sloping stack do not matter)."""
        if band.sum() < 10:
            return 0.0
        B = p[band]
        A = p[:, :2]
        share = []
        for b in B[:: max(1, len(B) // 60)]:
            near = np.hypot(A[:, 0] - b[0], A[:, 1] - b[1]) < 3.0
            share.append(b[2] >= p[near, 2].max() - 0.6 * LAYER_CM)
        return float(np.mean(share))

    def free_edge_share(self, q, ax, side, fr=None):
        p = self.to_frame(q, fr)
        lo, hi = np.percentile(p[:, ax], 3.0), np.percentile(p[:, ax], 97.0)
        edge_val = hi if side > 0 else lo
        band = np.abs(p[:, ax] - edge_val) < 3.0
        return self._local_top_share(p, band)

    def apply(self, q, action):
        """Return the cloud after `action` (a FoldAction), or None if the operator does not apply.
        Folds act along the GARMENT's axes: axis 0 = lateral (left/right), 1 = long (collar/hem)."""
        q = np.asarray(q, float)
        env = self.env
        fr = self.frame(q)
        p = self.to_frame(q, fr)
        if action.sleeve is not None:
            k = 0 if action.sleeve == "left" else 1
            oh = self.overhang(q, fr)
            if oh[k] <= 2.0:
                return None                                   # nothing sticking out to fold
            tip_i = env.sleeve_tip_idx[action.sleeve]
            near = np.hypot(p[:, 0] - p[tip_i, 0], p[:, 1] - p[tip_i, 1]) < 3.0
            if p[tip_i, 2] < p[near, 2].max() - 2.0 * LAYER_CM:
                return None                                   # sleeve tip buried under a layer (2 = the flat garment's own thickness)
            body = p[env.body_idx, 0]
            seam = np.percentile(body, 3.0) if k == 0 else np.percentile(body, 97.0)
            cx = float(np.median(body))
            only = np.zeros(len(p), bool); only[env.sleeve_set[action.sleeve]] = True
            moving = only & ((p[:, 0] < seam) if k == 0 else (p[:, 0] > seam))
            if moving.sum() < 20:
                return None
            if getattr(action, "travel", "seam") == "across":
                far = np.percentile(body, 97.0) if k == 0 else np.percentile(body, 3.0)
                line = 0.5 * (float(p[tip_i, 0]) + far)
            else:
                line = seam if k == 0 else cx
            out = p.copy()
            out[moving, 0] = 2.0 * line - p[moving, 0]
            out[moving, 2] = np.percentile(p[~moving, 2], 97.0) + LAYER_CM + (p[moving, 2] - p[moving, 2].min())
            return self.from_frame(out, fr)
        ax = action.axis
        lo, hi = np.percentile(p[:, ax], 3.0), np.percentile(p[:, ax], 97.0)
        line = lo + (hi - lo) * action.frac
        moving = (p[:, ax] > line) if action.side > 0 else (p[:, ax] < line)
        if moving.sum() < 50 or (~moving).sum() < 50:
            return None
        if FREE_EDGE_MIN_SHARE > 0 and self.free_edge_share(q, ax, action.side, fr) < FREE_EDGE_MIN_SHARE:
            return None                                       # the edge to grab is buried
        out = p.copy()
        out[moving, ax] = 2.0 * line - p[moving, ax]
        out[moving, 2] = np.percentile(p[~moving, 2], 97.0) + LAYER_CM + (p[moving, 2] - p[moving, 2].min())
        return self.from_frame(out, fr)


def goal_ok(pred, max_area=0.16, min_area=0.06, max_aspect=1.7, max_layers=9.0):
    """Goal on MEASURED predicates: small but not crushed, square-ish, sleeves in, not a wad."""
    if not (min_area <= pred["area_frac"] <= max_area):
        return False, "area %.0f%%" % (100 * pred["area_frac"])
    if pred["sleeves"] != "both_in":
        return False, "sleeves %s" % pred["sleeves"]
    if pred["aspect"] > max_aspect:
        return False, "aspect %.2f" % pred["aspect"]
    if pred["layers"] > max_layers:
        return False, "%.0f layers" % pred["layers"]
    return True, "ok"


def skeletons_measured(model, q0, actions, max_depth=4, max_area=0.16, limit=60):
    """Enumerate operator sequences whose GEOMETRIC outcome satisfies the measured goal.
    Depth-first over the operator names in `actions` (name -> FoldAction), pruning sequences
    whose operator does not apply to the current cloud. Returns sequences sorted by the
    outcome's area (smallest first), each as (seq, predicates)."""
    found = []
    names = list(actions)

    def rec(q, seq):
        if len(found) >= limit * 4:
            return
        pred = model.predicates(q)
        if seq:
            ok, _ = goal_ok(pred, max_area=max_area)
            if ok:
                found.append((tuple(seq), pred))
                return
        if len(seq) >= max_depth:
            return
        for n in names:
            q2 = model.apply(q, actions[n])
            if q2 is None:
                continue
            rec(q2, seq + [n])

    rec(np.asarray(q0, float), [])
    found.sort(key=lambda t: (len(t[0]), t[1]["area_frac"]))
    return found[:limit]
