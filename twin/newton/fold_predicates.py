"""Symbolic state for folding a labelled sheet: predicates over the garment's named parts, computed
from particle positions. This is what the planner searches over and what the twin audits with, so a
"fold" means the same thing in both. A wad satisfies none of these, however small its footprint.

    on(A, B)        label A lies on top of label B (xy within eps, A above B)
    aligned(E1, E2) two edges coincide: both corner pairs are `on`
    layers(n)       the sheet is n layers deep at its centre of footprint
    flat            the pile is no thicker than its layers allow
    footprint       3-97 percentile extents, for reporting and tie-breaking only
"""
from __future__ import annotations

import numpy as np

# edges of a rectangle, by the corner labels the twin's towel dump uses
EDGES = {
    "near": ("near_left_corner", "near_right_corner"),
    "far": ("far_left_corner", "far_right_corner"),
    "left": ("near_left_corner", "far_left_corner"),
    "right": ("near_right_corner", "far_right_corner"),
}
CORNERS = ("near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner")

# goals as conjunctions of (predicate, args)
GOALS = {
    "half_y": [("on", "near_left_corner", "far_left_corner"), ("on", "near_right_corner", "far_right_corner"),
               ("layers", 2), ("flat",)],
    "half_x": [("on", "near_left_corner", "near_right_corner"), ("on", "far_left_corner", "far_right_corner"),
               ("layers", 2), ("flat",)],
    "quarter": [("on", "near_left_corner", "far_left_corner"), ("on", "near_right_corner", "far_right_corner"),
                ("on", "near_left_corner", "near_right_corner"), ("on", "far_left_corner", "far_right_corner"),
                ("layers", 4), ("flat",)],
}


class SheetPredicates:
    def __init__(self, labels: dict, spacing_cm: float = 0.5, thickness_cm: float = 0.6,
                 eps_cm: float = 3.0, asset_layers: int = 1):
        """labels: {name: {"particle": index, ...}}. spacing: particle spacing of the asset (sets the
        particle count per layer). thickness: one layer's thickness. eps: how close two labels must
        be to count as coincident - the grasp is a 4cm pinch, so 3cm is 'landed on it'.
        asset_layers: 2 for a two-sheet asset (particles per layer doubles)."""
        self.L = {k: int(v["particle"]) for k, v in labels.items()}
        self.spacing = float(spacing_cm)
        self.t = float(thickness_cm)
        self.eps = float(eps_cm)
        self.asset_layers = int(asset_layers)
        self.per_cm2 = self.asset_layers / (self.spacing * self.spacing)

    # ---- primitives --------------------------------------------------------------------------
    def on(self, q, a, b):
        pa, pb = q[self.L[a]], q[self.L[b]]
        return bool(np.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= self.eps and pa[2] >= pb[2] - 0.5 * self.t)

    def coincident(self, q, a, b):
        pa, pb = q[self.L[a]], q[self.L[b]]
        return bool(np.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= self.eps)

    def layers_at(self, q, xy, r_cm: float = 2.0):
        d = np.hypot(q[:, 0] - xy[0], q[:, 1] - xy[1])
        n = int((d <= r_cm).sum())
        return n / (self.per_cm2 * np.pi * r_cm * r_cm)

    def layers(self, q):
        """layer count sampled over the interior of the footprint (median of a 3x3 grid)"""
        lo, hi = np.percentile(q[:, :2], 10, axis=0), np.percentile(q[:, :2], 90, axis=0)
        xs, ys = np.linspace(lo[0], hi[0], 3), np.linspace(lo[1], hi[1], 3)
        vals = [self.layers_at(q, (x, y)) for x in xs for y in ys]
        return float(np.median(vals))

    def thickness(self, q):
        return float(np.percentile(q[:, 2], 97) - np.percentile(q[:, 2], 3))

    def footprint(self, q):
        lo, hi = np.percentile(q[:, :2], 3, axis=0), np.percentile(q[:, :2], 97, axis=0)
        return float(hi[0] - lo[0]), float(hi[1] - lo[1])

    # ---- the state -----------------------------------------------------------------------------
    def state(self, q):
        q = np.asarray(q, float)
        s = {"on": set(), "aligned": set()}
        for a in CORNERS:
            for b in CORNERS:
                if a != b and self.on(q, a, b):
                    # canonical: when both orders hold (flush within tolerance), keep the pair with
                    # the HIGHER label first, so on(A,B) reads "A on top of B"
                    if (b, a) in s["on"]:
                        if q[self.L[a]][2] > q[self.L[b]][2]:
                            s["on"].discard((b, a)); s["on"].add((a, b))
                    else:
                        s["on"].add((a, b))
        for e1, (a1, b1) in EDGES.items():
            for e2, (a2, b2) in EDGES.items():
                if e1 < e2 and ((self.coincident(q, a1, a2) and self.coincident(q, b1, b2))
                                or (self.coincident(q, a1, b2) and self.coincident(q, b1, a2))):
                    s["aligned"].add((e1, e2))
        lay = self.layers(q)
        s["layers"] = int(round(lay)); s["layers_raw"] = lay
        thick = self.thickness(q)
        s["thickness"] = thick
        # flat: no thicker than its layers (rounded up) allow, with one layer of slack for drape
        s["flat"] = bool(thick <= (max(1, int(np.ceil(lay))) + 1) * self.t * self.asset_layers)
        s["footprint"] = self.footprint(q)
        return s

    def holds(self, s, pred):
        if pred[0] == "on":
            return (pred[1], pred[2]) in s["on"] or (pred[2], pred[1]) in s["on"]
        if pred[0] == "aligned":
            return tuple(sorted(pred[1:3])) in s["aligned"]
        if pred[0] == "layers":
            return s["layers"] == pred[1]
        if pred[0] == "flat":
            return s["flat"]
        raise ValueError(pred)

    def score(self, s, goal):
        """(satisfied count, total) for a goal conjunction"""
        preds = GOALS[goal] if isinstance(goal, str) else goal
        return sum(self.holds(s, p) for p in preds), len(preds)

    def unsatisfied(self, s, goal):
        preds = GOALS[goal] if isinstance(goal, str) else goal
        return [p for p in preds if not self.holds(s, p)]

    def suggest(self, s, goal):
        """Regression: every unsatisfied on(A,B) proposes the move A->B; two such moves whose grabs
        are the ends of one edge are offered as ONE two-handed fold."""
        singles = [(p[1], p[2]) for p in self.unsatisfied(s, goal) if p[0] == "on"]
        pairs = []
        for i, (a, b) in enumerate(singles):
            for a2, b2 in singles[i + 1:]:
                if any({a, a2} == set(e) for e in EDGES.values()):
                    pairs.append(((a, b), (a2, b2)))
        return pairs, [((a, b),) for a, b in singles]

    def describe(self, s):
        on = ", ".join(f"on({a.replace('_corner', '')},{b.replace('_corner', '')})" for a, b in sorted(s["on"])) or "none"
        al = ", ".join(f"{e1}={e2}" for e1, e2 in sorted(s["aligned"])) or "none"
        return (f"layers {s['layers']} ({s['layers_raw']:.1f}), thickness {s['thickness']:.1f}cm, "
                f"{'flat' if s['flat'] else 'NOT flat'}, footprint {s['footprint'][0]:.0f}x{s['footprint'][1]:.0f}cm; "
                f"on: {on}; aligned: {al}")


def goal_names():
    return list(GOALS)


# ---- containers: in(x, box) is geometric. box = {"center": (x, y), "half": h, "floor": z0, "rim": z1}
def in_box_fraction(points, box, margin_cm: float = -0.3):
    """fraction of points inside the box's interior volume (xy within the walls, z below the rim)"""
    q = np.asarray(points, float)
    cx, cy = box["center"]; h = float(box["half"]) - margin_cm
    inside = (np.abs(q[:, 0] - cx) <= h) & (np.abs(q[:, 1] - cy) <= h) & (q[:, 2] <= box["rim"] + 0.5) & (q[:, 2] >= box["floor"] - 1.0)
    return float(inside.mean()) if len(q) else 0.0


def block_in_box(center_xyz, dims_cm, box, margin_cm: float = 0.0):
    """a rigid block is in the box when its whole footprint is inside the walls and it sits below the rim"""
    cx, cy = box["center"]; h = float(box["half"]) - margin_cm
    bx, by, bz = [float(v) for v in center_xyz]; dx, dy, dz = [float(v) / 2.0 for v in dims_cm]
    return bool(abs(bx - cx) + dx <= h and abs(by - cy) + dy <= h and bz + dz <= box["rim"] + 0.5 and bz - dz >= box["floor"] - 1.0)


def task1_state(P, q_towel, box, block_center=None, block_dims=None, fold_goal="half_y", towel_in_thresh=0.9):
    """The symbolic state of Task 1: the towel's fold predicates plus the container facts."""
    st = P.state(q_towel)
    sat, tot = P.score(st, fold_goal)
    st["folded"] = bool(sat == tot)
    st["towel_in_box_frac"] = in_box_fraction(q_towel, box)
    st["in_towel_box"] = bool(st["towel_in_box_frac"] >= towel_in_thresh)
    st["in_block_box"] = bool(block_center is not None and block_in_box(block_center, block_dims, box))
    return st


def task1_score(st):
    """goal: folded(towel) & in(towel, box) & in(block, box)"""
    facts = [("folded", st["folded"]), ("in(towel,box)", st["in_towel_box"]), ("in(block,box)", st["in_block_box"])]
    return sum(v for _, v in facts), len(facts), facts
