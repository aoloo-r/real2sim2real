"""Task-2 TAMP: pack SEVERAL deformables and rigid objects into one box, choosing HOW MANY FOLDS
each garment gets so that everything fits.

    goal:  for every garment g:  in(g, box)        (which needs folded(g, level) for some level)
           for every rigid r:    in(r, box)

The task layer is the same regression as tamp_task1 (pack needs folded + fits; pick_place needs
fits). What is new is the GEOMETRIC grounding of fits: a joint choice of one fold level per
garment plus a 3D placement for every object on a 1 cm height-map of the box floor. The fold
level is the number of fold steps (0 none, 1 half, 2 quarter, 3 eighth): each halves a side and
doubles the layers, so fewer folds = fewer actions but more floor; the search prefers the
assignment with the FEWEST fold steps among those where everything fits under the rim, then the
lowest stack. Rigid objects may sit on a level garment stack (as in Task 1) or on each other; a
garment never goes on a rigid object (it would drape, not stack).

Scene file (json):
  {"box": {"inner_cm": [30, 30], "h_cm": 15},
   "garments": [{"name": "towelA", "kind": "towel", "flat_cm": [40, 30], "layer_cm": 0.6,
                 "fold_plans": {"half_y": [...label steps...], "quarter": [...], ...}}, ...],
   "rigids":   [{"name": "block1", "dims_cm": [10, 6, 4]}, ...]}

Fold plans per level come from label_tamp (physics-validated) where they exist; a level whose
plan is synthesised from the level below (one more corner-stack carry) is marked "synth" in the
output so the execution's audit, not the plan, is what certifies it.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# fold level -> (footprint fractions (x, y), layers, steps)  for a rectangular towel
TOWEL_LEVELS = {
    "none":    ((1.0, 1.0), 1, []),
    "half_y":  ((1.0, 0.5), 2, ["far_left_corner->near_left_corner + far_right_corner->near_right_corner"]),
    "half_x":  ((0.5, 1.0), 2, ["near_right_corner->near_left_corner + far_right_corner->far_left_corner"]),
    "quarter": ((0.5, 0.5), 4, ["far_left_corner->near_left_corner + far_right_corner->near_right_corner",
                                "near_left_corner->near_right_corner + far_left_corner->far_right_corner"]),
    "eighth":  ((0.25, 0.5), 8, ["far_left_corner->near_left_corner + far_right_corner->near_right_corner",
                                 "near_left_corner->near_right_corner + far_left_corner->far_right_corner",
                                 "near_right_corner->near_left_corner + far_right_corner->far_left_corner"]),
    # GATE: the bimanual fold for arms standing SIDE BY SIDE. Each hand takes one END of the near
    # edge - the ends are a full garment width apart, one on each arm's own side - and both carry
    # it to the middle; then the same for the far edge. Every carry is parallel to the other, no
    # path crosses, and neither arm ever reaches past the centre line. (The mirror image, both
    # hands on ONE side edge, keeps the TIPS apart but puts both arms in one volume: measured
    # -7.4 cm link overlap, 304 of 1429 samples, exec_s7.)
    "gate":    ((1.0, 0.5), 2, ["near_left_corner->left_edge_midpoint + near_right_corner->right_edge_midpoint",
                                "far_left_corner->left_edge_midpoint + far_right_corner->right_edge_midpoint"]),
    # both hands on one side edge: kept for comparison, never chosen (not in VALIDATED)
    "gate_x":  ((0.5, 1.0), 2, ["near_left_corner->near_edge_midpoint + far_left_corner->far_edge_midpoint",
                                "near_right_corner->near_edge_midpoint + far_right_corner->far_edge_midpoint"]),
}
# nominal label positions on a flat towel (fractions of width x, depth y; y=0 near the robots' far side)
LABEL_XY = {"near_left_corner": (0.0, 0.0), "near_right_corner": (1.0, 0.0), "far_left_corner": (0.0, 1.0), "far_right_corner": (1.0, 1.0),
            "near_edge_midpoint": (0.5, 0.0), "far_edge_midpoint": (0.5, 1.0), "left_edge_midpoint": (0.0, 0.5), "right_edge_midpoint": (1.0, 0.5),
            "near_left_quarter": (0.25, 0.0), "near_right_quarter": (0.75, 0.0), "far_left_quarter": (0.25, 1.0), "far_right_quarter": (0.75, 1.0)}
ARM_SEP_CM, CROSS_REACH_CM = 18.0, 12.0


def step_carries(step, flat_cm):
    """How many carries the twin needs for one plan step: 1 when both hands can do it together
    (grabs a gripper apart, paths not crossing, no drop past the centre line toward the other
    arm's base), else one per pinch (the twin's arm rules serialise it)."""
    moves = [m.strip() for m in step.split("+")]
    if len(moves) < 2:
        return 1
    pts = []
    for m in moves:
        a, b = m.split("->")
        if a not in LABEL_XY or b not in LABEL_XY:
            return len(moves)              # shirt labels etc.: assume serial-safe cost of 1 per pinch is pessimistic
        pts.append((np.array(LABEL_XY[a]) * flat_cm, np.array(LABEL_XY[b]) * flat_cm))
    (g0, d0), (g1, d1) = pts[:2]
    cx = flat_cm[0] / 2.0
    if min(np.linalg.norm(g0 - g1), np.linalg.norm(d0 - d1)) < ARM_SEP_CM:
        return 2
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if orient(g0, d0, g1) * orient(g0, d0, d1) < 0 and orient(g1, d1, g0) * orient(g1, d1, d0) < 0:
        return 2                            # crossing carries
    # hands: the left arm takes the left-most grab; its drop must not go past the centre toward the right
    (gl, dl), (gr, dr) = sorted(pts[:2], key=lambda p: p[0][0])
    if dl[0] > cx + CROSS_REACH_CM or dr[0] < cx - CROSS_REACH_CM:
        return 2
    return 1
# a t-shirt: sleeves+sides in (two-hand step) gives ~0.55 x 1.0 of the flat bbox, then halves
SHIRT_LEVELS = {
    "none":    ((1.0, 1.0), 1, []),
    "sides":   ((0.55, 1.0), 2, ["left_armpit->right_hem_corner + right_armpit->left_hem_corner"]),
    "half":    ((0.55, 0.5), 4, ["left_armpit->right_hem_corner + right_armpit->left_hem_corner",
                                 "left_shoulder->right_shoulder"]),
    "quarter": ((0.3, 0.5), 8, ["left_armpit->right_hem_corner + right_armpit->left_hem_corner",
                                "left_shoulder->right_shoulder",
                                "left_hem_corner->right_hem_corner"]),
}
LEVELS = {"towel": TOWEL_LEVELS, "shirt": SHIRT_LEVELS}
VALIDATED = {("towel", "half_y"), ("towel", "half_x"), ("towel", "quarter"), ("towel", "gate"),
             ("shirt", "sides"), ("shirt", "half")}   # half_x = half_y mirrored; gate = corner carries to the centre line   # physics-searched plans exist
SLACK_CM = 2.0            # a folded bundle is never exactly half: measured 22x13 for a nominal 20x15
T_FOLD_STEP, T_PACK, T_PICK, T_ASIDE = 28.0, 26.0, 19.0, 16.0


class HeightMap:
    """1 cm height-map of the box floor with a margin from the walls."""

    def __init__(self, inner, margin=1.0):
        self.nx, self.ny = int(round(inner[0])), int(round(inner[1])); self.m = margin
        self.h = np.zeros((self.nx, self.ny)); self.who = np.full((self.nx, self.ny), "", dtype=object)

    def place(self, w, d, height, max_h, allow_on=lambda under_name: True, prefer_floor=True, side=None, soft=False, wall_cm=0.0):
        """best (dx, dy, z_on) for a w x d footprint: lowest level first, then nearest the centre;
        the surface under it must be level (same height everywhere) and allowed."""
        W, D = int(round(w)), int(round(d))
        best = None
        _mw = int(round(self.m + wall_cm))                 # extra clearance from the walls for whatever the gripper holds there
        for x0 in range(_mw, self.nx - W - _mw + 1):
            for y0 in range(_mw, self.ny - D - _mw + 1):
                under = self.h[x0:x0 + W, y0:y0 + D]
                z = float(under.max())
                if soft:
                    # a GARMENT on a garment: cloth drapes, so it may overhang - the stack under it
                    # must be level where it is and support >= 70% of the footprint
                    top = under == z
                    if z > 1e-6 and (top.mean() < 0.7 or (under[~top].size and z - under[~top].max() > 3.0 and under[~top].max() > 1e-6)):
                        continue
                elif z - under.min() > 0.5:
                    continue                                  # not level
                names = set(self.who[x0:x0 + W, y0:y0 + D].flatten()) - {""}
                if z > 1e-6 and not all(allow_on(n) for n in names):
                    continue
                if z + height > max_h:
                    continue
                cx, cy = x0 + W / 2 - self.nx / 2, y0 + D / 2 - self.ny / 2
                # ON THE PICKING ARM'S SIDE: an object that starts on the right is carried by the
                # right arm, and a place point on the far left next to the base line made that arm
                # reach across into the other arm (v5: block1 at box (-10,+5), 11 overlapping samples)
                # the picking arm must not reach into the other arm's half: a right-side block
                # placed at box x=-2 made the right upper arm brush the left arm (s9, t=349-352)
                _m = float(os.environ.get("PACK_SIDE_MARGIN_CM", "3"))
                if side == "right" and cx < _m or side == "left" and cx > -_m:
                    continue
                # AWAY FROM THE ROBOTS: +y in the box is the side next to the arm bases; a block
                # placed there (v4: block1 at box +11) made the placing arm fold its forearm down
                # through the other arm's upper arm. Same level and distance -> prefer -y.
                # ... and fill that side from the left so a second block fits beside the first
                # (centring the first one left 12.5 cm for a 12.5 cm block: it went to +y, v5)
                key = (z if prefer_floor else 0.0, 1 if cy > 0.5 else 0, round(cy), cx)
                if best is None or key < best[0]:
                    best = (key, x0, y0, z)
        if best is None:
            return None
        _, x0, y0, z = best
        return x0, y0, z

    def commit(self, name, x0, y0, w, d, z, height):
        W, D = int(round(w)), int(round(d))
        self.h[x0:x0 + W, y0:y0 + D] = z + height
        self.who[x0:x0 + W, y0:y0 + D] = name
        return (x0 + W / 2 - self.nx / 2, y0 + D / 2 - self.ny / 2)     # offset of the centre from the box centre


def flat_of(g, rot):
    """the garment's flat size as it LIES on the table (rot=True: turned 90 deg)"""
    w, h = g["flat_cm"]
    return (h, w) if rot else (w, h)


def footprint(g, level, rot=False):
    (fx, fy), layers, steps = LEVELS[g["kind"]][level]
    fl = flat_of(g, rot)
    w, d = fl[0] * fx + SLACK_CM, fl[1] * fy + SLACK_CM
    return w, d, layers * g.get("layer_cm", 0.6), steps


RIGID_ON_CLOTH = True
DIRECT = True


def try_assignment(scene, levels, order_rigids_first=False, rots=None):
    """Ground one fold-level assignment: pack every object; return (ok, layout, steps, stats)."""
    box = scene["box"]; inner = box["inner_cm"]; H = box["h_cm"]
    hm = HeightMap(inner)
    layout, steps, t = {}, [], 0.0
    rots = rots or {}
    garments = sorted(scene["garments"], key=lambda g: -footprint(g, levels[g["name"]], rots.get(g["name"], False))[0]
                      * footprint(g, levels[g["name"]], rots.get(g["name"], False))[1])
    rigids = sorted(scene["rigids"], key=lambda r: -r["dims_cm"][0] * r["dims_cm"][1])
    is_rigid = {r["name"] for r in rigids}

    pack_steps = {}
    fold_steps = {}

    def do_garment(g):
        """place the garment in the box (packing order); its fold steps are emitted in PILE order below"""
        nonlocal t
        lv = levels[g["name"]]; rot = rots.get(g["name"], False)
        w, d, thick, fsteps = footprint(g, lv, rot)
        # a garment stacks only on the floor or on another garment (a level stack), never on a rigid
        p = hm.place(w, d, thick, H, allow_on=lambda n: n not in is_rigid, soft=True)
        if p is None:
            return f"{g['name']} folded '{lv}' ({w:.0f}x{d:.0f}cm) does not fit"
        x0, y0, z = p
        c = hm.commit(g["name"], x0, y0, w, d, z, thick)
        layout[g["name"]] = {"level": lv, "rot90": bool(rot), "flat_cm": list(flat_of(g, rot)),
                             "footprint_cm": [w, d], "thick_cm": thick, "dx": c[0], "dy": c[1], "z_on": z,
                             "validated": (g["kind"], lv) in VALIDATED or lv == "none"}
        pack_steps[g["name"]] = f"pack:{g['name']}:{c[0]:.0f},{c[1]:.0f}"
        # FOLD TOWARD THE SIDE IT WILL BE PACKED ON. A towel's sideways fold leaves the bundle on
        # one side of the table; folded to the right and packed on the LEFT of the box, the right
        # arm carried it across into the left arm's shoulder (5 cm, exec_s32). Mirroring the fold
        # leaves the bundle with the arm on the target's side.
        if g["kind"] == "towel" and c[0] < -2.0:
            fold_steps[g["name"]] = [s_.replace("left", "#").replace("right", "left").replace("#", "right") for s_ in fsteps]
            layout[g["name"]]["mirrored_fold"] = True
        t += T_FOLD_STEP * sum(step_carries(s_, flat_of(g, rot)) for s_ in fsteps) + T_PACK
        return None

    def do_rigid(r):
        nonlocal t
        bw, bd, bh = r["dims_cm"]
        best = None
        for (w, d) in ((bw, bd), (bd, bw)):                  # may be placed rotated 90 deg
            # 2 cm clearance around a rigid: the block is set down kinematically and a 0.5 cm gap
            # let its sweep crush the towel's edge against the floor (v5 NaN at the last place)
            # a block is set down by a gripper ~7 cm wide: keep it GRIPPER_WALL_CM from the walls or
            # the fingers end up inside them (69% of a finger link in a wall, exec_s31)
            p = hm.place(w + 2.0, d + 2.0, bh, H, allow_on=lambda n: RIGID_ON_CLOTH or n in is_rigid, side=r.get("side"),
                         wall_cm=float(os.environ.get("GRIPPER_WALL_CM", "4")))
            if p is not None and (best is None or p[2] < best[0][2]):
                best = (p, w, d)
        if best is None:
            return f"{r['name']} {bw}x{bd}x{bh} does not fit"
        (x0, y0, z), w, d = best
        c = hm.commit(r["name"], x0, y0, w + 2.0, d + 2.0, z, bh)
        layout[r["name"]] = {"dims_cm": [w, d, bh], "rot90": w != bw, "dx": c[0], "dy": c[1], "z_on": z}
        steps.append(f"pick_place:{r['name']}:{c[0]:.0f},{c[1]:.0f},{z:.1f}" + (":rot90" if w != bw else ""))
        t += T_PICK
        return None

    # clear(g): a garment that lies ON another (scene "on") must be folded and packed before the
    # one under it - the top of the pile first, whatever its size
    under = {g["name"]: g.get("on") for g in scene["garments"]}
    def depth(n):
        d = 0
        while under.get(n):
            n = under[n]; d += 1
        return d
    # DIRECT (default): each garment goes INTO THE BOX as soon as it is folded - no staging on
    # the table. The packing order is then the pile order (top of the pile lands first, on the
    # floor), and the layout below is built in that order. With DIRECT off the packing order is
    # biggest-first and garments folded early wait at a set-aside spot.
    if DIRECT:
        garments = sorted(scene["garments"], key=lambda g: -depth(g["name"]))
    seq = (rigids + garments) if order_rigids_first else (garments + rigids)
    rigid_steps_before = []
    for o in seq:
        if o["name"] in is_rigid:
            n0 = len(steps)
            why = do_rigid(o)
            if order_rigids_first:
                rigid_steps_before += steps[n0:]; del steps[n0:]
        else:
            why = do_garment(o)
        if why:
            return False, why, None, None
    # EXECUTION ORDER. Folds happen on the table in PILE order (top garment first: clear(g)).
    # A garment folded while another still lies under it cannot wait on the pile, so it is SET
    # ASIDE at a staging spot (one-hand bundle carry) and packed when its packing turn comes.
    pile_order = sorted(scene["garments"], key=lambda g: -depth(g["name"]))
    pack_order = [g for g in garments]                       # the order do_garment packed them
    g_steps = []
    packed = []
    for i, g in enumerate(pile_order):
        for s_ in fold_steps.get(g["name"], LEVELS[g["kind"]][levels[g["name"]]][2]):
            g_steps.append(f"{g['name']}: {s_}")
        still_under = [h for h in pile_order[i + 1:]]        # garments below it on the pile
        if DIRECT:
            g_steps.append(pack_steps[g["name"]]); packed.append(g["name"])
        elif still_under:
            g_steps.append(f"aside:{g['name']}"); t += T_ASIDE
            layout[g["name"]]["aside"] = True
        else:
            # the pile is clear: pack everything that is folded, in packing order
            for h in pack_order:
                if h["name"] not in packed:
                    g_steps.append(pack_steps[h["name"]]); packed.append(h["name"])
    for h in pack_order:
        if h["name"] not in packed:
            g_steps.append(pack_steps[h["name"]]); packed.append(h["name"])
    steps[:] = rigid_steps_before + g_steps + steps
    _fl = lambda g: flat_of(g, rots.get(g["name"], False))
    stats = {"fold_steps": sum(len(LEVELS[g["kind"]][levels[g["name"]]][2]) for g in scene["garments"]),
             "fold_carries": sum(step_carries(s_, _fl(g)) for g in scene["garments"] for s_ in LEVELS[g["kind"]][levels[g["name"]]][2]),
             "serial_steps": sum(1 for g in scene["garments"] for s_ in LEVELS[g["kind"]][levels[g["name"]]][2] if step_carries(s_, _fl(g)) > 1),
             "asides": sum(1 for s_ in steps if s_.startswith("aside:")),
             "actions": len(steps), "est_s": t, "max_h": float(hm.h.max()),
             "floor_used": float((hm.h > 0).mean()), "unvalidated": [n for n, L in layout.items() if "validated" in L and not L["validated"]]}
    return True, layout, steps, stats


def render_layout(scene, layout, path):
    """A top-view diagram of the packing (PNG via matplotlib if available, else skipped)."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    inner = scene["box"]["inner_cm"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(Rectangle((-inner[0] / 2, -inner[1] / 2), inner[0], inner[1], fill=False, lw=2, color="#8a6d3b"))
    for n, L in layout.items():
        if "level" in L:
            w, d = L["footprint_cm"]; col = "#9fb7d9"; txt = f"{n}\n{L['level']} ({L['thick_cm']:.1f}cm)"
        else:
            w, d = L["dims_cm"][:2]; col = "#e0908a"; txt = f"{n}\nz={L['z_on']:.1f}"
        ax.add_patch(Rectangle((L["dx"] - w / 2, L["dy"] - d / 2), w, d, color=col, alpha=0.75 if "level" in L else 0.9, ec="k"))
        ax.text(L["dx"], L["dy"], txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(-inner[0] / 2 - 2, inner[0] / 2 + 2); ax.set_ylim(-inner[1] / 2 - 2, inner[1] / 2 + 2)
    ax.set_aspect("equal"); ax.set_title(f"box {inner[0]:.0f}x{inner[1]:.0f}x{scene['box']['h_cm']:.0f} cm - top view")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-synth", action="store_true", help="allow fold levels without a physics-searched plan (marked)")
    ap.add_argument("--direct", type=int, default=1, help="1: pack each garment straight after folding it (no set-aside)")
    ap.add_argument("--rigid-on-cloth", type=int, default=1,
                    help="1: a rigid may sit on a packed garment (kinematic blocks); 0: only on the floor or other rigids "
                         "(dynamic blocks - the cloth-rigid coupling is one-way, cloth does not support a block)")
    a = ap.parse_args()
    global RIGID_ON_CLOTH, DIRECT
    RIGID_ON_CLOTH = bool(a.rigid_on_cloth)
    DIRECT = bool(a.direct)
    scene = json.load(open(a.scene))
    names = [g["name"] for g in scene["garments"]]
    level_sets = []
    for g in scene["garments"]:
        lv = [l for l in LEVELS[g["kind"]] if a.allow_synth or l == "none" or (g["kind"], l) in VALIDATED]
        # "min_level": this garment's OWN folding pattern - a t-shirt is folded sides-in then in half, a
        # bath towel in quarters, a hand towel once - whatever the box would allow. Levels with fewer
        # fold steps than the named one are not offered.
        if g.get("level"):
            lv = [g["level"]]                     # this garment's fold is prescribed (e.g. a bath towel "in four")
        if g.get("min_level"):
            need = len(LEVELS[g["kind"]][g["min_level"]][2])
            lv = [l for l in lv if len(LEVELS[g["kind"]][l][2]) >= need]
        level_sets.append(lv)
    print(f"[pack] box {scene['box']['inner_cm']}x{scene['box']['h_cm']}cm; {len(names)} garments, {len(scene['rigids'])} rigids; "
          f"fold options {dict(zip(names, level_sets))}", flush=True)
    results = []
    rot_sets = [[False, True] if g["kind"] == "towel" else [False] for g in scene["garments"]]
    for combo in itertools.product(*level_sets):
        levels = dict(zip(names, combo))
        for rcombo in itertools.product(*rot_sets):
          rots = dict(zip(names, rcombo))
          for rf in (False, True):
            ok, lay, steps, st = try_assignment(scene, levels, order_rigids_first=rf, rots=rots)
            tag = ", ".join(f"{n}={l}" + ("/rot" if rots[n] else "") for n, l in levels.items()) + (" [rigids first]" if rf else "")
            if not ok:
                continue
            print(f"[pack]   ok {tag}: {st['fold_steps']} fold steps ({st['fold_carries']} carries, {st['serial_steps']} one-armed), "
                  f"{st['actions']} actions, stack {st['max_h']:.1f}cm, floor {st['floor_used']:.0%}", flush=True)
            results.append((levels, rf, lay, steps, st, rots))
    if not results:
        print("[pack] NO assignment fits - a bigger box or an extra fold level (--allow-synth)", flush=True); sys.exit(2)
    # fewest fold steps, then fewest actions, then lowest stack, then validated plans first
    # fewest CARRIES (bimanual steps count once, steps the arm rules serialise count twice), then
    # fewest actions, then the lowest stack
    # ... and among equal carries, the plan with the fewest steps the arms must do ONE-HANDED:
    # a two-step gate fold (both arms, every step) beats a one-step half fold done serially
    # FEWEST ONE-ARMED STEPS FIRST: a plan the two arms can share throughout is the point of a
    # bimanual cell, even when it costs an extra fold step. Then carries (time), actions, stack.
    results.sort(key=lambda r: (r[4]["serial_steps"], r[4]["fold_carries"], r[4]["actions"], r[4]["max_h"], len(r[4]["unvalidated"])))
    levels, rf, lay, steps, st, rots = results[0]
    plan = {"task": "pack_many", "box": scene["box"], "levels": levels, "rot90": rots, "rigids_first": rf, "layout": lay,
            "plan": steps, "stats": st, "scene": scene,
            "alternatives": [{"levels": r[0], "rot90": r[5], "rigids_first": r[1], "stats": r[4]} for r in results[1:6]]}
    json.dump(plan, open(a.out, "w"), indent=1)
    png = render_layout(scene, lay, a.out[:-5] + "_layout.png" if a.out.endswith(".json") else a.out + "_layout.png")
    print(f"[pack] garment orientation: " + ", ".join(f"{n}={'turned 90deg' if r else 'as given'}" for n, r in rots.items()), flush=True)
    print(f"[pack] PLAN: fold levels {levels}; {st['fold_steps']} fold steps ({st['serial_steps']} one-armed), {st['actions']} actions, est {st['est_s']:.0f}s, "
          f"stack {st['max_h']:.1f}/{scene['box']['h_cm']}cm" + (f"; unvalidated fold plans: {st['unvalidated']}" if st["unvalidated"] else ""), flush=True)
    for s_ in steps:
        print("   ", s_)
    print(f"[pack] wrote {a.out}" + (f" and {png}" if png else ""), flush=True)


if __name__ == "__main__":
    main()
