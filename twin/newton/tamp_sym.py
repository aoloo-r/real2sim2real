"""Symbolic TAMP for the sled domain: the ORDER of actions is searched, not written.

State = symbolic facts + the geometry they are read from:
    on_towel(o) | in_box(o)      where each object is
    toppled(o)                   an object the pull knocked over (it cannot be top-grasped)
    folds(k), towel_in_box       the towel
    towel_dy, pos(o)             measured geometry: how far the towel has moved, where each rider stands
Derived predicates, evaluated on that geometry when an action is considered:
    reachable(o)        o within R_PICK - margin of the nearer arm's base
    corners_reachable   the corners the next fold grasps are within reach
    clear(towel)        nothing stands on the towel
    fits(k)             the towel folded k times fits the free strip of the box (the box layout)

Operators (two arms; the executor assigns them):
    pick_place(o)   pre on_towel(o), !toppled(o), reachable(o)          eff in_box(o), !on_towel(o)
    pull(d, a)      pre !towel_in_box, folds(0), pulls < MAX_PULLS        eff COMPUTED: a Newton rollout of the
                    (d from what is out of reach, a from ACCELS            pull with the current riders gives the
                     not above the riders' rigid-body limits)              new positions and who toppled
    fold            pre clear(towel), corners_reachable, folds(k), k < 2  eff folds(k+1)
    pack_towel      pre clear(towel), fits(folds), !towel_in_box          eff towel_in_box
Goal: a conjunction from goal_parse.py (in_box(o), folded(towel, k), towel_in_box, before(a, b)),
usually parsed from an instruction in words; without --goal: in_box(o) for every o, and
towel_in_box if the scene asks for it. before(a, b) is a precondition of pick_place(b). Folding or
packing the towel needs it clear, so then every object must come off it, whether the goal names it or not.

Search: A* on estimated execution time. A pull's effects are not known until it is simulated, so a
pull successor enters the queue with OPTIMISTIC effects (nobody topples, everyone moves by d) and
is simulated only when popped (lazy evaluation); the simulated state then re-enters the queue.
A rollout that topples an object leaves a state where that object can never be packed: a dead end,
so the search falls back to the next cheapest alternative - packing that object before the pull,
or a gentler pull. Cheap checks (reach, fit, rigid-body limits) run before any rollout; rollouts
are cached. A pull with nothing on the towel needs no rollout (nothing can topple).

Not in this planner: collision checking of the arm motions (the executor's motions are scripted
waypoints), and the towel is re-simulated flat at its new position for a second pull.

Usage: tamp_sym.py --scene scene.json --out plan.json [--goal goal.json] [--max-rollouts 12]
"""
import argparse
import copy
import heapq
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tamp_sled import (ACCELS, D_MAX, FOLD_SEQS, REACH_MARGIN, T_DRAG_FIXED, T_FOLD, T_PICK, VMAX,  # noqa: E402
                      analytic, choose_fold, drag_time, fold_moves, layout)

MAX_PULLS = 2
T_PACK_TOWEL = 14.0
MAX_FOLDS = len(FOLD_SEQS) - 1


class S:
    """A search state. `pos` holds rider positions (x, y) on the table; in-box objects sit at their slots."""
    __slots__ = ("on_towel", "in_box", "toppled", "folds", "towel_in_box", "towel_dy", "pos", "pulls")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def copy(self, **kw):
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kw)
        return S(**d)

    def key(self):
        return (self.on_towel, self.in_box, self.toppled, self.folds, self.towel_in_box, self.pulls, round(self.towel_dy),
                tuple(sorted((n, round(x), round(y)) for n, (x, y) in self.pos.items())))

    def facts(self):
        return sorted([f"on_towel({o})" for o in self.on_towel] + [f"in_box({o})" for o in self.in_box] +
                      [f"toppled({o})" for o in self.toppled] + [f"folds({self.folds})"] + (["towel_in_box"] if self.towel_in_box else []))


def default_goal(scene):
    return dict(in_box=[o["name"] for o in scene["objects"]], folds_min=0, towel_in_box=bool(scene.get("pack_towel", True)), before=[])


class Domain:
    def __init__(self, scene, goal=None):
        from twin_sled import BASES_XY, R_PICK, BOX_XY
        self.scene, self.BASES, self.R, self.BOX = scene, BASES_XY, R_PICK - REACH_MARGIN, BOX_XY
        self.cx, self.cy = [float(v) for v in scene["towel"]["centre_xy"]]
        self.W, self.L = [float(v) for v in scene["towel"]["size_cm"]]
        self.objs = {o["name"]: o for o in scene["objects"]}
        self.slot, self.strip, self.rows = layout(scene)
        self.goal_spec = goal or default_goal(scene)
        self.goal_objs = frozenset(self.goal_spec["in_box"])
        self.want_towel = bool(self.goal_spec["towel_in_box"])
        self.folds_min = int(self.goal_spec["folds_min"])
        self.touch_towel = self.want_towel or self.folds_min > 0     # then the towel must be cleared of every object
        self.after = {}                                              # before(a, b): b may be packed only once a is
        for a_, b_ in self.goal_spec["before"]:
            self.after.setdefault(b_, set()).add(a_)
        seq, _, self.fold_tried = choose_fold(scene, self.strip)
        self.k_fit = len(seq) if seq is not None else None          # fits(k) holds for k >= k_fit (fewest folds that fit)
        self.limits = analytic(scene)
        self.cache, self.rollouts, self.trace = {}, 0, []

    # ---- derived predicates -------------------------------------------------------------
    def reach(self, xy):
        return min(float(np.hypot(xy[0] - b[0], xy[1] - b[1])) for b in self.BASES.values())

    def reachable(self, xy):
        """The executor's rule (measured reach, no margin); the margin is used when choosing pull distances."""
        return self.reach(xy) <= self.R + REACH_MARGIN

    def y_need(self, x):
        """How far forward (y) a point at this x must come to be reachable by the nearer arm."""
        dx = min(abs(x - b[0]) for b in self.BASES.values())
        return list(self.BASES.values())[0][1] - math.sqrt(max(self.R ** 2 - dx * dx, 0.0))

    def fold_corners(self, st, k):
        """The two corners the k-th fold (1-based) grasps, on the table now."""
        v_far = 1.0 - 0.5 ** (k - 1)
        y = self.cy + st.towel_dy - self.L / 2 + v_far * self.L + 1.0
        return [(self.cx + s * (self.W / 2 - 1.5), y) for s in (-1.0, 1.0)]

    def corners_reachable(self, st, k):
        return all(self.reachable(c) for c in self.fold_corners(st, k))

    def fits(self, k):
        return self.k_fit is not None and k >= self.k_fit

    # ---- operators ----------------------------------------------------------------------
    def actions(self, st):
        """Applicable ground actions: (name, args, cost, successor or None if it needs a rollout)."""
        out = []
        for o in sorted(st.on_towel):
            if o not in st.toppled and self.reachable(st.pos[o]) and self.after.get(o, set()) <= st.in_box:
                nxt = st.copy(on_towel=st.on_towel - {o}, in_box=st.in_box | {o}, pos={n: p for n, p in st.pos.items() if n != o})
                out.append(("pick_place", (o,), T_PICK, nxt))
        if not st.towel_in_box and st.folds == 0 and st.pulls < MAX_PULLS:
            for d, acc in self.pull_options(st):
                out.append(("pull", (d, acc), T_DRAG_FIXED + drag_time(d, acc), None))
        clear = not st.on_towel
        if self.touch_towel and clear and not st.towel_in_box and st.folds < MAX_FOLDS and self.corners_reachable(st, st.folds + 1):
            out.append(("fold", (st.folds + 1,), T_FOLD, st.copy(folds=st.folds + 1)))
        if self.want_towel and clear and not st.towel_in_box and self.fits(st.folds) and st.folds >= self.folds_min:
            out.append(("pack_towel", (st.folds,), T_PACK_TOWEL, st.copy(towel_in_box=True)))
        return out

    def must_pack(self, st):
        """Objects still on the towel that have to go in the box: the goal's, or all of them if the towel is to be touched."""
        return st.on_towel if self.touch_towel else st.on_towel & self.goal_objs

    def pull_options(self, st):
        riders = [o for o in st.on_towel if o not in st.toppled]
        d_obj = max([0.0] + [self.y_need(st.pos[o][0]) - st.pos[o][1] for o in self.must_pack(st) if o not in st.toppled])
        d_fold = 0.0
        if self.touch_towel:
            d_fold = max(0.0, max(self.y_need(c[0]) - c[1] for c in self.fold_corners(st, 1)))
        ds = sorted({round(v, 1) for v in (d_obj, d_fold, max(d_obj, d_fold)) if v > 0.5})
        lim = min([min(self.limits[o].values()) for o in riders] + [1e9])   # cheap check: rigid-body slip / topple limits
        return [(d, a) for d in ds if d <= D_MAX for a in ACCELS if a <= lim]

    def h(self, st):
        """Admissible estimate of the remaining time: one pick per unpacked object, at least one pull
        if a rider or the first fold's corners are out of reach, the folds the goal or the box needs, the towel pack."""
        must = self.must_pack(st)
        t = T_PICK * len(must)
        need_pull = any(not self.reachable(st.pos[o]) for o in must) or \
            (self.touch_towel and not st.towel_in_box and st.folds == 0 and not self.corners_reachable(st, 1))
        if need_pull and st.pulls < MAX_PULLS:                     # a rider or a fold corner is still out of reach
            t += T_DRAG_FIXED + drag_time(1.0, ACCELS[0])
        if self.touch_towel and not st.towel_in_box:
            k_need = max(self.folds_min, (self.k_fit or 0) if self.want_towel else 0)
            t += T_FOLD * max(0, k_need - st.folds) + (T_PACK_TOWEL if self.want_towel else 0.0)
        return t

    def goal(self, st):
        return self.goal_objs <= st.in_box and st.folds >= self.folds_min and (st.towel_in_box or not self.want_towel)

    def dead(self, st):
        return bool(st.toppled & self.must_pack(st))               # a toppled object that must be packed no longer can be

    # ---- the expensive check: simulate the pull -----------------------------------------
    def pull_effect(self, st, d, acc, verbose=True):
        riders = sorted(o for o in st.on_towel if o not in st.toppled)
        if not riders:                                             # nothing to topple or leave behind
            return st.copy(towel_dy=st.towel_dy + d, pulls=st.pulls + 1), dict(sim=False)
        key = (round(st.towel_dy), tuple((o, round(st.pos[o][0]), round(st.pos[o][1])) for o in riders), d, acc)
        if key not in self.cache:
            from twin_sled import SledSim
            sc = copy.deepcopy(self.scene)
            ncy = self.cy + st.towel_dy
            sc["towel"]["centre_xy"] = [self.cx, ncy]
            sc["objects"] = [dict(self.objs[o], on_towel_xy=[st.pos[o][0] - self.cx, st.pos[o][1] - ncy]) for o in riders] + \
                            [self.objs[o] for o in sorted(st.in_box)]
            plan = {"scene": sc, "riders": riders, "preplaced": {o: self.slot[o] for o in st.in_box},
                    "plan": [drag_step(d, acc)]}
            t0 = time.time()
            sim = SledSim(plan, with_arm=False, record=False, verbose=False)
            y0 = float(sim.s0.particle_q.numpy()[:, 1].mean())
            res = sim.run()
            dy = float(sim.s0.particle_q.numpy()[:, 1].mean()) - y0
            self.rollouts += 1
            ob = {o: dict(xy=tuple(res["objects"][o]["xy"]), tilt=res["objects"][o]["tilt_deg"]) for o in riders}
            from twin_sled import TILT_OK
            toppled = sorted(o for o in riders if ob[o]["tilt"] > TILT_OK)
            self.cache[key] = dict(dy=dy, ob=ob, toppled=toppled, ok=bool(res["completed"]), wall_s=round(time.time() - t0, 1))
            r = self.cache[key]
            self.trace.append(dict(rollout=self.rollouts, state=st.facts(), pull_cm=d, accel=acc, riders=riders, toppled=toppled,
                                   towel_moved_cm=round(dy, 1), end={o: [round(v, 1) for v in ob[o]["xy"]] for o in riders}, wall_s=r["wall_s"]))
            if verbose:
                print(f"[sym] rollout {self.rollouts}: pull {d:.0f} cm at {acc:.0f} cm/s2 with {riders} on the towel -> "
                      + (f"TOPPLED {toppled}" if toppled else "all upright") + f"; towel moved {dy:.1f} cm ({r['wall_s']:.0f}s)", flush=True)
        r = self.cache[key]
        nxt = st.copy(towel_dy=st.towel_dy + r["dy"], pulls=st.pulls + 1, toppled=st.toppled | frozenset(r["toppled"]),
                      pos={o: r["ob"][o]["xy"] for o in riders})
        return nxt, dict(sim=True, toppled=r["toppled"])


def drag_step(d, acc):
    return {"op": "drag", "dir": [0.0, 1.0], "dist_cm": round(d, 1), "accel": acc, "vmax": VMAX, "lift_cm": 1.0,
            "grasp_uv": {"left": [0.25, 0.98], "right": [0.75, 0.98]}}


def search(scene, max_rollouts=12, verbose=True, goal=None):
    D = Domain(scene, goal)
    start = S(on_towel=frozenset(D.objs), in_box=frozenset(), toppled=frozenset(), folds=0, towel_in_box=False, towel_dy=0.0,
              pos={n: (D.cx + float(o["on_towel_xy"][0]), D.cy + float(o["on_towel_xy"][1])) for n, o in D.objs.items()}, pulls=0)
    if D.want_towel and D.k_fit is None:
        raise SystemExit(f"[sym] the towel fits the free strip with none of the folds: {D.fold_tried}")
    cnt = itertools.count()
    # queue entries: (f, -g, tie, g, state, path, pending pull or None)
    heap = [(D.h(start), 0.0, next(cnt), 0.0, start, [], None)]
    closed, expanded, pruned = set(), 0, dict(dead_end=0, budget=0)
    while heap:
        f, _, _, g, st, path, pend = heapq.heappop(heap)
        if pend is not None:                                       # lazy: simulate the pull now that it is the cheapest option
            if D.rollouts >= max_rollouts and any(o not in st.toppled for o in st.on_towel):
                pruned["budget"] += 1                              # out of rollouts: this pull cannot be checked
                continue
            d, acc = pend[0], pend[1]
            nxt, info = D.pull_effect(st, d, acc, verbose)
            step = ("pull", (d, acc), info)
            if D.dead(nxt):
                pruned["dead_end"] += 1
                continue
            heapq.heappush(heap, (g + D.h(nxt), -g, next(cnt), g, nxt, path + [step], None))
            continue
        if D.goal(st):
            return D, st, path, dict(expanded=expanded, rollouts=D.rollouts, pruned=pruned, est_s=round(g, 1))
        k = st.key()
        if k in closed:
            continue
        closed.add(k)
        expanded += 1
        for name, args, cost, nxt in D.actions(st):
            g2 = g + cost
            if name == "pull":                                     # optimistic successor: riders move by d, nobody topples
                opt = st.copy(towel_dy=st.towel_dy + args[0], pulls=st.pulls + 1,
                              pos={o: (p[0], p[1] + args[0]) for o, p in st.pos.items()})
                heapq.heappush(heap, (g2 + D.h(opt), -g2, next(cnt), g2, st, path, (args[0], args[1], False)))
            else:
                heapq.heappush(heap, (g2 + D.h(nxt), -g2, next(cnt), g2, nxt, path + [(name, args, {})], None))
    raise SystemExit(f"[sym] no plan: {expanded} states expanded, {D.rollouts} rollouts, pruned {pruned}")


def to_plan(D, path, stats):
    """The executor's plan format (twin_sled.py)."""
    steps, riders, size = [], None, [D.W, D.L]
    on = set(D.objs)
    for name, args, info in path:
        if name == "pick_place":
            steps.append({"op": "pick_place", "name": args[0], "to_xy": D.slot[args[0]]}); on.discard(args[0])
        elif name == "pull":
            if riders is None:
                riders = sorted(on)
            steps.append(drag_step(args[0], args[1]))
        elif name == "fold":
            size = [size[0], size[1] / 2]
            steps.append({"op": "fold", "name": f"far edge onto near edge ({args[0]})", "moves": fold_moves(args[0]), "to_cm": list(size)})
        elif name == "pack_towel":
            ys = D.strip[2] + 0.5 + size[1] / 2
            steps.append({"op": "pack_towel", "grasp_uv": {"left": [0.05, 1.0 - 0.5 * 0.5 ** args[0]], "right": [0.95, 1.0 - 0.5 * 0.5 ** args[0]]},
                          "band_half_cm": 4.0, "to_xy": [D.BOX[0], round(ys, 1)], "clear_cm": 10.0})
            towel_place = dict(rect=[D.BOX[0] - size[0] / 2, D.BOX[0] + size[0] / 2, ys - size[1] / 2, ys + size[1] / 2], size_cm=list(size),
                               folds=args[0], fold_options=D.fold_tried, strip=[round(v, 1) for v in D.strip])
    return {"task": "sled", "planner": "symbolic A* (tamp_sym.py)", "scene": D.scene, "goal": D.goal_spec, "plan": steps, "riders": riders or [],
            "towel_place": towel_place if D.want_towel else None, "rows": D.rows, "est_s": stats["est_s"],
            "search": dict(stats, trace=D.trace, symbolic_path=[f"{n}{tuple(a)}" for n, a, _ in path],
                           analytic={n: {q: round(v, 0) for q, v in d.items()} for n, d in D.limits.items()})}


LABEL = {"pick_place": lambda a: f"pack {a[0]}", "pull": lambda a: f"PULL {a[0]:.0f} cm at {a[1]:.0f} cm/s2",
         "fold": lambda a: f"fold {a[0]}", "pack_towel": lambda a: "pack towel"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rollouts", type=int, default=12)
    ap.add_argument("--goal", help="goal_parse.py output; default: every object in the box (+ the towel if the scene says so)")
    a = ap.parse_args()
    scene = json.load(open(a.scene))
    goal = None
    if a.goal:
        g = json.load(open(a.goal))
        if not g["ok"]:
            raise SystemExit(f"[sym] the goal was rejected by goal_parse.py: errors {g['errors']}, unsupported {g['unsupported']}")
        goal = g["goal"]
        print(f"[sym] goal: {g['predicates']}" + (f"  (from \"{g['text']}\")" if g.get("text") else ""))
    t0 = time.time()
    D, st, path, stats = search(scene, a.max_rollouts, goal=goal)
    stats["plan_wall_s"] = round(time.time() - t0, 1)
    p = to_plan(D, path, stats)
    json.dump(p, open(a.out, "w"), indent=1)
    print(f"[sym] PLAN: " + " > ".join(LABEL[n](a_) for n, a_, _ in path) + f"; est {stats['est_s']:.0f}s")
    print(f"[sym] search: {stats['expanded']} states expanded, {stats['rollouts']} rollouts, pruned {stats['pruned']}, {stats['plan_wall_s']:.0f}s")
    print(f"[sym] final state: {st.facts()}")
    print(f"[sym] wrote {a.out}")
