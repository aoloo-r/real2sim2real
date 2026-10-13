"""Task-1 TAMP: dual-arm planning over RIGID and DEFORMABLE objects in one problem.

    goal:  folded(towel) & in(towel, box) & in(block, box)

Three operator schemas, lifted, with symbolic preconditions and effects (the task layer) and a
geometric grounding (the motion layer) that decides whether a symbolic step can actually be
realised in this scene:

    fold(towel, g)            pre on_table(towel)                       eff folded(towel, g)
                              ground: the label-level fold plan for goal g (label_tamp, physics-validated)
    pack(towel, box, p)       pre folded(towel, g), on_table(towel), fits(footprint(g), box, p)
                              eff in(towel, box), !on_table(towel)
                              ground: a placement p inside the box's free floor area; both hands
    pick_place(block, box, p) pre on_table(block), fits(block, box, p)   eff in(block, box)
                              ground: a placement p on the free floor OR on top of the packed towel

Search is REGRESSION from the goal over the lifted schemas (which gives the causally relevant
skeletons and nothing else), then each skeleton is grounded forward: fold goals are filtered by
whether their folded footprint fits the box (that is where the symbolic choice "quarter, not
half" comes from - the half-folded towel is 40x15 cm and the box is 24 cm square), placements
come from a 1 cm occupancy grid of the box floor, and the fold plan comes from the physics
beam search of label_tamp.py (cached per goal). The twin executes the result:

    TWIN_SCENE=task1 TWIN_LABEL_PLAN=<out.json> TWIN_LABELS=<labels.json> ...

The deformable object has no closed-form pose; the planner carries its footprint (the fold
goal's halving of the measured flat size) and the executed result is audited with the same
predicates (fold_predicates.task1_state), so the plan's claims are checked, not assumed.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_predicates import GOALS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# the footprint each fold goal leaves (fraction of the flat towel along x, y)
FOLD_FOOTPRINT = {"half_y": (1.0, 0.5), "half_x": (0.5, 1.0), "quarter": (0.5, 0.5)}
FOLD_LAYERS = {"half_y": 2, "half_x": 2, "quarter": 4}

# twin stage durations (s of simulated time, from the pose streams in twin_stockfold.py):
# a two-handed label move is ~28 s on the twin clock (measured: t=2 -> t=30 -> complete 56-58);
# pack (two hands) 20 s; pick_place (one hand) 19 s. These are ESTIMATES used to rank plans;
# the execution reports the measured time.
T_FOLD_STEP = 28.0
T_PACK = 20.0
T_PICK_PLACE = 19.0


# ----------------------------------------------------------------------------------------- task layer
class Op:
    def __init__(self, name, params, pre, add, delete, hands):
        self.name, self.params, self.pre, self.add, self.delete, self.hands = name, params, pre, add, delete, hands

    def __repr__(self):
        return f"{self.name}({', '.join(self.params)})"


def lifted_ops(fold_goals):
    """The schemas, grounded over the small object universe (one towel, one block, one box)."""
    ops = []
    for g in fold_goals:
        ops.append(Op("fold", ("towel", g), pre={"on_table(towel)"}, add={f"folded(towel,{g})"}, delete=set(), hands=2))
        ops.append(Op("pack", ("towel", "box", g), pre={f"folded(towel,{g})", "on_table(towel)", f"fits(towel:{g},box)"},
                      add={"in(towel,box)"}, delete={"on_table(towel)"}, hands=2))
    ops.append(Op("pick_place", ("block", "box"), pre={"on_table(block)", "fits(block,box)"}, add={"in(block,box)"},
                  delete={"on_table(block)"}, hands=1))
    return ops


def regress(goal, init, ops, max_len=4):
    """Backward chaining: from the goal set, pick an open fact, apply every operator that adds it
    (and deletes nothing we still need), regress, repeat. Returns skeletons (forward order) whose
    open facts are all in the initial state. `fits(...)` facts are GEOMETRIC: they are left open
    for the grounding stage, which decides them (the task layer cannot).
    """
    out = []

    def rec(open_facts, suffix):
        geo = {f for f in open_facts if f.startswith("fits(")}
        need = open_facts - geo
        if need <= init:
            out.append((list(suffix), geo))
            return
        if len(suffix) >= max_len:
            return
        for f in sorted(need - init):
            for op in ops:
                if f not in op.add or (op.delete & open_facts):
                    continue
                new_open = (open_facts - op.add) | op.pre
                rec(new_open, [op] + suffix)
            break   # regress one open fact at a time (the others are handled deeper); avoids duplicate orders
    rec(frozenset(goal), [])
    # the single-fact rule misses interleavings; add the orderings of independent steps explicitly
    seen, skels = set(), []
    for s, geo in out:
        for perm in itertools.permutations(s):
            if tuple(map(repr, perm)) in seen:
                continue
            if forward_ok(perm, init):
                seen.add(tuple(map(repr, perm))); skels.append((list(perm), geo))
    return skels


def forward_ok(seq, init):
    st = set(init)
    for op in seq:
        if not all(p in st or p.startswith("fits(") for p in op.pre):
            return False
        st = (st - op.delete) | op.add
    return True


# ----------------------------------------------------------------------------------------- motion layer
class BoxFloor:
    """1 cm occupancy of the box floor, plus a height map for stacking."""

    def __init__(self, inner_cm, margin_cm=1.0):
        self.n = int(round(inner_cm)); self.m = margin_cm
        self.h = np.zeros((self.n, self.n), float)      # height of what sits on each floor cell

    def _cells(self, dx, dy, w, d):
        x0 = int(round(self.n / 2 + dx - w / 2)); y0 = int(round(self.n / 2 + dy - d / 2))
        x1 = x0 + int(round(w)); y1 = y0 + int(round(d))
        if x0 < self.m or y0 < self.m or x1 > self.n - self.m or y1 > self.n - self.m:
            return None
        return x0, x1, y0, y1

    def place(self, w, d, height, prefer="floor", max_h=None):
        """find (dx, dy, z_on) for a w x d footprint: on the floor if it fits, else on top of
        what is there (z_on = height under it). Nearest the centre wins."""
        best = None
        for dx in range(-self.n // 2, self.n // 2 + 1):
            for dy in range(-self.n // 2, self.n // 2 + 1):
                c = self._cells(dx, dy, w, d)
                if c is None:
                    continue
                x0, x1, y0, y1 = c
                under = self.h[x0:x1, y0:y1]
                if under.max() <= 1e-6:
                    key = (0, dx * dx + dy * dy)
                    z_on = 0.0
                elif prefer == "any" and np.ptp(under) < 0.5:      # level stack (all cells same height)
                    key = (1, dx * dx + dy * dy)
                    z_on = float(under.max())
                else:
                    continue
                if max_h is not None and z_on + height > max_h:
                    continue
                if best is None or key < best[0]:
                    best = (key, dx, dy, z_on, c)
        if best is None:
            return None
        _, dx, dy, z_on, (x0, x1, y0, y1) = best
        self.h[x0:x1, y0:y1] = z_on + height
        return dx, dy, z_on


def ground(skel, geo, scene, fold_plans):
    """Forward grounding of one skeleton. Returns the executable plan steps, placements and the
    time estimate, or (None, why)."""
    floor = BoxFloor(scene["box_inner_cm"])
    steps, placements, t_dual, t_single = [], {}, 0.0, 0.0
    folded_goal, towel_h = None, None
    for op in skel:
        if op.name == "fold":
            g = op.params[1]
            fp = fold_plans.get(g)
            if fp is None:
                return None, f"no fold plan for goal '{g}'"
            steps += fp["plan"]
            folded_goal = g
            two_hand_steps = sum(1 for s in fp["plan"] if "+" in s)
            t_dual += T_FOLD_STEP * len(fp["plan"])
            t_single += T_FOLD_STEP * (len(fp["plan"]) + two_hand_steps)     # one arm does the pairs in turn
        elif op.name == "pack":
            g = op.params[2]
            if folded_goal != g:
                return None, "pack before fold"
            fx, fy = FOLD_FOOTPRINT[g]
            w, d = scene["towel_cm"][0] * fx, scene["towel_cm"][1] * fy
            towel_h = scene["towel_layer_cm"] * FOLD_LAYERS[g]
            p = floor.place(w + 2.0, d + 2.0, towel_h, prefer="floor", max_h=scene["box_h_cm"])     # +2: fold slack
            if p is None:
                return None, f"folded towel {w:.0f}x{d:.0f}cm does not fit the {scene['box_inner_cm']:.0f}cm box"
            placements["towel"] = {"dx": p[0], "dy": p[1], "z_on": p[2], "footprint_cm": [w, d]}
            steps.append(f"pack:{p[0]:.0f},{p[1]:.0f}")
            t_dual += T_PACK; t_single += T_PACK
        elif op.name == "pick_place":
            bw, bd, bh = scene["block_cm"]
            p = floor.place(bw, bd, bh, prefer="any", max_h=scene["box_h_cm"])
            if p is None:
                return None, "block does not fit the box"
            placements["block"] = {"dx": p[0], "dy": p[1], "z_on": p[2], "footprint_cm": [bw, bd]}
            steps.append(f"pick_place:block:{p[0]:.0f},{p[1]:.0f},{p[2]:.1f}")
            t_dual += T_PICK_PLACE; t_single += T_PICK_PLACE
    return {"steps": steps, "placements": placements, "t_dual_s": t_dual, "t_single_s": t_single,
            "skeleton": [repr(o) for o in skel], "hands": max(o.hands for o in skel)}, ""


# ----------------------------------------------------------------------------------------- fold plans
def fold_plan_for(goal, a):
    """The label-level fold plan for a symbolic fold goal: cached json, else run label_tamp's
    physics beam search (minutes)."""
    f = os.path.join(a.fold_cache, f"plan_{goal}.json")
    if os.path.exists(f):
        return json.load(open(f))
    if not a.search_folds:
        return None
    os.makedirs(a.fold_cache, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "label_tamp.py"), "--labels", a.labels, "--start", a.start,
           "--cloth", "towel", "--goal", goal, "--min-grip", "20", "--band", str(a.band), "--depth", "3",
           "--beam", "3", "--out", f]
    print(f"[task1] searching fold plan for '{goal}': {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, env=dict(os.environ), capture_output=True, text=True)
    (open(os.path.join(a.fold_cache, f"label_tamp_{goal}.log"), "w")).write(r.stdout + r.stderr)
    return json.load(open(f)) if os.path.exists(f) else None


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--start", required=True, help="towel start state .npy (for fold search)")
    ap.add_argument("--fold-cache", required=True, help="dir with plan_<goal>.json (from label_tamp --goal)")
    ap.add_argument("--search-folds", action="store_true", help="run label_tamp for goals without a cached plan")
    ap.add_argument("--band", type=float, default=20.0)
    ap.add_argument("--towel-cm", default="40,30")
    ap.add_argument("--towel-layer-cm", type=float, default=0.65, help="two-layer asset: 0.6 gap + fabric")
    ap.add_argument("--box-inner-cm", type=float, default=24.0)
    ap.add_argument("--box-h-cm", type=float, default=12.0)
    ap.add_argument("--block-cm", default="10,6,4")
    ap.add_argument("--goal", default="folded,in_towel,in_block",
                    help="comma list of: folded, in_towel, in_block")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scene = {"towel_cm": tuple(float(v) for v in a.towel_cm.split(",")), "towel_layer_cm": a.towel_layer_cm,
             "box_inner_cm": a.box_inner_cm, "box_h_cm": a.box_h_cm,
             "block_cm": tuple(float(v) for v in a.block_cm.split(","))}
    init = {"on_table(towel)", "on_table(block)", "flat(towel)"}
    goal_facts = set()
    want = set(a.goal.split(","))
    fold_goals = list(GOALS)
    if "in_towel" in want:
        goal_facts.add("in(towel,box)")
    if "folded" in want and "in_towel" not in want:
        goal_facts |= {f"folded(towel,{g})" for g in ["quarter"]}       # a bare fold goal needs a named fold
    if "in_block" in want:
        goal_facts.add("in(block,box)")
    ops = lifted_ops(fold_goals)
    print(f"[task1] goal {sorted(goal_facts)}; init {sorted(init)}; {len(ops)} grounded operators", flush=True)

    # goal "in(towel,box)" regresses through pack(towel, box, g) for EVERY fold goal g, so the
    # skeleton set contains fold(half_y)+pack, fold(half_x)+pack, fold(quarter)+pack ... and the
    # geometric grounding below is what rejects the halves (their footprints do not fit the box).
    skels = regress(goal_facts, init, ops)
    print(f"[task1] {len(skels)} candidate skeletons from regression", flush=True)
    needed_goals = sorted({o.params[1] for s, _ in skels for o in s if o.name == "fold"})
    fold_plans = {}
    for g in needed_goals:
        # footprint gate BEFORE paying for a fold search: a plan that cannot be packed is useless here
        fx, fy = FOLD_FOOTPRINT[g]
        w, d = scene["towel_cm"][0] * fx + 2.0, scene["towel_cm"][1] * fy + 2.0
        if "in(towel,box)" in goal_facts and (w > scene["box_inner_cm"] - 2 or d > scene["box_inner_cm"] - 2):
            print(f"[task1] fold goal '{g}': folded footprint {w - 2:.0f}x{d - 2:.0f}cm > box {scene['box_inner_cm']:.0f}cm - pruned", flush=True)
            continue
        fp = fold_plan_for(g, a)
        if fp is None:
            print(f"[task1] fold goal '{g}': no fold plan (cache miss, --search-folds not given)", flush=True)
        else:
            fold_plans[g] = fp
            print(f"[task1] fold goal '{g}': {len(fp['plan'])} label steps  {fp['plan']}", flush=True)

    results = []
    for skel, geo in skels:
        res, why = ground(skel, geo, scene, fold_plans)
        tag = " > ".join(repr(o) for o in skel)
        if res is None:
            print(f"[task1]   x {tag}: {why}", flush=True)
            continue
        print(f"[task1]   ok {tag}: {len(res['steps'])} actions, est {res['t_dual_s']:.0f}s dual / {res['t_single_s']:.0f}s single; "
              f"placements {res['placements']}", flush=True)
        results.append(res)
    if not results:
        print("[task1] NO plan", flush=True); sys.exit(2)
    # fewest actions, then least estimated time, then prefer the block on the floor (stable) over on the towel
    results.sort(key=lambda r: (len(r["steps"]), r["t_dual_s"], r["placements"].get("block", {}).get("z_on", 0.0)))
    best = results[0]
    plan = {"garment": "towel", "task": "task1", "goal": sorted(goal_facts), "skeleton": best["skeleton"],
            "plan": best["steps"], "placements": best["placements"], "labels": a.labels,
            "estimate_s": {"dual": best["t_dual_s"], "single_arm": best["t_single_s"]},
            "scene": scene, "alternatives": [{"skeleton": r["skeleton"], "steps": r["steps"], "t_dual_s": r["t_dual_s"]} for r in results[1:]]}
    json.dump(plan, open(a.out, "w"), indent=1)
    print(f"[task1] PLAN ({len(best['steps'])} actions, est {best['t_dual_s']:.0f}s): " + " > ".join(best["steps"]), flush=True)
    print(f"[task1] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
