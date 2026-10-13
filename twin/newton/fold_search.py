"""Beam search over fold sequences, with the simulator as a black box.

No model of what a fold does — every candidate is rolled out in `fold_env` and judged
on the resulting garment. The batched env evaluates several (state, action) pairs at
once, one per world, which is what makes search affordable.

    python fold_search.py --beam 2 --depth 3 --worlds 8

State is the pair the planner reasons over: the garment extent and the landmark-based
predicates (are the sleeves in?). Cost is compactness plus a penalty for sleeves left
sticking out, so a plan that merely shrinks the bounding box by crumpling scores worse
than one that folds the sleeves in first.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from fold_env import FoldEnv, FoldAction


# The predefined action set: five folds a garment plausibly admits.
ACTIONS: dict[str, FoldAction] = {
    "half_hem_up":    FoldAction(axis=1, frac=0.5, side=+1),
    "half_collar_dn": FoldAction(axis=1, frac=0.5, side=-1),
    "half_right_in":  FoldAction(axis=0, frac=0.5, side=+1),
    "panel_left_in":  FoldAction(axis=0, frac=1.0 / 3.0, side=-1),
    "panel_right_in": FoldAction(axis=0, frac=2.0 / 3.0, side=+1),
    # Split side folds: same fold line, grasped high and low so both ends come over. One
    # mid-edge grasp carries only the middle - measured 55cm -> 38cm where a complete half
    # fold gives 27.5cm, because the far ends of a 60cm edge lag and drag on the table.
    "right_in_top":   FoldAction(axis=0, frac=0.5, side=+1, grab_at=0.75),
    "right_in_bot":   FoldAction(axis=0, frac=0.5, side=+1, grab_at=0.25),
    "left_panel_top": FoldAction(axis=0, frac=1.0 / 3.0, side=-1, grab_at=0.75),
    "left_panel_bot": FoldAction(axis=0, frac=1.0 / 3.0, side=-1, grab_at=0.25),
    # Length folds split the same way - a mid grab took 54cm -> 41cm against an ideal 27cm.
    "collar_dn_l":    FoldAction(axis=1, frac=0.5, side=-1, grab_at=0.25),
    "collar_dn_r":    FoldAction(axis=1, frac=0.5, side=-1, grab_at=0.75),
    "hem_up_l":       FoldAction(axis=1, frac=0.5, side=+1, grab_at=0.25),
    "hem_up_r":       FoldAction(axis=1, frac=0.5, side=+1, grab_at=0.75),
    # Sleeve folds, targeted by landmark rather than by geometry - the same actions the twin
    # performs, so a plan validated here is executable there.
    "sleeve_l_in":    FoldAction(sleeve="left"),
    "sleeve_r_in":    FoldAction(sleeve="right"),
}


# Derived from the box the shirt is actually packed into, not invented. twin_stockfold builds
# the reconstructed VIVO box at 25x26cm and its collision cage at half-extents (w/2 - 1) and
# (d/2 - 1), so the usable interior is 23 x 24cm; 1cm of clearance leaves 22 x 23 for the bundle.
#
# The old 22 x 28 was a number I chose, and because the thresholds apply to the SORTED
# dimensions it was structurally biased against square folds: a square result had to fit
# 22x22 = 484cm2 while an elongated one was allowed 22x28 = 616cm2. 22 x 23 is nearly
# symmetric, so shape no longer changes the area budget.
BOX_INTERIOR_CM = (23.0, 24.0)
PACK_CLEARANCE_CM = 1.0
GOAL = {
    # FOLDING goal, not a packing goal. The box target (22x23cm from the VIVO interior) belonged
    # to the pack phase; folding quality is a property of the garment, so it is expressed as a
    # fraction of the FLAT shirt (55.6 x 59.2cm = 3291cm2).
    "max_area_frac": 0.16,    # ~three folds' worth
    # A folded parcel, not a wad: the 2026-09-10 ablation "passed" a quarter bundle crushed to
    # 3-12% of flat because only the ceiling was tested. Floor on area, cap on thickness.
    "min_area_frac": 0.06,
    "max_thickness_cm": 8.0,
    "max_overhang_cm": 2.0,   # sleeves actually tucked
    "min_grip": 70,           # every grab on the path found real fabric
    "max_aspect": 1.7,        # a parcel, not a strip
}
FLAT_AREA_CM2 = 55.6 * 59.2


# Fabric particles a fixed-height gripper needs to find for a real hold. Measured on the twin:
# a working pinch takes 70-480 depending on how many layers sit under the tip; the crease grab
# that effectively did nothing took 2 and 21 on two runs of the same plan.
GRIP_OK = 70


def score(ext: np.ndarray, overhang: tuple[float, float],
          target=(22.0, 23.0),   # beam-mode cost only; the pass/fail goal is GOAL
          grip: int | None = None) -> float:
    """Lower is better: how far each dimension exceeds target, plus sleeve overhang.

    Both dimensions must come under target, so repeatedly folding one axis (which
    minimises AREA fastest) no longer wins - that produced a 9x59cm rolled strip.
    """
    vals = [float(v) for v in ext[:2]] + [float(overhang[0]), float(overhang[1])]
    if not all(np.isfinite(v) for v in vals):
        # A diverged rollout must never win. `max(0.0, nan)` is 0.0 in Python, so a NaN
        # extent used to score a PERFECT 0.000 and sort to the top of the beam - the
        # search was rewarding solver explosions.
        return float("inf")
    # Per-axis shortfall, not area: a 9x59cm strip has a small area but is rolled, not
    # folded. Sorted so the measure does not care which way round the garment lies.
    small, large = sorted(vals[:2])
    t_small, t_large = sorted(float(v) for v in target)
    over = max(0.0, small - t_small) / t_small + max(0.0, large - t_large) / t_large
    # Once both dimensions are under target `over` is 0 for every survivor and the ranking
    # saturates - several plans tied at 0.000. This keeps a gentle preference for smaller,
    # weighted far below the overhang term so it only ever breaks ties.
    tie = 0.01 * (small + large) / (t_small + t_large)
    # Graspability. Extent and overhang describe the CLOTH; they say nothing about whether the
    # gripper can take hold to perform the fold. Without this the search happily chose plans
    # whose second grab lands on the crease made by the first - free in sim (the virtual pinch
    # snaps to the nearest particle) but 2-21 particles on the twin, i.e. no grasp at all.
    # Ramped, not a cliff, so the search sees a gradient away from marginal holds.
    hold = 0.0
    if grip is not None and grip < GRIP_OK:
        hold = 0.5 * (GRIP_OK - max(int(grip), 0)) / GRIP_OK
    # Squareness. Both dimensions being under target says nothing about the SHAPE, so a
    # garment bunched along one axis scored the same as a neatly folded rectangle.
    aspect = large / max(small, 1e-6)
    squash = 0.3 * max(0.0, aspect - GOAL["max_aspect"])
    return over + 0.06 * float(overhang[0] + overhang[1]) + tie + hold + squash


# What counts as folded. These are REQUIREMENTS the rollout either meets or does not - not
# weights to be traded off against each other. A plan that leaves a sleeve hanging out has not
# folded the shirt, however small its bounding box; a plan whose gripper never took hold did
# not perform the fold at all, however good the cloth looks afterwards.


def succeeded(ext, overhang, worst_grip, goal=GOAL, flat_area=None, thickness=None):
    """Is this a FOLDED shirt? A pass/fail test, deliberately not a score.

    Ranking by a continuous cost means nothing ever fails and the winner is merely whatever
    scored lowest - which is how a 9x59cm rolled strip, a NaN solver explosion, and a plan that
    grabs a crease each came first at some point.
    """
    vals = [float(ext[0]), float(ext[1]), float(overhang[0]), float(overhang[1])]
    if not all(np.isfinite(v) for v in vals):
        return False, "diverged"
    small, large = sorted(vals[:2])
    # Against THIS garment's own flat area when the caller knows it - the shirt constant is
    # only a default, and scoring a pair of trousers against a t-shirt is meaningless.
    frac = (small * large) / float(flat_area or FLAT_AREA_CM2)
    if frac > goal["max_area_frac"]:
        return False, f"not folded enough {small:.0f}x{large:.0f}cm ({frac:.0%} of flat)"
    if frac < goal.get("min_area_frac", 0.0):
        return False, f"crushed {small:.0f}x{large:.0f}cm ({frac:.0%} of flat)"
    if thickness is not None and thickness > goal.get("max_thickness_cm", 1e9):
        return False, f"a wad, {thickness:.0f}cm thick"
    oh = vals[2] + vals[3]
    if oh > goal["max_overhang_cm"]:
        return False, f"sleeve out {oh:.1f}cm"
    aspect = large / max(small, 1e-6)
    if aspect > goal["max_aspect"]:
        return False, f"not square enough {aspect:.2f} ({small:.0f}x{large:.0f}cm)"
    if worst_grip < goal["min_grip"]:
        return False, f"grasp failed (worst {worst_grip})"
    return True, f"{small:.0f}x{large:.0f}cm ({frac:.0%} of flat) grip {worst_grip}"


def search_success(env, init, max_depth=3, names=None, verbose=True):
    """Try every fold sequence in sim; keep the ones that WORK.

    No sorting decides the winner - the success predicate does. Sequences are expanded
    breadth-first and a branch stops growing as soon as it succeeds, so the shortest working
    plans are found first and nothing is pruned by a heuristic I invented.
    """
    names = names or list(ACTIONS)
    frontier = [(init.copy(), [], 10 ** 6)]        # (state, path, worst grip so far)
    wins, tried = [], 0
    for d in range(max_depth):
        states, actions, paths, prev = [], [], [], []
        for st, path, pg in frontier:
            for nm in names:
                states.append(st), actions.append(ACTIONS[nm])
                paths.append(path + [nm]), prev.append(pg)
        if not states:
            break
        t0 = time.time()
        new_states, _, new_grips = FoldSearch(env, names=names)._rollout(states, actions)
        tried += len(states)
        nxt = []
        for j, st in enumerate(new_states):
            wg = min(prev[j], new_grips[j])
            lo, hi = np.percentile(st, 3.0, axis=0), np.percentile(st, 97.0, axis=0)
            ext = (hi - lo)[:2]
            sym = env.symbolic_state(0, q=st)
            ok, why = succeeded(ext, env.sleeve_overhang_of(st), wg)
            if ok:
                wins.append((paths[j], why))
                if verbose:
                    print(f"[success]  WORKS  {' > '.join(paths[j]):46s} {why}  "
                          f"[{sym['label']}]", flush=True)
            elif why != "diverged":
                nxt.append((st, paths[j], wg))     # keep growing what has not failed outright
        if verbose:
            print(f"[success] depth {d + 1}: {len(states)} rollouts in {time.time() - t0:.0f}s  "
                  f"-> {len(wins)} working plan(s), {len(nxt)} still growing", flush=True)
        frontier = nxt
    if verbose:
        print(f"[success] {tried} rollouts tried, {len(wins)} worked", flush=True)
        if not wins:
            print("[success] NOTHING WORKED - the action set cannot reach the goal from "
                  "this state, which is a result about the actions, not the search", flush=True)
    return wins


class FoldSearch:
    def __init__(self, env: FoldEnv, beam: int = 2, depth: int = 3,
                 target=(22.0, 23.0),
                 names: list[str] | None = None):
        self.env, self.beam, self.depth, self.target = env, beam, depth, target
        self.labels = []          # candidate names, for the live viewer
        self.names = names or list(ACTIONS)

    def _rollout(self, states: list[np.ndarray], actions: list[FoldAction]):
        """Roll out (state, action) pairs, env.n at a time. Returns states + scores."""
        out_states, out_scores, out_grips = [], [], []
        n = self.env.n
        for i in range(0, len(states), n):
            chunk_s, chunk_a = states[i:i + n], actions[i:i + n]
            k = len(chunk_s)
            if k < n:                                   # pad the batch, discard the extras
                chunk_s = chunk_s + [chunk_s[-1]] * (n - k)
                chunk_a = chunk_a + [chunk_a[-1]] * (n - k)
            self.env.set_particles(np.stack(chunk_s))
            if getattr(self.env, "viewer", None) is not None and self.labels:
                names = self.labels[i:i + n]
                print("[watch-search] worlds now trying: "
                      + ", ".join(f"w{w}={nm}" for w, nm in enumerate(names)), flush=True)
            self.env.step(list(chunk_a))
            q = self.env.particles()
            ext = self.env.extents()
            for w in range(k):
                if not np.isfinite(q[w]).all():
                    print(f"[search]   ! world {w} diverged (NaN particles) - discarded",
                          flush=True)
                g = int(self.env.grip_counts()[w])
                out_grips.append(g)
                out_states.append(q[w].copy())
                out_scores.append(score(ext[w], self.env.sleeve_overhang(w), self.target,
                                        grip=g))
        return out_states, out_scores, out_grips

    def run(self, init: np.ndarray, quiet: bool = False):
        """init: (per_world, 3) starting garment state. Returns the best (path, score).

        `init` is whatever state the garment is ACTUALLY in - flat, half-folded, rotated,
        displaced. Nothing here assumes a flat shirt at the origin; the plan is computed for
        the state handed in, which is the point of planning per episode.
        """
        beam = [(init.copy(), [],
                 score(self._ext_of(init), self.env.sleeve_overhang_of(init), self.target),
                 10 ** 6)]                          # no grab made yet
        best = beam[0]
        for d in range(self.depth):
            states, actions, paths, prev_grip = [], [], [], []
            for st, path, _, pg in beam:
                for name in self.names:
                    states.append(st)
                    actions.append(ACTIONS[name])
                    paths.append(path + [name])
                    prev_grip.append(pg)

            t0 = time.time()
            self.labels = [p[-1] for p in paths]
            new_states, new_scores, new_grips = self._rollout(states, actions)
            # The WORST grab anywhere on the path is what makes a plan unexecutable - a crease
            # grab at fold 2 is fatal however good the last grab is. `env.last_grip` only holds
            # the most recent step, so carry the running minimum instead of re-reading it.
            path_grip = [min(prev_grip[j], new_grips[j]) for j in range(len(new_grips))]
            new_scores = [score(self._ext_of(new_states[j]),
                                self.env.sleeve_overhang_of(new_states[j]),
                                self.target, grip=path_grip[j]) if np.isfinite(new_scores[j])
                          else float("inf") for j in range(len(new_scores))]
            order = np.argsort(new_scores)

            if not quiet:
                print(f"[search] depth {d + 1}: {len(states)} rollouts in {time.time() - t0:.0f}s",
                      flush=True)
                for j in order[:len(self.names)]:
                    e = self._ext_of(new_states[j])
                    print(f"[search]   {new_scores[j]:.3f}  {' > '.join(paths[j]):46s} "
                          f"extent {e[0]:.0f}x{e[1]:.0f}cm  worst-grip {path_grip[j]}", flush=True)

            beam = [(new_states[j], paths[j], new_scores[j], path_grip[j])
                    for j in order[:self.beam]]
            if beam[0][2] < best[2]:
                best = beam[0]
        return best[1], best[2]

    def _ext_of(self, q: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)
        return (hi - lo)[:2]


def watch(plan: list[str], worlds: int = 1, randomize: bool = False, seed: int = 0):
    """Execute a named fold sequence in the viewer, reporting the garment as it goes."""
    env = FoldEnv(n_worlds=worlds, view=True)
    env.reset(randomize=randomize, seed=seed)
    print(f"[watch] start  extent {env.extents()[0].round(1)}cm  "
          f"overhang {tuple(round(v, 1) for v in env.sleeve_overhang())}cm", flush=True)
    for name in plan:
        env.step([ACTIONS[name]] * worlds)
        print(f"[watch] {name:16s} extent {env.extents()[0].round(1)}cm  "
              f"overhang {tuple(round(v, 1) for v in env.sleeve_overhang())}cm  "
              f"score {score(env.extents()[0], env.sleeve_overhang()):.3f}", flush=True)
    print("[watch] done - close the window when finished", flush=True)
    while True:                                   # hold the final state on screen
        env.hold(0.1)                             # render only - do NOT keep solving


def make_start(env: FoldEnv, kind: str, seed: int, prefolds: int,
               randomize: bool = True) -> np.ndarray:
    """The state the garment is in when the pipeline starts, for episode `seed`.

    'flat'   - laid out, but placed at a random position/rotation when seed > 0
    'half'   - flat, then `prefolds` RANDOM folds applied, i.e. it arrives part-done
    a path   - a saved .npy particle state (e.g. dumped from the twin)
    """
    if kind.endswith(".npy"):
        q = np.load(kind).astype(np.float32)
        env.reset(q=q, randomize=False)
        return env.particles()[0].copy()

    # randomize=False keeps every shirt at the CAPTURED pose, so the states differ
    # only in how folded they are - which is what the twin's single RGBD capture
    # actually represents. Placement variation is a separate axis, opt in with
    # --randomize.
    env.reset(randomize=randomize, seed=seed)
    if kind == "half":
        rng = np.random.default_rng(seed)
        picks = [ACTIONS[n] for n in rng.choice(list(ACTIONS), size=prefolds)]
        for a in picks:                            # same action in every world: one shared state
            env.step([a] * env.n)
    return env.particles()[0].copy()


def make_start_set(env: FoldEnv, n: int, kind: str, prefolds: int, out_dir: str,
                   randomize: bool = True):
    """Generate start states ONCE and store them, so episodes are reproducible.

    A seed does not reproduce a start state: the RNG fixes WHICH random folds are applied, but
    the cloth solve is not bit-reproducible, so the same seed gave 38x58 on one run and a NaN
    shirt on the next. A benchmark whose episodes change between runs cannot be compared
    against itself, so the states themselves are the artefact, not the seed that made them.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    for seed in range(n * 3):                      # over-generate: some pre-folds diverge
        if kept >= n:
            break
        q = make_start(env, kind, seed=seed, prefolds=prefolds, randomize=randomize)
        if not np.isfinite(q).all():
            print(f"[starts] seed {seed} diverged during the pre-fold - skipped", flush=True)
            continue
        st = env.symbolic_state(0)
        path = os.path.join(out_dir, f"start_{kept:03d}_{st['size']}_{st['sleeves']}.npy")
        np.save(path, q)
        print(f"[starts] {path}  state {st['label']}  "
              f"{st['extent_cm'][0]}x{st['extent_cm'][1]}cm  area {st['area_frac']}", flush=True)
        kept += 1
    print(f"[starts] wrote {kept} reproducible start states to {out_dir}", flush=True)


def plan_episodes(env: FoldEnv, n_ep: int, kind: str, beam: int, depth: int, prefolds: int):
    """Roll out and plan ONCE PER START STATE - the pipeline's actual usage.

    A plan found for a flat shirt at the origin says nothing about a shirt that arrives
    half-folded and rotated. Each episode re-plans from what it is actually given.
    """
    results = []
    for ep in range(n_ep):
        init = make_start(env, kind, seed=ep, prefolds=prefolds)
        st0 = env.symbolic_state(0)
        print(f"\n[episode {ep}] start {kind}  STATE {st0['label']}  "
              f"{st0['extent_cm'][0]}x{st0['extent_cm'][1]}cm", flush=True)
        t0 = time.time()
        path, sc = FoldSearch(env, beam=beam, depth=depth).run(init, quiet=True)
        print(f"[episode {ep}] best: {' > '.join(path)}   score {sc:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        results.append((ep, path, sc))

    print("\n[summary] best plan per start state", flush=True)
    for ep, path, sc in results:
        print(f"[summary]  ep{ep}  {sc:.3f}  {' > '.join(path)}", flush=True)
    from collections import Counter
    firsts = Counter(p[0] for _, p, _ in results if p)
    whole = Counter(" > ".join(p) for _, p, _ in results if p)
    print(f"[summary] first action: {dict(firsts)}", flush=True)
    print(f"[summary] distinct plans: {len(whole)}/{len(results)}  "
          f"most common {whole.most_common(1)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--view", action="store_true", help="watch instead of search")
    ap.add_argument("--view-search", action="store_true",
                    help="watch the SEARCH itself: every world is a different candidate fold")
    ap.add_argument("--plan", type=str, default="",
                    help="comma-separated action names to watch, e.g. panel_left_in,half_right_in")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--randomize", action="store_true", help="random SE(2) start placement")
    ap.add_argument("--episodes", type=int, default=0,
                    help="plan once per start state; 0 = single search from a flat shirt")
    ap.add_argument("--start", type=str, default="flat",
                    help="start state per episode: flat | half | <path>.npy")
    ap.add_argument("--prefolds", type=int, default=1,
                    help="random folds applied before planning when --start half")
    ap.add_argument("--out", type=str, default="",
                    help="write the chosen plan here for twin_stockfold's TWIN_PLAN_FILE")
    ap.add_argument("--cloth", choices=("shirt", "sheet"), default="shirt",
                    help="shirt = unisex_shirt.usd (newton's own cloth_franka asset); "
                         "sheet = flat control, same material and particle count, no sleeves")
    ap.add_argument("--make-starts", type=str, default="",
                    help="generate reproducible start states into this directory and exit")
    ap.add_argument("--mode", choices=("success", "beam"), default="success",
                    help="success: try everything, keep what WORKS (pass/fail predicate). "
                         "beam: rank by the weighted cost and keep the top few.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.view_search:
        env = FoldEnv(n_worlds=a.worlds, cloth=a.cloth, view=True)
        if a.start.endswith(".npy"):
            env.reset(q=np.load(a.start).astype(np.float32), randomize=False)
        else:
            env.reset(randomize=a.randomize, seed=a.seed)
        st = env.symbolic_state(0)
        print(f"[watch-search] planning live from STATE {st['label']} "
              f"{st['extent_cm'][0]}x{st['extent_cm'][1]}cm - each world is one candidate fold",
              flush=True)
        path, sc = FoldSearch(env, beam=a.beam, depth=a.depth).run(env.particles()[0].copy())
        print(f"[watch-search] best: {' > '.join(path)}  score {sc:.3f}", flush=True)
        while True:
            env.hold(0.1)
    if a.view or a.plan:
        plan = [p for p in a.plan.split(",") if p] or ["panel_left_in", "panel_right_in",
                                                       "half_hem_up"]
        bad = [p for p in plan if p not in ACTIONS]
        if bad:
            raise SystemExit(f"unknown action(s) {bad}; choose from {list(ACTIONS)}")
        watch(plan, worlds=1, randomize=a.randomize, seed=a.seed)
        return

    env = FoldEnv(n_worlds=a.worlds, cloth=a.cloth)
    if a.start.endswith(".npy") and not a.make_starts and not a.episodes and a.mode == "beam":
        env.reset(q=np.load(a.start).astype(np.float32), randomize=False)
        st = env.symbolic_state(0)
        print(f"[search] start {a.start}  STATE {st['label']}  "
              f"{st['extent_cm'][0]}x{st['extent_cm'][1]}cm  beam={a.beam} depth={a.depth}",
              flush=True)
        path, sc = FoldSearch(env, beam=a.beam, depth=a.depth).run(env.particles()[0].copy())
        end = env.symbolic_state(0)
        print(f"[search] best: {' > '.join(path)}   score {sc:.3f}   "
              f"{st['label']} -> {end['label']}", flush=True)
        if a.out:
            open(a.out, "w").write(",".join(path))
        return
    if a.make_starts:
        make_start_set(env, a.episodes or 5, a.start, a.prefolds, a.make_starts,
                       randomize=a.randomize)
        return
    if a.mode == "success" and not a.episodes:
        if a.start.endswith(".npy"):
            env.reset(q=np.load(a.start).astype(np.float32), randomize=False)
            print(f"[success] planning from observed state {a.start}", flush=True)
        else:
            env.reset(randomize=a.randomize, seed=a.seed)
        e = env.extents()[0]
        print(f"[success] start {e[0]:.0f}x{e[1]:.0f}cm  goal {GOAL}", flush=True)
        wins = search_success(env, env.particles()[0].copy(), max_depth=a.depth)
        for path, why in wins:
            print(f"[success] plan ({len(path)} folds): {' > '.join(path)}   {why}", flush=True)
        if a.out:
            # Breadth-first, so wins[0] is a shortest working plan. Chosen by the predicate and
            # plan LENGTH - no weighted cost decides this.
            if wins:
                open(a.out, "w").write(",".join(wins[0][0]))
                print(f"[success] wrote {' > '.join(wins[0][0])} -> {a.out}", flush=True)
            else:
                open(a.out, "w").write("")
                print(f"[success] no working plan; wrote an EMPTY plan to {a.out}", flush=True)
        return
    if a.episodes:
        plan_episodes(env, a.episodes, a.start, a.beam, a.depth, a.prefolds)
        return
    env.reset(randomize=a.randomize, seed=a.seed)
    init = env.particles()[0].copy()
    print(f"[search] start extent {env.extents()[0].round(1)} cm, "
          f"beam={a.beam} depth={a.depth} actions={len(ACTIONS)}", flush=True)

    t0 = time.time()
    path, sc = FoldSearch(env, beam=a.beam, depth=a.depth).run(init)
    print(f"[search] best: {' > '.join(path)}   score {sc:.3f}   ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
