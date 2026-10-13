"""Task-level planning over symbolic garment states, validated by simulation.

Until now the planner enumerated action sequences and rolled every one out. With 13 actions
that is 2197 sequences at depth 3 - the search became the bottleneck and the symbolic state was
only used to LABEL results, never to produce them.

This is the TAMP split. Each fold is declared as an operator with symbolic PRECONDITIONS and
EFFECTS over the garment's state. A task-level search over those operators produces SKELETONS -
short action sequences that reach the goal symbolically. Only those skeletons are then handed to
the simulator, which is still the only authority on what a fold actually does: a skeleton that
the symbolic model says works is a HYPOTHESIS, and physics either confirms it or kills it.

The effects below are not invented. Each is what we measured on the twin:
  - `half_collar_dn` tucks BOTH sleeves in one move (overhang 14.9/14.0 -> 0.0/0.0), because the
    sleeves attach at the collar end and the fold carries them onto the torso.
  - `half_hem_up` does NOT (56x28cm with sleeves still splayed) - the hem half has no sleeves.
  - panel folds tuck ONE sleeve each (panel_left_in: 0.0/14.1, then panel_right_in: 0.0/0.0).
  - a fold only shrinks the axis it acts on, so it is useful only while that axis is the long one.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

SIZES = ["spread", "half", "quarter", "packed"]


def _shrink(size: str) -> str:
    i = SIZES.index(size)
    return SIZES[min(i + 1, len(SIZES) - 1)]


def _tuck(sleeves: str, which: str) -> str:
    """Symbolic sleeve bookkeeping: `which` in {left, right, both}."""
    have = set()
    if sleeves in ("left_in", "both_in"):
        have.add("left")
    if sleeves in ("right_in", "both_in"):
        have.add("right")
    have |= ({"left", "right"} if which == "both" else {which})
    if have >= {"left", "right"}:
        return "both_in"
    if "left" in have:
        return "left_in"
    if "right" in have:
        return "right_in"
    return "both_out"


# Operators: axis the fold acts on, which sleeves it brings in, and whether it shrinks.
# axis "y" = length (collar<->hem), "x" = width.
OPERATORS = {
    "half_collar_dn": {"axis": "y", "tucks": "both", "shrinks": True},
    "half_hem_up":    {"axis": "y", "tucks": None,   "shrinks": True},
    "half_right_in":  {"axis": "x", "tucks": None,   "shrinks": True},
    "panel_left_in":  {"axis": "x", "tucks": "left", "shrinks": True},
    "panel_right_in": {"axis": "x", "tucks": "right", "shrinks": True},
    "sleeve_l_in":    {"axis": None, "tucks": "left", "shrinks": False},
    "sleeve_r_in":    {"axis": None, "tucks": "right", "shrinks": False},
}


def applicable(state: dict, name: str) -> bool:
    """PRECONDITIONS - when is this fold worth doing at all?"""
    op = OPERATORS[name]
    if state["size"] == "packed":
        return False                                  # already at the goal size
    if op["tucks"] and not op["shrinks"]:
        # a sleeve fold is pointless once that sleeve is already in
        side = op["tucks"]
        if state["sleeves"] == "both_in":
            return False
        if side == "left" and state["sleeves"] in ("left_in",):
            return False
        if side == "right" and state["sleeves"] in ("right_in",):
            return False
        return True
    # a fold only shortens the axis it acts on, so it is only useful while that axis is long
    if op["axis"] and state["long_axis"] not in (op["axis"], "square"):
        return False
    return True


def apply_op(state: dict, name: str) -> dict:
    """EFFECTS - the symbolic model's prediction, which the simulator then checks."""
    op = OPERATORS[name]
    out = dict(state)
    if op["shrinks"]:
        out["size"] = _shrink(state["size"])
        if op["axis"] == "y":
            out["long_axis"] = "x"                    # halving the length makes width the long axis
        elif op["axis"] == "x":
            out["long_axis"] = "y"
    if op["tucks"]:
        out["sleeves"] = _tuck(state["sleeves"], op["tucks"])
    return out


def is_goal(state: dict) -> bool:
    return state["size"] == "packed" and state["sleeves"] == "both_in"


def skeletons(start: dict, max_depth: int = 4, limit: int = 40):
    """Every symbolic action sequence that reaches the goal, shortest first.

    These are HYPOTHESES. The symbolic model says they should work; only simulation can say
    whether they do. This is what replaces enumerating 13^depth sequences blindly.
    """
    out, frontier = [], [(start, [])]
    for _ in range(max_depth):
        nxt = []
        for st, path in frontier:
            for name in OPERATORS:
                if not applicable(st, name):
                    continue
                st2, p2 = apply_op(st, name), path + [name]
                if is_goal(st2):
                    out.append(p2)
                    if len(out) >= limit:
                        return out
                else:
                    nxt.append((st2, p2))
        frontier = nxt
        if not frontier:
            break
    return out


def state_from_env(env, world: int = 0) -> dict:
    """Read the symbolic state off the simulator, adding the orientation the labels lacked."""
    s = env.symbolic_state(world)
    ex = s["extent_cm"]
    long_axis = ("square" if abs(ex[0] - ex[1]) < 0.15 * max(ex)
                 else "x" if ex[0] > ex[1] else "y")
    return {"size": s["size"], "sleeves": s["sleeves"], "long_axis": long_axis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="", help="a saved .npy garment state")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--validate", type=int, default=0,
                    help="simulate this many skeletons and report what physics says")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=80.0,
                    help="cm between worlds; tighter keeps N shirts watchable")
    ap.add_argument("--view", action="store_true", help="watch the validation")
    ap.add_argument("--cloth", type=str, default="shirt",
                    help="'shirt' (stock asset), 'sheet' (flat control), or 'mask:<capture-id>' "
                         "to build the REAL garment from that capture's mask + depth; "
                         "'mask:<capture-id>:<object id|label>' picks one garment of several")
    ap.add_argument("--rec-usd", type=str, default="",
                    help="record the WHOLE parallel validation - every world's garment, side "
                         "by side - as one time-sampled USD for Isaac Sim playback. Newton "
                         "stays the physics; Isaac is just the screen.")
    ap.add_argument("--out", type=str, default="",
                    help="write the chosen plan here, ready for the robot")
    ap.add_argument("--scale", type=str, default="1,1",
                    help="per-axis rest-shape scale of the 'shirt' asset, sx,sy - the SAME "
                         "numbers the twin fitted (TWIN_FIT_SCALE), so a loaded state is not "
                         "pre-stretched against a stock rest shape")
    ap.add_argument("--goal-area", type=float, default=None,
                    help="override GOAL max_area_frac (fraction of the FLAT garment). From an "
                         "already-folded start the pinched folds on a thick stack land loose; "
                         "0.16 (three crisp folds from flat) rejected every plan at 20-28%%")
    a = ap.parse_args()
    _scale = tuple(float(v) for v in a.scale.split(","))

    from fold_env import FoldEnv
    from fold_search import ACTIONS, succeeded, FLAT_AREA_CM2 as FLAT_AREA, GOAL
    _goal = dict(GOAL)
    if a.goal_area is not None:
        _goal["max_area_frac"] = float(a.goal_area)
        print(f"[tamp] goal: folded area <= {a.goal_area:.0%} of flat (overridden)", flush=True)

    env = FoldEnv(n_worlds=a.worlds, view=a.view, spacing_cm=a.spacing, cloth=a.cloth,
                  scale_xy=_scale)
    # The garment's FLAT reference comes from its home (flat) state, measured BEFORE a start
    # state is loaded. Measuring after the load made a half-folded start (fold_state_fit) its
    # own "flat": the planner then read it as 'spread' and scored the goal against half a shirt.
    env.reset()
    env.measure_flat()
    if a.start.endswith(".npy"):
        _q0 = np.load(a.start).astype(np.float32).reshape(-1, 3)
        # The twin's table top is at z=20cm; this env's ground is z=0. A state loaded as-is was
        # dropped 20cm and re-settled - harmless for a flat shirt, but a FOLDED start state
        # (fold_state_fit) would scatter on landing. Rest it on the ground, xy untouched.
        _q0[:, 2] -= float(_q0[:, 2].min()) - 0.5
        env.reset(q=_q0, randomize=False)
        # The twin dumps the REAL robot frame next to the observed cloth. Without it the planner
        # has no idea where the arm is and happily proposes grasps at the collar edge that sit
        # a metre out - the twin then clamps the waypoint and folds something else, so the plan
        # "worked" in sim and was impossible on the robot.
        import os as _os
        _rj = a.start + ".robot.json"
        if _os.path.exists(_rj):
            import json as _j
            _r = _j.load(open(_rj))
            env.set_reach(_r["base_cm"], _r["reach_cm"])
        else:
            print(f"[tamp] no {_rj} - reach is NOT being checked; plans may be unreachable",
                  flush=True)
    _flat = env.FLAT_CM[0] * env.FLAT_CM[1]   # the garment's own flat, not a t-shirt constant
    st = state_from_env(env)
    print(f"[tamp] start state {st}", flush=True)

    sk = skeletons(st, max_depth=a.depth)
    print(f"[tamp] {len(sk)} symbolic skeleton(s) reach the goal "
          f"(vs {len(OPERATORS)}^{a.depth} = {len(OPERATORS) ** a.depth} blind sequences)",
          flush=True)
    for p in sk[:12]:
        print(f"[tamp]   {' > '.join(p)}", flush=True)

    if not a.validate:
        return
    # Validate several skeletons AT ONCE - one per world. A world that has finished its sequence
    # takes a no-op (side=0 selects no particles) and simply holds. This is the black box being
    # asked several questions in parallel.
    from fold_env import FoldAction
    # DIVERSE candidates, not the first N in enumeration order - those all share a first
    # action and fold identically for two steps, which wastes most of the batch.
    def _diverse(cands, k):
        out, seen_first, seen_pair = [], set(), set()
        for pool in (0, 1, 2):                       # first pass: unique openings, then pairs
            for c in cands:
                if len(out) >= k:
                    return out
                key1, key2 = c[0], tuple(c[:2])
                if pool == 0 and key1 in seen_first:
                    continue
                if pool == 1 and key2 in seen_pair:
                    continue
                if c in out:
                    continue
                out.append(c); seen_first.add(key1); seen_pair.add(key2)
        return out
    # fold_env implements the geometric folds only - sleeve_l_in / sleeve_r_in exist as TAMP
    # operators and on the twin, but not in the fast validator. Skip skeletons it cannot run
    # rather than crash, and say how many were dropped.
    runnable = [c for c in sk if all(n in ACTIONS for n in c)]
    if len(runnable) < len(sk):
        print(f"[tamp] {len(sk) - len(runnable)} skeleton(s) use sleeve folds, which fold_env "
              f"does not implement - not validated here", flush=True)
    picks = _diverse(runnable, min(a.validate, env.n))
    q0 = env.particles()[0].copy()
    # Each world's cloth lives at its own x offset (world i at i * spacing). Copying world 0's
    # ABSOLUTE positions into every world stacked all the shirts at x=0 - the camera then framed
    # a 69cm span and the other environments were simply not there.
    _copies = []
    for _w in range(env.n):
        _c = q0.copy()
        _dx, _dy = env.world_xy[_w]          # grid position, matching how the env was built
        _c[:, 0] += _dx
        _c[:, 1] += _dy
        _copies.append(_c)
    env.reset(q=np.concatenate(_copies, axis=0), randomize=False)
    env.frame_scene()          # aim at the cloth, not at where we guessed it would be
    rec = None
    if a.rec_usd:
        from pxr import Gf, Usd, UsdGeom, UsdLux
        _stage = Usd.Stage.CreateNew(a.rec_usd)
        UsdGeom.SetStageUpAxis(_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(_stage, 0.01)
        _every = 4
        _stage.SetTimeCodesPerSecond(env.FPS / _every)
        _stage.SetStartTimeCode(0)
        _w = UsdGeom.Xform.Define(_stage, "/World")
        _stage.SetDefaultPrim(_w.GetPrim())
        _tris_all = env.model.tri_indices.numpy().reshape(-1, 3)
        _tpw = len(_tris_all) // env.n
        _tris0 = _tris_all[:_tpw] % env.per_world          # world-0 topology, world-local ids
        _attrs = []
        for _i in range(env.n):
            _m = UsdGeom.Mesh.Define(_stage, f"/World/env_{_i}/garment")
            _m.CreateFaceVertexIndicesAttr(_tris0.flatten().tolist())
            _m.CreateFaceVertexCountsAttr([3] * _tpw)
            _m.CreateDoubleSidedAttr(True)
            _m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            _m.CreateDisplayColorAttr([Gf.Vec3f(0.83, 0.90, 0.28)])
            _attrs.append(_m.GetPointsAttr())
        _sun = UsdLux.DistantLight.Define(_stage, "/World/sun")
        _sun.CreateIntensityAttr(3000.0)
        UsdGeom.Xformable(_sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
        UsdLux.DomeLight.Define(_stage, "/World/dome").CreateIntensityAttr(400.0)
        _q0 = env.particles()
        _g = UsdGeom.Cube.Define(_stage, "/World/floor")
        _g.CreateSizeAttr(1.0)
        _xg = UsdGeom.Xformable(_g)
        _lo = _q0.reshape(-1, 3).min(0); _hi = _q0.reshape(-1, 3).max(0)
        _xg.AddTranslateOp().Set(Gf.Vec3d(float((_lo[0]+_hi[0])/2), float((_lo[1]+_hi[1])/2),
                                          float(_lo[2]) - 5.5))
        _xg.AddScaleOp().Set(Gf.Vec3f(float(_hi[0]-_lo[0]) + 160.0,
                                      float(_hi[1]-_lo[1]) + 160.0, 10.0))
        _g.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])
        _ctr = {"f": 0, "t": 0}

        def _rec_cb(e):
            _ctr["f"] += 1
            if _ctr["f"] % _every:
                return
            qs = e.state_0.particle_q.numpy()
            tc = Usd.TimeCode(_ctr["t"])
            for _i in range(e.n):
                qq = qs[_i * e.per_world:(_i + 1) * e.per_world]
                if np.isfinite(qq).all():
                    _attrs[_i].Set([Gf.Vec3f(*map(float, v)) for v in qq], tc)
            _ctr["t"] += 1
            if _ctr["t"] % 150 == 0:
                _stage.SetEndTimeCode(_ctr["t"] - 1)
                _stage.GetRootLayer().Save()

        env.frame_cb = _rec_cb
        import atexit as _ax

        def _rec_save():
            _stage.SetEndTimeCode(max(_ctr["t"] - 1, 1))
            _stage.GetRootLayer().Save()
            print(f"[tamp] recorded {_ctr['t']} frames of {env.n} worlds -> {a.rec_usd} "
                  f"(open in Isaac Sim, press PLAY)", flush=True)
        _ax.register(_rec_save)
        print(f"[tamp] recording validation to {a.rec_usd}", flush=True)
    print(f"\n[tamp] validating {len(picks)} skeleton(s) in parallel:", flush=True)
    for w, pp in enumerate(picks):
        print(f"[tamp]   w{w}: {' > '.join(pp)}", flush=True)
    worst = [10 ** 6] * env.n
    far = [0.0] * env.n              # furthest grasp each plan asked the arm to make
    for step_i in range(max(len(pp) for pp in picks)):
        acts, names = [], []
        for w in range(env.n):
            if w < len(picks) and step_i < len(picks[w]):
                acts.append(ACTIONS[picks[w][step_i]]); names.append(picks[w][step_i])
            else:
                acts.append(FoldAction(side=0)); names.append("-")
        env.step(acts)
        g = env.grip_counts()
        for w in range(env.n):
            if names[w] != "-":
                worst[w] = min(worst[w], int(g[w]))
                far[w] = max(far[w], float(env.last_reach[w]))
        print(f"[tamp] step {step_i + 1}: " + "  ".join(f"w{w}={n}" for w, n in enumerate(names)),
              flush=True)
    print("", flush=True)
    # SELECT. The system decides - it does not print a table for someone to read and rank by
    # hand. Survivors are ordered by how far they beat the goal (smallest folded area first,
    # then firmest worst grasp), and the winner is written out ready for the robot.
    survivors = []
    for w, pp in enumerate(picks):
        e = env.extents()[w]
        ok, why = succeeded(e, env.sleeve_overhang_of(env.particles()[w]), worst[w],
                            goal=_goal, flat_area=_flat)
        # Physics is not the only authority - the arm is too. A fold the robot cannot reach is
        # not a plan, however neatly it folds.
        if ok and env.reach_cm and far[w] > env.reach_cm:
            ok, why = False, f"grasp out of reach {far[w]:.0f}cm > {env.reach_cm:.0f}cm"
        print(f"[tamp] {'WORKS ' if ok else 'FAILS '} w{w} {' > '.join(pp):50s} {why}", flush=True)
        if ok:
            small, large = sorted(float(v) for v in e[:2])
            survivors.append({"plan": pp, "area": small * large, "grip": worst[w],
                              "ext": (small, large), "reach": far[w]})
    print("", flush=True)
    if not survivors:
        print("[tamp] NO PLAN WORKS from this state - a result about the action set, not a "
              "failure to search. Widen the actions or relax the goal.", flush=True)
        return
    survivors.sort(key=lambda d: (d["area"], -d["grip"]))
    best = survivors[0]
    print(f"[tamp] {len(survivors)}/{len(picks)} plans work. BEST PLAN:", flush=True)
    print(f"[tamp]   {' > '.join(best['plan'])}", flush=True)
    print(f"[tamp]   {best['ext'][0]:.0f}x{best['ext'][1]:.0f}cm "
          f"({best['area'] / _flat:.0%} of flat), worst grasp {best['grip']} particles"
          + (f", furthest reach {best['reach']:.0f}cm" if env.reach_cm else ""), flush=True)
    for d in survivors[1:4]:
        print(f"[tamp]   runner-up: {' > '.join(d['plan']):46s} "
              f"{d['ext'][0]:.0f}x{d['ext'][1]:.0f}cm", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(",".join(best["plan"]))
        print(f"[tamp] wrote the plan to {a.out} - ready for TWIN_PLAN_FILE", flush=True)
    if env.viewer is not None and not a.out:
        print("[tamp] holding the result on screen", flush=True)
        while True:
            env.hold(0.1)
    elif env.viewer is not None:
        for _ in range(40):        # a few seconds to see the outcome, then hand over
            env.hold(0.1)


if __name__ == "__main__":
    main()
