"""Label-based folding: operators are move(label A -> label B) over the garment's SEMANTIC labels
(named by a VLM, lifted to particle indices), searched with physics validation. No garment-specific
fold vocabulary: every candidate is "pinch the fabric at label A and carry it onto label B".

    python label_tamp.py --labels labels.json --start observed.npy --scale sx,sy [--depth 3 --beam 3]
                         [--goal-area 0.25] [--rec-usd out.usd] [--out plan.json]

Search: beam search over sequences of label moves. Candidates at each step are pruned with
garment-agnostic rules only (outline labels, sensible travel, move lands on the garment, moves
inward); symmetric left/right pairs are also offered as ONE two-handed move. Each candidate is
rolled out in fold_env (9 worlds in lockstep per batch) and scored by the same pass/fail test the
named-operator planner uses (area window, thickness, aspect, grip), minus the sleeve-overhang term
(which needs shirt landmarks). Progress for the beam = footprint area after the move.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from fold_env import FoldEnv, FoldAction
from fold_search import succeeded, GOAL
from fold_predicates import SheetPredicates, GOALS


def outline_labels(L):
    return [n for n, l in L.items() if l["kind"] in ("corner", "edge_midpoint")]


def candidates(L, q, pairs, min_cm=10.0, max_cm=50.0):
    """Garment-agnostic label moves from the CURRENT particle positions q (per_world, 3)."""
    names = outline_labels(L)
    pos = {n: q[L[n]["particle"], :2] for n in L}
    hull_c = np.median(q[:, :2], 0)
    lo, hi = np.percentile(q[:, :2], 3.0, axis=0), np.percentile(q[:, :2], 97.0, axis=0)
    out = []
    def ok(a, b):
        d = pos[b] - pos[a]; t = float(np.linalg.norm(d))
        if not (min_cm <= t <= max_cm):
            return False
        mid = pos[a] + 0.5 * d
        if not (np.all(mid >= lo - 2.0) and np.all(mid <= hi + 2.0)):
            return False                                   # the fold must land on the garment
        return float(d @ (hull_c - pos[a])) > 0.0            # and move inward, not off the edge
    for a, b in itertools.permutations(names, 2):
        if ok(a, b):
            out.append(((a, b),))
    # two-handed symmetric moves: (left_x -> left_y) together with (right_x -> right_y)
    pairmap = {a: b for a, b in pairs}; pairmap.update({b: a for a, b in pairs})
    for a, b in itertools.permutations(names, 2):
        a2, b2 = pairmap.get(a), pairmap.get(b)
        if a2 and b2 and a2 != a and b2 != b and a < a2 and ok(a, b) and ok(a2, b2):
            d1, d2 = pos[b] - pos[a], pos[b2] - pos[a2]
            n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
            if float(d1 @ d2) / max(n1 * n2, 1e-6) < 0.5:
                continue                                       # the two hands pull different ways
            sep0, sep1 = pos[a2] - pos[a], pos[b2] - pos[b]
            if float(sep0 @ sep1) <= 0.0 or np.linalg.norm(sep1) < 0.5 * np.linalg.norm(sep0):
                continue                                       # the hands cross or converge onto one point
            out.append(((a, b), (a2, b2)))
    return out


def name_of(mv):
    return " + ".join(f"{a}->{b}" for a, b in mv)


BAND_CM = 0.0


def to_action(mv, L):
    return FoldAction(moves=[(int(L[a]["particle"]), int(L[b]["particle"])) for a, b in mv], band_cm=BAND_CM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels json from lift_labels.py")
    ap.add_argument("--start", required=True, help="observed.npy of the garment (twin frame)")
    ap.add_argument("--scale", default="1,1")
    ap.add_argument("--cloth", default="shirt")
    ap.add_argument("--worlds", type=int, default=9)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-cands", type=int, default=18, help="candidates evaluated per parent")
    ap.add_argument("--goal-area", type=float, default=0.25)
    ap.add_argument("--goal", default="", choices=[""] + list(GOALS),
                    help="SYMBOLIC goal (fold_predicates.GOALS). With a goal, progress is the number of goal "
                         "predicates satisfied, success is all of them, and candidates are proposed from the "
                         "unsatisfied on(A,B) facts first (regression). Without one: the footprint search.")
    ap.add_argument("--eps", type=float, default=3.0, help="cm: labels this close count as on/coincident")
    ap.add_argument("--layer-thickness", type=float, default=0.6, help="cm: one layer of the asset")
    ap.add_argument("--band", type=float, default=0.0,
                    help="cm: grip a band of the edge this wide per hand, as the twin does (TWIN_FOLD_BAND_CM); "
                         "0 = the 4cm point pinch")
    ap.add_argument("--min-grip", type=int, default=None,
                    help="particles a pinch must hold to count (default fold_search.GOAL, 70 = the SHIRT's "
                         "density at a 4cm pinch). A towel corner at 0.5cm spacing holds ~55: scale it to the asset.")
    ap.add_argument("--max-thick", type=float, default=None,
                    help="cm; states thicker than this are DISCARDED from the beam (a folded towel of 4 "
                         "layers is ~2cm; the shirt default is 8). Footprint alone cannot tell a fold from a wad.")
    ap.add_argument("--spacing", type=float, default=130.0)
    ap.add_argument("--rec-usd", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--replay", default="", help="plan json from --out: skip the search, just record this plan")
    a = ap.parse_args()

    Lj = json.load(open(a.labels)); L = Lj["labels"]; pairs = [tuple(p) for p in Lj.get("pairs", [])]
    global BAND_CM
    BAND_CM = float(a.band)
    goal = dict(GOAL); goal["max_area_frac"] = a.goal_area; goal["max_overhang_cm"] = 1e9   # no shirt landmarks
    if a.max_thick is not None:
        goal["max_thickness_cm"] = float(a.max_thick)
    if a.min_grip is not None:
        goal["min_grip"] = int(a.min_grip)
    env = FoldEnv(n_worlds=a.worlds, spacing_cm=a.spacing, cloth=a.cloth,
                  scale_xy=tuple(float(v) for v in a.scale.split(",")))
    env.reset(); env.measure_flat()
    flat = float(env.FLAT_CM[0] * env.FLAT_CM[1])
    q0 = np.load(a.start).astype(np.float32).reshape(-1, 3); q0[:, 2] -= float(q0[:, 2].min()) - 0.5
    rj = a.start + ".robot.json"
    if os.path.exists(rj):
        r = json.load(open(rj)); env.set_reach(r["base_cm"], r["reach_cm"])
    print(f"[label-tamp] {Lj['garment']}: {len(L)} labels ({len(outline_labels(L))} on the outline), "
          f"{len(pairs)} symmetric pairs; flat {env.FLAT_CM[0]:.0f}x{env.FLAT_CM[1]:.0f}cm; goal area <= {a.goal_area:.0%}", flush=True)

    def area_frac(q):
        lo, hi = np.percentile(q[:, :2], 3.0, axis=0), np.percentile(q[:, :2], 97.0, axis=0)
        return float((hi - lo)[0] * (hi - lo)[1]) / flat

    def rollout_batch(parent_states, moves):
        """Each entry: (state, move). Runs len(entries) <= env.n worlds in lockstep for ONE move."""
        n = len(moves)
        copies = []
        for w in range(env.n):
            s = parent_states[min(w, n - 1)].copy()
            dx, dy = env.world_xy[w]; s[:, 0] += dx; s[:, 1] += dy
            copies.append(s)
        env.reset(q=np.concatenate(copies, 0), randomize=False)
        acts = [to_action(moves[w], L) if w < n else FoldAction(side=0) for w in range(env.n)]
        env.step(acts)
        qs = env.particles(); grips = env.grip_counts()
        res = []
        for w in range(n):
            q = qs[w].copy(); dx, dy = env.world_xy[w]; q[:, 0] -= dx; q[:, 1] -= dy
            ext = env.extents()[w]; oh = (0.0, 0.0)
            thick = float(np.percentile(q[:, 2], 97) - np.percentile(q[:, 2], 3)) if np.isfinite(q).all() else float("nan")
            okf, why = succeeded(ext, oh, int(grips[w]), goal=goal, flat_area=flat, thickness=thick)
            res.append(dict(q=q, grip=int(grips[w]), area=area_frac(q) if np.isfinite(q).all() else 9.9,
                            thick=thick, ok=okf, why=why, diverged=not np.isfinite(q).all()))
        return res

    P = None
    if a.goal:
        _e = __import__("os").environ
        P = SheetPredicates(L, spacing_cm=float(_e.get("TWIN_TOWEL_SPACING_CM", "0.5")),
                            thickness_cm=a.layer_thickness, eps_cm=a.eps)
        s0 = P.state(q0)
        print(f"[label-tamp] symbolic goal '{a.goal}' = {GOALS[a.goal]}", flush=True)
        print(f"[label-tamp] start state: {P.describe(s0)}  -> {P.score(s0, a.goal)[0]}/{P.score(s0, a.goal)[1]} satisfied", flush=True)

    t0 = time.time()
    beam = [dict(q=q0, seq=[], area=area_frac(q0), grip=10 ** 6, sat=(P.score(P.state(q0), a.goal)[0] if P else 0))]
    winners = []
    if a.replay:
        pj = json.load(open(a.replay)); winners = [dict(q=q0, seq=pj["plan"], area=pj.get("area_frac", 0.0), grip=0)]
        print(f"[label-tamp] replaying {' > '.join(pj['plan'])}", flush=True)
    for depth in range(1, 0 if a.replay else a.depth + 1):
        entries = []
        for b in beam:
            cands = candidates(L, b["q"], pairs)
            # prefer two-handed moves and longer travel first; cap per parent
            cands.sort(key=lambda mv: (-len(mv), -sum(np.linalg.norm(b["q"][L[y]["particle"], :2] - b["q"][L[x]["particle"], :2]) for x, y in mv)))
            if P:
                # REGRESSION from the goal: moves that would make an unsatisfied on(A,B) true go first
                sp, ss = P.suggest(P.state(b["q"]), a.goal)
                goal_first = [mv for mv in sp + ss if mv in cands] + [mv for mv in sp + ss if mv not in cands]
                cands = goal_first + [mv for mv in cands if mv not in goal_first]
            for mv in cands[:a.max_cands]:
                entries.append((b, mv))
        print(f"[label-tamp] depth {depth}: {len(entries)} candidate moves from {len(beam)} parent(s)", flush=True)
        results = []
        for i in range(0, len(entries), env.n):
            chunk = entries[i:i + env.n]
            res = rollout_batch([e[0]["q"] for e in chunk], [e[1] for e in chunk])
            for (b, mv), r in zip(chunk, res):
                seq = b["seq"] + [name_of(mv)]
                sat, tot, sym, st_ = 0, 0, "", None
                if P and not r["diverged"]:
                    st_ = P.state(r["q"]); sat, tot = P.score(st_, a.goal)
                    sym = f"  [{sat}/{tot}] {P.describe(st_)}"
                    r["ok"] = (sat == tot) and r["grip"] >= goal["min_grip"]
                tag = "WORKS" if r["ok"] else ("diverged" if r["diverged"] else (r["why"] if not P else "not yet"))
                print(f"[label-tamp]   {' > '.join(seq):70s} area {r['area']:.2f} grip {r['grip']:4d}  {tag}{sym}", flush=True)
                if r["diverged"] or r["grip"] < goal["min_grip"]:
                    continue
                if np.isfinite(r["thick"]) and r["thick"] > goal["max_thickness_cm"] and not (P and st_["flat"]):
                    # a pile is not progress, however small its footprint. With a symbolic goal the
                    # `flat` predicate owns thickness (layers x one layer's thickness, with drape
                    # slack): a quarter-folded towel measured 2.6-2.8cm and is flat; a fixed cap
                    # under that threw the correct plan away.
                    continue
                if P and sat < b["sat"]:
                    continue                      # a move that UNDOES goal facts is not progress either
                node = dict(q=r["q"], seq=seq, area=r["area"], grip=min(b["grip"], r["grip"]), sat=sat, thick=float(r["thick"]))
                if r["ok"]:
                    winners.append(node)
                results.append(node)
        if winners:
            break
        # beam: keep the most-progressed distinct states (area), but not crushed ones
        results = [n for n in results if n["area"] >= goal.get("min_area_frac", 0.0)]
        # with a symbolic goal: most goal predicates satisfied first, footprint only breaks ties
        results.sort(key=(lambda n: (-n["sat"], n["area"])) if P else (lambda n: n["area"]))
        beam = results[:a.beam]
        if not beam:
            print("[label-tamp] no viable candidates left", flush=True); break
        print(f"[label-tamp] beam after depth {depth}: " + "; ".join(f"{' > '.join(n['seq'])} ({n['area']:.2f})" for n in beam), flush=True)
    print(f"[label-tamp] search took {time.time() - t0:.0f}s", flush=True)
    if not winners:
        print("[label-tamp] NO plan reached the goal", flush=True)
        return 1
    # the saved plan: fewest actions, then the most two-handed (a two-hand step is one action
    # and one carry, so fewer sequential carries = less time), then the flattest, then the
    # smallest footprint. Footprint alone preferred a diagonal single-hand second move (0.18)
    # over the symmetric two-hand one (0.19).
    winners.sort(key=lambda n: (len(n["seq"]), -sum(s_.count("+") + 1 for s_ in n["seq"]),
                                n.get("thick", 0.0), n["area"]))
    best = winners[0]
    print(f"[label-tamp] PLAN: {' > '.join(best['seq'])}  (area {best['area']:.2f} of flat, worst grip {best['grip']})", flush=True)
    if a.out:
        json.dump({"garment": Lj["garment"], "plan": best["seq"], "area_frac": best["area"], "labels": a.labels}, open(a.out, "w"), indent=1)
    if a.rec_usd:
        # replay the winning sequence in world 0 with the recorder (same scheme as fold_tamp)
        from pxr import Gf, Usd, UsdGeom, UsdLux
        stage = Usd.Stage.CreateNew(a.rec_usd); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        every = 4; stage.SetTimeCodesPerSecond(env.FPS / every); stage.SetStartTimeCode(0)
        world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
        tris_all = env.model.tri_indices.numpy().reshape(-1, 3); tpw = len(tris_all) // env.n; tris0 = tris_all[:tpw] % env.per_world
        m = UsdGeom.Mesh.Define(stage, "/World/garment"); m.CreateFaceVertexIndicesAttr(tris0.flatten().tolist()); m.CreateFaceVertexCountsAttr([3] * tpw)
        m.CreateDoubleSidedAttr(True); m.CreateDisplayColorAttr([Gf.Vec3f(0.40, 0.10, 0.26)]); pts = m.GetPointsAttr()
        g = UsdGeom.Cube.Define(stage, "/World/floor"); g.CreateSizeAttr(1.0); xg = UsdGeom.Xformable(g)
        c = q0[:, :2].mean(0); xg.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), -5.5)); xg.AddScaleOp().Set(Gf.Vec3f(200.0, 200.0, 10.0))
        g.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])
        UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000.0); UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(400.0)
        # label markers as small spheres that follow their particles
        marks = {}
        for nm, l in L.items():
            sp = UsdGeom.Sphere.Define(stage, f"/World/labels/{nm}"); sp.CreateRadiusAttr(1.2)
            sp.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2) if l["kind"] == "corner" else Gf.Vec3f(0.1, 0.7, 1.0)])
            marks[nm] = (UsdGeom.Xformable(sp).AddTranslateOp(), l["particle"])
        ctr = {"f": 0, "t": 0}
        def cb(e):
            ctr["f"] += 1
            if ctr["f"] % every: return
            q = e.state_0.particle_q.numpy()[:e.per_world]; tc = Usd.TimeCode(ctr["t"])
            if np.isfinite(q).all():
                pts.Set([Gf.Vec3f(*map(float, v)) for v in q], tc)
                for nm, (op, j) in marks.items(): op.Set(Gf.Vec3d(*map(float, q[j] + [0, 0, 1.5])), tc)
            ctr["t"] += 1
        env.frame_cb = cb
        copies = []
        for w in range(env.n):
            s = q0.copy(); dx, dy = env.world_xy[w]; s[:, 0] += dx; s[:, 1] += dy; copies.append(s)
        env.reset(q=np.concatenate(copies, 0), randomize=False)
        for step_name in best["seq"]:
            mv = tuple(tuple(p.split("->")) for p in step_name.split(" + "))
            env.step([to_action(mv, L)] + [FoldAction(side=0)] * (env.n - 1))
        env.frame_cb = None
        stage.SetEndTimeCode(max(ctr["t"] - 1, 1)); stage.GetRootLayer().Save()
        print(f"[label-tamp] recorded {ctr['t']} frames -> {a.rec_usd}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
