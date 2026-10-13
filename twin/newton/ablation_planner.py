"""ABLATION: does the symbolic layer earn its place?

Two planners get the SAME operator set, the SAME physics (fold_env), the SAME goal test and the
SAME rollout budget, from the SAME real start states (fitted from RGBD, plus SE(2) perturbations):

  skeleton  symbolic skeletons that reach the goal in the abstract model (fold_tamp), validated in
            physics in diverse order until the budget is spent
  blind     random operator sequences of the same lengths, validated identically

Measured per start: whether a physics-valid plan was found, rollouts to the first one, the best
plan's folded area, and the symbolic model's precision (endorsed skeletons that pass physics).
If blind matches skeleton at equal budget, the abstraction is decoration.

    python ablation_planner.py --starts flat=/tmp/foldrun6/observed.npy half=/tmp/foldrun3/observed.npy \
        --perturb 2 --budget 18 --worlds 9 --goal-area 0.25 --out /tmp/ablation
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def se2_perturb(q, rng, rot_deg=15.0, shift_cm=8.0):
    q = q.copy()
    c = q[:, :2].mean(0)
    th = np.radians(rng.uniform(-rot_deg, rot_deg))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q[:, :2] = (q[:, :2] - c) @ R.T + c + rng.uniform(-shift_cm, shift_cm, size=2)
    return q


def diverse(cands, k):
    """fold_tamp's ordering: unique openings first, then unique pairs, then the rest."""
    out, seen1, seen2 = [], set(), set()
    for pool in (0, 1, 2):
        for c in cands:
            if len(out) >= k:
                return out
            if c in out:
                continue
            if pool == 0 and c[0] in seen1:
                continue
            if pool == 1 and tuple(c[:2]) in seen2:
                continue
            out.append(c); seen1.add(c[0]); seen2.add(tuple(c[:2]))
    return out


def validate(env, q0, seqs, goal, flat_area, log):
    """Run `seqs` in batches of env.n from start q0 (world-0 coords). Returns per-seq results."""
    from fold_env import FoldAction
    from fold_search import ACTIONS, succeeded
    out = []
    for b0 in range(0, len(seqs), env.n):
        batch = seqs[b0:b0 + env.n]
        tiles = []
        for w in range(env.n):
            t = q0.copy(); t[:, 0] += env.world_xy[w][0]; t[:, 1] += env.world_xy[w][1]
            tiles.append(t)
        env.reset(q=np.concatenate(tiles, 0), randomize=False, settle_s=0.5)
        worst = [10 ** 6] * env.n
        t0 = time.time()
        for step in range(max(len(s) for s in batch)):
            acts = [ACTIONS[batch[w][step]] if w < len(batch) and step < len(batch[w]) else FoldAction(side=0)
                    for w in range(env.n)]
            env.step(acts)
            g = env.grip_counts()
            for w in range(len(batch)):
                if step < len(batch[w]):
                    worst[w] = min(worst[w], int(g[w]))
        ext = env.extents(); qs = env.particles()
        for w, s in enumerate(batch):
            zz = qs[w][:, 2]
            thick = float(np.percentile(zz, 97) - np.percentile(zz, 3)) if np.isfinite(zz).all() else None
            ok, why = succeeded(ext[w], env.sleeve_overhang_of(qs[w]), worst[w], goal=goal, flat_area=flat_area,
                                thickness=thick)
            e = ext[w]
            area = float(e[0] * e[1] / flat_area) if np.isfinite(e).all() else float("nan")
            out.append({"seq": list(s), "ok": bool(ok), "why": why, "area_frac": area})
        log(f"      batch {b0 // env.n + 1}: {sum(r['ok'] for r in out[b0:])}/{len(batch)} pass ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", nargs="+", required=True, help="name=path/to/observed.npy ...")
    ap.add_argument("--perturb", type=int, default=2, help="extra SE(2)-perturbed copies per start")
    ap.add_argument("--budget", type=int, default=18, help="physics rollouts per planner per start")
    ap.add_argument("--worlds", type=int, default=9)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--goal-area", type=float, default=0.25)
    ap.add_argument("--scale", default="1.0059,0.9809")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/ablation")
    a = ap.parse_args()

    from fold_env import FoldEnv
    from fold_search import ACTIONS, GOAL
    from fold_tamp import OPERATORS, skeletons, state_from_env
    from fold_geom import GeomModel, skeletons_measured
    goal = dict(GOAL); goal["max_area_frac"] = a.goal_area
    op_names = list(OPERATORS)                       # the symbolic planner's vocabulary
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/ablation.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    env = FoldEnv(n_worlds=a.worlds, spacing_cm=130.0, cloth="shirt",
                  scale_xy=tuple(float(v) for v in a.scale.split(",")))
    env.reset(); env.measure_flat()
    flat_area = env.FLAT_CM[0] * env.FLAT_CM[1]
    geom = GeomModel(env)
    geom_actions = {n: ACTIONS[n] for n in op_names}      # same vocabulary as the symbolic planner
    log(f"[ablation] flat {env.FLAT_CM[0]:.0f}x{env.FLAT_CM[1]:.0f}cm, goal area <= {a.goal_area:.0%}, "
        f"budget {a.budget} rollouts/planner/start, operators {op_names}")

    results = []
    for spec in a.starts:
        name, path = spec.split("=", 1)
        q_base = np.load(path).astype(np.float32).reshape(-1, 3)
        q_base[:, 2] -= q_base[:, 2].min() - 0.5
        variants = [("orig", q_base)] + [(f"p{k}", se2_perturb(q_base, rng)) for k in range(a.perturb)]
        for vname, q0 in variants:
            # symbolic start state read off the settled cloth
            tiles = [q0.copy() for _ in range(env.n)]
            for w in range(env.n):
                tiles[w][:, 0] += env.world_xy[w][0]; tiles[w][:, 1] += env.world_xy[w][1]
            env.reset(q=np.concatenate(tiles, 0), randomize=False, settle_s=0.5)
            st = state_from_env(env)
            sk = [tuple(s) for s in skeletons(st, max_depth=a.depth)]
            sk = [s for s in sk if all(n in ACTIONS for n in s)]
            lengths = [len(s) for s in sk] or [2, 3, 4]
            sk_pick = diverse(sk, a.budget)
            blind = [tuple(random.choice(op_names) for _ in range(random.choice(lengths))) for _ in range(a.budget)]
            # MEASURED planner: geometric operator model, predicates re-measured every step
            t0 = time.time()
            q_cur = env.particles()[0].copy()
            gm = skeletons_measured(geom, q_cur, geom_actions, max_depth=a.depth, max_area=a.goal_area)
            gm_seqs = [g[0] for g in gm]
            gm_pick = diverse(gm_seqs, a.budget)
            gp = geom.predicates(q_cur)
            log(f"[ablation] start {name}/{vname}: state {st}, {len(sk)} skeletons ({len(sk_pick)} within budget); "
                f"measured: area {gp['area_frac']:.2f} layers {gp['layers']:.1f} sleeves {gp['sleeves']} -> "
                f"{len(gm)} geometric skeletons ({len(gm_pick)} within budget, {time.time() - t0:.1f}s)")
            rec = {"start": name, "variant": vname, "state": st, "n_skeletons": len(sk),
                   "measured_state": gp, "n_geom_skeletons": len(gm)}
            for planner, seqs in (("skeleton", sk_pick), ("measured", gm_pick), ("blind", blind)):
                if not seqs:
                    rec[planner] = {"found": False, "rollouts_to_first": None, "best_area": None,
                                    "passed": 0, "tried": 0}
                    log(f"   {planner}: no candidates"); continue
                log(f"   {planner}: {len(seqs)} sequences")
                res = validate(env, q0, seqs, goal, flat_area, log)
                oks = [i for i, r in enumerate(res) if r["ok"]]
                best = min((r["area_frac"] for r in res if r["ok"]), default=None)
                rec[planner] = {"found": bool(oks), "rollouts_to_first": (oks[0] + 1) if oks else None,
                                "best_area": best, "passed": len(oks), "tried": len(res),
                                "plans": [r for r in res if r["ok"]][:3]}
                log(f"   {planner}: found={bool(oks)} first@{(oks[0] + 1) if oks else '-'} "
                    f"passed {len(oks)}/{len(res)} best area {best if best is None else round(best, 3)}")
            results.append(rec)
            with open(f"{a.out}/results.json", "w") as fh:
                json.dump(results, fh, indent=1, default=float)

    # summary
    log("\n[ablation] SUMMARY (per start: found / rollouts to first / passed of tried / best area)")
    for planner in ("skeleton", "measured", "blind"):
        found = sum(r[planner]["found"] for r in results)
        firsts = [r[planner]["rollouts_to_first"] for r in results if r[planner]["found"]]
        prec = sum(r[planner]["passed"] for r in results) / max(1, sum(r[planner]["tried"] for r in results))
        log(f"   {planner:9s}: plans found on {found}/{len(results)} starts; median rollouts to first "
            f"{np.median(firsts) if firsts else float('nan'):.1f}; precision (passed/tried) {prec:.2f}")
    for r in results:
        log("   %-14s %-24s skel %s@%s (%d/%d) | measured %s@%s (%d/%d) | blind %s@%s (%d/%d)" % (
            f"{r['start']}/{r['variant']}", r["state"]["size"] + "/" + r["state"]["sleeves"],
            "Y" if r["skeleton"]["found"] else "n", r["skeleton"]["rollouts_to_first"] or "-",
            r["skeleton"]["passed"], r["skeleton"]["tried"],
            "Y" if r["measured"]["found"] else "n", r["measured"]["rollouts_to_first"] or "-",
            r["measured"]["passed"], r["measured"]["tried"],
            "Y" if r["blind"]["found"] else "n", r["blind"]["rollouts_to_first"] or "-",
            r["blind"]["passed"], r["blind"]["tried"]))
    log("ABLATION_DONE")


if __name__ == "__main__":
    main()
