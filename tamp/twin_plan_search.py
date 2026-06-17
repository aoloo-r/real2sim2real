"""TAMP plan-candidate generator — stage 1 of the parallel task-plan search engine.

Given a grounded goal (from tamp_plan.py) over ANY scene, enumerate/sample N candidate
PLANS that each achieve the goal a different way, varying BOTH task and motion:
  - task : the ORDER in which the goal placements are executed
  - motion: per-action GRASP strategy (rim height / center) and PLACEMENT offset

This is object-agnostic — it operates on the goal predicates + object labels, never on
specific object names. The parallel rollout engine (stage 2) then runs each candidate in
a cloned twin under physics and scores it (goals/stability/collisions/efficiency); the
best is selected and later evolved (Simify-style, stage 3).

Output: candidates.json = {"goal":..., "candidates":[{plan_id, order, actions:[...]}, ...]}

Usage:
  python twin_plan_search.py --task_plan /tmp/task_plan.json --n 8 --out /tmp/candidates.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random


# Discrete motion knobs searched per action (general, not object-specific).
GRASP_VARIANTS = [
    {"name": "rim_high", "rim_frac": 0.92},   # high on the wall (good for cups/bowls)
    {"name": "rim_mid",  "rim_frac": 0.70},
    {"name": "rim_low",  "rim_frac": 0.50},
    {"name": "center",   "rim_frac": 0.92, "grip_max": 0.12},  # force a center grasp
]
PLACE_OFFSETS = [                              # where within the target to release (m)
    {"name": "center", "dx": 0.0,  "dy": 0.0},
    {"name": "near",   "dx": -0.03, "dy": 0.0},
    {"name": "side",   "dx": 0.0,  "dy": 0.03},
]


def base_actions_from_goal(plan: dict) -> list[dict]:
    """The K goal placements as base actions (object -> relation -> target)."""
    acts = plan.get("actions") or []
    if acts:
        return [{"object_label": a.get("object_label"), "object": a.get("object"),
                 "target_label": a.get("target_label"), "target": a.get("target"),
                 "relation": a.get("relation", "on")} for a in acts]
    # fall back to goal_predicates if no actions were emitted
    out = []
    for g in plan.get("goal_predicates", []):
        out.append({"object": g.get("object"), "target": g.get("target"),
                    "relation": g.get("relation", "on"),
                    "object_label": None, "target_label": None})
    return out


def gen_candidates(plan: dict, n: int, seed: int = 0) -> list[dict]:
    base = base_actions_from_goal(plan)
    K = len(base)
    if K == 0:
        return []
    rng = random.Random(seed)

    # task-level: orderings of the K actions (cap permutations, sample if large)
    if K <= 3:
        orders = list(itertools.permutations(range(K)))
    else:
        orders = [tuple(range(K))]
        for _ in range(min(8, n)):
            o = list(range(K)); rng.shuffle(o); orders.append(tuple(o))
        orders = list(dict.fromkeys(orders))           # unique

    # build a large pool: order x (grasp,place per action), then sample N
    pool = []
    for order in orders:
        # each action independently picks a grasp + place variant
        per_action_choices = [list(itertools.product(GRASP_VARIANTS, PLACE_OFFSETS))
                              for _ in range(K)]
        # sample a few motion combos per order rather than full cross product
        for _ in range(max(1, n)):
            actions = []
            for ai in order:
                g, p = rng.choice(per_action_choices[ai])
                a = dict(base[ai])
                a["grasp"] = {k: v for k, v in g.items() if k != "name"}
                a["grasp_name"] = g["name"]
                a["place_offset"] = {"dx": p["dx"], "dy": p["dy"]}
                a["place_name"] = p["name"]
                actions.append(a)
            pool.append({"order": list(order), "actions": actions})

    # de-dup by a signature, then sample N
    seen, uniq = set(), []
    for c in pool:
        sig = (tuple(c["order"]),
               tuple((a["grasp_name"], a["place_name"]) for a in c["actions"]))
        if sig in seen:
            continue
        seen.add(sig); uniq.append(c)
    rng.shuffle(uniq)
    chosen = uniq[:n]
    for i, c in enumerate(chosen):
        c["plan_id"] = i
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_plan", required=True, help="task_plan.json from tamp_plan.py")
    ap.add_argument("--n", type=int, default=8, help="number of candidate plans")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/candidates.json")
    args = ap.parse_args()

    plan = json.load(open(args.task_plan))
    cands = gen_candidates(plan, args.n, args.seed)
    lbl = {o["id"]: o["label"] for o in plan.get("objects", [])}

    print(f"[PLAN-SEARCH] goal: {plan.get('instruction')}")
    print(f"[PLAN-SEARCH] {len(cands)} candidate plans (K={len(base_actions_from_goal(plan))} placements):\n")
    for c in cands:
        steps = " ; ".join(
            f"{(a.get('object_label') or lbl.get(a.get('object'),'?'))}"
            f"->{a['relation']}->{(a.get('target_label') or lbl.get(a.get('target'),'?'))}"
            f" [{a['grasp_name']},{a['place_name']}]"
            for a in c["actions"])
        print(f"  plan {c['plan_id']}: {steps}")

    out = {"instruction": plan.get("instruction"), "frame": plan.get("frame"),
           "objects": plan.get("objects", []), "candidates": cands}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[PLAN-SEARCH] wrote {len(cands)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
