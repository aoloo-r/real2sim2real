"""TAMP for the sled task: pack every object into the box when some of them are OUT OF REACH.

The objects stand on a towel; the arm can reach the towel's near edge but not its far end. Pulling
the towel brings the far objects within reach - if they survive the ride. So the plan interleaves
  pick_place(o -> box)   needs o within reach and upright
  drag(towel, D, accel)  moves every object still on the towel; it can topple or shift them
and the planner decides which reachable objects to pack BEFORE the pull (they then do not ride),
how far to pull and how hard. Rigid-body formulas give the first guess; every candidate is then
checked by a ROLLOUT of the pull in the twin, and refined from what failed (best-first, cheapest
estimated time first).

Usage: tamp_sled.py --scene scene.json --out plan.json [--max-rollouts 12]
"""
import argparse
import heapq
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACCELS = [300.0, 150.0, 75.0, 40.0, 20.0]      # pull acceleration levels [cm/s^2]
VMAX = 25.0
T_PICK, T_DRAG_FIXED = 10.0, 6.0
G = 981.0
REACH_MARGIN = 3.0
D_STEP, D_MAX = 6.0, 60.0


def drag_time(D, acc):
    v = min(VMAX, math.sqrt(acc * D))
    return 2 * v / acc + (D - v * v / acc) / v


SIDE_GAP, BACK_GAP, OBJ_GAP, STRIP_GAP = 1.5, 3.0, 1.2, 2.0
# folds a single gripper can do here, as (name, resulting size factor along the length, across, corner moves)
# Two-handed folds that keep each hand on its own side of the towel: the FAR edge is carried onto
# the NEAR edge (toward the arms), left corner by the left hand, right corner by the right hand.
# Folding across the width would send one hand across the other's side, so it is not offered.
FOLD_SEQS = [[], ["far_to_near"], ["far_to_near", "far_to_near"]]
T_FOLD = 12.0


def fold_moves(k):
    """Corner moves of the k-th far-to-near fold (material coordinates u across, v along; v=0 far)."""
    v_far = 1.0 - 0.5 ** (k - 1)                                # where the far edge (a crease after the first fold) is now
    return [{"hand": "left", "from": [0.05, v_far + 0.02], "to": [0.05, 0.98]}, {"hand": "right", "from": [0.95, v_far + 0.02], "to": [0.95, 0.98]}]


def layout(scene):
    """Objects in rows along the box's BACK wall, so the floor in front of them stays free as one
    strip for the folded towel. Returns the slots and the strip (x0, x1, y0, y1)."""
    from twin_sled import BOX_XY
    inner = float(scene["box"]["inner_cm"][0])
    import props
    foot = {o["name"]: props.footprint(o) for o in scene["objects"]}   # width, depth, body-centre offset (a mug's handle adds width)
    objs = sorted(scene["objects"], key=lambda o: -foot[o["name"]][1])
    rows, cur, wsum = [], [], 0.0
    for o in objs:
        w = foot[o["name"]][0]
        if cur and wsum + OBJ_GAP + w > inner - 2 * SIDE_GAP:
            rows.append(cur); cur, wsum = [], 0.0
        wsum += (OBJ_GAP if cur else 0.0) + w
        cur.append(o)
    rows.append(cur)
    slot, y_back = {}, BOX_XY[1] + inner / 2 - BACK_GAP
    for row in rows:
        depth = max(foot[o["name"]][1] for o in row)
        tot = sum(foot[o["name"]][0] for o in row) + OBJ_GAP * (len(row) - 1)
        x = BOX_XY[0] - tot / 2
        for o in row:
            fw, _, dx = foot[o["name"]]
            slot[o["name"]] = [round(x + fw / 2 + dx, 1), round(y_back - depth / 2, 1)]   # where the body's centre goes
            x += fw + OBJ_GAP
        y_back -= depth + OBJ_GAP
    strip = (BOX_XY[0] - inner / 2 + SIDE_GAP, BOX_XY[0] + inner / 2 - SIDE_GAP, BOX_XY[1] - inner / 2 + 1.0, y_back - STRIP_GAP + OBJ_GAP)
    return slot, strip, len(rows)


def choose_fold(scene, strip):
    """Fewest corner moves that make the towel fit the free strip (long side along the strip)."""
    W, L = scene["towel"]["size_cm"]                             # width along x, length along y
    sw, sd = strip[1] - strip[0], strip[3] - strip[2]
    tried = []
    for seq in FOLD_SEQS:
        a, b = W, L / (2 ** len(seq))                             # folded size: along x, along y
        fits = a <= sw - 2.0 and b <= sd - 1.0
        tried.append(dict(folds=list(seq), size_cm=[a, b], pinches=len(seq), fits=bool(fits)))
        if fits:
            return seq, (a, b), tried
    return None, None, tried


def make_plan(scene, first, acc, extra_D=0.0):
    from twin_sled import BASES_XY, R_PICK, BOX_XY
    cx, cy = scene["towel"]["centre_xy"]
    W, L = scene["towel"]["size_cm"]
    objs = {o["name"]: o for o in scene["objects"]}
    pos = {n: np.array([cx + o["on_towel_xy"][0], cy + o["on_towel_xy"][1]]) for n, o in objs.items()}
    riders = [n for n in objs if n not in first]
    slot, strip, n_rows = layout(scene)
    want_towel = scene.get("pack_towel", True)
    seq, size, tried = choose_fold(scene, strip) if want_towel else ([], None, [])
    if want_towel and seq is None:
        raise SystemExit(f"[sled-plan] the towel fits the free strip {strip[1] - strip[0]:.0f} x {strip[3] - strip[2]:.0f} cm with none of the folds: {tried}")

    def y_need(x):                                               # how near (in y) a point at this x must be for the closer arm
        dx = min(abs(x - b[0]) for b in BASES_XY.values())
        return list(BASES_XY.values())[0][1] - math.sqrt(max((R_PICK - REACH_MARGIN) ** 2 - dx * dx, 0.0))
    # pull along +y (toward the arms) until (a) the farthest rider and (b) every corner the folds touch are within reach
    D_obj = max([0.0] + [y_need(pos[n][0]) - pos[n][1] for n in riders])
    D_fold = max(y_need(cx + s_ * (W / 2 - 1.5)) - (cy - L / 2 + 1.0) for s_ in (-1.0, 1.0)) if (want_towel and seq) else 0.0
    D = max(D_obj, D_fold, 0.0)
    D = D + extra_D if D > 0 else 0.0
    steps = [{"op": "pick_place", "name": n, "to_xy": slot[n]} for n in first]
    if D > 0:
        steps.append({"op": "drag", "dir": [0.0, 1.0], "dist_cm": round(D, 1), "accel": acc, "vmax": VMAX, "lift_cm": 1.0,
                      "grasp_uv": {"left": [0.25, 0.98], "right": [0.75, 0.98]}})
    steps += [{"op": "pick_place", "name": n, "to_xy": slot[n]} for n in sorted(riders, key=lambda n: -pos[n][1])]
    towel_place = None
    if want_towel:
        cur = [W, L]
        for k in range(1, len(seq) + 1):
            cur = [cur[0], cur[1] / 2]
            steps.append({"op": "fold", "name": f"far edge onto near edge ({k})", "moves": fold_moves(k), "to_cm": list(cur)})
        ys = strip[2] + 0.5 + size[1] / 2                           # against the box's front wall
        v_mid = 1.0 - 0.5 * 0.5 ** len(seq)
        steps.append({"op": "pack_towel", "grasp_uv": {"left": [0.05, v_mid], "right": [0.95, v_mid]}, "band_half_cm": 4.0,
                      "to_xy": [BOX_XY[0], round(ys, 1)], "clear_cm": 10.0})
        towel_place = dict(rect=[BOX_XY[0] - size[0] / 2, BOX_XY[0] + size[0] / 2, ys - size[1] / 2, ys + size[1] / 2], size_cm=list(size),
                           folds=len(seq), fold_options=tried, strip=[round(v, 1) for v in strip])
    est = (T_PICK * len(objs) + (T_DRAG_FIXED + drag_time(D, acc) if D > 0 else 0.0) + (T_FOLD * len(seq) + 14.0 if towel_place else 0.0))
    return {"task": "sled", "scene": scene, "plan": steps, "riders": riders, "first": list(first), "accel": acc, "dist_cm": round(D, 1),
            "pull_for": dict(objects_cm=round(D_obj, 1), folding_cm=round(D_fold, 1)), "towel_place": towel_place, "rows": n_rows,
            "est_s": round(est, 1)}, {n: slot[n] for n in first}


def analytic(scene):
    import props
    mu_t = scene["towel"].get("mu", 0.5)
    return {o["name"]: dict(slip=math.sqrt(mu_t * o.get("mu", 0.5)) * G, topple=G * min(o["dims_cm"][0], o["dims_cm"][1]) / (2.0 * props.com_height(o)))
            for o in scene["objects"]}


def search(scene, max_rollouts=12, verbose=True):
    from twin_sled import SledSim, reachable
    cx, cy = scene["towel"]["centre_xy"]
    names = [o["name"] for o in scene["objects"]]
    start = {o["name"]: (cx + o["on_towel_xy"][0], cy + o["on_towel_xy"][1]) for o in scene["objects"]}
    near = [n for n in names if reachable(start[n])]
    far = [n for n in names if n not in near]
    an = analytic(scene)
    lim = min(min(an[n].values()) for n in names)
    a0 = next((i for i, a in enumerate(ACCELS) if a <= 0.7 * lim), len(ACCELS) - 1)
    heap, seen, trace, k, best = [], set(), [], 0, None

    def push(first, ai, xd):
        key = (first, ai, xd)
        if key in seen or ai >= len(ACCELS):
            return
        p, _ = make_plan(scene, first, ACCELS[ai], xd)
        if p["dist_cm"] > D_MAX:
            return
        seen.add(key)
        heapq.heappush(heap, (p["est_s"], len(seen), first, ai, xd))

    push((), a0, 0.0)                                             # first guess: pull first, with everything on board
    while heap and k < max_rollouts:
        est, _, first, ai, xd = heapq.heappop(heap)
        plan, pre = make_plan(scene, first, ACCELS[ai], xd)
        k += 1
        roll = dict(plan); roll["preplaced"] = pre
        roll["plan"] = [s for s in plan["plan"] if s["op"] == "drag"]
        res = SledSim(roll, with_arm=False, record=False, verbose=False).run()
        bad = {n: v["status"] for n, v in res["objects"].items() if v["status"] != "ok"}
        trace.append(dict(first=list(first), accel=ACCELS[ai], dist_cm=plan["dist_cm"], est_s=est, ok=not bad, failed=bad,
                          end={n: v["xy"] for n, v in res["objects"].items()}, wall_s=res["wall_s"]))
        if verbose:
            print(f"[sled-plan] rollout {k}: pack {list(first) or 'nothing'} first, pull {plan['dist_cm']:.0f} cm at {ACCELS[ai]:.0f} cm/s2 -> "
                  + ("OK" if not bad else "FAIL: " + ", ".join(f"{n} {s}" for n, s in bad.items())), flush=True)
        if not bad and res["completed"]:
            best = plan
            break
        toppled = [n for n, s in bad.items() if s == "toppled"]
        if [n for n in toppled if n in near]:
            push(tuple(sorted(set(first) | {n for n in toppled if n in near})), ai, xd)   # pack the fragile reachable ones before pulling
        if toppled:
            push(first, ai + 1, xd)                                                       # or pull more gently
        if any(s == "unreachable" for s in bad.values()):
            push(first, ai, xd + D_STEP)                                                  # it slipped back: pull further
    if best is None:
        raise SystemExit(f"[sled-plan] no plan passed in {k} rollouts")
    best["search"] = dict(rollouts=k, trace=trace, reachable_at_start=near, out_of_reach=far,
                          analytic={n: {q: round(v, 0) for q, v in d.items()} for n, d in an.items()})
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rollouts", type=int, default=12)
    a = ap.parse_args()
    scene = json.load(open(a.scene))
    for n, d in analytic(scene).items():
        print(f"[sled-plan] {n}: rigid-body limits - slips above {d['slip']:.0f} cm/s2, topples above {d['topple']:.0f} cm/s2")
    p = search(scene, a.max_rollouts)
    json.dump(p, open(a.out, "w"), indent=1)
    s = p["search"]
    print(f"[sled-plan] within reach at the start: {s['reachable_at_start']}; out of reach: {s['out_of_reach']} "
          f"(picking alone packs {len(s['reachable_at_start'])} of {len(scene['objects'])})")
    tp = p.get("towel_place")
    if tp:
        print(f"[sled-plan] box: objects in {p['rows']} row(s) at the back; free strip {tp['strip'][1] - tp['strip'][0]:.0f} x {tp['strip'][3] - tp['strip'][2]:.0f} cm in front")
        for o in tp["fold_options"]:
            print(f"[sled-plan]   towel with {len(o['folds'])} fold(s) = {o['size_cm'][0]:.0f} x {o['size_cm'][1]:.0f} cm: "
                  + ("FITS - chosen" if o["fits"] else "does not fit"))
        print(f"[sled-plan] pull distance: {p['pull_for']['objects_cm']:.0f} cm brings the objects within reach, {p['pull_for']['folding_cm']:.0f} cm brings the "
              f"whole towel within reach for folding -> {p['dist_cm']:.0f} cm")
    label = {"pick_place": lambda st: f"pack {st['name']}", "drag": lambda st: f"PULL towel {st['dist_cm']:.0f} cm at {st['accel']:.0f} cm/s2",
             "fold": lambda st: f"fold to {st['to_cm'][0]:.0f}x{st['to_cm'][1]:.0f}", "pack_towel": lambda st: "pack towel"}
    print(f"[sled-plan] PLAN ({s['rollouts']} rollouts): " + " > ".join(label[st["op"]](st) for st in p["plan"]) + f"; est {p['est_s']:.0f}s")
