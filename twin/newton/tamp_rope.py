"""TAMP for coiling a rope into a box with two arms.

The motion primitive is a LAY STROKE: a hand grasps the rope one loop-length further along than
the last point already in the box, lifts it above that point so the loop hangs, then lowers while
circling once - the hanging rope lays itself down as one loop of the coil. Hands alternate
(hand-over-hand); only the laying hand is in the box.

The planner chooses the coil radius, which fixes the loop length, the number of loops (= stack
height) and the number of strokes, subject to:
  bend      coil radius >= MIN_BEND_DIAMETERS * rope diameter
  hang      the hanging loop must fit under the highest tip pose:  HANG * loop <= lift_max
  regrasp   the next grasp point must lie on the table outside the front wall, so one loop must
            be long enough to run from the coil's front point up the wall and down to the table
  footprint coil + rope + gripper clearance to the walls fits the box floor (minus `reserve_cm`,
            a strip at the back kept free for other objects)
  height    loops * diameter stays below the rim
Objective: fewest strokes, then smallest footprint (leaves floor for other objects).

Usage: tamp_rope.py --scene scene.json --out plan.json
"""
import argparse
import json
import math

MIN_BEND_DIAMETERS = 4.0     # tightest coil radius, in rope diameters
HANG = 0.85                  # hand height above the anchor, as a fraction of the hanging length
GRIPPER_WALL_CM = 5.0        # coil's nearest point to a wall (gripper narrow side faces the wall)
TABLE_GRASP_CLEAR_CM = 8.0   # the next grasp point lies at least this far outside the wall
RIM_MARGIN_CM = 2.0          # coil top stays this far below the rim
FINGER_CLEAR_CM = 1.5        # between the former block's corners and the rope ring
WALL_GAP_CM = 2.0            # block edge to a wall
BLOCK_GAP_CM = 2.0           # between blocks placed side by side
STROKE_S = 14.0              # rough duration of one stroke [s], for the time estimate


def plan(scene, core_order="first"):
    box, rope = scene["box"], scene["rope"]
    inner_x, inner_y = [float(v) for v in box["inner_cm"]]
    h = float(box["h_cm"])
    wall = float(box.get("wall_cm", 1.0))
    L, d = float(rope["length_cm"]), float(rope["diameter_cm"])
    lift_max = float(scene.get("lift_max_cm", 50.0))
    reserve = float(scene.get("reserve_cm", 0.0))
    regrasp_min = GRIPPER_WALL_CM + 2.0 * h + wall + TABLE_GRASP_CLEAR_CM
    rigids = scene.get("rigids", [])
    # A rigid block can serve as the FORMER: it stands in the middle of the ring and the rope is
    # coiled around it. It must fit inside the ring with room for the fingers, so it bounds the
    # coil radius from below. The tallest block that fits is chosen (it holds the most loops).
    core, core_min_rc = None, 0.0
    for g in sorted(rigids, key=lambda g: -g["dims_cm"][2]):
        need = 0.5 * math.hypot(g["dims_cm"][0], g["dims_cm"][1]) + 0.5 * d + FINGER_CLEAR_CM
        if need <= min(inner_x, inner_y) / 2.0 - GRIPPER_WALL_CM - 0.5 * d and HANG * 2 * math.pi * need <= lift_max:
            core, core_min_rc = g, need
            break
    cands, rejected = [], {}
    for i in range(int(2 * MIN_BEND_DIAMETERS * d), 2 * 40 + 1):
        rc = 0.5 * i
        loop = 2.0 * math.pi * rc
        loops = L / loop
        strokes = math.ceil(L / loop - 1e-6)
        stack = math.ceil(loops) * d
        why = None
        if rc < MIN_BEND_DIAMETERS * d:
            why = "bend"
        elif rc < core_min_rc:
            why = "core"
        elif HANG * loop > lift_max:
            why = "hang"
        elif loop < regrasp_min:
            why = "regrasp"
        elif 2 * rc + d + 2 * GRIPPER_WALL_CM > min(inner_x, inner_y - reserve):
            why = "footprint"
        elif stack > h - RIM_MARGIN_CM:
            why = "height"
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        cands.append((strokes, rc, loop, loops, stack))
    if not cands:
        raise SystemExit(f"[rope-plan] no feasible coil: rejected {rejected} (regrasp needs loop >= {regrasp_min:.0f} cm, "
                         f"hang needs loop <= {lift_max / HANG:.0f} cm)")
    cands.sort(key=lambda c: (c[0], c[1]))
    strokes, rc, loop, loops, stack = cands[0]
    # coil sits against the FRONT wall (the side the rope is fed from) so the tail leaves the box
    # by the shortest route; `centre_dy` is its centre relative to the box centre
    centre_dy = -inner_y / 2.0 + GRIPPER_WALL_CM + rc
    grasps = [min(k * loop, L - 2.0) for k in range(1, strokes + 1)]
    steps, placed = [], {}
    if core:
        # "first": the block is a FORMER, in place before the rope is coiled around it.
        # "last":  the block is a KEEPER, set into the hollow of the finished coil.
        placed[core["name"]] = {"dxy": [0.0, round(centre_dy, 2)], "z_on": 0.0, "role": "former" if core_order == "first" else "keeper"}
    if core and core_order == "first":
        steps.append({"op": "pick_place", "name": core["name"], "dxy": placed[core["name"]]["dxy"], "z_on": 0.0, "hand": "right"})
    steps.append({"op": "place_end", "s_cm": 0.0, "hand": "left"})
    for k, s in enumerate(grasps):
        steps.append({"op": "lay_loop", "s_cm": round(s, 2), "hand": "right" if k % 2 == 0 else "left",
                      "arc_cm": round(s - (grasps[k - 1] if k else 0.0), 2)})
    # the other blocks: on the floor BEHIND the coil if they fit there, else stacked on the former.
    # They go in AFTER the rope: the laying hand and the trailing rope sweep that floor.
    y_lo, y_hi = centre_dy + rc + 0.5 * d + 2.0, inner_y / 2.0 - WALL_GAP_CM
    rest = [g for g in rigids if g is not core]
    total_w = sum(g["dims_cm"][0] for g in rest) + BLOCK_GAP_CM * max(0, len(rest) - 1)
    stack_z = core["dims_cm"][2] if core else 0.0
    if rest and total_w <= inner_x - 2 * WALL_GAP_CM and all(g["dims_cm"][1] <= y_hi - y_lo for g in rest):
        x = -total_w / 2.0
        for g in rest:
            placed[g["name"]] = {"dxy": [round(x + g["dims_cm"][0] / 2.0, 2), round(y_hi - g["dims_cm"][1] / 2.0, 2)], "z_on": 0.0, "role": "beside"}
            x += g["dims_cm"][0] + BLOCK_GAP_CM
    else:
        for g in sorted(rest, key=lambda g: -g["dims_cm"][0] * g["dims_cm"][1]):
            if not core or stack_z + g["dims_cm"][2] > h - RIM_MARGIN_CM + 2.0:
                raise SystemExit(f"[rope-plan] no place for {g['name']}: floor strip {y_hi - y_lo:.0f} cm deep, stack would reach {stack_z + g['dims_cm'][2]:.0f} cm")
            placed[g["name"]] = {"dxy": [0.0, round(centre_dy, 2)], "z_on": stack_z, "role": "stacked"}
            stack_z += g["dims_cm"][2]
    for g in ([core] if core and core_order != "first" else []) + rest:
        steps.append({"op": "pick_place", "name": g["name"], "dxy": placed[g["name"]]["dxy"], "z_on": placed[g["name"]]["z_on"], "hand": "right"})
    out = {
        "task": "coil_rope", "scene": scene, "rigids": placed,
        "coil": {"radius_cm": rc, "loop_cm": round(loop, 2), "loops": round(loops, 2), "stack_cm": stack,
                 "centre_dxy": [0.0, round(centre_dy, 2)], "direction": "ccw", "hang_frac": HANG},
        "plan": steps,
        "stats": {"actions": len(steps), "strokes": strokes, "est_s": round(10.0 + STROKE_S * strokes + 11.0 * len(rigids), 1),
                  "floor_used": round((2 * rc + d) ** 2 / (inner_x * inner_y), 3),
                  "free_depth_cm": round(inner_y - (2 * rc + d + GRIPPER_WALL_CM), 1),
                  "feasible_radii": [min(c[1] for c in cands), max(c[1] for c in cands)], "rejected": rejected},
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", choices=["first", "last"], default="first",
                    help="centre block before the rope (former) or after it (keeper)")
    a = ap.parse_args()
    p = plan(json.load(open(a.scene)), a.core)
    json.dump(p, open(a.out, "w"), indent=1)
    c, s = p["coil"], p["stats"]
    print(f"[rope-plan] coil radius {c['radius_cm']} cm -> loop {c['loop_cm']} cm, {c['loops']} loops, stack {c['stack_cm']} cm; "
          f"{s['actions']} actions (~{s['est_s']:.0f}s); floor used {s['floor_used']:.0%}, {s['free_depth_cm']} cm of depth left free; "
          f"feasible radii {s['feasible_radii'][0]}..{s['feasible_radii'][1]} cm, rejected {s['rejected']}")
    for n, r in p["rigids"].items():
        print(f"[rope-plan] {n}: {r['role']} at box{tuple(r['dxy'])}" + (f", on a {r['z_on']:.0f} cm stack" if r["z_on"] else ""))
    for st in p["plan"]:
        print("   ", st)
    print(f"[rope-plan] wrote {a.out}")
