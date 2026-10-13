"""TAMP for slinging a rigid block with a rope and lifting it by the rope (two arms).

Plan: the rope lies straight on the table; the block is set on its middle; each hand drapes its
end over the block to the far side (so the rope encircles the block and both strands cross its
top); each hand then takes the end that now lies on ITS side, and both lift. The block hangs in
the loop, is carried over the box and lowered in.

The planner decides how the block sits on the rope (which horizontal side the rope runs under:
the wrap uses the smaller perimeter, leaving more free rope), where each end lands, where the
hands re-grasp, and how high they must lift - and rejects the task when the rope is too short,
the hands would be too close, or the lift does not clear the box rim within reach.

Usage: tamp_sling.py --scene scene.json --out plan.json
"""
import argparse
import json

DRAPE_SLACK = 0.90        # a draped strand uses at most this fraction of its free length
LAND_CM = 12.0            # where a draped end is laid, beyond the far face
REGRASP_CM = 6.0          # re-grasp point on the table, beyond the face
MIN_HAND_SEP_CM = 18.0
RIM_CLEAR_CM = 5.0
END_MARGIN_CM = 6.0       # grasp this far from a rope end


def plan(scene, knot="none"):
    box, rope, obj = scene["box"], scene["rope"], scene["object"]
    L, d = float(rope["length_cm"]), float(rope["diameter_cm"])
    w0, d0, h = [float(v) for v in obj["dims_cm"]]
    tip_max = float(scene.get("tip_max_above_table_cm", 52.0))
    best, why = None, []
    for rot90 in (False, True):
        w = d0 if rot90 else w0                      # extent along the rope
        if max(w0, d0) + 2 > min(box["inner_cm"]):
            why.append("object does not fit the box"); continue
        half_free = L / 2.0 - w / 2.0                # rope from under an edge to the end
        arc_grasp = half_free - END_MARGIN_CM        # ... to the drape grasp
        land = min(LAND_CM, DRAPE_SLACK * arc_grasp - (2 * h + w))
        if land < REGRASP_CM + 2:
            why.append(f"rope too short for a wrap around {w:.0f}x{h:.0f} (needs {w + 2 * (2 * h + w + REGRASP_CM + 2) / DRAPE_SLACK + 2 * END_MARGIN_CM:.0f} cm)"); continue
        hand_sep = w + 2 * REGRASP_CM
        if hand_sep < MIN_HAND_SEP_CM:
            why.append(f"hands {hand_sep:.0f} cm apart when lifting"); continue
        lift = float(box["h_cm"]) + 0.5 + RIM_CLEAR_CM          # block bottom above the table
        strand_up = (REGRASP_CM ** 2 + h ** 2) ** 0.5            # top edge -> hand, once taut
        tip_z = lift + h + strand_up
        if tip_z > tip_max:
            why.append(f"lift needs the tip {tip_z:.0f} cm above the table (max {tip_max:.0f})"); continue
        cand = dict(rot90=rot90, along_cm=w, wrap_cm=2 * (w + h), land_cm=round(land, 1), hand_sep_cm=hand_sep,
                    lift_cm=lift, tip_above_table_cm=round(tip_z, 1), free_end_cm=round(half_free - (2 * h + w), 1))
        if best is None or cand["wrap_cm"] < best["wrap_cm"]:
            best = cand
    if best is None:
        raise SystemExit("[sling-plan] infeasible: " + "; ".join(why))
    w = best["along_cm"]
    s_mid = L / 2.0
    under_l, under_r = s_mid - w / 2.0, s_mid + w / 2.0
    regrasp_arc = 2 * h + w + REGRASP_CM                          # from an under-edge, around, to the re-grasp point
    steps = [
        {"op": "place_on_rope", "name": obj["name"], "hand": "right", "rot90": best["rot90"]},
        {"op": "drape", "hand": "left", "s_cm": END_MARGIN_CM, "anchor_s_cm": under_l, "to_side": 1, "land_cm": best["land_cm"]},
        {"op": "drape", "hand": "right", "s_cm": L - END_MARGIN_CM, "anchor_s_cm": under_r, "to_side": -1, "land_cm": best["land_cm"]},
        {"op": "lift_carry_lower", "right_s_cm": under_l - regrasp_arc, "left_s_cm": under_r + regrasp_arc,
         "lift_cm": best["lift_cm"], "hold_s": 3.0},
    ]
    if knot == "half":
        # HALF KNOT from lay-over moves only (a tool-down gripper cannot pass under a strand):
        #   1. end A over the block, landing front-right          (bottom layer)
        #   2. end B over the block, landing front-left            -> B crosses OVER A on the top
        #   3. A's end swung from front-right to back-right        -> A crosses OVER B's standing part
        # Along A: under B, then over B. Two alternating crossings = the strands are twisted once,
        # the loop cannot open, and the ends can be pulled from far apart.
        # the strands must stay ON the top face: with ends landing 8 cm to the front, strand B slid
        # off the front edge and looped around the block at table level (no wrap, no lift)
        dy = float(scene.get("knot_dy_cm", min(3.0, 0.3 * (w0 if best["rot90"] else d0))))
        swing = float(scene.get("knot_swing_cm", 8.0))
        steps[1]["land_dy"] = -dy
        steps[2]["land_dy"] = -dy
        steps.insert(3, {"op": "swing_end", "hand": "right", "s_cm": END_MARGIN_CM + 2.0, "to_dy": swing, "x_side": 1,
                         "land_cm": best["land_cm"]})
        steps[4]["regrasp"] = "table"
    return {"task": "sling", "scene": scene, "sling": best, "plan": steps, "knot": knot,
            "stats": {"actions": len(steps), "rejected": why}}


def plan_girth(scene):
    """CHOKER hitch with the eye on TOP of the block, lifted by ONE hand.

    The rope lies doubled on the table: two strands under the block's place, a round bight (the
    eye) on the left, the two ends on the right.
      1. set the block on the two strands                     (two supports: it does not rock)
      2,3. lay each end over the block to the left side        (the ends cross the whole top)
      4. lay the eye over the block's left face onto the top, over the ends, tip at the centre
      5. one hand takes both ends where they hang on the left and lifts above the centre
    The ends rise through the eye, so the loop under-left-top + top-right-under is closed, has no
    slack (the eye is cut to length h + w/2), and hangs from the middle of the top.
    Lifting from the SIDE with a slack eye rolled the block over before the hitch tightened."""
    box, rope, obj = scene["box"], scene["rope"], scene["object"]
    d = float(rope["diameter_cm"])
    w, dep, h = [float(v) for v in obj["dims_cm"]]
    cx, cy = [float(v) for v in scene.get("work_xy", [0.0, -50.0])]
    g = float(scene.get("strand_half_gap_cm", 2.5))               # strands under the block
    eye_r = float(scene.get("eye_radius_cm", 4.5))                # wider than the strand gap: the ends pass inside it
    why = []
    if eye_r <= g + d:
        why.append("eye not wider than the two strands")
    # where a strand leaves the underside it U-turns sideways before climbing the face (measured:
    # about 3 cm outward). On a 10 cm block that put it on the edge of the top, and laying the end
    # down pressed on that edge and rolled the block over.
    side_shift = float(scene.get("uturn_shift_cm", 3.5))
    if g + side_shift + 1.5 > dep / 2:
        why.append(f"block {dep:.0f} cm deep is too narrow: a strand climbs up to {g + side_shift:.1f} cm off the mid-line")
    if 2 * eye_r + d > dep:
        why.append(f"eye {2 * eye_r + d:.0f} cm wide does not sit on a {dep:.0f} cm top")
    eye_arc = h + w / 2 + float(scene.get("eye_extra_cm", 1.5))   # under-left edge -> tip, taut when the tip is at the top centre
    end_land = 8.0                                                # ends land this far beyond the left face (inside the eye);
    #                                                               4 cm wedged the rope against the block and rolled it over
    # 0.85 of the free length was too tight here: the rope does not hug the corners, and laying the
    # first end pulled the block over (tilt 93 deg). 0.7 leaves room for the bulges.
    end_arc = (h + w + h + end_land) / float(scene.get("end_slack", 0.92)) + 2.0   # under-right edge -> grasp near the end
    ends_x = cx + w / 2 + end_arc + 3.0
    tip_x = cx - w / 2 - eye_arc + 3.0                            # the round eye is ~3 cm longer than a straight run
    if cx - w / 2 - end_land - 4.0 < tip_x:
        end_land = cx - w / 2 - 4.0 - tip_x                       # keep the laid ends inside the eye
    lift = float(box["h_cm"]) + 0.5 + RIM_CLEAR_CM
    hang = w / 2 + 4.0 + min(8.0, end_arc - (h + w + h + end_land))       # top centre -> grasp beside the top edge, plus the slack that runs out
    need = d + h + hang + lift
    tip_max = float(scene.get("tip_max_above_table_cm", 52.0))
    if need > tip_max:
        why.append(f"lift needs the tip {need:.0f} cm above the table (max {tip_max:.0f})")
    if max(w, dep) + 2 > min(box["inner_cm"]):
        why.append("object does not fit the box")
    if why:
        raise SystemExit("[sling-plan] infeasible: " + "; ".join(why))
    sc = json.loads(json.dumps(scene))
    fold = bool(scene.get("robot_folds", True))
    tip_to_face = eye_arc - 3.0
    strand = tip_to_face + w + end_arc + 3.0                       # eye tip -> rope end, along one strand
    steps = []
    layout = {"type": "keyhole", "tip_x": round(tip_x, 1), "ends_x": round(ends_x, 1), "y": cy, "half_gap": g, "eye_r": eye_r,
              "neck_x": cx - w / 2 - 1.0}
    if fold:
        # The rope starts STRAIGHT. The left hand carries the left end over to the right, laying the
        # left half beside the right half; where the rope doubles back is the eye. The left arm
        # cannot reach the far end's place, so it hands the end to the right hand half-way.
        L = 2.0 * strand + 4.0
        layout = {"type": "straight", "x0": round(tip_x - L / 2, 1), "x1": round(tip_x + L / 2, 1), "y": cy - g}
        sc["rope"]["length_cm"] = round(L, 1)
        steps.append({"op": "fold_rope", "hand": "left", "then": "right", "y_lay": cy + g, "handover_dx": 16.0,
                      "strand_cm": round(strand, 1)})
    steps += [
        {"op": "place_on_rope", "name": obj["name"], "hand": "right", "rot90": False,
         **({"tip_to_centre_cm": round(tip_to_face + w / 2, 1)} if fold else {})},
        # both ends in ONE grip: laid one at a time they ended 5-9 cm apart on the top, wider than
        # the eye's tip, and nudging one beside the other afterwards did not hold
        {"op": "ends_over", "hand": "right", "land_cm": end_land},
        {"op": "eye_on_top", "hand": "left"},
        {"op": "hitch_lift", "hand": "right" if fold else "left", "beyond_cm": 3.5, "lift_cm": lift, "hold_s": 3.0},
    ]
    return {"task": "sling", "knot": "girth", "scene": sc, "plan": steps,
            "layout": layout,
            "sling": dict(rot90=False, along_cm=w, eye_arc_cm=eye_arc, end_arc_cm=round(end_arc, 1), lift_cm=lift, tip_above_table_cm=round(need, 1)),
            "stats": {"actions": len(steps), "rejected": why}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knot", choices=["none", "half", "girth"], default="none")
    a = ap.parse_args()
    p = plan_girth(json.load(open(a.scene))) if a.knot == "girth" else plan(json.load(open(a.scene)), a.knot)
    json.dump(p, open(a.out, "w"), indent=1)
    s = p["sling"]
    if a.knot == "girth":
        print(f"[sling-plan] choker hitch, eye on top: eye {s['eye_arc_cm']:.1f} cm, ends {s['end_arc_cm']} cm beyond the block; one-hand lift {s['lift_cm']:.1f} cm, "
              f"tip up to {s['tip_above_table_cm']} cm above the table; layout {p['layout']}")
        for st in p["plan"]:
            print("   ", st)
        raise SystemExit(0)
    print(f"[sling-plan] rope under the block's {s['along_cm']:.0f} cm side (rot90={s['rot90']}), wrap {s['wrap_cm']:.0f} cm, ends land {s['land_cm']} cm beyond the faces, "
          f"{s['free_end_cm']} cm of each end to spare; lift {s['lift_cm']:.1f} cm with hands {s['hand_sep_cm']:.0f} cm apart, tip {s['tip_above_table_cm']} cm above the table"
          + (f"; rejected: {p['stats']['rejected']}" if p["stats"]["rejected"] else ""))
    for st in p["plan"]:
        print("   ", st)
