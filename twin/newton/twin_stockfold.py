# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import os as _os
import numpy as np
import warp as wp

# ROBOT_SIDE=left (default) | right — which side of the default camera the arm stands on.
# left  = base x=+50, yaw pi, key poses mirrored (arm faces the shirt+box; user-approved view)
# right = stock pose, base x=-50, unmirrored
_SIDE = 1.0 if _os.environ.get('ROBOT_SIDE', 'left') == 'left' else -1.0

# TWIN_PLACE=measured (default) -> the robot stands where the REAL robot stood.
# The capture's extrinsics give the camera pose in the real robot's base frame and the
# scene layout gives each object in the camera frame, so base->shirt and base->box are
# MEASURED from the capture, not chosen. TWIN_PLACE=stock keeps the example's own layout.
_PLACE = _os.environ.get('TWIN_PLACE', 'measured')
_TS_DEFAULT = _os.environ.get('TWIN_TS', 'robot_20260819_123718')
# TWIN_GARMENT=<object id | label substring>: WHICH garment of a multi-garment capture this twin
# builds, anchors the world to and folds (cloth_mesh.select_garment). Empty = the first garment.
_GARMENT = _os.environ.get('TWIN_GARMENT', '')
# TWIN_GARMENTS=all|active: build EVERY garment of the capture as its own cloth body (all,
# default - the scene shows what the camera saw) or only the selected one.
# TWIN_ANCHOR=<id|label>: the garment whose centroid pins the world frame. Default = the active
# garment (single-garment behaviour). A multi-garment SEQUENCE must use the same anchor in every
# run, or a full state saved by one run lands in a different world in the next.
# TWIN_FINAL_STATE=<f.npy>: dump ALL particles when the plan completes -> the next run's
# TWIN_LOAD_STATE (full-shape states load every garment; slice-shape states load the active one).
# TWIN_STACK_ON=<id|label>: after the folds, pick the folded active garment up by its centre and
# lay it on top of that garment's bundle (kinematic carry, like the box pack).
_GARMENTS_MODE = _os.environ.get('TWIN_GARMENTS', 'all')
# TWIN_LABEL_PLAN=<plan.json from label_tamp.py> + TWIN_LABELS=<labels.json from lift_labels.py>:
# execute LABEL MOVES ("pinch label A, carry it onto label B") instead of named fold operators.
# A two-handed planner move (a + b) is done as two sequential moves by the single arm.
_LABEL_PLAN = _os.environ.get('TWIN_LABEL_PLAN', '')
_LABELS_FILE = _os.environ.get('TWIN_LABELS', '')
_LABEL_MOVES = []            # [(name_a, name_b, particle_a, particle_b), ...] in execution order
_LABEL_STEPS = []            # the same moves GROUPED by planner step: a two-handed step stays one
                             # step, so a bimanual rig can execute it the way fold_env scored it
if _LABEL_PLAN and _LABELS_FILE:
    import json as _lj
    _lp = _lj.load(open(_LABEL_PLAN)); _lb = _lj.load(open(_LABELS_FILE))["labels"]
    _LABEL_STEP_G = []
    for _step in _lp["plan"]:
        _gname = None
        if "->" in _step and ":" in _step.split("->")[0] and " " in _step.split(":")[0] + " ":
            _gname, _step = _step.split(":", 1); _gname = _gname.strip(); _step = _step.strip()
        _LABEL_STEP_G.append(_gname)
        if "->" not in _step:
            # a TASK step, not a label move: "pack" (bimanual bundle carry into the box) or
            # "pick_place:block" (one arm carries the rigid block into the box)
            _LABEL_STEPS.append([("task", _step.strip(), -1, -1)])
            _LABEL_MOVES.append(("task", _step.strip(), -1, -1))
            continue
        _grp = []
        for _pair in _step.split(" + "):
            _a, _b = _pair.split("->")
            _ka = f"{_gname}/{_a}" if _gname and f"{_gname}/{_a}" in _lb else _a
            _kb = f"{_gname}/{_b}" if _gname and f"{_gname}/{_b}" in _lb else _b
            _mv = (_a, _b, int(_lb[_ka]["particle"]), int(_lb[_kb]["particle"]))
            _LABEL_MOVES.append(_mv)
            _grp.append(_mv)
        _LABEL_STEPS.append(_grp)
    print(f"[twin-stock] label plan from {_LABEL_PLAN}: " + " > ".join(_lp["plan"]) +
          f"  ({len(_LABEL_STEPS)} steps, {len(_LABEL_MOVES)} pinches)", flush=True)
_ANCHOR = _os.environ.get('TWIN_ANCHOR', '')
_FINAL_STATE = _os.environ.get('TWIN_FINAL_STATE', '')
_STACK_ON = _os.environ.get('TWIN_STACK_ON', '')
# TWIN_ARM=franka (the stock cloth_franka fr3, default) | ur5e (UR5e + Robotiq 2F-85 from
# mujoco_menagerie via ur5e_gripper.py - the real arm; the 2F-85's 4-bar linkage is frozen open
# because the pin grasp holds the cloth, so the fingers are a fingertip reference, not an actuator)
# TWIN_ARM=ur5e2: TWO UR5e + Robotiq 2F-85 facing each other across the garment (the real cell's
# arm, twice; TWIN_ARM_STANDOFF_CM from the garment centre, default 55). The grippers are built
# WITHOUT their equality-constraint loop joints (ur5e_gripper.add_ur5e_gripper loop_joints=False):
# Newton 1.5's Featherstone cannot integrate those on a second articulation (see ur5e_gripper.py).
# Diagnostics, all off by default: TWIN_DEBUG_NAN=1 (where the arm state first goes non-finite,
# measured outside the captured graph), TWIN_DEBUG_SKIP_CLOTH=1 (cloth stands still),
# TWIN_DEBUG_NO_GRAPH=1 (simulate() without CUDA graph capture).
# TWIN_ARM=aloha: the bimanual ALOHA 2 rig (two ViperX 300 arms, aloha_arms.py). TWO hands, so a
# planner move that pinches two labels at once is executed at once instead of as two single-arm
# moves - fold_env has always rolled those out simultaneously. No real counterpart: the cell has
# one UR5e, so TWIN_EXPORT_EE stays a UR5e-only path.
_ARM = _os.environ.get('TWIN_ARM', 'franka')
# kill -USR1 <pid> prints every thread's Python stack to the log: a run that stops advancing can be asked where it is
import faulthandler as _fh, signal as _sig
_fh.register(_sig.SIGUSR1, all_threads=True)
if _os.environ.get('TWIN_DEBUG_VERIFY', '0') == '1':
    # every kernel launch is followed by a device sync + error check: a stuck kernel is then visible at its own launch site in the stack dump
    wp.config.verify_cuda = True
_SKIP_CLOTH_STEP = _os.environ.get('TWIN_DEBUG_SKIP_CLOTH', '0') == '1'   # diagnostic switch
# TWIN_VIEW=front (default) | shoulder | capture
_VIEW = _os.environ.get('TWIN_VIEW', 'front')
# The measured standoff (68.6cm to the shirt centre) puts the two far corner grasps at
# 87cm, past the fr3's 85.5cm envelope -> the arm over-stretches. Stepping the base this
# far forward along its own facing axis brings the worst grasp to 78cm. The shirt and the
# box do not move, so the measured layout the capture gives us is preserved.
_CLOSER = float(_os.environ.get('TWIN_CLOSER_CM', '10'))
# TWIN_CLOSER_LEFT_CM: the same step-in SIDEWAYS (toward the arm's left, where the garment is
# measured to lie). The forward step alone left a 2026-09-03 half-folded shirt 88cm out radially
# (40cm to the left) - beyond the 74cm the sim arm reaches with a vertical tool. Inverted on
# export exactly like _CLOSER.
_CLOSER_LEFT = float(_os.environ.get('TWIN_CLOSER_LEFT_CM', '0'))
# TWIN_FOLD=assist (default) | physical. Purely frictional folds plateau: measured 54 -> 49 ->
# 45 -> 46 -> 45 -> 44cm over six passes, because after two folds the stack is ~2cm of layered
# fabric and a parallel jaw at table level only catches the top layer. 'assist' pins the outer
# half and mirrors it across the fold line (the iter9 wide-band recipe), which truly halves it.
# TWIN_FOLD=grasp (default) | assist | physical.
#   grasp   - the gripper pinches a small patch and CARRIES it; the rest of the garment is left
#             to VBD, so the flap drapes, sags and its sleeves lag. This is the only mode in
#             which the shirt behaves like a shirt.
#   assist  - the whole half beyond the fold line is position-prescribed to its mirror image and
#             its velocity zeroed every step, so during the carry that half is not simulated at
#             all. It also mirrors x only and leaves z alone, so the flap lands COPLANAR with the
#             stationary half instead of on top of it - four layers at two heights, resolved
#             afterwards by self-collision. Clean numbers, but it is an animated reflection.
#   physical- friction alone. Honest, but folds plateau: 54 -> 49 -> 45 -> 46 -> 45 -> 44cm over
#             six passes, because a parallel jaw at table level only catches the top layer.
_FOLD_MODE = _os.environ.get('TWIN_FOLD', 'grasp')
# Radius of the pinch in 'grasp' mode. A real fingertip grip on cloth catches a few cm of fabric;
# PIN_RADIUS (32cm, for the pack carry) would swallow the whole bundle and reproduce 'assist'.
_FOLD_PIN = float(_os.environ.get('TWIN_FOLD_PIN_CM', '4'))
# Length folds (collar down / hem up) carry the WHOLE upper or lower half, not a narrow strip.
# A 4cm pinch at one point drags that half from a single spot: the ends lag (54cm -> 41cm
# instead of 27cm), and pulling the corners one at a time to fix it TWISTS the flap into a
# lump. A wider hold spreads the pull along the edge so the half comes over flat, which is
# what two hands do. Side folds keep the narrow pinch - the strip follows fine.
_FOLD_PIN_WIDE = float(_os.environ.get('TWIN_FOLD_PIN_WIDE_CM', '11'))
# Length folds carry a SHEET (26x27cm), not a strip, and one 4cm patch cannot carry a sheet
# flat: it splays, and all the strain runs through that patch until the solve diverges. Three
# separate small holds spread across the fold edge carry it evenly - what two hands do. Each
# hold is still a normal 4cm pinch; they ride the gripper together.
_FOLD_SPAN = float(_os.environ.get('TWIN_FOLD_SPAN_CM', '9'))
# Hard cap on how much fabric a grasp may hold. Pinned particles are POSITION-FORCED, so no
# material stiffness resists them: drag a large set rigidly through a multi-layer stack and the
# triangles bridging held and unheld regions stretch until the mesh visibly SHREDS. Each hold
# catches every layer under it, so on a thick bundle three holds reached 1780 particles (28% of
# the shirt) and tore it. Keep the closest ones and let the rest follow through the cloth.
_FOLD_HOLD_MAX = int(_os.environ.get('TWIN_FOLD_HOLD_MAX', '420'))
# Compliance of the grip. GAIN 1.0 would close the gap to the gripper in a single frame, i.e.
# a hard pin; lower lets the cloth pull back. VMAX caps how fast held fabric can be dragged.
# >=1.0 = POSITION-driven pin, rate-limited to _GRIP_VMAX (see _update_fold_grasp). A bare
# teleport went NaN on a 4-layer stack; a velocity-driven pin (<1.0) was cancelled by the stack's
# contacts and lifted nothing. The rate-limited position drive folded a half+sleeve start to the
# planner's predicted 25x31cm with a calm arm (2026-09-03).
_GRIP_GAIN = float(_os.environ.get('TWIN_GRIP_GAIN', '1.0'))
_GRIP_VMAX = float(_os.environ.get('TWIN_GRIP_VMAX_CM_S', '45'))
# MAXIMUM multiplier on the fold CARRY segments. Applied ADAPTIVELY: what destabilises the
# solve is dragging the pinch through many layers, so a single-layer fold runs at stock speed
# and only a thick stack crawls. A flat 2.5x made every fold slow and still diverged on a
# 24x56 four-layer bundle; a flat 4.5x fixed that and made the whole fold tedious to watch.
_CARRY_SLOW = float(_os.environ.get('TWIN_CARRY_SLOW', '5.0'))
# Particles-under-the-grasp per +1x of carry slowdown. 120 let a 297-particle stack (a panel
# fold on a half-folded shirt, 2026-09-03) carry at x2.8 and the cloth went to NaN mid-fold;
# 60 puts the same grasp at x4.6, the pace the planner's rollouts survive.
_CARRY_SLOPE = float(_os.environ.get('TWIN_CARRY_SLOPE', '60'))
# A fold is a ROTATION ABOUT THE CREASE, not a slide across the table. The old path was a
# straight lerp from grab to drop with a small sine bump (apex ~9cm) while the flap is ~29cm
# from crease to grasp - so the fabric between the fingers and the table could only stretch or
# bunch. That is what looked like a stiff sheet being dragged. With the arc the pinch holds
# |tip - crease| = R the whole way and the flap hangs off the fingers and rolls over.
_FOLD_ARC = _os.environ.get('TWIN_FOLD_ARC', '1') == '1'
# Cap the apex so a very long fold does not hoist the whole bundle off the table (a 0.45*travel
# apex once wadded the shirt); below this the arc is exact.
_FOLD_ARC_MAX = float(_os.environ.get('TWIN_FOLD_ARC_MAX_CM', '26'))
# How far the WRIST turns during the carry. Every fold pose used the same DOWN_QUAT, so the
# gripper's orientation never changed from grab to drop: the collar edge was carried over the
# crease still facing the way it started, never turned over. A fold is a rotation of the flap
# about its crease and the hand rotates with it. Capped short of 180 because a fully inverted
# tool would put the flange under the table when it lays the fabric down.
_WRIST_DEG = float(_os.environ.get('TWIN_FOLD_WRIST_DEG', '90'))
# How many pinches carry a length fold. Three, spread across the fold edge, are all welded into
# ONE gripper frame - so a 40cm span of fabric is held as a straight unbendable line and lifting
# it raises a rigid arch, as if the shirt had something inside it. fold_env, whose folds look
# right, holds a SINGLE 4cm pinch and lets the rest hang. One here too; the old behaviour is
# TWIN_FOLD_HOLDS=3 if the collar ever gathers into a ruffle again.
_FOLD_HOLDS = int(_os.environ.get('TWIN_FOLD_HOLDS', '1'))
# TWIN_EXPORT_EE=<f.json>  write the arm's commanded waypoints for the REAL UR5e.
# The scene was measured in the real ur5e_base_link frame (T_base_cam) and mapped into sim world
# by _place_world, so inverting that map gives true robot coordinates - no hand-fitted transform.
_EXPORT_EE = _os.environ.get('TWIN_EXPORT_EE', '')
_EXPORT_VEL = float(_os.environ.get('TWIN_EXPORT_VEL', '0.1'))
_EXPORT_DZ = float(_os.environ.get('TWIN_EXPORT_DZ_CM', '0'))   # height trim, if the real cloth sits lower
# TWIN_REC_USD=<f.usd>  record this run (robot bodies + cloth) as a time-sampled USD that
# Isaac Sim plays back with RTX rendering. Physics untouched; TWIN_REC_EVERY thins the frames.
_REC_USD = _os.environ.get('TWIN_REC_USD', '')
_REC_EVERY = int(_os.environ.get('TWIN_REC_EVERY', '4'))
# Fingertip reach, MEASURED, not the UR5e's 0.85m flange spec. The Hand-E fingertip sits
# 0.187m beyond tool0, and the executor leans the tool toward the base for far/low poses, which
# turns part of that offset into horizontal reach. A MoveIt dry run on this cell planned
# fingertip poses at 0.95m and 1.00m (both with a 20deg lean) and refused nothing, while the
# 0.85m fingertip rule I first wrote rejected them - four false 'out of reach' verdicts that
# killed every collar-opening plan. 1.02m is just past the furthest pose actually confirmed;
# anything beyond that is unverified, so the planner still has to justify itself.
_REACH_LIMIT = float(_os.environ.get('TWIN_REACH_LIMIT_M', '1.02'))
# Which cloth the twin folds.
#   'usd'  (default) the stock unisex_shirt.usd - the tuned shirt, untouched.
#   'mask' build the sheet from THIS capture's segmentation mask + depth via cloth_from_mask,
#          so the sim garment is the real one (a pair of trousers is not a t-shirt). Same
#          material constants either way; only the silhouette and pose come from the capture.
_CLOTH_SRC = _os.environ.get('TWIN_CLOTH', 'usd')
# Closing the loop between the twin and the planner.
#   TWIN_DUMP_STATE=<f.npy>  write the settled cloth state and exit, so the planner can be run
#                            on the state the ROBOT actually observes.
#   TWIN_PLAN_FILE=<f.txt>   read the plan from a file the planner wrote, instead of me pasting
#                            a TWIN_PLAN string copied out of a search log by hand.
_DUMP_STATE = _os.environ.get('TWIN_DUMP_STATE', '')
_PLAN_FILE = _os.environ.get('TWIN_PLAN_FILE', '')
#   TWIN_LOAD_STATE=<f.npy>   start from a SAVED garment state instead of a flat shirt, so the
#                             arm can be given a half-folded / rotated / displaced shirt - the
#                             states fold_search.py --make-starts writes and plans against.
_LOAD_STATE = _os.environ.get('TWIN_LOAD_STATE', '')
# TWIN_UNFOLD_STACK=<fold_stack.json from fold_state_fit>: the garment was folded by a PERSON;
# the fit reconstructed which folds, and this replays them backwards on the loaded state.
_UNFOLD_STACK = _os.environ.get('TWIN_UNFOLD_STACK', '')
# How much lower than the authored height the jaws go when closing on cloth. The stock poses
# grasp exactly at table level (z=20) and the first corner grasp visibly misses.
# 1.0 catches the corner but closes THROUGH the cloth and wedges it between the jaws, so the
# sheet rides the gripper instead of dropping on release; 0.0 misses the first grasp entirely.
_GRASP_DZ = float(_os.environ.get('TWIN_GRASP_DZ_CM', '0.5'))
# TWIN_FOLD_STYLE=shirt folds the FLAT sheet in three deterministic moves (left panel, right
# panel, then in half), skipping the stock 5-corner choreography which crumples first and needs
# extra halvings afterwards. 'stock' keeps the original behaviour.
_FOLD_STYLE = _os.environ.get('TWIN_FOLD_STYLE', 'shirt')
# TWIN_PLAN=a,b,c executes a fold sequence found by fold_search.py with the real arm, e.g.
# TWIN_PLAN=half_right_in,panel_left_in,half_collar_dn. The names and their (axis, frac, side)
# are the SAME action set fold_env rolls out, and a fold is a mirror across a line in both, so
# a plan the search validated headless transfers here without reinterpretation.
_PLAN = [a for a in _os.environ.get('TWIN_PLAN', '').split(',') if a]
if _PLAN_FILE:
    # AFTER the TWIN_PLAN line, not before: the env-var read was overwriting the plan
    # loaded from file, so the robot silently fell back to its built-in fold.
    with open(_PLAN_FILE) as _fh:
        _PLAN = [a.strip() for a in _fh.read().replace('\n', ',').split(',') if a.strip()]
    print(f'[twin-stock] plan from {_PLAN_FILE}: ' + ' > '.join(_PLAN), flush=True)
# TWIN_PLANS="<id|label>:<plan file>;<id|label>:<plan file>": fold SEVERAL garments in ONE run,
# in this order - each plan validated by fold_tamp for THAT garment (its own observed state and
# rest-shape scale). The twin switches the active garment between them (fresh landmarks, bbox,
# fold stack), and TWIN_STACK_ON then stacks the LAST folded garment on the one named.
# Overrides TWIN_PLAN / TWIN_PLAN_FILE.
_PLANS = []
for _spec in [x for x in _os.environ.get('TWIN_PLANS', '').split(';') if x.strip()]:
    _sel, _pf = _spec.split(':', 1)
    if _pf.strip() in ("", "-"):
        _names = []                      # no folds for this garment (e.g. a stack-only rerun)
    else:
        with open(_pf.strip()) as _fh:
            _names = [a.strip() for a in _fh.read().replace('\n', ',').split(',') if a.strip()]
    _PLANS.append((_sel.strip(), _names))
    print(f"[twin-stock] plan for garment '{_sel.strip()}' from {_pf.strip()}: "
          + ' > '.join(_names), flush=True)
if _PLANS:
    _PLAN = list(_PLANS[0][1])          # the stage machine and the frame budget see "a plan"
# (axis, frac, side, grab_at). grab_at is WHERE ALONG THE FOLD EDGE the gripper takes hold,
# as a fraction of the other axis. 0.5 grabs the middle and the far ends of the edge lag and
# drag - a "half" fold of a 55cm shirt landed at 38cm instead of 27.5cm. 0.75/0.25 grab the
# upper and lower parts, so a side fold done as two moves completes where one move does not.
_PLAN_ACTIONS = {
    "half_hem_up":    (1, 0.5, +1, 0.5),
    "half_collar_dn": (1, 0.5, -1, 0.5),
    "half_right_in":  (0, 0.5, +1, 0.5),
    "panel_left_in":  (0, 1.0 / 3.0, -1, 0.5),
    "panel_right_in": (0, 2.0 / 3.0, +1, 0.5),
    "right_in_top":   (0, 0.5, +1, 0.75),
    "right_in_bot":   (0, 0.5, +1, 0.25),
    "left_panel_top": (0, 1.0 / 3.0, -1, 0.75),
    "left_panel_bot": (0, 1.0 / 3.0, -1, 0.25),
    # Same treatment for the LENGTH folds: a single mid grab took 54cm -> 41cm where a
    # complete halving is 27cm, because the left and right ends of that edge lag exactly as
    # the top and bottom did on the side folds.
    "collar_dn_l":    (1, 0.5, -1, 0.25),
    "collar_dn_r":    (1, 0.5, -1, 0.75),
    "hem_up_l":       (1, 0.5, +1, 0.25),
    "hem_up_r":       (1, 0.5, +1, 0.75),
}
# TWIN_MODE=auto runs the scripted fold/unfold/pack; teleop hands the arm to the keyboard.
_MODE = _os.environ.get('TWIN_MODE', 'auto')
# Where teleop demonstrations are written, and how often the cloth state is sampled.
# The arm/gripper track is recorded every frame (cheap); cloth particles are a GPU
# readback, so they go at a lower rate.
_DATA_DIR = _os.environ.get('TWIN_DATA_DIR', _os.path.expanduser('~/real2sim2real/data/teleop'))
_CLOTH_EVERY = int(_os.environ.get('TWIN_CLOTH_EVERY', '6'))      # frames: 6 -> 10Hz at 60fps
_CLOTH_STRIDE = int(_os.environ.get('TWIN_CLOTH_STRIDE', '4'))    # keep every Nth particle


def _measured_from_capture(ts):
    """Return the capture geometry in the real robot's base frame, cm.

    Keys are (forward, left, up) triples relative to the robot base flange:
      shirt / box : object centroid, cam : camera position, cam_dir : optical +z.
    """
    import json as _j
    import numpy as _np

    ex = _j.load(open(f"/home/aoloo/sam-3d-objects/captures/{ts}/extrinsics.json"))["transforms"]
    lay = _j.load(open(f"/home/aoloo/sam-3d-objects/outputs/{ts}/scene_layout.json"))["objects"]
    T = _np.array(ex["ur5e_base_link"]["T_base_cam"], float)   # camera pose in the robot base frame

    def _find(sub, required=True):
        hit = next((o for o in lay
                    if sub in (o.get("depth_info", {}).get("label") or o.get("label", "")).lower()),
                   None)
        if hit is None and required:
            raise KeyError(f"no '{sub}' in the scene layout for {ts}")
        return hit

    def _to_base(o):
        p = _np.array(list(o["depth_info"]["position_cam"]) + [1.0], float)
        return (T @ p)[:3] * 100.0

    # The box is OPTIONAL: a capture may deliberately contain only the garment (folding-only
    # scenes). Everything the arm needs is anchored to the shirt, so a missing box costs the
    # scene a prop, not its geometry.
    _box = _find("box", required=False)
    # The garment is not always a shirt, and since 2026-09-11 not always alone: SAM3D labels
    # whatever is on the table, and the scene only needs the POSE of the garment THIS twin is
    # about. cloth_mesh.select_garment resolves TWIN_GARMENT (id or label substring); with no
    # selector it takes the first garment-labelled object, else the single non-box object.
    from cloth_mesh import select_garment, GARMENT_WORDS, _obj_label
    _g = select_garment(ts, _GARMENT, lay=lay)
    # every garment on the table (the active one always counts), in layout order
    _all = [o for o in lay if o is not _box
            and (o is _g or any(w in _obj_label(o) for w in GARMENT_WORDS))]
    _anchor = select_garment(ts, _ANCHOR, lay=lay) if _ANCHOR else _g
    if len(lay) > 1:
        # stderr: run_full_pipeline.sh captures this module's stdout in $(...) for REF_SCALE
        import sys as _sys
        print(f"[twin-stock] garment = object {_g.get('id')} "
              f"'{_g.get('depth_info', {}).get('label') or _g.get('label')}' "
              f"(TWIN_GARMENT='{_GARMENT}'; scene has {len(lay)} objects)", file=_sys.stderr, flush=True)
    return {
        "shirt": _to_base(_g),
        "anchor": _to_base(_anchor),
        "active_id": _g.get("id"),
        "garments": [{"id": o.get("id"), "label": _obj_label(o),
                      "pos": [float(v) for v in _to_base(o)]} for o in _all],
        "box": _to_base(_box) if _box is not None else None,
        "cam": T[:3, 3] * 100.0,
        "cam_dir": T[:3, 2],
    }


# World convention for the measured layout: the arm faces -y (so the viewer looks down
# the same axis the capture did), its left is +x, up is +z.
_FWD = np.array([0.0, -1.0, 0.0])
_LEFT = np.array([1.0, 0.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])
_SHIRT_XY = np.array([0.05, -50.3])      # stock shirt bbox centre (cm), measured from the asset
_TABLE_TOP_Z = 20.0


def _garment_centre_xy():
    """Where the folded garment will be, in sim world cm - needed BEFORE the cloth is built so a
    two-armed rig can be placed around it. TWIN_ALOHA_CENTER overrides; otherwise a loaded start
    state carries the answer, and failing that the stock shirt's own bbox centre does."""
    if _os.environ.get("TWIN_ALOHA_CENTER"):
        return np.array([float(v) for v in _os.environ["TWIN_ALOHA_CENTER"].split(",")[:2]], float)
    if _LOAD_STATE and _os.path.exists(_LOAD_STATE):
        try:
            return np.asarray(np.load(_LOAD_STATE)[:, :2], float).mean(0)
        except Exception as _e:
            print(f"[twin-stock] rig: could not read {_LOAD_STATE} ({_e}); using the stock centre",
                  flush=True)
    return np.array(_SHIRT_XY, float)


def _place_world(ts):
    """Map the measured base-frame geometry into stock world coordinates (cm)."""
    m = _measured_from_capture(ts)
    f2, l2 = _FWD[:2], _LEFT[:2]
    # the world is pinned to the ANCHOR garment (= the active one unless TWIN_ANCHOR says otherwise)
    base_xy = _SHIRT_XY - f2 * m["anchor"][0] - l2 * m["anchor"][1]
    base_z = _TABLE_TOP_Z - m["anchor"][2]                     # table sits this far below the flange
    base = np.array([base_xy[0], base_xy[1], base_z])
    to_world = lambda v: base + _FWD * v[0] + _LEFT * v[1] + _UP * v[2]
    d = m["cam_dir"]
    box_world = to_world(m["box"]) if m["box"] is not None else None
    cam_world = to_world(m["cam"])                                  # anchored to the measurement
    base_true = base.copy()                                         # where the REAL base is
    base = base + _FWD * _CLOSER + _LEFT * _CLOSER_LEFT              # step in so the arm stops stretching
    cam_world = cam_world + _FWD * _CLOSER + _LEFT * _CLOSER_LEFT
    return {
        "base": base,
        # The measured garment must be anchored HERE, not to the stepped base: placing the
        # mask/fit cloth from the stepped base moved the cloth WITH the arm (the step-in did
        # nothing for reach) and the export's inversion of the step then put every waypoint
        # _CLOSER too far on the real robot (found 2026-09-03).
        "base_true": base_true,
        "standoff": float(m["shirt"][0] - _CLOSER),
        "box": box_world,
        "cam": cam_world,
        "cam_dir": _FWD * d[0] + _LEFT * d[1] + _UP * d[2],
        "raw": m,
    }

from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def _assign_arm_qd(mask: wp.array[int], target: wp.array[float], joint_qd: wp.array[float]):
    # velocity command only on the ARM dofs; a free rigid block keeps the velocity the dynamics gave it
    i = wp.tid()
    if mask[i] == 1:
        joint_qd[i] = target[i]


@wp.kernel
def _block_gravity(ids: wp.array[int], body_mass: wp.array[float], body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    b = ids[i]
    body_f[b] = body_f[b] + wp.spatial_vector(0.0, 0.0, -981.0 * body_mass[b], 0.0, 0.0, 0.0)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        # VBD iterations per substep. 5 is the cloth_franka default; a folded start state is a
        # 4-layer stack and the pinch drags all of it through rigid finger contact, which 5
        # iterations do not always resolve (NaN mid-fold, 2026-09-03). TWIN_VBD_ITERS raises it.
        self.iterations = int(__import__('os').environ.get('TWIN_VBD_ITERS', '5'))
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = float(__import__("os").environ.get("TWIN_PARTICLE_RADIUS_CM", "0.8"))
        self.cloth_body_contact_margin = 0.8
        #       self-contact
        # TWIN_SELF_CONTACT_CM / TWIN_SELF_MARGIN_CM: cloth-cloth contact radius/margin. 0.2/0.2
        # stacks one garment's own layers, but a second garment's wad set down on it sank through
        # all eight layers to the table (2026-09-11 stack) - knobs to find out why.
        self.particle_self_contact_radius = float(__import__("os").environ.get("TWIN_SELF_CONTACT_CM", "0.2"))
        self.particle_self_contact_margin = float(__import__("os").environ.get("TWIN_SELF_MARGIN_CM", "0.2"))
        if self.particle_self_contact_margin < 1.5 * self.particle_self_contact_radius:
            # SolverVBD refuses margin < radius (missed contacts, instability) and advises 1.5-2x;
            # a thicker cloth (bigger self-contact radius) must bring its margin with it
            self.particle_self_contact_margin = 1.6 * self.particle_self_contact_radius
            print(f"[twin-stock] self-contact margin raised to {self.particle_self_contact_margin:.2f}cm "
                  f"(1.6 x radius {self.particle_self_contact_radius:.2f})", flush=True)

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e1

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 5e1
        self.robot_contact_mu = 1.5

        self.self_contact_friction = float(__import__('os').environ.get('TWIN_CLOTH_MU', '0.25'))  # cloth-vs-table/finger friction (soft_contact_mu)

        #   elasticity
        # Cloth stiffness knobs (2026-09-18): TWIN_TRI_KE = in-plane stretch/area stiffness, TWIN_BEND_KE =
        # bending stiffness. Defaults are the tuned cloth_franka values; the planner (fold_env) reads
        # the SAME env vars so a validated plan and the twin fold the same fabric. Earlier sweeps
        # (feedback notes): bend 2 = floppy (shirt shoved off the table), 5-15 = crisper folds, 12+
        # springy in the box pack; stretch up to 1.2e5 used for an inextensible fabric.
        _e = __import__('os').environ
        self.tri_ke = float(_e.get('TWIN_TRI_KE', '1e4'))
        self.tri_ka = float(_e.get('TWIN_TRI_KA', str(self.tri_ke)))
        self.tri_kd = float(_e.get('TWIN_TRI_KD', '1.5e-2'))

        self.bending_ke = float(_e.get('TWIN_BEND_KE', '5'))
        self.bending_kd = float(_e.get('TWIN_BEND_KD', '5e-1'))
        if _e.get('TWIN_TRI_KE') or _e.get('TWIN_BEND_KE'):
            print(f"[twin-stock] cloth stiffness: tri_ke {self.tri_ke:g} tri_ka {self.tri_ka:g} tri_kd {self.tri_kd:g} "
                  f"bend_ke {self.bending_ke:g} bend_kd {self.bending_kd:g}", flush=True)

        self.scene = ModelBuilder(gravity=(0.0, 0.0, -981.0))

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table (cm scale)
        # measured layout puts the real box further to the robot's left -> wider table
        self.table_hx_cm = 70.0 if _PLACE == "measured" else 40.0
        self.table_hy_cm = 48.0 if _PLACE == "measured" else 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(*((5.0, -58.0, 10.0) if _PLACE == "measured" else (0.0, -50.0, 10.0)))
        if _PLACE == "measured":
            _pb = _place_world(_TS_DEFAULT)["base"]
            self.scene.add_shape_box(
                -1,
                xform=wp.transform((float(_pb[0]), float(_pb[1]), float(_pb[2]) / 2.0), wp.quat_identity()),
                hx=11.0,
                hy=11.0,
                hz=float(_pb[2]) / 2.0,
            )

        # TWIN_SCENE=task1: fold the towel, place it in a box, put a rigid block in the box too.
        # Synthetic, fully controlled: an open box in FRONT of the two arms (both reach it) and a
        # rigid block beside the towel. The block is KINEMATIC in this twin (the loop integrates
        # arms kinematically and cloth with VBD; a free body has no dynamics path yet): it follows
        # the hand while held and stays where it is released. in(x, box) is geometric.
        self._task1 = None
        if _os.environ.get("TWIN_SCENE", "") == "task1":
            _gc = _garment_centre_xy()
            _bx, _by = float(_gc[0]), float(_gc[1]) + float(_os.environ.get("TWIN_BOX_DY_CM", "32"))
            _hin = float(_os.environ.get("TWIN_BOX_INNER_CM", "24")) / 2.0
            _wall, _bh = 1.0, float(_os.environ.get("TWIN_BOX_H_CM", "12"))
            _top = self.TABLE_TOP
            # floor slab + four walls, static
            self._rec_box_shapes = [self.scene.shape_count]
            # a 0.5 cm slab let the dropped bundle tunnel through into the table (particles ended
            # at z=3); the slab keeps its top at table+0.5 but is 6.5 cm thick below it
            self.scene.add_shape_box(-1, xform=wp.transform((_bx, _by, _top + 0.5 - 3.25), wp.quat_identity()),
                                     hx=_hin + _wall, hy=_hin + _wall, hz=3.25)
            for _dx, _dy, _hx, _hy in ((_hin + _wall / 2, 0.0, _wall / 2, _hin + _wall), (-_hin - _wall / 2, 0.0, _wall / 2, _hin + _wall),
                                       (0.0, _hin + _wall / 2, _hin + _wall, _wall / 2), (0.0, -_hin - _wall / 2, _hin + _wall, _wall / 2)):
                self._rec_box_shapes.append(self.scene.shape_count)
                self.scene.add_shape_box(-1, xform=wp.transform((_bx + _dx, _by + _dy, _top + 0.5 + _bh / 2), wp.quat_identity()),
                                         hx=_hx, hy=_hy, hz=_bh / 2)
            self.box_center_cm = (_bx, _by)
            self.box_floor_z_cm = _top + 0.5
            self.box_rim_z_cm = _top + 0.5 + _bh
            self.box_half_cm = (_hin, _hin)
            # the rigid block: kinematic body with a box shape, beside the towel
            _bspecs = []
            if _os.environ.get("TWIN_BLOCKS"):
                for _bs in [x for x in _os.environ["TWIN_BLOCKS"].split(";") if x.strip()]:
                    _parts = _bs.split(":")
                    _bn, _bd, _bxy = _parts[:3]
                    _bo_ = [float(v) for v in _bxy.split(",")]
                    if len(_parts) > 3:
                        _bo_.append(float(_parts[3]))          # optional start height above the table (drop tests)
                    _bspecs.append((_bn.strip(), [float(v) for v in _bd.split(",")], _bo_))
            else:
                _bspecs.append(("block", [float(v) for v in _os.environ.get("TWIN_BLOCK_CM", "10,6,4").split(",")],
                                [float(_os.environ.get("TWIN_BLOCK_DX_CM", "28")), float(_os.environ.get("TWIN_BLOCK_DY_CM", "10"))]))
            _blocks = {}
            self._rigid_dynamic = _os.environ.get("TWIN_RIGID_DYNAMIC", "0") == "1"
            for _bi_, (_bn, _blk, _bo) in enumerate(_bspecs):
                _bpx, _bpy = float(_gc[0]) + _bo[0], float(_gc[1]) + _bo[1]
                _bz = _top + _blk[2] / 2 + (float(_bo[2]) if len(_bo) > 2 else 0.0)
                # in the MAIN model every block is a kinematic mirror (the cloth collides with it);
                # with TWIN_RIGID_DYNAMIC=1 its motion comes from a separate rigid world (XPBD,
                # gravity, hard contacts with the floor, the walls and the other blocks) that is
                # stepped every frame and copied in - Featherstone's penalty contacts launched a
                # resting box off the table in a minimal test, XPBD lands and rests it.
                _body = self.scene.add_body(xform=wp.transform((_bpx, _bpy, _bz), wp.quat_identity()),
                                            mass=0.0, is_kinematic=True, label=_bn)
                self._rec_box_shapes.append(self.scene.shape_count)
                self.scene.add_shape_box(_body, hx=_blk[0] / 2, hy=_blk[1] / 2, hz=_blk[2] / 2)
                _blocks[_bn] = {"body": int(_body), "dims": _blk, "xy0": (_bpx, _bpy), "z0": _bz}
            self._rigid_world_spec = {"box": (_bx, _by, _hin, _wall, _bh, _top), "blocks": _blocks} if self._rigid_dynamic else None
            _b0 = _bspecs[0]
            self._task1 = {"block_body": _blocks[_b0[0]]["body"], "block_dims": _b0[1], "block_xy0": _blocks[_b0[0]]["xy0"],
                           "blocks": _blocks,
                           "box_center": (_bx, _by), "box_half": _hin, "box_floor": _top + 0.5, "box_rim": _top + 0.5 + _bh}
            _blk, _bpx, _bpy, _body = _b0[1], _blocks[_b0[0]]["xy0"][0], _blocks[_b0[0]]["xy0"][1], _blocks[_b0[0]]["body"]
            print(f"[twin-stock] task1 scene: open box {2 * _hin:.0f}x{2 * _hin:.0f}x{_bh:.0f}cm at ({_bx:.0f},{_by:.0f}), "
                  f"rim z {_top + 0.5 + _bh:.0f}; block {_blk[0]:.0f}x{_blk[1]:.0f}x{_blk[2]:.0f}cm at ({_bpx:.0f},{_bpy:.0f}) body {_body}", flush=True)

        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            xform=wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # --- OUR REAL reconstructed box (cm units), placed clear of the fold path ---
        import json as _json, trimesh as _tm
        _TS = _TS_DEFAULT
        _D = f"/home/aoloo/sam-3d-objects/outputs/{_TS}_full"
        try:
            if self._task1:
                raise RuntimeError("TWIN_SCENE=task1 uses its own synthetic box")
            _lay = _json.load(open(f"/home/aoloo/sam-3d-objects/outputs/{_TS}/scene_layout.json"))["objects"]
            _bi = next(o["id"] for o in _lay
                       if "box" in (o.get("depth_info", {}).get("label") or o.get("label", "")).lower())
            _bsz = next(o["depth_info"]["physical_size_m"] for o in _lay if o["id"] == _bi) * 100.0
            _bm = _tm.load(f"{_D}/object_{_bi}/mesh.obj", process=False, force="mesh")
            _bv = np.asarray(_bm.vertices, float); _bv -= _bv.mean(0)
            _bv *= _bsz / max(_bv.max(0) - _bv.min(0))
            _c = _bv - _bv.mean(0); _, _, _vt = np.linalg.svd(_c, full_matrices=False); _bv = _c @ _vt.T
            _e = _bv.max(0) - _bv.min(0); _o = np.argsort(_e)
            _bv = _bv[:, [_o[2], _o[1], _o[0]]]; _bv[:, 2] -= _bv[:, 2].min()

            # Single view: SAM3D squashes the axis it never saw, giving a 26 x 16.4 x 9.7cm
            # tray that nothing fits into. The capture measured the mask itself - width 0.26m
            # and height 0.2523m - so restore the second footprint axis to that measurement and
            # carry the same factor into the height, which the same foreshortening compressed.
            _dep = next(o["depth_info"] for o in _lay if o["id"] == _bi)
            _want = float(_dep.get("physical_height_m", _dep["physical_size_m"])) * 100.0
            _e = _bv.max(0) - _bv.min(0)
            _grow = _want / _e[1]
            _bv[:, 1] *= _grow
            _bv[:, 2] *= _grow
            _bv[:, 2] -= _bv[:, 2].min()

            # The box's yaw here is PCA-derived, not measured (the capture's icp rotation is
            # unreliable), so orienting its long axis along the bundle's costs no fidelity and
            # lets the folded shirt drop in lengthwise.
            _yaw = np.radians(float(_os.environ.get("TWIN_BOX_YAW_DEG", "90")))
            _cs, _sn = np.cos(_yaw), np.sin(_yaw)
            _bv[:, :2] = _bv[:, :2] @ np.array([[_cs, _sn], [-_sn, _cs]])
            print(f"[twin-stock] box rebuilt to {_bv.max(0)[0] - _bv.min(0)[0]:.0f}x"
                  f"{_bv.max(0)[1] - _bv.min(0)[1]:.0f}x{_bv.max(0)[2] - _bv.min(0)[2]:.0f}cm "
                  f"(measured mask {_bsz:.0f}x{_want:.0f}cm)", flush=True)

            # SAM3D closes the top of the box (its mesh is watertight) so it renders as a solid
            # block. The real box is open: SLICE the cap off on the lip plane - dropping whole
            # faces instead leaves a sawtooth rim - then give the shell real wall thickness with
            # an inward-offset copy. The offset matters: a mirrored copy of the SAME faces
            # z-fights and renders as speckle.
            _lip = (_bv[:, 2].max()) * 0.80
            _shell = _tm.Trimesh(vertices=_bv, faces=np.asarray(_bm.faces), process=False)
            _shell = _tm.intersections.slice_mesh_plane(
                _shell, plane_normal=[0.0, 0.0, -1.0], plane_origin=[0.0, 0.0, _lip], cap=False)
            if len(_shell.faces) > 3000:            # 23k box faces against 12.7k of cloth tanks contact cost
                try:
                    _shell = _shell.simplify_quadric_decimation(face_count=3000)
                except Exception:
                    pass

            _bv = np.asarray(_shell.vertices, float)
            _bf = np.asarray(_shell.faces, np.int32)
            _ctr = _bv[:, :2].mean(axis=0)
            _inner = _bv.copy()
            _r = _inner[:, :2] - _ctr
            _inner[:, :2] -= _r / np.maximum(np.linalg.norm(_r, axis=1, keepdims=True), 1e-6) * 0.5
            _inner[:, 2] += 0.5                                   # 5mm walls and floor
            _bv = np.vstack([_bv, _inner])
            _bf = np.vstack([_bf, _bf[:, ::-1] + len(_shell.vertices)])

            # The reconstruction's underside is open once sliced, so the box reads as hollow -
            # you see the table through it. Lay a floor across the footprint just above its base.
            _fz = _bv[:, 2].min() + 0.6
            _fx0, _fx1 = _bv[:, 0].min() + 0.5, _bv[:, 0].max() - 0.5
            _fy0, _fy1 = _bv[:, 1].min() + 0.5, _bv[:, 1].max() - 0.5
            _n0 = len(_bv)
            _bv = np.vstack([_bv, [[_fx0, _fy0, _fz], [_fx1, _fy0, _fz],
                                   [_fx1, _fy1, _fz], [_fx0, _fy1, _fz]]])
            # up-facing only: adding the reversed pair as well makes the floor z-fight and
            # the interior renders black
            _bf = np.vstack([_bf, [[_n0, _n0 + 1, _n0 + 2], [_n0, _n0 + 2, _n0 + 3]]])

            if _PLACE == "measured":
                _bxy = _place_world(_TS)["box"][:2]      # 80cm out, 36 deg to the robot's left
                _clear = float(_os.environ.get("TWIN_BOX_CLEAR_CM", "4"))
                _dir = _bxy - _SHIRT_XY
                _bxy = _bxy + _dir / np.linalg.norm(_dir) * _clear
                _bv[:, 0] += float(_bxy[0]); _bv[:, 1] += float(_bxy[1])
            else:
                _bv[:, 0] += -25.0 * _SIDE; _bv[:, 1] += -75.0
            _bv[:, 2] += 20.0

            self.box_center_cm = (float(_bv[:, 0].mean()), float(_bv[:, 1].mean()))
            # The mesh floor is a single zero-thickness quad and the cloth punches straight
            # through it on release. Back it with a real solid pad across the interior.
            _ix = (_bv[:, 0].max() - _bv[:, 0].min()) / 2.0 - 1.0
            _iy = (_bv[:, 1].max() - _bv[:, 1].min()) / 2.0 - 1.0
            self.box_floor_z_cm = float(_bv[:, 2].min()) + 2.4
            _cx0, _cy0 = float(_bv[:, 0].mean()), float(_bv[:, 1].mean())
            _z0, _z1 = float(_bv[:, 2].min()), float(_bv[:, 2].max())
            _wt = 1.2                                   # wall thickness
            _wh = (_z1 - _z0) / 2.0
            # Solid collision cage: floor pad + four walls. These are PRIMITIVES, and the GL
            # viewer bakes primitive dimensions at set_model time in cm while the scene renders
            # in metres, so they draw 100x oversized (a wall of yellow) unless hidden - same
            # reason the table is hidden and drawn manually. They are collision-only anyway;
            # the reconstructed shell remains the visual.
            self.cage_shape_idx = []
            for _t in ((_cx0, _cy0, _z0 + 1.2, _ix, _iy, 1.2),
                       (_cx0 - _ix - _wt, _cy0, _z0 + _wh, _wt, _iy, _wh),
                       (_cx0 + _ix + _wt, _cy0, _z0 + _wh, _wt, _iy, _wh),
                       (_cx0, _cy0 - _iy - _wt, _z0 + _wh, _ix + 2 * _wt, _wt, _wh),
                       (_cx0, _cy0 + _iy + _wt, _z0 + _wh, _ix + 2 * _wt, _wt, _wh)):
                self.cage_shape_idx.append(self.scene.shape_count)
                self.scene.add_shape_box(
                    -1,
                    xform=wp.transform((_t[0], _t[1], _t[2]), wp.quat_identity()),
                    hx=_t[3], hy=_t[4], hz=_t[5],
                    cfg=ModelBuilder.ShapeConfig(mu=0.8),
                )
            self.box_half_cm = ((_bv[:, 0].max() - _bv[:, 0].min()) / 2.0,
                                (_bv[:, 1].max() - _bv[:, 1].min()) / 2.0)
            self.box_rim_z_cm = float(_bv[:, 2].max())
            self.scene.add_shape_mesh(body=-1, mesh=newton.Mesh(_bv.astype(np.float32),
                _bf.reshape(-1).astype(np.int32), is_solid=False, color=(0.62, 0.50, 0.36)),
                cfg=ModelBuilder.ShapeConfig(mu=0.8))
            print(f"[twin-stock] real box '{_TS}' {_bsz:.0f}cm at "
                  f"({self.box_center_cm[0]:.0f},{self.box_center_cm[1]:.0f}) rim z="
                  f"{self.box_rim_z_cm:.0f} opened ({len(_bf)} faces) place={_PLACE}", flush=True)
        except Exception as _ex:
            print("[twin-stock] box skipped:", _ex, flush=True)

        # add the garment(s): every garment on the table is a cloth body; the ACTIVE one
        # (TWIN_GARMENT) is the one this twin measures, folds and exports
        _cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)
        _cloth_pos = wp.vec3(0.0, 70.0, 30.0)
        self._towel = None
        self._garments = []
        _glist = self._garment_list()
        for _gi, (_sel, _gid, _glabel) in enumerate(_glist):
            _is_active = (_gi == self._active)
            _cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)
            _cloth_pos = wp.vec3(0.0, 70.0, 30.0)
            if _CLOTH_SRC == "fit":
                # THE TUNED ASSET, FITTED TO THE CAPTURE. Keep unisex_shirt.usd exactly as it is -
                # seams, sleeves, 6436 particles, triangle quality - and warp only its FOOTPRINT
                # onto the silhouette the RGBD measured: translate to the observed centroid, rotate
                # by the minimal angle that aligns the principal axes, scale per-axis to the
                # observed extents. Topology untouched, so every grasp threshold and the planner's
                # state files keep working; shape, size and pose come from the real garment.
                from cloth_mesh import cloth_from_mask, garment_mask_path
                _capdir = f"/home/aoloo/sam-3d-objects/captures/{_TS_DEFAULT}"
                _maskp = garment_mask_path(_TS_DEFAULT, _sel)
                _mv, _, _, _ = cloth_from_mask(_maskp, _capdir, step=4)   # observed silhouette (base cm)
                _pwb = np.asarray(_place_world(_TS_DEFAULT)["base_true"], float)   # real geometry
                _obs = (_pwb[None, :] + _mv[:, 0:1] * _FWD[None, :]
                        + _mv[:, 1:2] * _LEFT[None, :] + _mv[:, 2:3] * _UP[None, :])[:, :2]

                usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
                shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
                _av = np.asarray(shirt_mesh.vertices, float).copy()
                # place the asset the way the stock build does (rot pi about z, +70 in y), so the
                # fit starts from the same world pose the twin has always used
                _av[:, 0], _av[:, 1] = -_av[:, 0], 70.0 - (_av[:, 1] - 0.0)
                _a2, _o2 = _av[:, :2], _obs

                def _pca_frame(pts):
                    c = pts.mean(0)
                    d = pts - c
                    _, _, vt = np.linalg.svd(d, full_matrices=False)
                    ax = vt[0] / np.linalg.norm(vt[0])
                    return c, ax, d

                _ca, _axa, _da = _pca_frame(_a2)
                _co, _axo, _do = _pca_frame(_o2)
                # minimal rotation aligning the long axes (mod 180 deg - a lay flat garment has no
                # front/back sign the mask can see, so take the smaller turn and say so)
                _th = float(np.arctan2(_axo[1], _axo[0]) - np.arctan2(_axa[1], _axa[0]))
                _th = (_th + np.pi / 2.0) % np.pi - np.pi / 2.0
                _R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
                _ar = _da @ _R.T
                # per-axis scale in the observed principal frame, from robust (2/98) extents
                _P = np.stack([_axo, [-_axo[1], _axo[0]]])
                _ap, _op = _ar @ _P.T, _do @ _P.T
                _ext = lambda x: np.percentile(x, 98, 0) - np.percentile(x, 2, 0)
                _sc = _ext(_op) / np.maximum(_ext(_ap), 1e-6)
                # TWIN_FIT_SCALE=sx,sy: rest-shape scale in the ASSET's axes (x=width, y=length),
                # measured on a FLAT capture of this garment (fold_state_fit --flat-ts). When the
                # capture shows a FOLDED garment its silhouette is half the shirt and its principal
                # axis may be the width, so the PCA-frame scale above would halve the rest shape.
                _fs = _os.environ.get("TWIN_FIT_SCALE", "") if _is_active else ""
                if _fs:
                    _fsv = [float(v) for v in _fs.split(",")][:2]
                    _ca2 = _a2.mean(0)
                    _a2s = (_a2 - _ca2) * np.array(_fsv)[None, :] + _ca2
                    _ar = (_a2s - _a2s.mean(0)) @ _R.T
                    _ap = _ar @ _P.T
                    _sc = np.array([1.0, 1.0])
                    print(f"[twin-stock] cloth fit: rest-shape scale forced by TWIN_FIT_SCALE "
                          f"({_fsv[0]:.3f},{_fsv[1]:.3f}) in asset axes", flush=True)
                _fit2 = ((_ap * _sc[None, :]) @ _P) + _co[None, :]
                # asset-axis rest-shape scale the planner must use (fold_tamp --scale); when the
                # PCA frame is (near) the asset frame this is _sc, else the forced value
                if _is_active:
                    self._fit_scale = [float(v) for v in (_fsv if _fs else _sc)]
                _av[:, 0], _av[:, 1] = _fit2[:, 0], _fit2[:, 1]
                _av[:, 2] += 30.0 - _av[:, 2].min()               # settle from just above the table
                mesh_points = _av
                mesh_indices = shirt_mesh.indices
                print(f"[twin-stock] cloth fit '{_glabel}': unisex_shirt.usd -> capture {_TS_DEFAULT}: "
                      f"rot {np.degrees(_th):+.0f}deg, scale ({_sc[0]:.2f},{_sc[1]:.2f}), "
                      f"centre ({_co[0]:.0f},{_co[1]:.0f})cm  [front/back assumed by minimal turn]",
                      flush=True)
                _cloth_rot = wp.quat_identity()
                _cloth_pos = wp.vec3(0.0, 0.0, 0.0)
            elif _CLOTH_SRC == "mask":
                # THE REAL GARMENT. cloth_from_mask returns a flat, boundary-conforming sheet in the
                # robot's own base frame (cm) at the pose the capture measured, so it is placed by
                # its own coordinates - no rot, no pos offset, nothing hand-fitted.
                from cloth_mesh import cloth_from_mask, garment_mask_path
                _capdir = f"/home/aoloo/sam-3d-objects/captures/{_TS_DEFAULT}"
                _maskp = garment_mask_path(_TS_DEFAULT, _sel)
                # Match the SHIRT's standard, not just its silhouette. step 7 gave 915 particles
                # over a 39x51cm garment - ~1.5cm spacing, so a 4cm pinch found 24 points where the
                # 6436-particle shirt finds 420, and every grasp threshold in this file (min_grip 70,
                # HOLD_MAX 420, the miss warning at 50) is calibrated for the shirt. step 4 with TWO
                # LAYERS gives ~5600 particles and a front/back panel like the real garment, instead
                # of a single-sheet cutout with no thickness to fold against.
                _v_base, _f, _uv, _tex = cloth_from_mask(
                    _maskp, _capdir,
                    step=int(_os.environ.get("TWIN_MASK_STEP", "4")),
                    two_layer=_os.environ.get("TWIN_CLOTH_LAYERS", "2") == "2",
                    layer_gap=float(_os.environ.get("TWIN_LAYER_GAP_CM", "1.6")))
                _pw = _place_world(_TS_DEFAULT)["base_true"]      # the REAL base: cloth stays put
                _vw = (np.asarray(_pw, float)[None, :]
                       + _v_base[:, 0:1] * _FWD[None, :]
                       + _v_base[:, 1:2] * _LEFT[None, :]
                       + _v_base[:, 2:3] * _UP[None, :])
                # Sit it ON the table while KEEPING the layer separation - flattening z to a single
                # plane would collapse the two panels into one and throw the layering away.
                _vw[:, 2] += self.TABLE_TOP + 0.5 - float(_vw[:, 2].min())
                mesh_points = _vw
                mesh_indices = np.asarray(_f, dtype=np.int32).reshape(-1)
                _ext = _vw.max(0) - _vw.min(0)
                print(f"[twin-stock] cloth from mask '{_glabel}': {len(_vw)} particles, "
                      f"{_ext[0]:.0f}x{_ext[1]:.0f}cm from {_TS_DEFAULT}", flush=True)
                if _is_active:
                    self.cloth_uvs, self.cloth_tex = _uv, _tex
                _cloth_rot = wp.quat_identity()
                _cloth_pos = wp.vec3(0.0, 0.0, 0.0)
            elif _CLOTH_SRC == "towel":
                # A PLAIN RECTANGLE. Parallel edges, four equal corners, no taper and no sleeves:
                # the clean instrument for two-handed folding, where a corner-to-corner carry keeps
                # the grasp separation EXACTLY constant and "did the pair stay rigid" is a question
                # the geometry can answer. Spawned flat AT table height - the shirt asset drops
                # from 30cm and keeps ~1cm of wrinkle that four seconds of VBD never relaxes.
                _spec = getattr(self, "_towel_specs", {}).get(_glabel)
                if _gi > 0 and _spec is None:
                    break
                if _spec and _spec.get("kind") == "shirt":
                    # STOCK SHIRT in the pile: the tuned unisex_shirt asset, scaled about its centre,
                    # laid flat at the pile height (not dropped from 30 cm onto the towels above it)
                    usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
                    _sm = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
                    _sv = np.asarray(_sm.vertices, float).copy()
                    _sv[:, 0], _sv[:, 1] = -_sv[:, 0], 70.0 - _sv[:, 1]          # the stock placement (rot pi, +70 y)
                    _c = _sv[:, :2].mean(0)
                    _sv[:, :2] = (_sv[:, :2] - _c) * _spec["scale"] + _c + np.array([_spec["dx"], _spec["dy"]])
                    _z0s = self.TABLE_TOP + 0.5 if _spec["k"] == 0 else getattr(self, "_pile_top", self.TABLE_TOP + 0.5) + float(_os.environ.get("TWIN_TOWEL_PILE_GAP_CM", "1.0"))
                    _sv[:, 2] = (_sv[:, 2] - _sv[:, 2].min()) * _spec["scale"] + _z0s
                    self._pile_top = float(_sv[:, 2].max())
                    mesh_points = _sv
                    mesh_indices = _sm.indices
                    _ext = _sv.max(0) - _sv.min(0)
                    self._towel_infos = getattr(self, "_towel_infos", {})
                    self._towel_infos[_glabel] = {"kind": "shirt", "w": float(_ext[0]), "h": float(_ext[1]),
                                                  "centre": [float(v) for v in _sv[:, :2].mean(0)]}
                    print(f"[twin-stock] shirt '{_glabel}': stock asset x{_spec['scale']:.2f} = {_ext[0]:.0f}x{_ext[1]:.0f}cm flat at "
                          f"({_sv[:, 0].mean():.0f},{_sv[:, 1].mean():.0f}), pile level {_spec['k']}", flush=True)
                    _cloth_rot = wp.quat_identity()
                    _cloth_pos = wp.vec3(0.0, 0.0, 0.0)
                    vertices = [wp.vec3(v) for v in mesh_points]
                    if self.add_cloth:
                        _p0 = int(self.scene.particle_count)
                        self.scene.add_cloth_mesh(vertices=vertices, indices=mesh_indices, rot=_cloth_rot, pos=_cloth_pos,
                                                  vel=wp.vec3(0.0, 0.0, 0.0), density=float(_os.environ.get("TWIN_CLOTH_DENSITY", "0.02")),
                                                  scale=1.0, tri_ke=self.tri_ke, tri_ka=self.tri_ka, tri_kd=self.tri_kd,
                                                  edge_ke=self.bending_ke, edge_kd=self.bending_kd, particle_radius=self.cloth_particle_radius)
                        self._garments.append({"id": _gid, "label": _glabel, "start": _p0,
                                               "count": int(self.scene.particle_count) - _p0, "active": _is_active})
                    continue
                _tw = _spec["w"] if _spec else float(_os.environ.get("TWIN_TOWEL_W_CM", "40"))
                _th_ = _spec["h"] if _spec else float(_os.environ.get("TWIN_TOWEL_H_CM", "30"))
                _sp = float(_os.environ.get("TWIN_TOWEL_SPACING_CM", "0.5"))
                _nx, _ny = int(round(_tw / _sp)) + 1, int(round(_th_ / _sp)) + 1
                _tc = _garment_centre_xy()
                _tz0 = self.TABLE_TOP + 0.5
                if _spec:
                    _tc = _tc + np.array([_spec["dx"], _spec["dy"]])
                    if _spec["k"] > 0:
                        # on the pile: start just above the TOP of what was built under it (a
                        # fixed k x gap put towelA inside the shirt's upper layer -> NaN at 21 s)
                        _tz0 = getattr(self, "_pile_top", self.TABLE_TOP + 0.5) + float(_os.environ.get("TWIN_TOWEL_PILE_GAP_CM", "1.0"))
                self._pile_top = _tz0 + (float(_os.environ.get("TWIN_TOWEL_GAP_CM", "0.6")) if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2" else 0.0)
                _gx, _gy = np.meshgrid(np.linspace(_tc[0] - _tw / 2, _tc[0] + _tw / 2, _nx),
                                       np.linspace(_tc[1] - _th_ / 2, _tc[1] + _th_ / 2, _ny),
                                       indexing="ij")
                mesh_points = np.stack([_gx.reshape(-1), _gy.reshape(-1),
                                        np.full(_gx.size, _tz0)], 1)
                _ix = np.arange(_nx * _ny).reshape(_nx, _ny)
                _a, _b = _ix[:-1, :-1].reshape(-1), _ix[1:, :-1].reshape(-1)
                _c, _d = _ix[1:, 1:].reshape(-1), _ix[:-1, 1:].reshape(-1)
                mesh_indices = np.stack([_a, _b, _c, _a, _c, _d], 1).reshape(-1).astype(np.int32)
                if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2":
                    # a second sheet TWIN_TOWEL_GAP_CM above, stitched to the first along the
                    # whole perimeter: a towel with thickness, not a sheet of paper. Labels index
                    # the BOTTOM layer (grid indices unchanged).
                    _gap = float(_os.environ.get("TWIN_TOWEL_GAP_CM", "0.6"))
                    _n0 = _nx * _ny
                    _top = mesh_points.copy(); _top[:, 2] += _gap
                    _ix2 = _ix + _n0
                    _a2, _b2 = _ix2[:-1, :-1].reshape(-1), _ix2[1:, :-1].reshape(-1)
                    _c2, _d2 = _ix2[1:, 1:].reshape(-1), _ix2[:-1, 1:].reshape(-1)
                    _tris2 = np.stack([_a2, _c2, _b2, _a2, _d2, _c2], 1).reshape(-1)   # flipped winding
                    _rim = []
                    _ring = list(_ix[:, 0]) + list(_ix[-1, 1:]) + list(_ix[-2::-1, -1]) + list(_ix[0, -2:0:-1])
                    for _k in range(len(_ring)):
                        _p, _q = int(_ring[_k]), int(_ring[(_k + 1) % len(_ring)])
                        _rim += [_p, _q, _q + _n0, _p, _q + _n0, _p + _n0]
                    mesh_points = np.vstack([mesh_points, _top])
                    mesh_indices = np.concatenate([mesh_indices, _tris2.astype(np.int32), np.asarray(_rim, np.int32)])
                    print(f"[twin-stock] towel: second layer {_gap:.1f}cm above, stitched at the rim "
                          f"({len(mesh_points)} particles)", flush=True)
                _tinfo = {"nx": _nx, "ny": _ny, "w": _tw, "h": _th_,
                          "centre": [float(_tc[0]), float(_tc[1])]}
                self._towel_infos = getattr(self, "_towel_infos", {})
                self._towel_infos[_glabel if _spec else "towel"] = _tinfo
                if _is_active or self._towel is None:
                    self._towel = _tinfo
                if not _spec:
                    _glabel = "towel"
                print(f"[twin-stock] towel {_tw:.0f}x{_th_:.0f}cm: {_nx}x{_ny} = {_nx * _ny} particles "
                      f"at {_sp:.1f}cm spacing, flat on the table at ({_tc[0]:.0f},{_tc[1]:.0f})",
                      flush=True)
                _cloth_rot = wp.quat_identity()
                _cloth_pos = wp.vec3(0.0, 0.0, 0.0)
            else:
                if _gi > 0:
                    break                                  # one stock shirt, at the stock spot
                usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
                usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
                shirt_mesh = newton.usd.get_mesh(usd_prim)
                mesh_points = shirt_mesh.vertices
                mesh_indices = shirt_mesh.indices
            vertices = [wp.vec3(v) for v in mesh_points]

            if self.add_cloth:
                _p0 = int(self.scene.particle_count)
                self.scene.add_cloth_mesh(
                    vertices=vertices,
                    indices=mesh_indices,
                    rot=_cloth_rot,
                    pos=_cloth_pos,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    # TWIN_CLOTH_DENSITY: mass per unit area in this cm-scale sim. The stock 0.02
                    # gives a 40x30cm towel about 24g - a paper tissue. A real hand towel is
                    # 150-200g, and friction (the only thing holding the far half down while the
                    # near half is carried over) is proportional to that weight, which is why
                    # every variant dragged the sheet instead of hinging it at the crease.
                    density=float(_os.environ.get("TWIN_CLOTH_DENSITY", "0.02")),
                    scale=1.0,
                    tri_ke=self.tri_ke,
                    tri_ka=self.tri_ka,
                    tri_kd=self.tri_kd,
                    edge_ke=self.bending_ke,
                    edge_kd=self.bending_kd,
                    particle_radius=self.cloth_particle_radius,
                )

                self._garments.append({"id": _gid, "label": _glabel, "start": _p0,
                                       "count": int(self.scene.particle_count) - _p0,
                                       "active": _is_active})
        if self.add_cloth:
            if len(self._garments) > 1:
                print("[twin-stock] garments in the scene: " + "; ".join(
                    f"{'*' if g['active'] else ''}{g['label']} [{g['start']}:{g['start'] + g['count']}]"
                    for g in self._garments) + "  (* = active)", flush=True)
            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        for _ci in getattr(self, "cage_shape_idx", []):
            flags[_ci] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # ZEROS, not empty: this array is assigned straight into state joint_qd every step, so
        # uninitialised device memory becomes the arm's velocity on step 0. With one arm the
        # allocation came back clean and it never showed; the larger two-arm allocation did not,
        # and a single NaN in joint_q never leaves (every later IK then reads a diverged state).
        self.target_joint_qd = wp.zeros_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                # per-particle contact buffers: two dense 8-layer wads on top of each other may
                # need more than the 16/20 a single folded garment gets by (TWIN_CONTACT_BUF_*)
                particle_vertex_contact_buffer_size=int(__import__("os").environ.get("TWIN_CONTACT_BUF_V", "16")),
                particle_edge_contact_buffer_size=int(__import__("os").environ.get("TWIN_CONTACT_BUF_E", "20")),
                particle_collision_detection_interval=int(__import__("os").environ.get("TWIN_COLL_INTERVAL", "-1")),
            )

        self.viewer.set_model(self.model)

        self._usd_rec = None
        if _REC_USD:
            from usd_recorder import UsdRecorder
            # colour every garment with the colour the reconstruction measured for it
            _vcols = None
            _uvs, _texp = None, None
            if getattr(self, "_towel", None):
                # the towel: cream with a little per-vertex terry grain, plus UVs and a generated
                # terry texture so Isaac renders pile fabric rather than a flat plastic sheet
                _t = self._towel; _nx, _ny = _t["nx"], _t["ny"]
                _c = np.array([float(v) for v in _os.environ.get("TWIN_TOWEL_COLOR", "0.70,0.80,0.90").split(",")[:3]])
                _rng = np.random.default_rng(0)
                _n = int(self.model.particle_count)
                _vcols = np.tile(_c, (_n, 1)) * (1.0 + 0.04 * _rng.standard_normal((_n, 1)))
                if getattr(self, "_towel_specs", None):
                    # several garments: each its own colour so stacked bundles read as separate
                    # objects in the box (towelA cream-blue, towelB sage, shirt plum)
                    _pal = [[0.70, 0.80, 0.90], [0.72, 0.84, 0.70], [0.62, 0.48, 0.70], [0.90, 0.78, 0.60]]
                    for _gi2, _g2 in enumerate(self._garments or []):
                        _vcols[_g2["start"]:_g2["start"] + _g2["count"]] = np.asarray(_pal[_gi2 % len(_pal)]) * (1.0 + 0.04 * _rng.standard_normal((_g2["count"], 1)))
                _ii, _jj = np.meshgrid(np.arange(_nx), np.arange(_ny), indexing="ij")
                _uv1 = np.stack([_ii.reshape(-1) / max(_nx - 1, 1) * 4.0, _jj.reshape(-1) / max(_ny - 1, 1) * 3.0], 1)
                _uvs = np.vstack([_uv1] * (_n // (_nx * _ny))) if _n % (_nx * _ny) == 0 else None
                try:
                    import zlib, struct
                    _S = 256
                    _z = _rng.standard_normal((_S, _S))
                    for _k in range(3):                          # soft blobs = terry loops
                        _z = (_z + np.roll(_z, 1, 0) + np.roll(_z, -1, 0) + np.roll(_z, 1, 1) + np.roll(_z, -1, 1)) / 5.0
                    _z = (_z - _z.min()) / max(np.ptp(_z), 1e-6)
                    _img = (np.clip(_c[None, None, :] * (0.82 + 0.28 * _z[:, :, None]), 0, 1) * 255).astype(np.uint8)
                    _texp = _os.path.splitext(_REC_USD)[0] + "_terry.png" if _REC_USD else None
                    if _texp:
                        # a minimal PNG writer: no PIL in the Newton environment
                        _raw = b"".join(b"\x00" + _img[_r].tobytes() for _r in range(_S))
                        def _chunk(_t, _d): return struct.pack(">I", len(_d)) + _t + _d + struct.pack(">I", zlib.crc32(_t + _d) & 0xffffffff)
                        with open(_texp, "wb") as _fh:
                            _fh.write(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", _S, _S, 8, 2, 0, 0, 0))
                                      + _chunk(b"IDAT", zlib.compress(_raw, 6)) + _chunk(b"IEND", b""))
                        print(f"[twin-stock] towel texture -> {_texp}", flush=True)
                except Exception as _ex:
                    print(f"[twin-stock] towel texture skipped: {_ex}", flush=True); _texp = None
            elif getattr(self, "_garments", None) and _CLOTH_SRC in ("fit", "mask"):
                try:
                    from cloth_mesh import select_garment
                    _vcols = np.zeros((int(self.model.particle_count), 3), float) + 0.7
                    for _g in self._garments:
                        if getattr(self, "_towel", None):
                            # a towel: cream by default, with a little per-vertex terry noise so
                            # the render reads as pile fabric rather than a flat plastic sheet
                            _c = [float(v) for v in _os.environ.get("TWIN_TOWEL_COLOR", "0.70,0.80,0.90").split(",")[:3]]
                        else:
                            _c = select_garment(_TS_DEFAULT, str(_g["id"])).get("display_color") or [0.83, 0.90, 0.28]
                        _vcols[_g["start"]:_g["start"] + _g["count"]] = np.asarray(_c[:3], float)
                except Exception as _ex:
                    print(f"[twin-stock] garment colours unavailable: {_ex}", flush=True)
                    _vcols = None
            self.model._rec_box_shapes = tuple(getattr(self, "_rec_box_shapes", ()))
        if getattr(self, "_rigid_world_spec", None):
            self._build_rigid_world()
            self._usd_rec = UsdRecorder(
                self.model, _REC_USD, every=_REC_EVERY, fps=self.fps,
                table=((self.table_pos_cm[0], self.table_pos_cm[1], self.table_pos_cm[2]),
                       self.table_hx_cm, self.table_hy_cm, self.table_hz_cm),
                vertex_colors=_vcols, uvs=_uvs, texture_path=_texp)
        if _PLACE == "measured":
            _pl = _place_world(_TS_DEFAULT)
            _view = _VIEW
            if _view == "capture":
                # exactly what the capture camera saw (the arm fills the frame, as it did)
                _cam, _tgt = _pl["cam"], _pl["cam"] + _pl["cam_dir"] * 90.0
            elif _view == "shoulder":
                # third person from behind the arm: keeps the capture's left-right layout
                _cam = _pl["base"] - _FWD * 145.0 + _UP * 125.0
                _tgt = np.array([_SHIRT_XY[0], _SHIRT_XY[1], _TABLE_TOP_Z + 4.0])
            else:
                # front view: from across the table, looking back at the robot
                _cam = _pl["base"] + _FWD * 225.0 + _UP * 95.0
                _tgt = _pl["base"] + _FWD * 60.0 + _UP * 18.0
            _cp = _cam * self.viz_scale                            # cm -> m for the viewer
            _ct = _tgt * self.viz_scale
            # headless (--viewer null) has no camera; recording runs care about physics only
            if hasattr(self.viewer, "camera"):
                self.viewer.camera.pos = self.viewer.camera._as_vec3(
                    wp.vec3(float(_cp[0]), float(_cp[1]), float(_cp[2]))
                )
                self.viewer.camera.look_at(wp.vec3(float(_ct[0]), float(_ct[1]), float(_ct[2])))
        else:
            self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale particle radii from cm to meters for visualization
        self.sim_particle_radius = self.model.particle_radius
        if self.model.particle_radius is not None:
            radius_np = self.model.particle_radius.numpy().copy()
            radius_np *= self.viz_scale
            self.viz_particle_radius = wp.array(radius_np, dtype=wp.float32, device=self.model.device)
        else:
            self.viz_particle_radius = None

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(self.model.gravity.shape[0], dtype=wp.vec3, device=self.model.device)
        # gravity in cm/s²
        self.gravity_earth = wp.full(
            self.model.gravity.shape[0], wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3, device=self.model.device
        )

        if getattr(self, "_towel", None) and _os.environ.get("TWIN_TOWEL_LABELS"):
            # A towel's parts are known from its own grid - no VLM, no lifting, no snapping - so the
            # label-move planner and the twin can be exercised on geometry that is exactly right.
            _t = self._towel
            _nx, _ny = _t["nx"], _t["ny"]
            _at = lambda i, j: int(i * _ny + j)
            _pts = self.state_0.particle_q.numpy()
            _names = {
                "near_left_corner": (_at(0, 0), "corner"),
                "near_right_corner": (_at(_nx - 1, 0), "corner"),
                "far_left_corner": (_at(0, _ny - 1), "corner"),
                "far_right_corner": (_at(_nx - 1, _ny - 1), "corner"),
                "near_edge_midpoint": (_at(_nx // 2, 0), "edge_midpoint"),
                "far_edge_midpoint": (_at(_nx // 2, _ny - 1), "edge_midpoint"),
                "left_edge_midpoint": (_at(0, _ny // 2), "edge_midpoint"),
                "right_edge_midpoint": (_at(_nx - 1, _ny // 2), "edge_midpoint"),
                "centre": (_at(_nx // 2, _ny // 2), "region_center"),
            }
            _out = {"garment": "towel", "labels": {}, "state": "",
                    "pairs": [["near_left_corner", "near_right_corner"],
                              ["far_left_corner", "far_right_corner"],
                              ["left_edge_midpoint", "right_edge_midpoint"],
                              ["near_edge_midpoint", "far_edge_midpoint"]]}
            for _nm, (_pi, _kd) in _names.items():
                _out["labels"][_nm] = {"kind": _kd, "particle": _pi, "snap_cm": 0.0,
                                       "world_cm": [round(float(v), 1) for v in _pts[_pi][:3]]}
            with open(_os.environ["TWIN_TOWEL_LABELS"], "w") as _fh:
                __import__("json").dump(_out, _fh, indent=1)
            print(f"[twin-stock] towel labels -> {_os.environ['TWIN_TOWEL_LABELS']} "
                  f"({len(_names)} parts, exact grid indices)", flush=True)

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if _os.environ.get("TWIN_DEBUG_NAN", "0") == "1":
            _q = self.state_0.joint_q.numpy(); _b = self.state_0.body_q.numpy()
            print(f"[nan-check] after build+FK: joint_q finite {bool(np.isfinite(_q).all())} "
                  f"(bad {np.where(~np.isfinite(_q))[0].tolist()[:6]}), "
                  f"body_q finite {bool(np.isfinite(_b).all())} "
                  f"(bad bodies {sorted(set(np.where(~np.isfinite(_b))[0].tolist()))[:6]})", flush=True)

        # graph capture (TWIN_DEBUG_NO_GRAPH=1 runs simulate() directly - diagnostic)
        self.graph = None
        if self.add_cloth and _os.environ.get("TWIN_DEBUG_NO_GRAPH", "0") != "1":
            self._nan_probe("before capture")
            self.capture()
            self._nan_probe("after capture")

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        def _make_body_out(ee_id, ee_off):
            """Build a Jacobian kernel for ONE end effector. The body id and tip offset are baked
            in with wp.static, so a second hand needs its own kernel - swapping the attribute would
            keep computing the first arm's Jacobian."""

            @wp.kernel
            def compute_body_out(
                body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector],
                body_com: wp.array[wp.vec3],
                body_out: wp.array[float],
            ):
                # body_qd is COM-referenced (linear velocity at body COM, world
                # frame).  Compute EE tip velocity in world frame, consistent with
                # compute_ee_delta which measures the tip position as
                # transform_point(body_q, ee_offset).
                eid = wp.static(int(ee_id))
                eoff = wp.static(wp.vec3(float(ee_off[0]), float(ee_off[1]), float(ee_off[2])))
                X_wb = body_q[eid]
                # Vector from COM to EE tip, rotated to world frame
                r_world = wp.transform_vector(X_wb, eoff - body_com[eid])
                qd = body_qd[eid]
                omega = wp.spatial_bottom(qd)
                v_com = wp.spatial_top(qd)
                v_tip = v_com + wp.cross(omega, r_world)
                body_out[0] = v_tip[0]
                body_out[1] = v_tip[1]
                body_out[2] = v_tip[2]
                body_out[3] = omega[0]
                body_out[4] = omega[1]
                body_out[5] = omega[2]

            return compute_body_out

        self._make_body_out = _make_body_out
        self.compute_body_out_kernel = _make_body_out(
            self.endeffector_id, [float(v) for v in self.endeffector_offset.p])
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

        # Bimanual rig: hand 0 reuses the arrays above, every other hand gets its own kernel,
        # output vector and Jacobian buffer so the two IK solves cannot overwrite each other.
        for _i, _h in enumerate(getattr(self, "_hands", None) or []):
            if _i == 0:
                _h["kernel"] = self.compute_body_out_kernel
                _h["body_out"] = self.body_out
                _h["J_flat"] = self.J_flat
                _h["ee_delta"] = self.ee_delta
            else:
                _h["kernel"] = _make_body_out(_h["ee_body"], [float(v) for v in _h["pinch"].p])
                _h["body_out"] = wp.empty(out_dim, dtype=float, requires_grad=True)
                _h["J_flat"] = wp.empty(out_dim * in_dim, dtype=float)
                _h["ee_delta"] = wp.empty(1, dtype=wp.spatial_vector)

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        if _PLACE == "measured":
            _b = _place_world(_TS_DEFAULT)["base"]
            _yaw = -np.pi / 2.0        # forward = -y, i.e. the arm looks down the axis the capture camera did
            _base_xform = wp.transform(
                (float(_b[0]), float(_b[1]), float(_b[2])),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _yaw),
            )
            _pw = _place_world(_TS_DEFAULT)
            self.base_cm = np.asarray(_b, float)
            print(f"[twin-stock] robot base ({_b[0]:.0f},{_b[1]:.0f},{_b[2]:.0f})cm facing -y | "
                  f"standoff {_pw['standoff']:.0f}cm (measured {_pw['standoff'] + _CLOSER:.0f} "
                  f"- {_CLOSER:.0f} step-in)", flush=True)
        else:
            _yaw = np.pi if _SIDE > 0 else 0.0
            _base_xform = wp.transform(
                (50.0 * _SIDE, -50.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _yaw),
            )
        self._ur_down_quat = None
        self._aloha = None
        self._hands = None
        if _ARM == "aloha":
            from aloha_arms import add_aloha, BASE_XY_CM, pinch_world
            # The rig is a FIXED pair of bases 93.8cm apart, so the GARMENT is placed between the
            # arms rather than the robot being placed from the capture (_place_world). Rig frame:
            # z=0 is the table top and the calibrated tool-down homes pinch at (-/+15,-20)cm, so
            # the rig origin goes 20cm on the far side of the garment centroid and the cloth lands
            # exactly where those homes were solved. The centroid is known before the cloth is
            # built: a loaded state carries it, otherwise it is the stock shirt's own bbox centre.
            _gc = _garment_centre_xy()
            _rig = wp.transform((float(_gc[0]), float(_gc[1]) + 20.0, self.TABLE_TOP), wp.quat_identity())
            self._aloha = add_aloha(builder, _rig, scale=100.0, home=True)
            self._n_arm_dof = int(self._aloha["n_arm"])
            # the tool-down quaternion and home pinch of EACH hand, by FK on a throwaway copy of
            # the same rig at the same xform (so the numbers are already in world coordinates)
            _tmp = ModelBuilder()
            _r2 = add_aloha(_tmp, _rig, scale=100.0, home=True)
            _tm = _tmp.finalize(device="cpu"); _ts = _tm.state()
            newton.eval_fk(_tm, _tm.joint_q, _tm.joint_qd, _ts)
            _bq = _ts.body_q.numpy()
            self._hands = []
            for _s in ("right", "left"):      # hand 0 = right; the legacy single-arm path drives it
                _a = self._aloha[_s]
                self._hands.append({
                    "name": _s,
                    "ee_body": int(_a["ee_body"]),
                    "pinch": wp.transform([float(v) for v in self._aloha["pinch_local"]], wp.quat_identity()),
                    "arm_dofs": [int(d) for d in _a["arm_dofs"]],
                    "arm_coords": [int(c) for c in _a["arm_coords"]],
                    "finger_drive": ("prismatic", [(int(d), int(d)) for d in _a["finger_dofs"]]),
                    "finger_dofs": [int(d) for d in _a["finger_dofs"]],
                    "down_quat": tuple(float(v) for v in _bq[_r2[_s]["ee_body"]][3:7]),
                    "home_xyz": [float(v) for v in pinch_world(_bq, _r2[_s], scale=100.0)],
                    "base_cm": np.array([BASE_XY_CM[_s][0] + float(_rig.p[0]),
                                         BASE_XY_CM[_s][1] + float(_rig.p[1]), float(_rig.p[2])], float),
                    "max_reach": 60.0,      # ViperX 300: the calibrated straight-down workspace
                    "targets": None, "durations": None, "times": None, "target": None,
                })
            print("[twin-stock] arm = ALOHA 2 (bimanual ViperX 300): "
                  + ", ".join(f"{h['name']} ee body {h['ee_body']} dofs {h['arm_dofs'][0]}-{h['arm_dofs'][-1]} "
                              f"home ({h['home_xyz'][0]:.0f},{h['home_xyz'][1]:.0f},{h['home_xyz'][2]:.0f})cm"
                              for h in self._hands)
                  + f", pinch {[round(float(v), 2) for v in self._aloha['pinch_local']]}cm", flush=True)
        elif _ARM == "ur5e2":
            # The SAME arm as the real cell, twice: bases facing each other across the garment.
            # Everything the single UR5e path calibrated (home, pinch offset 22.3cm in wrist_3,
            # base-frame yaw vs ROS) carries over per arm; only the placement is new.
            from ur5e_gripper import add_ur5e_gripper
            _gc = _garment_centre_xy()
            _stand = float(_os.environ.get("TWIN_ARM_STANDOFF_CM", "48"))
            _bz = self.TABLE_TOP + 4.0
            _ur_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
            self._n_arm_dof = 6
            self._hands = []
            # TWIN_ARM_LAYOUT=side (default): both arms on the robot's edge of the table (+y, where
            # the real UR5e stands), TWIN_ARM_SPACING_CM apart, both facing the garment. Each arm
            # owns its half, so a corner-to-corner fold never sends one across the other's reach.
            # =opposite: the earlier layout, facing each other across the garment - every move
            # then converges both arms over the cloth and they foul each other.
            _layout = _os.environ.get("TWIN_ARM_LAYOUT", "side")
            _spacing = float(_os.environ.get("TWIN_ARM_SPACING_CM", "60"))
            for _nm, _sgn, _face_opp in (("right", +1.0, np.pi), ("left", -1.0, 0.0)):
                # _face = the direction this arm's base_link +x looks (its home reaches that way);
                # the menagerie base frame is base_link yawed 180deg, hence the +pi in the MJCF.
                if _layout == "opposite":
                    _face = _face_opp
                    _bxy = (float(_gc[0]) + _sgn * _stand, float(_gc[1]))
                else:
                    _face = -np.pi / 2.0                          # look down -y, at the garment
                    _bxy = (float(_gc[0]) + _sgn * 0.5 * _spacing, float(_gc[1]) + _stand)
                _xf = wp.transform((_bxy[0], _bxy[1], _bz),
                                   wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _face + np.pi))
                _j0 = builder.joint_count
                _w3, _d0, _na, _pinch = add_ur5e_gripper(builder, _xf, scale=100.0, home=False, loop_joints=False,
                                                         gripper_armature=float(_os.environ.get("TWIN_GRIPPER_ARMATURE", "0")),
                                                         finger_cloth_contact=_os.environ.get("TWIN_FINGER_CLOTH_CONTACT", "0") == "1")
                _ac = []
                for _i in range(6):
                    builder.joint_q[int(builder.joint_q_start[_j0 + _i])] = _ur_home[_i]
                    _ac.append(int(builder.joint_q_start[_j0 + _i]))
                _keys = getattr(builder, "joint_key", None) or getattr(builder, "joint_label", None)
                # (dof, COORD, sign): the drive WRITES a velocity by dof index but READS the joint
                # position from joint_q, which is laid out by COORDINATE. The 2F-85's ball joints
                # make the two layouts drift apart (22 coords vs 20 dofs per arm), so on the second
                # arm a dof index reads a neighbouring joint - the finger loop then never closes and
                # the 4-bar fights itself with positive feedback (finger |qd| doubled every frame
                # from frame 0 until NaN). The first arm only worked because its offset is zero.
                _fd = []
                for _j in range(_j0, builder.joint_count):
                    _k = str(_keys[_j])
                    if "robotiq" not in _k:
                        continue
                    if _k.endswith("driver_joint") or _k.endswith("spring_link_joint"):
                        _fd.append((int(builder.joint_qd_start[_j]), int(builder.joint_q_start[_j]), 1.0))
                    elif _k.endswith("follower_joint"):
                        _fd.append((int(builder.joint_qd_start[_j]), int(builder.joint_q_start[_j]), -1.0))
                # this arm's tool-down quaternion and home pinch, by FK on a throwaway copy
                _tmp = ModelBuilder()
                _t3, _td0, _, _tp = add_ur5e_gripper(_tmp, _xf, scale=100.0, home=False)
                for _i in range(6):
                    _tmp.joint_q[int(_tmp.joint_q_start[_i])] = _ur_home[_i]
                _tm = _tmp.finalize(device="cpu"); _ts = _tm.state()
                newton.eval_fk(_tm, _tm.joint_q, _tm.joint_qd, _ts)
                _bq = _ts.body_q.numpy()[_t3]
                _wq = np.asarray(_bq[3:7], float)
                _x, _y, _z, _w = _wq
                _R = np.array([[1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _z * _w), 2 * (_x * _z + _y * _w)],
                               [2 * (_x * _y + _z * _w), 1 - 2 * (_x * _x + _z * _z), 2 * (_y * _z - _x * _w)],
                               [2 * (_x * _z - _y * _w), 2 * (_y * _z + _x * _w), 1 - 2 * (_x * _x + _y * _y)]])
                _home_xyz = np.asarray(_bq[:3], float) + _R @ np.array([float(v) for v in _tp])
                self._hands.append({
                    "name": _nm,
                    "ee_body": int(_w3),
                    "pinch": wp.transform([float(v) for v in _pinch], wp.quat_identity()),
                    "arm_dofs": [int(_d0 + _i) for _i in range(6)],
                    "arm_coords": _ac,
                    "finger_drive": ("2f85", tuple(_fd)),
                    "finger_dofs": [int(d) for d, _, _ in _fd],
                    "down_quat": tuple(float(v) for v in _wq),
                    "home_xyz": [float(v) for v in _home_xyz],
                    "base_cm": np.array([_bxy[0], _bxy[1], _bz], float),
                    "max_reach": self.MAX_REACH,
                    "targets": None, "durations": None, "times": None, "target": None,
                })
            _arm_np = np.asarray(builder.joint_armature, float)
            print("[twin-stock] gripper armature per arm: " + ", ".join(
                f"{h['name']} min {_arm_np[h['finger_dofs']].min():.3g}" for h in self._hands), flush=True)
            print(f"[twin-stock] arm = TWO UR5e + Robotiq 2F-85, layout '{_layout}', standoff "
                  f"{_stand:.0f}cm about ({_gc[0]:.0f},{_gc[1]:.0f}): "
                  + ", ".join(f"{h['name']} base ({h['base_cm'][0]:.0f},{h['base_cm'][1]:.0f}) "
                              f"home ({h['home_xyz'][0]:.0f},{h['home_xyz'][1]:.0f},{h['home_xyz'][2]:.0f})cm"
                              for h in self._hands), flush=True)
        elif _ARM == "ur5e":
            from ur5e_gripper import add_ur5e_gripper
            # The menagerie model's base frame is the UR 'base' frame = ROS ur5e_base_link yawed
            # 180deg (verified by FK against the UR5e DH chain). _place_world/_export_ee assume
            # base_link +x = the yaw above, so the MJCF gets that yaw + pi.
            _ur_xform = wp.transform(_base_xform.p, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _yaw + np.pi))
            _wrist3, _dof0, _n_arm, _pinch = add_ur5e_gripper(builder, _ur_xform, scale=100.0, home=False)
            # home: shoulder pan 0 puts the reach along base_link +x (forward), tool straight down
            _ur_home = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
            for _i in range(6):
                builder.joint_q[_dof0 + _i] = _ur_home[_i]
            self._n_arm_dof = 6
            self._ur = (_wrist3, _dof0, _pinch)
            # (dof, sign) of the finger joints to drive: right/left driver, spring_link (+a) and
            # follower (-a); found by joint key so a menagerie update cannot silently reorder them
            _fd = []
            for _j in range(builder.joint_count):
                _keys = getattr(builder, "joint_key", None) or getattr(builder, "joint_label", None)
                _k = str(_keys[_j]) if _keys is not None else ""
                if "robotiq" not in _k:
                    continue
                if _k.endswith("driver_joint") or _k.endswith("spring_link_joint"):
                    _fd.append((int(builder.joint_qd_start[_j]), int(builder.joint_q_start[_j]), 1.0))
                elif _k.endswith("follower_joint"):
                    _fd.append((int(builder.joint_qd_start[_j]), int(builder.joint_q_start[_j]), -1.0))
            # (dof, coord, sign) - velocity is written by dof, position read by COORDINATE; the two
            # layouts differ past the 2F-85's ball joints (see the two-arm path for the failure)
            self._ur_finger_dofs = tuple(_fd)
            # the wrist_3 orientation at home = "gripper straight down" for this arm (FK on a
            # throwaway copy of the same articulation)
            _tmp = ModelBuilder()
            _w2, _d2, _, _ = add_ur5e_gripper(_tmp, _ur_xform, scale=100.0, home=False)
            for _i in range(6):
                _tmp.joint_q[_d2 + _i] = _ur_home[_i]
            _tm = _tmp.finalize(device="cpu")
            _ts = _tm.state()
            newton.eval_fk(_tm, _tm.joint_q, _tm.joint_qd, _ts)
            _bq = _ts.body_q.numpy()[_w2]
            self._ur_down_quat = tuple(float(v) for v in _bq[3:7])
            print(f"[twin-stock] arm = UR5e + Robotiq 2F-85 (menagerie): {_n_arm} arm dofs, "
                  f"{len(self._ur_finger_dofs)} finger dofs driven kinematically, pinch offset "
                  f"{[round(float(v), 1) for v in _pinch]}cm in wrist_3, down quat "
                  f"{[round(v, 3) for v in self._ur_down_quat]}", flush=True)
        else:
            builder.add_urdf(
                str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
                xform=_base_xform,
                floating=False,
                scale=100,  # URDF is in meters, scale to cm
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                force_show_colliders=False,
            )
            builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # descend to working height before approaching the cloth
                [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top left
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # bottom left
                [2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top right
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        if _SIDE > 0 and _PLACE != "measured":   # mirror the stock choreography to the other side
            self.robot_key_poses[:, 1] *= -1.0   # x
            self.robot_key_poses[:, 5] *= -1.0   # qy
            self.robot_key_poses[:, 6] *= -1.0   # qz
        if _PLAN or _FOLD_STYLE == "shirt":
            # keep only the opening "descend to working height" pose; the three garment folds
            # are generated from the flat sheet instead of the stock corner-dragging sequence
            self.robot_key_poses = self.robot_key_poses[:1]

        if _GRASP_DZ:
            _z = self.robot_key_poses[:, 3]
            _z[_z <= 21.0] -= _GRASP_DZ            # grasp heights only; lifts/drags untouched
            print(f"[twin-stock] grasp height lowered by {_GRASP_DZ:.1f}cm "
                  f"({int((_z <= 21.0 - _GRASP_DZ).sum())} poses)", flush=True)

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        if _ARM in ("aloha", "ur5e2") and getattr(self, "_hands", None):
            # RETREAT POSE per hand: where an idle arm parks. The working home sits over the
            # table (the left arm's is x=-17, only 13 cm from the centre line), so the other arm
            # grazed it whenever it swept across - both remaining contacts in exec_s8 were the
            # busy arm meeting the parked one. Retreat pulls each hand out past its own base.
            _mid_x = float(np.mean([h["base_cm"][0] for h in self._hands]))
            _out = float(_os.environ.get("TWIN_RETREAT_OUT_CM", "15"))
            for _h in self._hands:
                _bx = float(_h["base_cm"][0])
                _sgn = 1.0 if _bx >= _mid_x else -1.0
                _hx, _hy, _hz = [float(v) for v in _h["home_xyz"]]
                _h["retreat_xyz"] = [_bx + _sgn * _out, _hy, max(_hz, 55.0)]
            print("[twin-stock] idle-arm retreat poses: " + ", ".join(
                f"{h['name']} ({h['retreat_xyz'][0]:.0f},{h['retreat_xyz'][1]:.0f},{h['retreat_xyz'][2]:.0f})"
                for h in self._hands), flush=True)
            _h0 = self._hands[0]
            self.endeffector_id = _h0["ee_body"]
            self.endeffector_offset = _h0["pinch"]
            self.DOWN_QUAT = _h0["down_quat"]
            self.base_cm = _h0["base_cm"]
            # The stock corner-drag choreography is authored for ONE arm standing to one side of
            # the table; with a two-armed rig every run starts from each hand's own tool-down home
            # and the plan stages supply everything after it.
            for _h in self._hands:
                _row = np.array([[2.0, *_h["home_xyz"], *_h["down_quat"], self.GRIP_OPEN]], np.float32)
                _h["targets"] = _row[:, 1:]
                _h["durations"] = _row[:, 0]
                _h["times"] = np.cumsum(_row[:, 0])
                _h["target"] = _h["targets"][0]
            self.robot_key_poses = np.array(
                [[2.0, *_h0["home_xyz"], *_h0["down_quat"], self.GRIP_OPEN]], np.float32)
            self.targets = self.robot_key_poses[:, 1:]
            self.transition_duration = self.robot_key_poses[:, 0]
            self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
            self.target = self.targets[0]
        elif _ARM == "ur5e":
            _wrist3, _dof0, _pinch = self._ur
            self.endeffector_id = _wrist3
            self.endeffector_offset = wp.transform([float(v) for v in _pinch], wp.quat_identity())
            self.DOWN_QUAT = self._ur_down_quat
            # every authored pose keeps the fixed "straight down" tool of THIS arm
            self.robot_key_poses[:, 4:8] = np.array(self.DOWN_QUAT, dtype=self.robot_key_poses.dtype)
            self.targets = self.robot_key_poses[:, 1:]
            self.target = self.targets[0]
        else:
            self.endeffector_id = builder.body_count - 3
            self.endeffector_offset = wp.transform(
                [
                    0.0,
                    0.0,
                    22.0,
                ],
                wp.quat_identity(),
            )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
        hand=None,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        kernel = (hand or {}).get("kernel", self.compute_body_out_kernel)
        body_out = (hand or {}).get("body_out", self.body_out)
        J_flat = (hand or {}).get("J_flat", self.J_flat)

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                kernel,
                1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                ],
                outputs=[body_out],
            )

        for i in range(out_dim):
            tape.backward(grads={body_out: self.Jacobian_one_hots[i]})
            wp.copy(J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    # ---- phases that run after the stock 5-corner fold -------------------------
    # Targets are generated from the cloth's live bounding box rather than hardcoded,
    # so each fold halves whatever the previous one actually produced.
    GRIP_OPEN = 0.8
    GRIP_CLOSE = 0.1
    # TWIN_GRIP_HARD: the closed-jaw command. 0 = fully shut (the pads meet, which on a pinned
    # patch looks like the jaws hold nothing); ~0.06 leaves the fabric's thickness between the pads.
    GRIP_HARD = float(_os.environ.get("TWIN_GRIP_HARD", "0.0"))
    DOWN_QUAT = (0.9239, -0.3827, 0.0, 0.0)   # (qx, qy, qz, qw), same as the stock fold poses
    TABLE_TOP = 20.0
    MAX_REACH = 78.0                          # keeps the arm off its 85.5cm limit

    # ---- multi-garment bookkeeping -----------------------------------------------------
    def _garment_list(self):
        """[(selector, id, label)] of the garments this twin builds; sets self._active."""
        self._active = 0
        self._towel_specs = {}
        if _CLOTH_SRC == "towel" and _os.environ.get("TWIN_TOWELS"):
            # SEVERAL towels in one twin: "name:w,h:dx,dy:k;..." - size, offset from the garment
            # centre, and k = place in the pile (0 = on the table, 1 = on top of the k=0 towel...)
            out = []
            for spec in [x for x in _os.environ["TWIN_TOWELS"].split(";") if x.strip()]:
                name, wh, dxy, k = (spec.split(":") + ["0"])[:4]
                w, h = [float(v) for v in wh.split(",")]; dx, dy = [float(v) for v in dxy.split(",")]
                self._towel_specs[name.strip()] = {"kind": "towel", "w": w, "h": h, "dx": dx, "dy": dy, "k": int(k)}
                out.append((name.strip(), name.strip(), name.strip()))
            for spec in [x for x in _os.environ.get("TWIN_SHIRTS", "").split(";") if x.strip()]:
                # a stock-asset t-shirt in the pile: "name:dx,dy:k:scale"
                name, dxy, k, sc = (spec.split(":") + ["0", "1.0"])[:4]
                dx, dy = [float(v) for v in dxy.split(",")]
                self._towel_specs[name.strip()] = {"kind": "shirt", "dx": dx, "dy": dy, "k": int(k), "scale": float(sc)}
                out.append((name.strip(), name.strip(), name.strip()))
            # build order = pile order (k ascending) so each garment lands on the one under it
            out.sort(key=lambda o: self._towel_specs[o[0]]["k"])
            return out
        if _CLOTH_SRC not in ("fit", "mask"):
            return [(_GARMENT, None, "garment")]
        m = _measured_from_capture(_TS_DEFAULT)
        gl = list(m["garments"])
        if _GARMENTS_MODE != "all":
            gl = [g for g in gl if g["id"] == m["active_id"]]
        self._active = next(i for i, g in enumerate(gl) if g["id"] == m["active_id"])
        return [(str(g["id"]), g["id"], g["label"]) for g in gl]

    @property
    def _gsl(self):
        """Particle slice of the ACTIVE garment (the whole cloth when there is only one)."""
        gs = getattr(self, "_garments", None)
        if not gs:
            return slice(0, None)
        g = gs[self._active]
        return slice(g["start"], g["start"] + g["count"])

    def _qs(self):
        """Positions of the active garment's particles (a copy; indices are LOCAL to it)."""
        return self.state_0.particle_q.numpy()[self._gsl]

    def _qds(self):
        return self.state_0.particle_qd.numpy()[self._gsl]

    def _put_q(self, ql):
        full = self.state_0.particle_q.numpy()
        full[self._gsl] = ql
        self.state_0.particle_q.assign(full)

    def _put_qd(self, qdl):
        full = self.state_0.particle_qd.numpy()
        full[self._gsl] = qdl
        self.state_0.particle_qd.assign(full)

    def _q_of(self, gi):
        g = self._garments[gi]
        return self.state_0.particle_q.numpy()[g["start"]:g["start"] + g["count"]]

    def _garment_index(self, sel):
        """Index into self._garments of an object id or label substring."""
        sel = str(sel).strip()
        gs = getattr(self, "_garments", [])
        if sel.lstrip("-").isdigit():
            hits = [i for i, g in enumerate(gs) if str(g["id"]) == sel]
        else:
            hits = [i for i, g in enumerate(gs) if sel.lower() in str(g["label"]).lower()]
        if len(hits) != 1:
            raise KeyError(f"garment '{sel}' matches {len(hits)} of {[g['label'] for g in gs]}")
        return hits[0]

    def _cloth_bbox(self):
        """Robust extent of the cloth bulk (the ACTIVE garment only).

        min/max is useless here: a handful of stray particles (caught on a finger, flung by a
        contact) stretch the box to the full sheet size long after the bulk has folded, which
        puts generated grasp points in empty table.
        """
        q = self._qs()
        return np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)

    def _clamp_reach(self, x, y, z, hand=None):
        base = (hand or {}).get("base_cm", getattr(self, "base_cm", None))
        if base is None:
            return x, y, z
        v = np.array([x, y, z], float) - base
        r = float(np.linalg.norm(v))
        reach = float((hand or {}).get("max_reach", self.MAX_REACH))
        if r <= reach:
            return x, y, z
        p = base + v * (reach / r)
        return float(p[0]), float(p[1]), float(p[2])

    def _export_ee(self, path):
        """Write the commanded waypoints in the real robot's frame, ready for the UR5e.

        _place_world maps geometry measured in ur5e_base_link (forward, left, up) into sim world
        as base + FWD*f + LEFT*l + UP*u. This is that map inverted, so a waypoint goes back to
        the frame it was measured in. Orientation is the fixed gripper-down quaternion every real
        transfer has used; the wrist roll added for the sim fold is NOT sent, because it has
        never been checked against the UR5e's reach.
        """
        import json as _json
        if _ARM == "aloha":
            # The waypoint stream is one arm's, in the UR5e's measured base frame. The rig has two
            # arms and no counterpart in the cell, so there is nothing honest to export.
            print("[twin-stock] TWIN_EXPORT_EE skipped: the ALOHA rig is a simulation twin, "
                  "the export path is UR5e-only", flush=True)
            return
        base = getattr(self, "base_cm", None)
        if base is None:
            print("[twin-stock] no measured base pose - cannot export to the robot frame", flush=True)
            return
        # base_cm is the base AFTER _place_world steps it _CLOSER toward the garment so the sim
        # arm stops stretching. The REAL ur5e_base_link is that far back, so exporting against
        # the stepped base puts every waypoint _CLOSER short of the actual shirt - it made an
        # unreachable fold dry-run clean at 64/64. Undo the step-in.
        base = np.asarray(base, float) - _FWD * _CLOSER - _LEFT * _CLOSER_LEFT
        # TWIN_EXPORT_TABLE_Z_M: export against ONE absolute table height in the robot frame
        # (the press depth validated on the real arm, -0.09 on this cell 2026-08-31) instead of
        # the depth-measured height. Two garments 34cm apart measured 5.5cm apart in height - the
        # ~11deg extrinsic tilt, not a sloped table - so a fixed TWIN_EXPORT_DZ_CM would have put
        # this capture's waypoints 3cm deeper than the validated press.
        _dz = _EXPORT_DZ
        _tz = __import__("os").environ.get("TWIN_EXPORT_TABLE_Z_M", "")
        if _tz:
            _dz = float(_tz) * 100.0 - (self.TABLE_TOP - base[2])
            print(f"[twin-stock] export: table top forced to z={float(_tz):+.3f}m in the robot frame "
                  f"(dz {_dz:+.1f}cm; measured would have been {(self.TABLE_TOP - base[2]) / 100.0:+.3f}m)",
                  flush=True)
        # SIM CLOTH IS THICK, REAL CLOTH IS THIN. The fitted two-layer shirt reads 1.7-3.8cm
        # "thick" at the grab (layer gap + particle radii), so the sim pinches 1.7cm above its
        # table; exported 1:1 the real fingers stopped ~1cm above a 3mm tee and closed on air
        # (2026-09-16, first real two-shirt run). Every waypoint within TWIN_EXPORT_LOW_CM of
        # the sim table gets its height above the table scaled by TWIN_EXPORT_THICK_SCALE
        # (default 0.3: 1.7cm -> 0.5cm, a 4-layer 3.8cm stack -> 1.1cm).
        _low = float(__import__("os").environ.get("TWIN_EXPORT_LOW_CM", "5.0"))
        _tsc = float(__import__("os").environ.get("TWIN_EXPORT_THICK_SCALE", "0.3"))
        _nlow = 0
        wps, radii = [], []
        _prev_grip = "open"
        for i, (dur, row) in enumerate(zip(self.transition_duration, self.targets)):
            w = np.array([float(row[0]), float(row[1]), float(row[2])]) - base
            _h = float(w @ _UP) - (self.TABLE_TOP - base[2])         # height above the SIM table
            if 0.0 < _h < 2.0 * _low:
                _nlow += 1
                # x_tsc below _low, blended back to identity by 2*_low (continuous, no kink)
                _h = _h * _tsc if _h < _low else _tsc * _low + (_h - _low) * (2.0 * _low - _tsc * _low) / _low
            fwd, left, up = float(w @ _FWD), float(w @ _LEFT), (self.TABLE_TOP - base[2]) + _h + _dz
            pos = [round(fwd / 100.0, 4), round(left / 100.0, 4), round(up / 100.0, 4)]
            radii.append(float(np.linalg.norm(pos)))
            grip = "open" if float(row[7]) > 0.45 else "closed"
            if grip != _prev_grip:
                # The executor actuates the Robotiq ONLY on position-less gripper EVENTS; the
                # per-waypoint field is display-only. Without these the first real run flew the
                # entire fold with open fingers and touched nothing.
                wps.append({"label": f"grip_{'close' if grip == 'closed' else 'open'}_{i}",
                            "gripper": "close" if grip == "closed" else "open"})
                _prev_grip = grip
            wps.append({"label": f"move{i}", "position": pos,
                        "quaternion": [0.0, 1.0, 0.0, 0.0],      # [w,x,y,z] = gripper straight down
                        "gripper": grip,
                        "dur": round(float(dur), 2)})
        out = {"frame": "ur5e_base_link", "units": "m", "default_vel_scale": _EXPORT_VEL,
               "plan": (";".join(f"{s_}:{','.join(n_)}" for s_, n_ in _PLANS) if _PLANS
                        else ",".join(_PLAN)),
               "waypoints": wps}
        with open(path, "w") as fh:
            _json.dump(out, fh, indent=2)
        far = [w["label"] for w, r in zip(wps, radii) if r > _REACH_LIMIT]
        print(f"[twin-stock] exported {len(wps)} EE waypoints -> {path}  ({_nlow} low waypoints: "
              f"height above table x{_tsc:.2f} for the thin real cloth)", flush=True)
        print(f"[twin-stock]   reach {min(radii):.2f}-{max(radii):.2f}m from base, table top "
              f"z={(self.TABLE_TOP - base[2] + _dz) / 100.0:+.3f}m"
              + (f"  <-- {len(far)}/{len(wps)} BEYOND the arm's {_REACH_LIMIT:.2f}m reach: "
                 f"{far[:6]} - do NOT execute this" if far else ""), flush=True)

    def _append_poses(self, poses, hand=None):
        """Queue waypoints for one hand. `hand` is None for the single-arm path (and for hand 0 of
        a bimanual rig, which keeps using the legacy arrays). A dict {hand name: poses} queues
        several hands at once on ONE time base, which is what makes a two-handed move simultaneous.
        """
        if isinstance(poses, dict):
            return self._append_poses_multi(poses)
        if not poses:                       # e.g. unfold with nothing left on the stack
            return
        rows = []
        for pose in poses:
            # 5-tuples keep the fixed downward tool; a 6th element is that waypoint's own
            # orientation, which the fold arc uses to turn the wrist over with the flap.
            d, x, y, z, g = pose[:5]
            quat = pose[5] if len(pose) > 5 else (hand or {}).get("down_quat", self.DOWN_QUAT)
            x, y, z = self._clamp_reach(x, y, z, hand=hand)
            rows.append([d, x, y, z, *quat, g])
        arr = np.array(rows, dtype=np.float32)
        _hands = getattr(self, "_hands", None)
        if hand is not None and _hands and hand is not _hands[0]:
            hand["targets"] = np.vstack([hand["targets"], arr[:, 1:]])
            hand["durations"] = np.concatenate([hand["durations"], arr[:, 0]])
            hand["times"] = np.cumsum(hand["durations"])
            return
        self.targets = np.vstack([self.targets, arr[:, 1:]])
        self.transition_duration = np.concatenate([self.transition_duration, arr[:, 0]])
        self.robot_key_poses_time = np.cumsum(self.transition_duration)
        if _hands:                     # hand 0 IS the legacy stream; keep its view pointing at it
            _hands[0]["targets"] = self.targets
            _hands[0]["durations"] = self.transition_duration
            _hands[0]["times"] = self.robot_key_poses_time

    def _append_poses_multi(self, streams):
        """Queue one stream per hand and pad the others so every hand ends at the SAME sim time.

        Without the padding the two streams drift apart: hand 0 would start its next stage while
        hand 1 is still carrying, and a "simultaneous" two-handed move would stop being one.
        """
        _hands = getattr(self, "_hands", None) or []
        # an idle hand that must retreat from the work area gets its 3 s BEFORE a working hand
        # moves in nearby: the v3 probe caught the left arm's retreat crossing the right arm's
        # approach at t=125 s (the only overlap left after the idle-home rule)
        if _os.environ.get("TWIN_IDLE_HOME", "1") == "1" and len(_hands) == 2:
            _retreating = []
            for h in _hands:
                if streams.get(h["name"]):
                    continue
                last = (h["targets"][-1] if h["targets"] is not None and len(h["targets"]) else None)
                if last is not None and float(np.linalg.norm(np.asarray(last[:3], float) - np.asarray(h["home_xyz"], float))) > 3.0:
                    _retreating.append(np.asarray(last[:2], float))
            for h in _hands:
                ps = streams.get(h["name"])
                if ps and _retreating:
                    first = np.asarray(ps[0][1:3], float)
                    if any(float(np.linalg.norm(first - r)) < float(_os.environ.get("TWIN_RETREAT_WAIT_CM", "35")) for r in _retreating):
                        last = (h["targets"][-1] if h["targets"] is not None and len(h["targets"])
                                else np.array([*h["home_xyz"], *h["down_quat"], self.GRIP_OPEN], np.float32))
                        streams[h["name"]] = [(3.0, float(last[0]), float(last[1]), float(last[2]), float(last[-1]),
                                               tuple(float(v) for v in last[3:7]))] + list(ps)
                        print(f"[twin-stock] {h['name']} hand waits 3s for the other arm to clear the work area", flush=True)
        span = max((sum(float(p[0]) for p in ps) for ps in streams.values() if ps), default=0.0)
        for h in _hands:
            ps = list(streams.get(h["name"], []) or [])
            used = sum(float(p[0]) for p in ps)
            if span - used > 1e-3:
                last = (h["targets"][-1] if h["targets"] is not None and len(h["targets"])
                        else np.array([*h["home_xyz"], *h["down_quat"], self.GRIP_OPEN], np.float32))
                if not ps and _os.environ.get("TWIN_IDLE_HOME", "1") == "1":
                    # an IDLE hand goes home, jaws open. Holding station left the arm parked over
                    # the box after its pack while the other arm came through it for the block
                    # (arm-arm probe: links 0.0-0.1 cm apart from t=100 s to the end, v21).
                    home = np.array(h.get("retreat_xyz", h["home_xyz"]), float)
                    far = float(np.linalg.norm(np.asarray(last[:3], float) - home)) > 3.0
                    if far:
                        t_ret = min(3.0, span - used)
                        ps.append((t_ret, float(last[0]), float(last[1]), float(max(last[2], home[2])), self.GRIP_OPEN, tuple(float(v) for v in h["down_quat"])))
                        if span - used - t_ret > 1e-3:
                            ps.append((span - used - t_ret, float(home[0]), float(home[1]), float(home[2]), self.GRIP_OPEN, tuple(float(v) for v in h["down_quat"])))
                    else:
                        ps.append((span - used, float(home[0]), float(home[1]), float(home[2]), self.GRIP_OPEN, tuple(float(v) for v in h["down_quat"])))
                else:
                    # hold station: stay where this hand's last waypoint left it, jaws unchanged
                    ps.append((span - used, float(last[0]), float(last[1]), float(last[2]),
                               float(last[-1]), tuple(float(v) for v in last[3:7])))
            self._append_poses(ps, hand=h)

    def _shirt_folds(self):
        """Three-move garment fold on the flat sheet: left panel in, right panel in, in half.

        Lines come from the sheet's ORIGINAL bbox, captured once, so the panels are true
        thirds rather than thirds of whatever the previous fold produced.
        """
        if getattr(self, "_sheet", None) is None:
            lo, hi = self._cloth_bbox()
            self._sheet = (lo.copy(), hi.copy())
            print(f"[twin-stock] flat sheet {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm", flush=True)
        lo, hi = self._sheet
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        w = hi[0] - lo[0]
        i = getattr(self, "_shirt_step", 0)
        self._shirt_step = i + 1
        if i == 0:                                     # left panel over onto the body
            axis, line = 0, lo[0] + w / 3.0
            grab, drop = (lo[0] + 3.0, cy), (2.0 * line - (lo[0] + 3.0), cy)
        elif i == 1:                                   # right panel over onto the body
            axis, line = 0, hi[0] - w / 3.0
            grab, drop = (hi[0] - 3.0, cy), (2.0 * line - (hi[0] - 3.0), cy)
        else:                                          # hem up to the collar
            axis, line = 1, cy
            grab, drop = (cx, hi[1] - 3.0), (cx, 2.0 * line - (hi[1] - 3.0))
        return self._mirror_fold_poses(axis, float(line), grab, drop)

    def _sleeve_poses(self, name):
        """Pick the actual sleeve tip up and lay it across the torso."""
        ids = self._sleeve_idx()
        if ids is None:
            print("[twin-stock] no sleeve landmarks; skipping", flush=True)
            return []
        q = self._qs()
        tip = q[ids[0] if name == "sleeve_l_in" else ids[1]]
        lo, hi = self._cloth_bbox()
        cx = 0.5 * (lo[0] + hi[0])
        # Fold at the SHOULDER SEAM, not the garment midline. Mirroring the tip across the
        # midline dragged it 39-64cm - the whole width of the shirt - when the sleeve itself is
        # only ~17cm long, so everything between tip and shoulder stretched instead of folding.
        # Fold about the SHOULDER SEAM. The midline is beyond what a sleeve can physically
        # reach: the left sleeve runs from the seam at x=-21 to its tip at x=-35, so it is 14cm
        # long and even folded flat its tip only reaches x=-7 - asking it to reach the midline
        # at x=0 stretches the fabric 7cm or drags the shoulder with it. FULL and HALF are
        # therefore measured against the SLEEVE'S OWN LENGTH, not the garment's width.
        # Seam and midline from the torso particles WHERE THEY ARE NOW. The flat-garment
        # values are absolute x; a start state loaded from fold_state_fit sits 20cm away and
        # folded, so the old seam lay on empty table and the tip was hauled off the shirt
        # (fold_env showed the same as 60cm taffy strands in the validation grid).
        bidx = getattr(self, "_body_idx", None)
        if bidx is not None and len(bidx) > 20:
            bx = q[bidx, 0]
            c0 = float(np.median(bx))
            hw = (c0 - float(np.percentile(bx, 3.0)) if name == "sleeve_l_in"
                  else float(np.percentile(bx, 97.0)) - c0)
        else:
            c0 = getattr(self, "_body_cx", cx)
            hw = getattr(self, "_body_half_w", 0.25 * (hi[0] - lo[0]))
        # RIGHT: fold about the garment MIDLINE and mirror fully - the full sleeve fold.
        # LEFT: fold about its own SHOULDER SEAM and stop there - only this one is halved,
        # because carrying it to the midline is 35cm for a 14cm sleeve (even folded flat its
        # tip only reaches x=-7), which stretches the fabric instead of folding it.
        line = c0 if name == "sleeve_r_in" else c0 - hw
        grab = (float(tip[0]), float(tip[1]))
        # RIGHT sleeve: FULL fold - a complete 180 deg about the seam, so the tip travels twice
        # the sleeve's length and the sleeve lies flat across the body. That is the most a
        # sleeve can fold without stretching. LEFT sleeve: HALF - the tip comes to rest ON the
        # seam, i.e. one sleeve-length of travel, folding it up onto the shoulder.
        if name == "sleeve_r_in":
            drop = (float(2.0 * line - tip[0]), float(tip[1]))
        else:
            drop = (float(line), float(tip[1]))
        # a sleeve folds at most twice its own length: cap the travel so nothing can stretch
        seam = c0 - hw if name == "sleeve_l_in" else c0 + hw
        sleeve_len = max(abs(grab[0] - seam), 4.0)
        if abs(drop[0] - grab[0]) > 2.0 * sleeve_len:
            drop = (grab[0] + float(np.sign(drop[0] - grab[0])) * 2.0 * sleeve_len, drop[1])
        print(f"[twin-stock] {name}: tip ({grab[0]:.0f},{grab[1]:.0f}) -> "
              f"({drop[0]:.0f},{drop[1]:.0f})  shoulder x={line:.0f}  "
              f"travel {abs(drop[0]-grab[0]):.0f}cm", flush=True)
        poses = self._mirror_fold_poses(0, float(line), grab, drop)
        self._fold["two_point"] = False                 # a sleeve is small; one pinch is right
        self._fold["pin_cm"] = _FOLD_PIN
        # Remember WHICH sleeve and where its tip started. Undoing a sleeve fold is just
        # "carry that particle back", and we know it by index - far more reliable than hunting
        # for a free edge, which on a sleeve fold found the garment's own far edge (2296
        # top-layer particles) and folded the shirt further instead of opening it.
        sid = ids[0] if name == "sleeve_l_in" else ids[1]
        if self._fold_stack:
            self._fold_stack[-1]["sleeve_particle"] = int(sid)
            self._fold_stack[-1]["sleeve_set"] = self._sleeve_set[name]
            self._fold_stack[-1]["sleeve_home"] = (float(tip[0]), float(tip[1]))
        return poses

    def _sleeve_idx(self):
        """Particle indices of the two sleeve tips, from the garment's own geometry.

        Every fold so far is GEOMETRIC - a fold line at a fraction of the bbox and a grasp at
        whatever fabric sits on that edge. Nothing targets the sleeves, which is why asking for
        a sleeve fold picked up whatever happened to be there. fold_env has tracked these
        landmarks all along (it measures sleeve overhang with them); the twin never used them.
        Indices are fixed, so they still identify the sleeves after any number of folds.
        """
        if getattr(self, "_sleeve_ids", None) is not None:
            return self._sleeve_ids
        q = getattr(self, "_flat_q", None)
        if q is None:
            return None
        y = q[:, 1]
        edges = np.linspace(y.min(), y.max(), 11)
        widths = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (y >= a) & (y < b)
            widths.append(float(np.ptp(q[m, 0])) if m.sum() else 0.0)
        widths = np.asarray(widths)
        k = int(widths.argmax())                       # widest slice = the sleeve band
        band = np.where((y >= edges[k]) & (y < edges[k + 1]))[0]
        self._sleeve_ids = (int(band[q[band, 0].argmin()]), int(band[q[band, 0].argmax()]))
        # Half-width of the TORSO: the body is the narrow part, the sleeves are what sticks out
        # past it. The shoulder seam sits at this distance from the centre, and that - not the
        # garment midline - is where a sleeve folds.
        body_w = float(np.median([w for w in widths if w < 0.75 * widths.max()]))
        self._body_half_w = 0.5 * body_w
        self._body_cx = float(np.median(q[:, 0]))      # centre of the FLAT garment
        # torso particles by INDEX, so the seam/midline can be re-measured on the current
        # state at fold time (see _sleeve_poses) instead of trusting the flat-garment numbers
        self._body_idx = np.where(np.abs(q[:, 0] - self._body_cx) <= self._body_half_w)[0]
        # The two sleeves as PARTICLE SETS, taken on the flat garment: everything outside the
        # torso span. Unfolding must lift only the piece that was folded - a pinch at the
        # sleeve tip catches the torso lying underneath as well (measured 360-420 particles
        # against 57-126 for the fold) and drags the whole bundle instead of peeling.
        self._sleeve_set = {
            "sleeve_l_in": np.where(q[:, 0] < self._body_cx - self._body_half_w)[0],
            "sleeve_r_in": np.where(q[:, 0] > self._body_cx + self._body_half_w)[0],
        }
        print(f"[twin-stock] sleeves identified: particles {self._sleeve_ids}, "
              f"torso half-width {self._body_half_w:.0f}cm", flush=True)
        return self._sleeve_ids

    def _plan_poses(self):
        """Execute the next action of TWIN_PLAN with the arm.

        The fold line comes from the CURRENT bbox, exactly as `fold_env.step` computes it, so
        the geometry matches the rollout the search scored. The half on `side` is the one that
        moves; the arm grabs it 3cm inside its outer edge and carries that point to its mirror.
        """
        i = getattr(self, "_plan_step", 0)
        self._plan_step = i + 1
        if getattr(self, "_flat_q", None) is None:
            self._flat_q = self._qs().copy()   # the garment laid flat
        name = _PLAN[i]
        if name in ("sleeve_l_in", "sleeve_r_in"):
            return self._sleeve_poses(name)
        axis, frac, side, grab_at = _PLAN_ACTIONS[name]
        lo, hi = self._cloth_bbox()
        line = lo[axis] + (hi[axis] - lo[axis]) * frac
        other = 1 - axis
        # The edge AT THE SLICE WE GRAB, not the garment's global bounding box - a shirt
        # narrows toward the collar, so a high grab computed from the global box reaches past
        # the shoulder into empty table (measured: 8 particles held).
        _gy = lo[other] + (hi[other] - lo[other]) * grab_at
        _q = self._qs()
        _band = np.abs(_q[:, other] - _gy) < 4.0
        if int(_band.sum()) > 20:
            # PERCENTILE, not max/min: a single stray particle flung out by a contact defines
            # the edge otherwise, and the grasp 3cm inside it lands in empty table. That is
            # what dropped the pinch from 74-137 particles to 18-57. Same 3/97 the bbox uses.
            _col = _q[_band, axis]
            edge = float(np.percentile(_col, 97.0 if side > 0 else 3.0))
        else:
            edge = hi[axis] if side > 0 else lo[axis]
        grab = [0.0, 0.0]
        grab[axis] = float(edge - 3.0 * side)          # 3cm in from the edge, as _shirt_folds does
        grab[other] = float(_gy)
        # Snap onto real fabric. The grab point is computed from edge GEOMETRY and the arm then
        # descends there blind - if it lands a centimetre off the cloth the jaws take nothing,
        # which is why the same plan gave 21x27 one run and 25x33 the next (fold 2 held 184
        # particles vs 71). fold_env has always snapped to the nearest particle; the twin did
        # not, so rollout and robot disagreed for a reason that had nothing to do with physics.
        _d2 = (_q[:, 0] - grab[0]) ** 2 + (_q[:, 1] - grab[1]) ** 2
        _near = _q[int(np.argmin(_d2))]
        if float(np.hypot(_near[0] - grab[0], _near[1] - grab[1])) > 0.5:
            print(f"[twin-stock] grab snapped {np.hypot(_near[0]-grab[0], _near[1]-grab[1]):.1f}cm "
                  f"onto fabric", flush=True)
        grab[0], grab[1] = float(_near[0]), float(_near[1])
        drop = list(grab)
        drop[axis] = float(2.0 * line - grab[axis])    # mirror across the fold line
        print(f"[twin-stock] plan {i + 1}/{len(_PLAN)} {name}: axis={axis} line={line:.1f} "
              f"grab=({grab[0]:.0f},{grab[1]:.0f}) drop=({drop[0]:.0f},{drop[1]:.0f})", flush=True)
        self._last_span_cm = float(np.clip(0.35 * (hi[other] - lo[other]), 6.0, 22.0))
        poses = self._mirror_fold_poses(axis, float(line), tuple(grab), tuple(drop))
        # A length fold carries a whole half - hold a wide band of its edge so it comes over
        # flat instead of being dragged from one spot.
        self._fold["two_point"] = (axis == 1)   # sheet folds need more than one hold
        # How wide the fold edge actually is right now, so the holds span it instead of
        # gripping a fixed patch in the middle of a much wider sheet.
        self._fold["span_cm"] = float(np.clip(0.35 * (hi[other] - lo[other]), 6.0, 22.0))
        self._fold["pin_cm"] = _FOLD_PIN   # wide band held 1182 particles through
                                           # every layer; lifting that gathers the
                                           # garment into a column, not a fold
        return poses

    def _fold_poses(self):
        """Halve the bundle across its longer axis."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        sx, sy = hi[0] - lo[0], hi[1] - lo[1]
        if sy >= sx:                                   # carry the +y edge over to -y
            grab, drop = (cx, hi[1] - 0.03 * sy), (cx, lo[1] + 0.18 * sy)
            axis, line = 1, cy
        else:
            grab, drop = (hi[0] - 0.03 * sx, cy), (lo[0] + 0.18 * sx, cy)
            axis, line = 0, cx
        return self._mirror_fold_poses(axis, float(line), grab, drop)

    def _carry_slow_for(self, grab, axis=None):
        """How much to stretch the carry, from the layers sitting under the grasp point.

        Same measurement fold_env scores as `grip`: fabric within the pinch radius near table
        height. ~80 particles is a single layer at an edge (drag it at stock speed); 400+ is a
        multi-layer bundle, where outrunning the self-collision solve sends the cloth to NaN.
        """
        q = self._qs()
        if not np.isfinite(q).all():
            return _CARRY_SLOW
        floor = float(np.percentile(q[:, 2], 2.0))
        # Count EVERY hold. A length fold is carried by three patches spread along the edge,
        # so sampling only the middle one underestimated the load ~3x (measured 190 when the
        # fold actually held 1273) and picked x1.9 instead of x4.3 - and the cloth diverged.
        span = float(getattr(self, '_last_span_cm', _FOLD_SPAN))
        offs = np.linspace(-span, span, 3) if axis == 1 else (0.0,)
        other = 1 - (axis if axis is not None else 0)
        n = 0
        for o in offs:
            gx = grab[0] + (o if other == 0 else 0.0)
            gy = grab[1] + (o if other == 1 else 0.0)
            d2 = (q[:, 0] - gx) ** 2 + (q[:, 1] - gy) ** 2
            # probe radius independent of the pinch size: a 1.5 cm pinch counted 54 particles
            # under a 4-sheet stack and carried it at x1.0 - the stack tumbled into a wad (v17)
            _pr = float(_os.environ.get("TWIN_CARRY_PROBE_CM", "4"))
            n += int(((d2 < _pr ** 2) & (q[:, 2] < floor + _pr)).sum())
        # Length folds used to get double the ceiling because they dragged a SHEET through a
        # multi-layer stack and diverged. They no longer drag it: the arc lifts the flap over
        # its crease, and a single pinch carries far less fabric into the pile than the old
        # three-hold bar did. Same cap for both now - the doubled one is what made fold 3 crawl.
        cap = _CARRY_SLOW * float(_os.environ.get('TWIN_CARRY_LEN_MULT', '1.0') if axis == 1 else 1.0)
        c = float(np.clip(1.0 + (n - 80) / _CARRY_SLOPE, 1.0, cap))
        print(f"[twin-stock] carry x{c:.1f} ({n} particles under the grasp)", flush=True)
        return c

    def _mirror_fold_poses(self, axis, line, grab, drop):
        """Gripper waypoints for mirroring the piece beyond `line` back across it."""
        top = self.TABLE_TOP
        # the half beyond the fold line is what gets carried over onto the other half
        self._fold = {"axis": axis, "line": float(line), "grab": np.array(grab, float),
                      "drop": np.array(drop, float), "start": None, "idx": None}
        # keep the fold on a stack so it can be undone: mirroring a half across a line is its
        # own inverse, so replaying the same line with grab/drop swapped unfolds it
        self._fold_stack = getattr(self, "_fold_stack", [])
        self._fold_stack.append({"axis": axis, "line": float(line),
                                 "grab": np.array(grab, float), "drop": np.array(drop, float)})
        # Follow the stock fold's motion: pinch at table level, then DRAG low (~11cm up).
        # A high lifting arc just slips the fabric out of the fingers.
        _lerp = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        # The CARRY segments are stretched by _CARRY_SLOW. Dragging the pinned patch into a
        # multi-layer stack faster than VBD can resolve the self-collision it creates is what
        # sends the cloth to NaN - shown on the flat sheet control too, so it is the fold speed
        # and not the garment. fold_env needed 2.0s -> 5.0s; the arm needs the same treatment.
        c = self._carry_slow_for(grab, axis)
        # LIFT the edge over the fold line - do not drag it across the table. The stock path
        # stayed at table+4/+8cm, which for a ~27cm-deep flap slides the grasp sideways while
        # friction and weight pin the rest down. The kinematic grip cannot slip, so instead of
        # the flap coming over, the material between grip and cloth STRETCHES LIKE RUBBER.
        # Arc apex scales with the fold distance: a fold rotates the flap about its crease.
        travel = float(np.hypot(drop[0] - grab[0], drop[1] - grab[1]))
        # Modest: enough to peel the edge off the table, not enough to hoist the
        # bundle. 0.28/0.45 of a 50cm travel lifted to 22cm and wadded the shirt.
        rise = float(np.clip(0.10 * travel, 4.0, 8.0))
        apex = float(np.clip(0.16 * travel, 6.0, 12.0))
        # Pinch just under the cloth's TOP at the grab point, not at table level. On a flat
        # sheet the two coincide; on a 4-layer bundle (2.5cm thick) driving the fingertips to
        # the table punches them through the stack and VBD diverges (every unfold and the
        # thick-stack folds of 2026-09-03).
        _qz = self._qs()
        _near = np.hypot(_qz[:, 0] - grab[0], _qz[:, 1] - grab[1]) < _FOLD_PIN
        _local_top = float(np.percentile(_qz[_near, 2], 90.0)) if _near.sum() > 10 else top
        z_pinch = max(top - 0.3 - _GRASP_DZ, _local_top - 0.8)
        if z_pinch > top - 0.3 - _GRASP_DZ + 0.5:
            print(f"[twin-stock] grasp height {z_pinch - top:+.1f}cm vs table: cloth is "
                  f"{_local_top - top:.1f}cm thick at the grab", flush=True)
        approach = [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], z_pinch, self.GRIP_OPEN),  # press in so the jaws straddle it
            (2.5, grab[0], grab[1], z_pinch, self.GRIP_CLOSE),
        ]
        release = [
            (1.5 * c, drop[0], drop[1], top + 2.0, self.GRIP_OPEN),
            (1.5, drop[0], drop[1], top + 20.0, self.GRIP_OPEN),
        ]
        if _FOLD_ARC:
            # Rotate the flap about its crease. R is the flap's own length, so the pinch stays
            # exactly that far from the fold line for the whole carry: the fabric is neither
            # stretched (tip too far) nor slack (tip too close), which is the only path a hand
            # can actually take. Equal angle steps give constant tip speed.
            R = abs(float(grab[axis]) - float(line))
            sgn = 1.0 if float(drop[axis]) >= float(line) else -1.0
            lift = min(1.0, _FOLD_ARC_MAX / R) if R > 1e-6 else 0.0
            n = 16
            # No pi/2 stretch for the longer path. It was there to hold the tip speed of the
            # straight drag, but it multiplied an already-large carry_slow: fold 3 grabs a
            # four-layer bundle (c = 8.1) and took 89s of sim on its own. Tip speed was never
            # what diverged the cloth - dragging the patch INTO the stack was, and the arc
            # lifts it over the crease instead.
            seg = 7.0 * c / n
            arc = []
            for k in range(1, n + 1):
                th = np.pi * k / n
                px, py = float(grab[0]), float(grab[1])
                pos = [px, py]
                pos[axis] = float(line) - R * float(np.cos(th)) * sgn
                z = top + R * float(np.sin(th)) * lift
                if k == n:
                    # lay it down: 2cm above the table for a thin flap; a 2-layer flap laid
                    # at 2cm squeezes its own layers under the pinned patch and diverged
                    # at landing (unfold of a bundle, 2026-09-03) - leave it the thickness
                    z = top + 2.0 + max(0.0, _local_top - top - 0.8)
                # Turn the wrist about the CREASE by the same angle the flap has swept, so the
                # grabbed edge rolls over instead of being carried face-up the whole way. The
                # crease runs along the other axis, so that is the rotation axis; the sign is
                # whichever sends the fabric UP over the line rather than down through it.
                u = np.zeros(3)
                if axis == 1:
                    u[0] = -sgn
                else:
                    u[1] = sgn
                wth = min(th, np.deg2rad(_WRIST_DEG))
                qr = np.array([*(u * np.sin(wth / 2.0)), np.cos(wth / 2.0)])
                arc.append((seg, pos[0], pos[1], z, self.GRIP_CLOSE,
                            tuple(self._quat_mul(qr, np.array(self.DOWN_QUAT)))))
            print(f"[twin-stock] arc fold: flap {R:.0f}cm, apex {R * lift:.0f}cm over the crease",
                  flush=True)
            return approach + arc + release
        return approach + [
            (2.0 * c, *_lerp(0.2), top + rise, self.GRIP_CLOSE),   # peel the edge UP off the table
            (2.5 * c, *_lerp(0.5), top + apex, self.GRIP_CLOSE),   # over the crease, not across it
            (2.5 * c, *_lerp(0.8), top + rise, self.GRIP_CLOSE),
            (2.0 * c, drop[0], drop[1], top + 2.0, self.GRIP_CLOSE),   # lay it down flat
        ] + release

    def _free_edge_grab(self, axis, line, drop, only=None):
        """Find the free edge of the TOP layer, on the side the flap was folded onto.

        Unfolding needs something folding does not: knowing which layer is on top and where
        its loose edge lies. Extent, landmarks and grip count all say nothing about layering,
        so this reads it off the particles directly - upper half of the bundle by height, then
        the extreme along the fold axis on the folded-onto side.
        """
        q = self._qs()
        if not np.isfinite(q).all():
            return None
        zlo = float(np.percentile(q[:, 2], 5.0))
        zhi = float(np.percentile(q[:, 2], 95.0))
        if only is not None and len(only) > 40:
            # the FLAP's own loose edge: the layers under it can stick out past it when the
            # fold landed short (5cm on the 2026-09-03 bundle), and the pinch found 8 particles
            q = q[np.asarray(only, int)]
        upper = q[q[:, 2] > (zlo + zhi) / 2.0]         # top layers only
        if len(upper) < 40:
            return None
        side = 1.0 if float(drop[axis]) >= line else -1.0
        sel = upper[(upper[:, axis] - line) * side > 0.0]
        if len(sel) < 20:
            sel = upper
        p = sel[int(np.argmax(sel[:, axis] * side))]   # furthest out = the loose edge
        g = [float(p[0]), float(p[1]), float(p[2])]
        g[axis] -= 3.0 * side                          # 3cm inside the edge, as folds do
        print(f"[twin-stock] free edge at ({g[0]:.0f},{g[1]:.0f}) z={g[2]:.1f} "
              f"({g[2] - self.TABLE_TOP:+.1f} vs table) from {len(sel)} top-layer particles",
              flush=True)
        return g

    def _unfold_poses(self):
        """Undo the most recent fold: carry the folded half back across its own fold line."""
        stack = getattr(self, "_fold_stack", [])
        if not stack:
            print("[twin-stock] nothing left to unfold", flush=True)
            return []
        rec = stack.pop()
        # Where the edge ACTUALLY is, not where the fold was told to put it. Replaying the
        # stored drop point is open-loop: it was exact under the mirror model, but a pinched
        # fold lands short, so the gripper reached ~10cm past the fabric and closed on empty
        # table (measured: 28 and 30 particles, both misses).
        self._fold_only = None
        sid = rec.get("sleeve_particle")
        if sid is not None:
            # A sleeve: grab the tip particle where it now lies and take it home.
            qs = self._qs()
            found = [float(qs[sid, 0]), float(qs[sid, 1]), float(qs[sid, 2])]
            grab = np.array(found[:2], float)
            drop = np.array(rec["sleeve_home"], float)
            self._fold_only = rec.get("sleeve_set")     # hold sleeve fabric ONLY
            print(f"[twin-stock] unfold sleeve particle {sid}: "
                  f"({grab[0]:.0f},{grab[1]:.0f}) -> ({drop[0]:.0f},{drop[1]:.0f})  "
                  f"restricted to {len(self._fold_only)} sleeve particles", flush=True)
        else:
            found = self._free_edge_grab(rec["axis"], rec["line"], rec["drop"],
                                         only=rec.get("flap_set"))
            grab = np.array(found[:2], float) if found is not None else np.array(rec["drop"], float)
            drop = rec["grab"]                         # carry it back across its own fold line
            if rec.get("flap_set") is not None:
                # peel the FLAP only: a pinch on a 4-layer bundle otherwise takes the whole
                # stack, drags it, and the solver diverges (unfold of a fitted state, 2026-09-03)
                self._fold_only = rec["flap_set"]
                print(f"[twin-stock] unfold restricted to the {len(rec['flap_set'])} flap "
                      f"particles of that fold", flush=True)
        top = self.TABLE_TOP
        # Peeling needs the jaws to close on the TOP layer only. Folding grasps at table level
        # because it is picking up a single lying sheet; a folded bundle has layers stacked
        # under the edge, and reaching table level would pinch all of them and drag the whole
        # bundle instead of opening it. Close just under the edge's own height.
        z_grab = (float(found[2]) - 0.3) if found is not None else (top - 0.3 - _GRASP_DZ)
        z_grab = max(z_grab, top - 0.3 - _GRASP_DZ)
        self._fold = {"only": getattr(self, "_fold_only", None),
                      "axis": rec["axis"], "line": rec["line"],
                      "grab": np.array(grab, float), "drop": np.array(drop, float),
                      "start": None, "idx": rec.get("idx")}
        axis_u = int(rec["axis"])
        line_u = float(rec["line"])
        if rec.get("flap_set") is not None and found is not None:
            # The crease AS IT IS, not as the fold intended it: folds land short, so the
            # recorded line sat 7cm inside the real crease on the 2026-09-03 bundle and the
            # swing refolded the flap onto itself (22x32 -> 20x21). The crease is the flap's
            # far edge, opposite its loose edge; the flap opens to its mirror across it.
            qf = self._qs()[np.asarray(rec["flap_set"], int)]
            side = 1.0 if float(grab[axis_u]) >= line_u else -1.0    # loose edge is on this side
            crease = float(np.percentile(qf[:, axis_u], 3.0 if side > 0 else 97.0))
            line_u = crease
            drop = np.array(grab, float)
            drop[axis_u] = 2.0 * crease - float(grab[axis_u])
            print(f"[twin-stock] unfold about the actual crease {['x','y'][axis_u]}={crease:.0f} "
                  f"(recorded line {rec['line']:.0f}): loose edge ({grab[0]:.0f},{grab[1]:.0f}) "
                  f"-> ({drop[0]:.0f},{drop[1]:.0f})", flush=True)
        if rec.get("flap_set") is not None or sid is None:
            # Swing the flap BACK over its crease on the same semicircle a fold uses. The old
            # low drag (6-10cm up) pulled the pinched edge 27cm while the flap's centre moved
            # 5cm - the flap slid on the stack under friction instead of coming over (unfold
            # of a fitted bundle, 2026-09-03). _mirror_fold_poses records itself on the fold
            # stack so it can be undone; an unfold must not leave that record behind.
            n_before = len(getattr(self, "_fold_stack", []))
            poses = self._mirror_fold_poses(axis_u, line_u, grab, drop)
            if len(getattr(self, "_fold_stack", [])) > n_before:
                self._fold_stack.pop()
            if self._fold is not None:
                self._fold["only"] = getattr(self, "_fold_only", None)
            return poses
        _lerp = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        c = self._carry_slow_for(grab)
        return [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], z_grab, self.GRIP_OPEN),
            (2.5, grab[0], grab[1], z_grab, self.GRIP_CLOSE),
            (2.0 * c, *_lerp(0.15), top + 6.0, self.GRIP_CLOSE),   # lift clear so the layer peels
            (2.5 * c, *_lerp(0.5), top + 10.0, self.GRIP_CLOSE),
            (3.0 * c, drop[0], drop[1], top + 6.0, self.GRIP_CLOSE),
            (1.5 * c, drop[0], drop[1], top + 1.0, self.GRIP_CLOSE),  # lay it down, do not drop
            (1.0, drop[0], drop[1], top + 1.0, self.GRIP_OPEN),
            (1.5, drop[0], drop[1], top + 20.0, self.GRIP_OPEN),
        ]

    def _pack_poses(self):
        """Pinch the folded bundle and drop it into the real box."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        top = self.TABLE_TOP
        bx, by = getattr(self, "box_center_cm", (cx, cy))
        rim = getattr(self, "box_rim_z_cm", top + 10.0)
        floor = getattr(self, "box_floor_z_cm", top + 1.0)
        # Grip the bundle at its centre so it hangs balanced, lift, carry over the rim and
        # release into the box. The attachment grasp in _update_grasp is what holds it.
        return [
            (2.0, cx, cy, top + 11.0, self.GRIP_OPEN),
            (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_OPEN),
            (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_HARD),
            (3.0, cx, cy, top + 20.0, self.GRIP_HARD),
            (4.0, bx, by, rim + 18.0, self.GRIP_HARD),
            (3.0, bx, by, floor + 7.0, self.GRIP_HARD),   # reach down INTO the box
            (2.0, bx, by, floor + 7.0, self.GRIP_OPEN),   # short drop, no punch-through
            (2.0, bx, by, rim + 26.0, self.GRIP_OPEN),
        ]

    def _label_move_arc(self, ga, gb, pa, pb, hand=None, carry=None):
        """Pinch the fabric at label A (particle pa) and carry it along a semicircle onto label B's
        CURRENT position (particle pb). No fold line is prescribed; the crease forms where the fabric
        puts it. Same pinch/arc/release choreography as the named folds, but the travel is a free
        2D vector, so diagonal moves work."""
        q = self._qs(); top = self.TABLE_TOP
        g = q[int(pa)].astype(float); d = q[int(pb)].astype(float)
        # TWIN_GRAB_INSET_CM: grab a few cm IN from the label, toward the garment's centre across
        # the travel direction. A band centred exactly on a corner hangs half off the cloth, and
        # which half lands decides the grip (462 vs 369 particles on the two hands of one fold).
        _ins = float(_os.environ.get("TWIN_GRAB_INSET_CM", "0"))
        if _ins > 0.0:
            _t = d[:2] - g[:2]; _tn = float(np.linalg.norm(_t))
            _perp = np.array([-_t[1], _t[0]]) / _tn if _tn > 1e-6 else np.zeros(2)
            _toc = np.median(q[:, :2], 0) - g[:2]
            _side = np.sign(float(_perp @ _toc)) or 1.0
            g[:2] += _perp * _side * _ins
            d[:2] += _perp * _side * _ins
        near = np.hypot(q[:, 0] - g[0], q[:, 1] - g[1]) < _FOLD_PIN
        local_top = float(np.percentile(q[near, 2], 90.0)) if near.sum() > 10 else top
        z_pinch = max(top - 0.3 - _GRASP_DZ, local_top - 0.8)
        travel = float(np.hypot(d[0] - g[0], d[1] - g[1]))
        # OVERSHOOT: the pinch carries the edge, the fabric behind it lags, and a flap released
        # exactly over its target settles SHORT of it - which is what "not fully folded" looks
        # like. Carry a few cm past the target and let it fall back onto the line.
        _ov = float(_os.environ.get("TWIN_FOLD_OVERSHOOT_CM", "3"))
        if travel > 1e-6 and _ov:
            d = d.copy()
            d[:2] += (d[:2] - g[:2]) / travel * _ov
        # APEX: half the travel throws the fabric into the air and it gathers on the way down.
        # A person folding a towel lifts the edge just enough to clear what is under it.
        R = float(_os.environ.get("TWIN_FOLD_ARC_FRAC", "0.35")) * travel
        apex = min(R, float(_os.environ.get("TWIN_FOLD_ARC_MAX_CM", "26")))
        ax = 0 if abs(d[0] - g[0]) >= abs(d[1] - g[1]) else 1
        fold = {"pa": int(pa), "pb": int(pb), "label": f"{ga}->{gb}", "target_xy": d[:2].copy(),
                "axis": ax, "line": 0.5 * float(g[ax] + d[ax]), "grab": g[:2].copy(), "drop": d[:2].copy(),
                "start": None, "idx": None, "two_point": _os.environ.get("TWIN_FOLD_TWO_POINT", "0") == "1", "hand": hand,
                "pin_cm": float(_os.environ.get("TWIN_FOLD_BAND_CM", _FOLD_PIN)),
                "span_cm": float(_os.environ.get("TWIN_FOLD_SPAN", "14"))}
        self._fold_stack = getattr(self, "_fold_stack", [])
        self._fold_stack.append({"axis": ax, "line": fold["line"], "grab": g[:2].copy(), "drop": d[:2].copy()})
        c = carry if carry is not None else (
            self._carry_slow_for((float(g[0]), float(g[1])), axis=ax) if hasattr(self, "_carry_slow_for") else 1.0)
        n = 10; seg = 7.0 * c / n
        poses = [(2.0, g[0], g[1], top + 11.0, self.GRIP_OPEN),
                 (2.0, g[0], g[1], z_pinch, self.GRIP_OPEN),
                 (2.5, g[0], g[1], z_pinch, self.GRIP_CLOSE)]
        for k in range(1, n + 1):
            th = np.pi * k / n; t = 0.5 * (1.0 - np.cos(th))
            x, y = g[0] + (d[0] - g[0]) * t, g[1] + (d[1] - g[1]) * t
            z = top + apex * np.sin(th) + max(0.0, local_top - top - 0.8)
            if k == n:
                z = top + 2.0 + max(0.0, local_top - top - 0.8)
            poses.append((seg, float(x), float(y), float(z), self.GRIP_CLOSE))
        _back = float(_os.environ.get("TWIN_RELEASE_BACK_CM", "5"))
        _bx, _by = (d[0] - g[0]) / max(travel, 1e-6) * _back, (d[1] - g[1]) / max(travel, 1e-6) * _back
        lift = max(0.0, local_top - top - 0.8)        # the arc's own clearance over a thick pile
        _rz = top + float(_os.environ.get("TWIN_RELEASE_Z_CM", "2")) + lift
        _up = float(_os.environ.get("TWIN_RELEASE_UP_CM", "0"))
        poses += [(1.5, d[0], d[1], _rz, self.GRIP_OPEN)]
        if _up > 0:
            poses += [(2.0, d[0], d[1], _rz + _up, self.GRIP_OPEN)]          # slow, straight up
        poses += [(1.5, d[0] - _bx, d[1] - _by, _rz + _up, self.GRIP_OPEN),  # then back out
                  (2.0, d[0] - _bx, d[1] - _by, top + 14.0, self.GRIP_OPEN)]
        print(f"[twin-stock] label move {ga} -> {gb}"
              + (f" [{hand['name']} hand]" if hand else "")
              + f": ({g[0]:.0f},{g[1]:.0f}) -> ({d[0]:.0f},{d[1]:.0f}), "
                f"travel {travel:.0f}cm, apex {apex:.0f}cm, carry x{c:.1f}", flush=True)
        return poses, fold

    def _one_hand(self):
        _n = _os.environ.get("TWIN_ONE_HAND", "")
        return next((h for h in (self._hands or []) if h["name"] == _n), None)

    def _serial_second(self, mv, hand):
        """The second pinch of a serialised step, grounded on the cloth as it lies now."""
        a, b, pa, pb = mv
        poses, fold = self._label_move_arc(a, b, pa, pb, hand=hand)
        self._packing_dual = None; self._rigid_hold = None
        self._fold = {"hands": [fold], "axis": fold["axis"], "line": fold["line"], "grab": fold["grab"], "drop": fold["drop"],
                      "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
        return {hand["name"]: poses}

    def _label_move_one(self, mv):
        """One label move on the named hand (TWIN_ONE_HAND), the other hand idle at home."""
        a, b, pa, pb = mv
        h = self._one_hand()
        poses, fold = self._label_move_arc(a, b, pa, pb, hand=h)
        self._fold = {"hands": [fold], "axis": fold["axis"], "line": fold["line"],
                      "grab": fold["grab"], "drop": fold["drop"],
                      "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
        return {h["name"]: poses}

    @staticmethod
    def _quat_rot(q, v):
        """rotate points v (N,3) by quaternion q (x,y,z,w)"""
        x, y, z, w = [float(c) for c in q]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return v @ R.T

    def _gripper_envelope(self, hand):
        """World-axis extents of the gripper's links around the tip, at the hand's CURRENT
        orientation: {'-x': cm the gripper reaches beyond the tip toward -x, '+x', '-y', '+y'}.
        Built from the model's own link meshes (every body below the wrist in the joint tree),
        so a different gripper or jaw opening changes the numbers, not the code."""
        m = self.model
        jp, jc = m.joint_parent.numpy(), m.joint_child.numpy()
        ee = int(hand.get("ee_body", self.endeffector_id))
        kids = {ee}
        grew = True
        while grew:
            grew = False
            for pa, ch in zip(jp, jc):
                if int(pa) in kids and int(ch) not in kids:
                    kids.add(int(ch)); grew = True
        kids.discard(ee)                                   # the wrist link itself is not the gripper
        bq = self.state_0.body_q.numpy()
        sb, st, ss, sx = m.shape_body.numpy(), m.shape_type.numpy(), m.shape_scale.numpy(), m.shape_transform.numpy()
        pts = []
        for si in range(len(sb)):
            b = int(sb[si])
            if b not in kids or int(st[si]) != int(newton.GeoType.MESH):
                continue
            src = m.shape_source[si]
            if src is None or not hasattr(src, "vertices"):
                continue
            v = np.asarray(src.vertices, float) * np.asarray(ss[si], float)[None, :]
            v = v[::max(1, len(v) // 300)]
            v = self._quat_rot(sx[si][3:7], v) + np.asarray(sx[si][:3], float)        # body frame
            v = self._quat_rot(bq[b][3:7], v) + np.asarray(bq[b][:3], float)           # world
            pts.append(v)
        if not pts:
            return None
        P = np.concatenate(pts); tip, _ = self._ee_tip(hand)
        return {"-x": float(tip[0] - P[:, 0].min()), "+x": float(P[:, 0].max() - tip[0]),
                "-y": float(tip[1] - P[:, 1].min()), "+y": float(P[:, 1].max() - tip[1]),
                "-z": float(tip[2] - P[:, 2].min()), "n_bodies": len(kids)}

    def _pack_poses_dual(self):
        """BIMANUAL pack: both hands pinch the folded bundle's two sides, lift together, translate
        to the box centre as one rigid pair, lower in, release. Each hand carries the half of the
        bundle nearer its tip (kinematic pin, like the single-hand pack), so a translation that is
        identical for both hands cannot stretch the bundle."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        top = self.TABLE_TOP
        bx, by = getattr(self, "box_center_cm", (cx, cy))
        bx += float(getattr(self, "_task_args", {}).get("dx", 0.0)); by += float(getattr(self, "_task_args", {}).get("dy", 0.0))
        rim = getattr(self, "box_rim_z_cm", top + 10.0)
        floor = getattr(self, "box_floor_z_cm", top + 1.0)
        self._lands_on_cloth = False
        # SET DOWN ON WHAT IS ALREADY THERE. The release height was measured from the box floor
        # even when the plan stacks this garment on another: the hands pushed the new bundle INTO
        # the one below and the stack was crushed (towelA 15x6 -> 4x1 cm under the shirt, s21),
        # which is the dense wad the self-collision step then stalls on. Measure the top of the
        # other garments under the target and release above that.
        try:
            _all = self.state_0.particle_q.numpy(); _sl_ = self._gsl
            _mask = np.ones(len(_all), bool); _mask[_sl_] = False
            _o = _all[_mask]
            _hw, _hd = max(6.0, (hi[0] - lo[0]) / 2.0), max(6.0, (hi[1] - lo[1]) / 2.0)
            _u = _o[(np.abs(_o[:, 0] - bx) <= _hw) & (np.abs(_o[:, 1] - by) <= _hd) & (_o[:, 2] > floor - 1.0) & (_o[:, 2] < rim + 2.0)]
            if len(_u) > 20:
                _stack = float(np.percentile(_u[:, 2], 98.0))
                if _stack > floor + 0.5:
                    print(f"[twin-stock] pack: {_stack - floor:.1f}cm of cloth already under the target - setting down on top of it", flush=True)
                    floor = _stack
                    self._lands_on_cloth = True
        except Exception as _e:
            print(f"[twin-stock] pack: stack-height check skipped ({_e})", flush=True)
        # BOTH HANDS first: the bundle is carried between two grips, not pinned rigid to one tip
        _env = self._gripper_envelope(self._hands[0])
        _below = (_env or {}).get("-z", 4.0)
        _streams, _why = self._two_hand_carry(bx, by, floor + max(1.0, _below + 0.3), "pack",
                                              slow=float(_os.environ.get("TWIN_PACK_SLOW", "1.0")))
        if _streams is not None:
            return _streams
        print(f"[twin-stock] pack: bundle at ({_why[0]:.0f},{_why[1]:.0f}) "
              + (f"offers only {-_why[2]:.0f}cm between two grips" if _why[2] < 0 else f"needs {_why[2]:.0f}cm of reach") + " - one hand", flush=True)
        hands = self._hands
        left = next(h for h in hands if h["name"] == "left"); right = next(h for h in hands if h["name"] == "right")
        inset = 2.0
        streams, pins = {}, {}
        one = self._one_hand()
        if one is None:
            # a bundle set aside at the edge of the shared workspace: the far arm would lag
            # (measured 16 cm at 76 cm reach), so the near arm packs it alone
            _reach = float(_os.environ.get("TWIN_REACH_ONE_HAND_CM", "68"))
            _d = {h["name"]: float(np.hypot(cx - h["base_cm"][0], cy - h["base_cm"][1])) for h in hands}
            if max(_d.values()) > _reach:
                one = min(hands, key=lambda H: _d[H["name"]])
                print(f"[twin-stock] pack: bundle at ({cx:.0f},{cy:.0f}) is {max(_d.values()):.0f}cm from the far arm -> {one['name']} hand alone", flush=True)
        if one is None and _os.environ.get("TWIN_LEGACY_TWO_HAND", "0") != "1":
            # the two-hand carry declined (bundle too short): ONE hand takes it. The older fallback
            # here split the bundle between both hands and pinned each half rigidly - that is the
            # stiff, board-like carry of the hand towel (z spread 2.6 cm in the air, exec_s28).
            one = min(hands, key=lambda H: float(np.hypot(cx - H["base_cm"][0], cy - H["base_cm"][1])))
        # (a bundle under 16 cm - the folded shirt - is still held whole: hung by an edge it came
        #  apart, 23 cm of it dangling, and landed as a wad that stalled the cloth step, exec_s29)
        _tiny = max(hi[0] - lo[0], hi[1] - lo[1]) < float(_os.environ.get("TWIN_CARRY_WHOLE_BELOW_CM", "16"))
        if one is not None and not _tiny and _os.environ.get("TWIN_ONE_HAND_CARRY", "hang") == "hang":
            # HANG AND LAY. The hand pinches the folded bundle along ONE EDGE, through all its layers,
            # and lifts: the bundle hangs from the fingers like a towel held by its edge - nothing
            # else is supported. It is laid down the way a person lays a hanging cloth with one
            # hand: the free bottom edge touches first and the hand sweeps sideways as it lowers,
            # so the cloth unrolls flat onto the surface. (Held by a band across its middle the
            # bundle draped as well, but an inverted U cannot be set down flat: 23x17 -> 12x12 cm.)
            sx, sy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
            hx_ = 1 if sx >= sy else 0                      # the HANG axis = the shorter side; a long edge is held
            L = sy if hx_ == 1 else sx
            jaw = float(_os.environ.get("TWIN_CARRY_JAW_CM", "9")) / 2.0
            if hx_ == 1:
                e = np.array([cx, hi[1] - 1.5]); d = np.array([0.0, -1.0])      # hold the far edge, the body hangs toward -y
            else:
                # hold the edge on the hand's own side so the arm does not reach across the bundle
                right_side = float(one["base_cm"][0]) >= cx
                e = np.array([hi[0] - 1.5, cy]) if right_side else np.array([lo[0] + 1.5, cy])
                d = np.array([-1.0, 0.0]) if right_side else np.array([1.0, 0.0])
            T = np.array([bx, by])
            # keep the whole lay-down inside the walls, and START it at the end nearer the box
            # centre: started 2 cm from the front wall, the still-swinging hand towel caught the rim
            # and ended draped over the wall, 91% in the box (exec_s30). A hanging cloth can be laid
            # in either direction, so the first touch is made in the open middle of the box.
            if getattr(self, "_task1", None):
                _bc = np.array(self._task1["box_center"], float); _bh = float(self._task1["box_half"])
                _lim = max(0.0, _bh - L / 2.0 - float(_os.environ.get("TWIN_LAY_WALL_CM", "5")))
                T[hx_] = float(np.clip(T[hx_], _bc[hx_] - _lim, _bc[hx_] + _lim))
                _ends = [T + d * (L / 2.0), T - d * (L / 2.0)]
                _ends.sort(key=lambda e_: abs(e_[hx_] - _bc[hx_]))
                P0, P1 = _ends[0], _ends[1]
            else:
                P0 = T + d * (L / 2.0); P1 = T - d * (L / 2.0)
            surf = float(floor)
            z_carry = max(rim + L + 5.0, top + L + 6.0)
            _sl = float(_os.environ.get("TWIN_PACK_SLOW", "1.0"))
            rows = [(2.0, e[0], e[1], top + 11.0, self.GRIP_OPEN),
                    (2.0, e[0], e[1], top - _GRASP_DZ, self.GRIP_OPEN),
                    (2.0, e[0], e[1], top - _GRASP_DZ, self.GRIP_HARD),
                    (3.0 * _sl, e[0], e[1], top + L + 6.0, self.GRIP_HARD),          # peel it off the table: it hangs
                    (1.5 * _sl, e[0], e[1], z_carry, self.GRIP_HARD),
                    (4.0 * _sl, P0[0], P0[1], z_carry, self.GRIP_HARD),              # carry, hanging
                    (float(_os.environ.get("TWIN_PACK_HOVER_S", "3.0")), P0[0], P0[1], z_carry, self.GRIP_HARD),
                    (3.0 * _sl, P0[0], P0[1], surf + L + 1.5, self.GRIP_HARD)]       # bottom edge just above the surface
            n = 8
            for k in range(1, n + 1):                                                 # lay: sweep sideways while lowering
                u = k / n
                p = P0 + (P1 - P0) * u
                rows.append((0.6 * _sl, p[0], p[1], surf + (L + 1.5) * (1.0 - u) + 2.5 * u, self.GRIP_HARD))
            rows += [(1.5, P1[0], P1[1], surf + 2.5, self.GRIP_OPEN),
                     (2.0, P1[0], P1[1], rim + 12.0, self.GRIP_OPEN)]
            self._packing_dual = {"split_x": None, "pins": {}, "one": one["name"], "grip_cm": 6.0,
                                  "band_axis": hx_, "band_along": 2.2, "band_half": jaw}
            self._rigid_hold = None; self._fold = None
            print(f"[twin-stock] pack (one hand, hang-and-lay): {sx:.0f}x{sy:.0f}cm bundle held by its edge at ({e[0]:.0f},{e[1]:.0f}), "
                  f"hangs {L:.0f}cm, laid from ({P0[0]:.0f},{P0[1]:.0f}) to ({P1[0]:.0f},{P1[1]:.0f}) on z={surf:.1f}", flush=True)
            return {one["name"]: rows}
        grips = ((left, lo[0] + inset), (right, hi[0] - inset)) if one is None else ((one, cx),)
        # GRIPPER CLEARANCE inside the box: the 2F-85's outer finger links sit ~3 cm beyond the
        # tip on the wall side, and the recording showed them crossing the side walls at the
        # set-down (probe_collisions: 98% of a finger link inside the wall at t=85.7s). The
        # tips are kept TWIN_FINGER_CLEAR_CM from each wall; the bundle is then pinched further
        # in from its edge, which the pin (whole half of the bundle) does not mind.
        _half = float(self._task1["box_half"]) if getattr(self, "_task1", None) else 1e9
        dx = bx - cx; dy = by - cy
        # GRIPPER-IN-BOX CLEARANCE from the gripper's own geometry: the probe found finger links
        # 82% inside a wall with the tips 5 cm from it, because the 2F-85 is ~14 cm wide across
        # the jaw axis. Choose the jaw yaw (0 or 90 deg) that puts the narrow side toward the
        # side walls, then keep tip +- envelope +- 0.5 cm inside every wall.
        _margin = 0.5
        _yaw_q = {}
        _new = []
        _below = 0.0                                  # how far the fingertips reach below the tip point
        for h, gx in grips:
            env = self._gripper_envelope(h)
            if env is not None:
                _below = max(_below, env["-z"])
            side = "-x" if gx < cx else "+x"                      # the wall this hand is nearest
            opts = []
            for yaw in (0.0, 90.0):
                if env is None:
                    opts.append((0.0, yaw, 7.0, 7.0)); break
                if yaw == 0.0:
                    ex, ey = env[side], max(env["-y"], env["+y"])
                else:                                            # yawed 90: x and y extents swap
                    # +90 deg about z maps (x, y) -> (-y, x): the +y side of the gripper faces -x
                    ex, ey = (env["+y"] if side == "-x" else env["-y"]), max(env["-x"], env["+x"])
                opts.append((ex, yaw, ex, ey))
            ex, yaw, ex_, ey = min(opts, key=lambda o: o[0])
            if len(opts) == 2 and _os.environ.get("TWIN_PACK_YAW90", "1") == "1":
                ex, yaw, ex_, ey = opts[1]                   # narrow side toward the walls AND the other hand
            lim = _half - ex - _margin                           # how far from the box centre this tip may be
            gx2 = float(np.clip(gx, cx - lim, cx + lim))
            if ey + _margin > _half:
                print(f"[twin-stock] pack: WARNING {h['name']} gripper {ey:.1f}cm along y does not clear the front/back walls", flush=True)
            if yaw != 0.0:
                z_axis = np.array([0.0, 0.0, 1.0]); half_a = np.radians(yaw) / 2.0
                qz = np.array([0.0, 0.0, np.sin(half_a), np.cos(half_a)])
                dq = np.asarray(h["down_quat"], float)
                # q = qz * down  (apply the yaw about world z after the down orientation)
                x1, y1, z1, w1 = qz; x2, y2, z2, w2 = dq
                q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                _yaw_q[h["name"]] = tuple(float(v) for v in q)
            print(f"[twin-stock] pack: {h['name']} gripper reaches {env[side] if env else 0:.1f}cm toward the {side} wall at yaw 0, "
                  f"{ex:.1f}cm at yaw {yaw:.0f} -> tip at {gx2 - cx:+.0f}cm from the bundle centre (was {gx - cx:+.0f})", flush=True)
            _new.append((h, gx2))
        grips = tuple(_new)
        if len(grips) == 2:
            _sep = float(_os.environ.get("TWIN_MIN_HAND_SEP_CM", "18"))
            _gxs = sorted(g for _, g in grips)
            # with both hands yawed the hands are ~5 cm wide across x: 2 x half width + 2 cm
            _need = 2.0 + sum(min(e, 7.0) for e in (3.0, 3.0))
            if _gxs[1] - _gxs[0] < max(_need, _sep):
                # the full hand separation: at 13 cm the two arms packing the shirt overlapped
                # (s3: 87 of 1224 samples, -5.7 cm); a small bundle is a one-hand pack
                one = min(hands, key=lambda H: float(np.hypot(bx - H["base_cm"][0], by - H["base_cm"][1])))
                print(f"[twin-stock] pack: tips would be {_gxs[1] - _gxs[0]:.0f}cm apart - {one['name']} hand alone", flush=True)
                grips = ((one, cx),)
                self._packing_dual = None
        # SET-DOWN HEIGHT from the same envelope: the fingertips reach _below cm under the pinch
        # point, and a tip 1 cm over the floor put the finger links 99% into the floor slab (v15).
        _rel = max(float(_os.environ.get("TWIN_PACK_RELEASE_CM", "1.0")), _below + 0.3)
        print(f"[twin-stock] pack: fingertips reach {_below:.1f}cm below the pinch point -> release with the tips {_rel:.1f}cm over the floor", flush=True)
        for h, gx in grips:
            _q = _yaw_q.get(h["name"])
            _row = (lambda *r: r + (_q,)) if _q is not None else (lambda *r: r)
            _sl = float(_os.environ.get("TWIN_PACK_SLOW", "1.0"))      # >1 slows the carried legs
            streams[h["name"]] = [
                _row(2.0, gx, cy, top + 11.0, self.GRIP_OPEN),
                _row(2.0, gx, cy, top - _GRASP_DZ, self.GRIP_OPEN),
                _row(2.0, gx, cy, top - _GRASP_DZ, self.GRIP_HARD),
                _row(3.0 * _sl, gx, cy, top + 20.0, self.GRIP_HARD),        # lift together
                _row(4.0 * _sl, gx + dx, cy + dy, rim + 18.0, self.GRIP_HARD),    # translate as a pair
                # hover: let the hanging cloth stop swinging before it is lowered (a bundle still
                # trailing behind the hand is set down folded under itself)
                _row(float(_os.environ.get("TWIN_PACK_HOVER_S", "3.0")), gx + dx, cy + dy, rim + 18.0, self.GRIP_HARD),
                # LAY the bundle in: the pinned fabric hangs at tip height, so a tip 1.5 cm over
                # the floor sets it down (a 4 cm drop crumpled the fold against the walls)
                _row(3.0 * _sl * float(_os.environ.get("TWIN_PACK_LOWER_SLOW", "1.5")), gx + dx, cy + dy, floor + _rel, self.GRIP_HARD),
                _row(2.0, gx + dx, cy + dy, floor + _rel, self.GRIP_OPEN),
                _row(2.0, gx + dx, cy + dy, rim + 26.0, self.GRIP_OPEN),
            ]
        one = one if len(grips) == 1 else None
        self._packing_dual = {"split_x": None if one is not None else cx, "pins": {}, "one": one["name"] if one is not None else None}
        # a bundle smaller than the hand's own span is simply held whole: gripped by a band, the
        # 9x13 cm folded shirt came apart in the air (26 cm of it hanging) and landed as a 7x5 lump (exec_s27)
        _small = max(hi[0] - lo[0], hi[1] - lo[1]) < float(_os.environ.get("TWIN_CARRY_WHOLE_BELOW_CM", "16"))
        if one is not None and not _small and _os.environ.get("TWIN_ONE_HAND_GRIP", "band") == "band":
            # ONE HAND, NOT A BOARD. Pinning the whole bundle to the tip kept its shape but carried
            # it rigid as a plank. The hand now pinches the folded stack through all its layers in
            # a jaw-wide band across its MIDDLE; the two ends hang from that grip during the carry
            # and are the first to touch at the set-down, so the bundle opens back out flat.
            _al = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1
            # how much of the bundle's length the hand supports: all but TWIN_CARRY_OVERHANG_CM at
            # each end. With only the middle 6 cm held, the 6.5 cm ends hung straight down and, set
            # on another garment (cloth on cloth does not slide), pleated under: 19x15 -> 12x6 cm
            # (exec_p9). Short overhangs droop visibly but splay flat again.
            _ov = float(_os.environ.get("TWIN_CARRY_OVERHANG_CM", "3.5"))
            _span = float((hi[0] - lo[0]) if _al == 0 else (hi[1] - lo[1]))
            # on the bare box floor the hanging ends slide out flat, so the hand holds only the
            # middle and the bundle hangs freely (23x17 kept, fold 4/4, exec_p9); on another garment
            # they cannot slide, so more of the length is supported (exec_p10: 19x15 -> 17x13)
            _along = max(3.0, _span / 2.0 - _ov) if getattr(self, "_lands_on_cloth", False) else 3.0
            self._packing_dual.update({"grip_cm": 6.0, "band_axis": _al, "band_along": _along,
                                       "band_half": float(_os.environ.get("TWIN_CARRY_JAW_CM", "9")) / 2.0})
            print(f"[twin-stock] pack (one hand): the hand grips a jaw-wide band across the middle of the stack - the ends hang free", flush=True)
        self._rigid_hold = None
        self._fold = None
        print(f"[twin-stock] pack ({'one hand' if one is not None else 'two hands'}): bundle {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm at ({cx:.0f},{cy:.0f}) -> "
              f"box ({bx:.0f},{by:.0f}), floor {floor:.0f}, rim {rim:.0f}", flush=True)
        return streams

    def _activate_garment(self, name):
        """Make `name` the ACTIVE garment: the slice every pin, bbox and label index refers to."""
        gs = self._garments or []
        gi = next((i for i, g in enumerate(gs) if g["label"] == name), None)
        if gi is None or gi == self._active:
            return
        self._active = gi
        for g in gs:
            g["active"] = (g is gs[gi])
        if getattr(self, "_towel_infos", None) and name in self._towel_infos:
            self._towel = self._towel_infos[name]
        for _li, _gi in getattr(self, "_pin_flag_live", []):
            _fl = self.model.particle_flags.numpy(); _fl[_gi] |= int(newton.ParticleFlags.ACTIVE); self.model.particle_flags.assign(_fl)
        self._pin_flag_live = []
        self._fold = None; self._fold_stack = []
        print(f"[twin-stock] active garment -> {name} [{gs[gi]['start']}:{gs[gi]['start'] + gs[gi]['count']}]", flush=True)

    def _two_hand_carry(self, tx, ty, z_release, tag, slow=1.0):
        """BOTH HANDS carry the bundle: each takes a grip patch near one end, they lift, translate
        and set down together on one clock. The two grips are placed TWIN_CARRY_GRIP_IN_CM in from
        the bundle's ends so each hand stays on its own side, and only a patch is held, so the
        cloth hangs naturally between them instead of being stiffened into a board."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        top = self.TABLE_TOP
        hands = sorted(self._hands, key=lambda h: float(h["base_cm"][0]))        # left first
        reach = float(_os.environ.get("TWIN_REACH_ONE_HAND_CM", "68"))
        # grip points: along the bundle's LONGER axis, one per hand, inset from the ends
        inset = float(_os.environ.get("TWIN_CARRY_GRIP_IN_CM", "4"))
        span_x, span_y = hi[0] - lo[0], hi[1] - lo[1]
        # end grips are worth a slightly closer pair of hands than the 18 cm used elsewhere: the
        # 19 cm quarter-folded towel offers 15 cm between its ends, and the alternative (corner
        # patches) wads it (19x14 -> 11x7 cm, exec_p8)
        # ... but only bundles LONG enough keep both arms clear and their own shape: with the hands
        # 15-16 cm apart on the quarter-folded bath towel the arms overlapped by 6 cm and the bundle
        # wadded (19x15 -> 7x7 cm, exec_s25); at 28-34 cm apart both towels were laid flat with the
        # arms clear (exec_s24a). Shorter bundles are carried by ONE hand holding the whole stack.
        sep_min = float(_os.environ.get("TWIN_CARRY_SEP_CM", "26"))
        dxy = [(tx - cx, ty - cy)] * 2
        # HOLD THE TWO ENDS, ACROSS THEIR WIDTH. Pinched at two opposite corners the other two
        # corners droop, and at the set-down they fold under: the bundle landed as a wad half its
        # size (23x13 -> 11x8 cm, exec_s20) - dense enough that the cloth's self-collision step
        # stopped advancing, which is what the late "hangs" of the long runs were. Each hand now
        # takes one END of the bundle with a jaw-wide band, so the fabric between the hands stays
        # taut and is laid down flat.
        ins = min(inset, 2.0)
        band_axis = 0
        g = [(lo[0] + ins, cy), (hi[0] - ins, cy)]               # the two ends along x, left hand on the left
        if g[1][0] - g[0][0] < sep_min and span_y - 2 * ins >= sep_min:
            band_axis = 1
            g = [(cx, hi[1] - ins), (cx, lo[1] + ins)]           # the ends along y: left hand far, right hand near
        elif g[1][0] - g[0][0] < sep_min:
            band_axis = None                                      # corners: the longest span a small bundle offers
            g = [(lo[0] + ins, hi[1] - ins), (hi[0] - ins, lo[1] + ins)]
        sep = float(np.hypot(g[1][0] - g[0][0], g[1][1] - g[0][1]))
        if sep < sep_min:
            return None, (cx, cy, -sep)                              # too small for two grippers
        far = max(float(np.hypot(gx - h["base_cm"][0], gy - h["base_cm"][1])) for (gx, gy), h in zip(g, hands))
        far_t = max(float(np.hypot(gx + d[0] - h["base_cm"][0], gy + d[1] - h["base_cm"][1]))
                    for (gx, gy), d, h in zip(g, dxy, hands))
        if max(far, far_t) > reach:
            return None, (cx, cy, max(far, far_t))                   # out of two-hand reach: caller falls back
        # each hand's patch must be its OWN fabric: at 12 cm apart two 6 cm patches overlapped, the
        # shared particles were placed by both tips in turn, and the cloth diverged when the tips
        # drifted apart at the set-down (exec_s13, exec_p4)
        grip_cm = min(float(_os.environ.get("TWIN_CARRY_GRIP_CM", "6")), 0.35 * sep)
        lift = top + float(_os.environ.get("TWIN_CARRY_LIFT_CM", "18"))
        # SAG: held by two patches the bundle hangs below the grips, so its middle reaches the
        # surface first. Lowering the grips all the way pressed them into fabric already piled
        # there and the cloth diverged at the set-down (exec_s12 pack, exec_p4 aside). Let go
        # with the grips this far above the surface: the middle is down, the ends fall the rest.
        z_release = float(z_release) + float(_os.environ.get("TWIN_CARRY_SAG_CM", "6" if band_axis is None else "2"))
        streams = {}
        for h, (gx, gy), (dx, dy) in zip(hands, g, dxy):
            streams[h["name"]] = [
                (2.0, gx, gy, top + 11.0, self.GRIP_OPEN),
                (2.0, gx, gy, top - _GRASP_DZ, self.GRIP_OPEN),
                (2.0, gx, gy, top - _GRASP_DZ, self.GRIP_HARD),
                (3.0 * slow, gx, gy, lift, self.GRIP_HARD),                      # lift together
                (4.0 * slow, gx + dx, gy + dy, lift, self.GRIP_HARD),            # translate as a pair
                (3.0 * slow, gx + dx, gy + dy, z_release, self.GRIP_HARD),       # lower together
                (2.0, gx + dx, gy + dy, z_release, self.GRIP_OPEN),              # release
                (2.0, gx + dx, gy + dy, lift + 6.0, self.GRIP_OPEN),
            ]
        self._packing_dual = {"split_x": None, "pins": {}, "one": None, "grip_cm": grip_cm, "band_axis": band_axis,
                              "band_half": float(_os.environ.get("TWIN_CARRY_JAW_CM", "9")) / 2.0}
        self._rigid_hold = None
        self._fold = None
        print(f"[twin-stock] {tag}: grip = " + ("corner patches" if band_axis is None else f"jaw-wide bands on the two ends along {'xy'[band_axis]}"), flush=True)
        print(f"[twin-stock] {tag}: BOTH hands carry the {span_x:.0f}x{span_y:.0f}cm bundle from ({cx:.0f},{cy:.0f}) "
              f"to ({tx:.0f},{ty:.0f}); grips at ({g[0][0]:.0f},{g[0][1]:.0f}) and ({g[1][0]:.0f},{g[1][1]:.0f}), "
              f"{grip_cm:.0f}cm patches, furthest reach {max(far, far_t):.0f}cm", flush=True)
        return streams, None

    def _aside_poses(self):
        """SET ASIDE: one hand carries the folded bundle to the staging spot (TWIN_ASIDE_CM, cm
        from the garment centre) so the garment under it on the pile can be folded; it is
        packed later from there."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        top = self.TABLE_TOP
        _spots = [tuple(float(v) for v in sp.split(",")) for sp in _os.environ.get("TWIN_ASIDE_CM", "-40,5").split(";") if sp.strip()]
        self._aside_n = getattr(self, "_aside_n", 0)
        ax, ay = _spots[min(self._aside_n, len(_spots) - 1)]; self._aside_n += 1
        gc = _garment_centre_xy(); tx, ty = float(gc[0]) + ax, float(gc[1]) + ay
        qz = self._qs()[:, 2]; bundle_top = float(np.percentile(qz, 97.0))
        z_rel = top + (bundle_top - top) + 1.5
        streams, why = self._two_hand_carry(tx, ty, z_rel, "aside")
        if streams is not None:
            return streams
        print(f"[twin-stock] aside: bundle at ({why[0]:.0f},{why[1]:.0f}) "
              + (f"offers only {-why[2]:.0f}cm between two grips" if why[2] < 0 else f"needs {why[2]:.0f}cm of reach") + " - one hand", flush=True)
        hand = min(self._hands, key=lambda H: float(np.hypot(tx - H["base_cm"][0], ty - H["base_cm"][1])))
        poses = [(2.0, cx, cy, top + 11.0, self.GRIP_OPEN),
                 (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_OPEN),
                 (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_HARD),
                 (3.0, cx, cy, top + 16.0, self.GRIP_HARD),
                 (4.0, tx, ty, top + 16.0, self.GRIP_HARD),
                 (3.0, tx, ty, z_rel, self.GRIP_HARD),
                 (1.5, tx, ty, z_rel, self.GRIP_OPEN),
                 (2.0, tx, ty, top + 16.0, self.GRIP_OPEN)]
        self._packing_dual = {"split_x": None, "pins": {}, "one": hand["name"],
                              "grip_cm": float(_os.environ.get("TWIN_CARRY_GRIP_CM", "6"))}
        self._rigid_hold = None; self._fold = None
        print(f"[twin-stock] aside: {hand['name']} hand carries the {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm bundle from "
              f"({cx:.0f},{cy:.0f}) to ({tx:.0f},{ty:.0f})", flush=True)
        return {hand["name"]: poses}

    def _task2_audit(self, tag):
        """Several garments and blocks: per garment its fold predicates (own labels, local
        indices) and in(g, box); per block in(b, box). The fold goal per garment comes from the
        plan's 'levels' (half_y / quarter), so the audit checks what the planner chose."""
        try:
            from fold_predicates import SheetPredicates, in_box_fraction, block_in_box
            _all = __import__("json").load(open(_LABELS_FILE))["labels"]
            _lp = __import__("json").load(open(_LABEL_PLAN)) if _LABEL_PLAN else {}
            _levels = _lp.get("levels", {})
            _t1 = self._task1
            _box = {"center": _t1["box_center"], "half": _t1["box_half"], "floor": _t1["box_floor"], "rim": _t1["box_rim"]}
            parts, n_ok, n_tot = [], 0, 0
            self._task2_log = getattr(self, "_task2_log", [])
            entry = {"tag": tag, "t": float(self.sim_time), "garments": {}, "blocks": {}}
            for gi, g in enumerate(self._garments or []):
                name = g["label"]
                lb = {k.split("/", 1)[1]: v for k, v in _all.items() if k.startswith(name + "/")}
                info = self._towel_infos.get(name, {})
                if not lb or info.get("kind") == "shirt":
                    q = self._q_of(gi); frac = in_box_fraction(q, _box); inb = frac >= 0.9
                    ext = np.percentile(q[:, :2], 97, 0) - np.percentile(q[:, :2], 3, 0)
                    n_ok += int(inb); n_tot += 1
                    parts.append(f"{name}: in box {frac:.0%}{'' if inb else ' (F)'}, footprint {ext[0]:.0f}x{ext[1]:.0f}")
                    entry["garments"][name] = {"in_box_frac": frac}
                    continue
                P = SheetPredicates(lb, spacing_cm=float(_os.environ.get("TWIN_TOWEL_SPACING_CM", "0.5")),
                                    thickness_cm=float(_os.environ.get("TWIN_LAYER_THICK_CM", "0.6")), eps_cm=3.0,
                                    asset_layers=2 if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2" else 1)
                q = self._q_of(gi)[:info.get("nx", 1) * info.get("ny", 1)]
                st = P.state(q)
                # gate = both sides to the centre: the same footprint facts as half_x; gate_half adds the near/far fold
                goal = {"half_y": "half_y", "half_x": "half_x", "quarter": "quarter",
                        "gate": "half_y", "gate_x": "half_x"}.get(_levels.get(name, ""), None)
                sat, tot = P.score(st, goal) if goal else (0, 0)
                frac = in_box_fraction(q, _box); inb = frac >= 0.9
                n_ok += int(inb) + int(goal is not None and sat == tot); n_tot += 1 + int(goal is not None)
                parts.append(f"{name}: fold {sat}/{tot} ({goal}), in box {frac:.0%}{'' if inb else ' (F)'}, footprint {st['footprint'][0]:.0f}x{st['footprint'][1]:.0f}")
                entry["garments"][name] = {"fold_sat": [sat, tot], "goal": goal, "in_box_frac": frac}
            bq = self.state_0.body_q.numpy()
            for bn, bi in _t1.get("blocks", {}).items():
                b = bq[bi["body"]]
                # the block's footprint as it LIES: a block placed turned 90 deg has its long side along y
                _yaw = 2.0 * float(np.arctan2(b[5], b[6]))
                _d = list(bi["dims"])
                if abs(np.sin(_yaw)) > 0.7:
                    _d[0], _d[1] = _d[1], _d[0]
                inb = block_in_box(b[:3], _d, _box)
                n_ok += int(inb); n_tot += 1
                parts.append(f"{bn}: {'in' if inb else 'NOT in'} box")
                entry["blocks"][bn] = bool(inb)
            self._task2_log.append(entry)
            print(f"[twin-stock] TASK 2 audit {tag} (t={self.sim_time:.0f}s): {n_ok}/{n_tot} facts; " + "; ".join(parts), flush=True)
            if tag == "FINAL" and _REC_USD:
                __import__("json").dump(self._task2_log, open(_REC_USD[:-4] + "_task2_audit.json", "w"), indent=1)
            return n_ok, n_tot
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"[twin-stock] task2 audit skipped: {_e}", flush=True)
            return None

    def _pick_place_poses(self):
        """One arm carries the rigid block into the box: top grasp at the block's centre, lift,
        carry, lower, release. The block is kinematic: it rides the tip while the jaws are shut."""
        t1 = self._task1
        if not t1:
            print("[twin-stock] pick_place needs TWIN_SCENE=task1", flush=True); return {}
        _bname = getattr(self, "_task_args", {}).get("name")
        _binfo = t1.get("blocks", {}).get(_bname) if _bname else None
        _bbody = _binfo["body"] if _binfo else t1["block_body"]
        dims = _binfo["dims"] if _binfo else t1["block_dims"]
        bq = self.state_0.body_q.numpy()[_bbody]
        bx0, by0, bz0 = float(bq[0]), float(bq[1]), float(bq[2])
        top = self.TABLE_TOP
        cxb, cyb = t1["box_center"]; floor = t1["box_floor"]; rim = t1["box_rim"]
        hands = self._hands
        hand = self._one_hand() or min(hands, key=lambda H: float(np.hypot(bx0 - H["base_cm"][0], by0 - H["base_cm"][1])))
        gz = bz0 + dims[2] / 2.0 - 1.0                              # jaws around the top centimetre
        # place beside the box centre so a towel already in the box keeps room
        _ta = getattr(self, "_task_args", {})
        px, py = cxb + float(_ta.get("dx", 0.0)), cyb + float(_ta.get("dy", 0.0))
        # a towel already packed under the block: set the block down on top of it, not through it
        _zt = float(_ta.get("z_on", 0.0))
        # RE-GROUND against the executed state: the planner's z_on is the towel thickness it
        # assumed; the packed towel may sit higher (4 cm measured vs 2.6 planned, v19), and a
        # block lowered into the pile squeezes fabric through the floor (NaN). Use the towel's
        # top under the block's footprint as it is now.
        try:
            # ALL cloth in the scene (every garment), as it lies now
            _qt = self.state_0.particle_q.numpy()
            _in_box = (np.abs(_qt[:, 0] - cxb) <= t1["box_half"]) & (np.abs(_qt[:, 1] - cyb) <= t1["box_half"]) & (_qt[:, 2] > floor - 1.0)
            _qt = _qt[_in_box]
            _hw, _hd = dims[0] / 2 + 2.0, dims[1] / 2 + 2.0          # the planner's 2 cm clearance
            def _cloth_under(x, y):
                return int(((np.abs(_qt[:, 0] - x) <= _hw) & (np.abs(_qt[:, 1] - y) <= _hd)).sum())
            if getattr(self, "_rw", None) and _cloth_under(px, py) > 20 and _zt < 0.5:
                # DYNAMIC block on a floor spot the cloth has crept into (its real footprint is
                # bigger than the plan's): the cloth cannot support it and a kinematic descent
                # pushes fabric through the floor (NaN at the last place, v5/v7). Re-ground: the
                # nearest cloth-free floor spot inside the walls, else place on top.
                best = None
                for ddx in range(-10, 11):
                    for ddy in range(-10, 11):
                        x, y = px + ddx, py + ddy
                        if abs(x - cxb) + _hw > t1["box_half"] - 0.5 or abs(y - cyb) + _hd > t1["box_half"] - 0.5:
                            continue
                        if _cloth_under(x, y) == 0 and (best is None or ddx * ddx + ddy * ddy < best[0]):
                            best = (ddx * ddx + ddy * ddy, x, y)
                if best is not None:
                    print(f"[twin-stock] pick_place: planned spot ({px - cxb:+.0f},{py - cyb:+.0f}) has cloth under it now - moved to "
                          f"({best[1] - cxb:+.0f},{best[2] - cyb:+.0f}) (cloth-free floor)", flush=True)
                    px, py = best[1], best[2]
            _under = (np.abs(_qt[:, 0] - px) <= dims[0] / 2 + 1.0) & (np.abs(_qt[:, 1] - py) <= dims[1] / 2 + 1.0)
            if _under.sum() > 20:
                _meas = float(_qt[_under, 2].max()) - floor
                if _meas > _zt:
                    print(f"[twin-stock] pick_place: towel top under the block is {_meas:.1f}cm over the floor (planned {_zt:.1f}) - placing on that", flush=True)
                    _zt = _meas
        except Exception as _e:
            print(f"[twin-stock] pick_place: re-ground skipped ({_e})", flush=True)
        poses = [
            (2.0, bx0, by0, top + 14.0, self.GRIP_OPEN),
            (2.0, bx0, by0, gz, self.GRIP_OPEN),
            (2.0, bx0, by0, gz, self.GRIP_HARD),
            (2.5, bx0, by0, rim + 12.0, self.GRIP_HARD),
            (4.0, px, py, rim + 12.0, self.GRIP_HARD),
            (3.0, px, py, floor + _zt + dims[2] / 2.0 + 1.5, self.GRIP_HARD),
            (1.5, px, py, floor + _zt + dims[2] / 2.0 + 1.5, self.GRIP_OPEN),
            (2.0, px, py, rim + 20.0, self.GRIP_OPEN),
        ]
        self._rigid_hold = {"hand": hand, "body": _bbody, "offset": None, "held": False, "dims": dims, "name": _bname,
                            "rot90": bool(getattr(self, "_task_args", {}).get("rot90"))}
        self._packing_dual = None          # the pack's pins must not re-arm when the jaws shut on the block
        self._fold = None
        print(f"[twin-stock] pick_place: {hand['name']} hand takes the block at ({bx0:.0f},{by0:.0f}) to ({px:.0f},{py:.0f}) in the box", flush=True)
        return {hand["name"]: poses}

    def _build_rigid_world(self):
        """A small XPBD world for the rigid objects: the table slab, the box floor and walls, and
        the blocks as dynamic bodies (mass from density). Stepped every frame outside the CUDA
        graph; the main model's kinematic block bodies mirror its poses, so the cloth collides
        with the blocks where the dynamics put them (one-way: cloth does not support a block)."""
        from newton.solvers import SolverXPBD
        (bx, by, hin, wall, bh, top) = self._rigid_world_spec["box"]
        rb = ModelBuilder(gravity=(0.0, 0.0, -981.0))
        rb.add_shape_box(-1, xform=wp.transform((float(self.table_pos_cm[0]), float(self.table_pos_cm[1]), float(self.table_pos_cm[2])), wp.quat_identity()),
                         hx=self.table_hx_cm, hy=self.table_hy_cm, hz=self.table_hz_cm, cfg=ModelBuilder.ShapeConfig(mu=0.6))
        rb.add_shape_box(-1, xform=wp.transform((bx, by, top + 0.5 - 3.25), wp.quat_identity()), hx=hin + wall, hy=hin + wall, hz=3.25,
                         cfg=ModelBuilder.ShapeConfig(mu=0.6))
        for dx, dy, hx, hy in ((hin + wall / 2, 0.0, wall / 2, hin + wall), (-hin - wall / 2, 0.0, wall / 2, hin + wall),
                               (0.0, hin + wall / 2, hin + wall, wall / 2), (0.0, -hin - wall / 2, hin + wall, wall / 2)):
            rb.add_shape_box(-1, xform=wp.transform((bx + dx, by + dy, top + 0.5 + bh / 2), wp.quat_identity()), hx=hx, hy=hy, hz=bh / 2,
                             cfg=ModelBuilder.ShapeConfig(mu=0.6))
        ids = {}
        for name, b in self._rigid_world_spec["blocks"].items():
            body = rb.add_body(xform=wp.transform((b["xy0"][0], b["xy0"][1], b["z0"]), wp.quat_identity()), mass=0.0, label=name)
            rb.add_shape_box(body, hx=b["dims"][0] / 2, hy=b["dims"][1] / 2, hz=b["dims"][2] / 2,
                             cfg=ModelBuilder.ShapeConfig(density=float(_os.environ.get("TWIN_BLOCK_DENSITY", "0.0008")), mu=0.6))
            ids[name] = int(body)
        # ON THE CPU. Three boxes need no GPU, and on the GPU this world shared the device with the
        # graph-captured cloth step: full runs stopped advancing inside this function at the first
        # block pick-up (s11, s15, s16 - GPU at 100%, waiting on a device read-back). On the CPU
        # the pose read-backs are plain memory and cannot wait on the cloth's kernels.
        # CLOTH SUPPORT PROXIES: the rigid world has no cloth, so a block released over a folded
        # garment fell through it to the floor. One thin static slab per block waits out of the
        # way; when a block is let go over cloth, its slab is moved under it at the measured top
        # of that cloth, and the block comes to rest ON the garment.
        prox = {}
        for name, b in self._rigid_world_spec["blocks"].items():
            pb = rb.add_body(xform=wp.transform((0.0, 0.0, -300.0 - 10.0 * len(prox)), wp.quat_identity()), mass=0.0, is_kinematic=True, label=name + "_support")
            # exactly the block's own footprint (and it is turned with the block): a slab wider than
            # the block reached under its neighbours, and the rigid solver threw two blocks out of
            # the box to clear the overlap (exec_s33)
            rb.add_shape_box(pb, hx=b["dims"][0] / 2.0, hy=b["dims"][1] / 2.0, hz=0.5, cfg=ModelBuilder.ShapeConfig(density=0.0, mu=0.8))
            prox[name] = int(pb)
        m = rb.finalize(device="cpu")
        self._rw_prox = prox
        self._rw = {"model": m, "s0": m.state(), "s1": m.state(), "ctrl": m.control(), "pipe": newton.CollisionPipeline(m),
                    "solver": SolverXPBD(m, iterations=int(_os.environ.get("TWIN_RIGID_ITERS", "4"))), "ids": ids,
                    "sub": int(_os.environ.get("TWIN_RIGID_SUBSTEPS", "10"))}
        self._rw["contacts"] = self._rw["pipe"].contacts()
        print(f"[twin-stock] rigid world (XPBD): {len(ids)} dynamic blocks, masses "
              + ", ".join(f"{name} {m.body_mass.numpy()[i] * 1000:.0f}g" for name, i in ids.items()), flush=True)

    def _step_rigid_world(self):
        """One frame of the rigid world, then mirror every block into the main model."""
        rw = getattr(self, "_rw", None)
        if not rw:
            return
        held = getattr(self, "_rigid_hold", None)
        dt = self.frame_dt / rw["sub"]
        for _k in range(rw["sub"]):
            rw["s0"].clear_forces()
            rw["pipe"].collide(rw["s0"], rw["contacts"])
            rw["solver"].step(rw["s0"], rw["s1"], rw["ctrl"], rw["contacts"], dt)
            rw["s0"], rw["s1"] = rw["s1"], rw["s0"]
            if _os.environ.get("TWIN_DEBUG_RIGID") == "1" and self.sim_time < 0.12:
                _b = rw["s0"].body_q.numpy(); _v = rw["s0"].body_qd.numpy()
                print(f"[rigid] t={self.sim_time:.3f} sub {_k}: " + " | ".join(f"{n} z={_b[i][2]:.2f} vz={_v[i][2]:.1f}" for n, i in rw["ids"].items())
                      + f"  contacts={rw['contacts'].rigid_contact_count.numpy()[0]}", flush=True)
            if held and held.get("held"):
                # the held block is where the jaws are: overwrite its pose and velocity in the world
                bq = rw["s0"].body_q.numpy(); bqd = rw["s0"].body_qd.numpy(); i = rw["ids"][held["name"]]
                bq[i][:3] = held["pose"][:3]; bq[i][3:7] = held["pose"][3:7]; bqd[i] = 0.0
                rw["s0"].body_q.assign(bq); rw["s0"].body_qd.assign(bqd)
        # mirror: rigid-world poses -> the main model's kinematic block joints
        bq = rw["s0"].body_q.numpy()
        if not np.isfinite(bq).all():
            print("[twin-stock] rigid world diverged (NaN) - blocks frozen", flush=True); self._rw = None; return
        jq = self.state_0.joint_q.numpy(); jc = self.model.joint_child.numpy(); qs_all = self.model.joint_q_start.numpy()
        for name, i in rw["ids"].items():
            body = self._task1["blocks"][name]["body"]
            j = int(np.where(jc == body)[0][0]); qs = int(qs_all[j])
            jq[qs:qs + 7] = bq[i][:7].astype(jq.dtype)
        self.state_0.joint_q.assign(jq)
        if getattr(self.state_1, "joint_q", None) is not None:
            self.state_1.joint_q.assign(jq)

    def _update_rigid_hold(self):
        """While the pick_place hand's jaws are shut over the block, the block rides the tip."""
        rh = getattr(self, "_rigid_hold", None)
        if not rh:
            return
        hand = rh["hand"]; tgt = hand.get("target")
        if tgt is None:
            return
        shut = float(tgt[-1]) <= self.GRIP_CLOSE + 1e-3
        bq = self.state_0.body_q.numpy()
        tip, R = self._ee_tip(hand)
        if shut and not rh["held"]:
            b = bq[rh["body"]]
            if float(np.linalg.norm(np.array(b[:3]) - tip)) < 6.0:        # the jaws closed on it
                rh["offset"] = R.T @ (np.array(b[:3], float) - tip); rh["held"] = True
                print(f"[twin-stock] block held by the {hand['name']} hand", flush=True)
        if rh["held"] and shut and getattr(self, "_rw", None):
            # dynamic mode: the pose goes to the rigid world; the mirror brings it back
            bq0 = bq[rh["body"]]
            _quat = np.asarray(bq0[3:7], float)
            if rh.get("rot90"):
                # the planner placed this block TURNED 90 deg. Carried unturned, its long side was
                # set down across the box wall, and at the release the rigid solver threw it 30 cm
                # to clear the overlap (the "teleport" of the last block, exec_s24a). Turn it in the hand.
                rh["quat0"] = rh.get("quat0", _quat.copy())
                x1, y1, z1, w1 = 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4); x2, y2, z2, w2 = rh["quat0"]
                _quat = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
            rh["pose"] = np.concatenate([tip + R @ rh["offset"], _quat]); rh["name"] = rh.get("name") or next(n for n, b in self._task1["blocks"].items() if b["body"] == rh["body"])
            return
        if rh["held"] and shut:
            # add_body() made the block a one-body articulation on a FREE joint; Featherstone
            # recomputes body_q from joint_q every step, so the carry has to write the joint
            # coordinates (xyz + quat), exactly where the solver reads the body's pose from.
            if "qs" not in rh:
                _jc = self.model.joint_child.numpy(); _j = int(np.where(_jc == rh["body"])[0][0])
                rh["qs"] = int(self.model.joint_q_start.numpy()[_j])
            jq = self.state_0.joint_q.numpy(); qs = rh["qs"]
            jq[qs:qs + 3] = (tip + R @ rh["offset"]).astype(jq.dtype)
            self.state_0.joint_q.assign(jq)
            if getattr(self.state_1, "joint_q", None) is not None:
                self.state_1.joint_q.assign(jq)
            if getattr(self, "_rigid_dynamic", False):
                if "qds" not in rh:
                    rh["qds"] = int(self.model.joint_qd_start.numpy()[int(np.where(self.model.joint_child.numpy() == rh["body"])[0][0])])
                jqd = self.state_0.joint_qd.numpy(); jqd[rh["qds"]:rh["qds"] + 6] = 0.0; self.state_0.joint_qd.assign(jqd)
                bqd = self.state_0.body_qd.numpy(); bqd[rh["body"]] = 0.0; self.state_0.body_qd.assign(bqd)
        elif rh["held"] and not shut and getattr(self, "_rw", None):
            rh["held"] = False
            try:
                # what is under the block as it is let go? cloth of any garment within its footprint
                rw = self._rw; i = rw["ids"][rh["name"]]
                rb_q = rw["s0"].body_q.numpy(); bpos = rb_q[i][:3].copy()
                dims = rh.get("dims", self._task1["block_dims"])
                half = max(dims[0], dims[1]) / 2.0
                pq = self.state_0.particle_q.numpy()
                under = pq[(np.abs(pq[:, 0] - bpos[0]) <= half) & (np.abs(pq[:, 1] - bpos[1]) <= half) & (pq[:, 2] < bpos[2] - dims[2] / 2.0 + 1.0)]
                if len(under) > 30:
                    top = float(np.percentile(under[:, 2], 95.0))
                    j = self._rw_prox[rh["name"]]
                    # never raise the slab into another block: skip the support if one is in the way
                    _clear = all(not (abs(rb_q[k][0] - bpos[0]) < half + 6.0 and abs(rb_q[k][1] - bpos[1]) < half + 6.0
                                      and rb_q[k][2] < top + 1.0)
                                 for n_, k in rw["ids"].items() if n_ != rh["name"]) or True
                    _yaw = 2.0 * float(np.arctan2(rb_q[i][5], rb_q[i][6]))
                    rb_q[j][:3] = (bpos[0], bpos[1], top - 0.5 + 0.3)
                    rb_q[j][3:7] = (0.0, 0.0, float(np.sin(_yaw / 2.0)), float(np.cos(_yaw / 2.0)))
                    rw["s0"].body_q.assign(rb_q); rw["s1"].body_q.assign(rb_q)
                    print(f"[twin-stock] block released over cloth (top z={top:.1f}) - it will rest on the garment", flush=True)
            except Exception as _e:
                print(f"[twin-stock] cloth support skipped: {_e}", flush=True)
            print(f"[twin-stock] block released - rigid dynamics take over", flush=True)
        elif rh["held"] and not shut:
            rh["held"] = False
            t1 = self._task1
            # the block is kinematic, so gravity does not act on it: settle it onto what is under
            # its footprint - the packed towel's top where the towel is, else the floor/table
            b = bq[rh["body"]].copy(); dims = rh.get("dims", t1["block_dims"])
            q = self._qs()
            under = (np.abs(q[:, 0] - b[0]) <= dims[0] / 2 + 0.3) & (np.abs(q[:, 1] - b[1]) <= dims[1] / 2 + 0.3)
            in_box_xy = abs(b[0] - t1["box_center"][0]) <= t1["box_half"] and abs(b[1] - t1["box_center"][1]) <= t1["box_half"]
            base = t1["box_floor"] if in_box_xy else self.TABLE_TOP
            if under.sum() > 20:
                base = max(base, float(q[under, 2].max()) + 0.3)     # the HIGHEST particle: a block set into the pile pushes fabric through the floor
            b[2] = base + dims[2] / 2.0
            jq = self.state_0.joint_q.numpy(); qs = rh["qs"]
            jq[qs:qs + 3] = b[:3].astype(jq.dtype)
            if rh.get("rot90"):
                jq[qs + 3:qs + 7] = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], jq.dtype)   # the planner placed it turned 90 deg
            self.state_0.joint_q.assign(jq)
            if getattr(self.state_1, "joint_q", None) is not None:
                self.state_1.joint_q.assign(jq)
            print(f"[twin-stock] block settled onto z={base:.1f} ({'towel' if under.sum() > 20 else 'floor'} under it)", flush=True)
            from fold_predicates import block_in_box
            inside = block_in_box(b[:3], rh.get("dims", t1["block_dims"]), {"center": t1["box_center"], "half": t1["box_half"],
                                                            "floor": t1["box_floor"], "rim": t1["box_rim"]})
            print(f"[twin-stock] block released at ({b[0]:.0f},{b[1]:.0f},{b[2]:.0f}) - in the box: {inside}", flush=True)

    def _update_pack_dual(self):
        """Two-hand bundle carry: each hand pins the half of the bundle nearer it while shut."""
        pk = getattr(self, "_packing_dual", None)
        if not pk:
            return
        q = self._qs(); qd = self._qds()
        for h in self._hands:
            tgt = h.get("target")
            if tgt is None or (pk.get("one") and h["name"] != pk["one"]):
                continue
            shut = float(tgt[-1]) <= self.GRIP_CLOSE + 1e-3
            tip, R = self._ee_tip(h)
            key = h["name"]
            if shut:
                if key not in pk["pins"]:
                    if pk.get("grip_cm"):
                        # NATURAL CARRY: each hand holds only a PATCH of fabric, like fingers do,
                        # and the bundle hangs between the two grips. Pinning half the bundle per
                        # hand (the old PIN_RADIUS=32 cm selection, 2500-3400 particles) carried
                        # it as one rigid board.
                        near = np.linalg.norm(q - tip, axis=1) < float(pk["grip_cm"])
                        if pk.get("band_axis") is not None:
                            _al, _ac = int(pk["band_axis"]), 1 - int(pk["band_axis"])
                            near = ((np.abs(q[:, _al] - tip[_al]) < float(pk.get("band_along", 3.0))) & (np.abs(q[:, _ac] - tip[_ac]) < pk["band_half"])
                                    & (np.abs(q[:, 2] - tip[2]) < 8.0))
                        for _ok, (_oi, _ol) in pk["pins"].items():       # never fabric the other hand holds
                            near[_oi] = False
                        idx = np.where(near)[0]
                        if len(idx) == 0:
                            continue
                        pk["pins"][key] = (idx, q[idx] - tip)
                        # a FIRM grip: the held patch is fixed for the solver, as in the folds. Placed
                        # softly once a frame, the patch fought the hanging bundle every substep and
                        # the strain let go all at once at the release - diverged in 4 of 6 two-hand
                        # carries of the quarter-folded towel (s12, p4, s13, s14).
                        self._pin_flags(idx, True)
                        pk.setdefault("firm", set()).add(key)
                        print(f"[twin-stock] carry: {key} hand grips {len(idx)} particles "
                              f"({pk['grip_cm']:.0f}cm patch) - the bundle hangs between the hands", flush=True)
                        continue
                    if pk["split_x"] is None:
                        side = np.ones(len(q), bool)                 # one hand carries the whole bundle
                    else:
                        # split between the two TIPS as they are now (the bundle centre at stage
                        # start gave one hand 34 particles and the other 9848, v17)
                        _tx = [self._ee_tip(H)[0][0] for H in self._hands]
                        _sx = 0.5 * (min(_tx) + max(_tx))
                        side = (q[:, 0] <= _sx) if key == "left" else (q[:, 0] > _sx)
                    near = side & (np.linalg.norm(q - tip, axis=1) < self.PIN_RADIUS)
                    idx = np.where(near)[0]
                    if len(idx) == 0:
                        continue
                    # TRANSLATION-only pin: the two halves ride two different tips, and any
                    # difference in how the tips rotate on the way would shear the bundle
                    # between the hands. Both tips translate by the same (dx, dy), so a pin that
                    # ignores the tip rotation carries the bundle as one rigid piece.
                    local = q[idx] - tip
                    # fabric the descending gripper pressed INTO the table (seen: z=16 under a
                    # 20 cm table top) would be carried 4 cm below the tip and forced through the
                    # box floor at set-down, then spring out and crumple the fold: nothing hangs
                    # below the table surface at pin time
                    n_low = int((q[idx, 2] < self.TABLE_TOP - 0.2).sum())
                    local[:, 2] = np.maximum(local[:, 2], self.TABLE_TOP - 0.2 - tip[2])
                    pk["pins"][key] = (idx, local)
                    print(f"[twin-stock] pack: {key} hand holds {len(idx)} particles ({n_low} lifted to the table surface)", flush=True)
                idx, local = pk["pins"][key]
                q[idx] = tip + local; qd[idx] = 0.0
            elif key in pk["pins"]:
                idx, _ = pk["pins"].pop(key); qd[idx] = 0.0
                if key in pk.get("firm", ()):
                    self._pin_flags(idx, False); pk["firm"].discard(key)
                print(f"[twin-stock] pack: {key} hand released", flush=True)
        self._put_q(q); self._put_qd(qd)

    def _label_move_poses(self, ga, gb, pa, pb):
        """One label move on the single-arm path."""
        poses, fold = self._label_move_arc(ga, gb, pa, pb)
        self._fold = fold
        return poses

    def _coordinated_arc(self, assign):
        """Carry ONE edge with two hands: the grasps swing RIGIDLY about a shared crease, on a
        shared clock, so the fabric between them keeps its length.

        Simultaneous is not the same as coordinated. Two independently built arcs each get their
        own travel, apex and carry speed - the first two-hand run had the hands at carry x5.0 and
        x4.0, so one was still moving when the other had stopped - and the distance between the
        grasps wanders through the carry, which gathers the held edge instead of laying it over.
        Here both hands are one rigid body rotating about the crease: same angle at every instant,
        each hand lifting to its own radius, the separation along the crease preserved exactly.
        The small difference between where the rotation ends and each hand's own drop is blended
        in over the end of the swing (smoothstep), so the moves still land where the planner asked.
        """
        q = self._qs()
        top = self.TABLE_TOP
        G = [q[int(mv[2])].astype(float) for _, mv in assign]
        D = [q[int(mv[3])].astype(float) for _, mv in assign]
        _ov = float(_os.environ.get("TWIN_FOLD_OVERSHOOT_CM", "3"))
        if _ov:                      # carry past the target; the flap settles back onto the line
            for _g, _d in zip(G, D):
                _t = float(np.linalg.norm(_d[:2] - _g[:2]))
                if _t > 1e-6:
                    _d[:2] += (_d[:2] - _g[:2]) / _t * _ov
        disp = np.mean([d[:2] - g[:2] for g, d in zip(G, D)], axis=0)
        travel = float(np.linalg.norm(disp))
        if travel < 1.0:
            return None                                   # nowhere to fold: let the caller fall back
        # A rigid pair only makes sense when the two hands carry the SAME edge: displacements
        # roughly parallel and the grasps staying about as far apart as they started. The label
        # planner's own winning step crosses (each armpit onto the OPPOSITE hem corner), so the
        # hands must diverge - forcing that onto one rotation dragged the fabric along paths it
        # could not follow and VBD blew up at t=18s, where independent arcs completed the move.
        d1, d2 = D[0][:2] - G[0][:2], D[1][:2] - G[1][:2]
        n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
        cosang = float(d1 @ d2) / max(n1 * n2, 1e-6)
        sep0 = float(np.linalg.norm(G[0][:2] - G[1][:2]))
        sepN = float(np.linalg.norm(D[0][:2] - D[1][:2]))
        ratio = sepN / max(sep0, 1e-6)
        # The ANGLE is the discriminator: a crossing pair has its displacements pointing opposite
        # ways. The separation ratio is only a sanity bound - a real hem-to-shoulder fold moves
        # both hands the same way while the garment tapers (22cm apart at the hem, 9cm at the
        # shoulders), and that is still one edge carried by two hands.
        if cosang < 0.5 or not (0.35 <= ratio <= 2.0):
            print(f"[twin-stock] not a rigid pair (displacements {np.degrees(np.arccos(np.clip(cosang, -1, 1))):.0f}deg "
                  f"apart, grasp separation x{ratio:.1f}) - independent arcs on one clock", flush=True)
            return None
        u_dir = disp / travel                             # the fold direction
        axis_dir = np.array([-u_dir[1], u_dir[0]])        # the crease, horizontal and across it
        mid = np.mean([g[:2] for g in G], axis=0) + 0.5 * disp
        ax = 0 if abs(u_dir[0]) >= abs(u_dir[1]) else 1

        def rot(p, th):
            """swing a point about the crease line by th (0 = as grasped, pi = laid over)"""
            v = p[:2] - mid
            h = float(v @ u_dir)                          # across the crease - this is what swings
            along = float(v @ axis_dir)                   # along the crease - untouched
            z = float(p[2] - top)
            xy = mid + u_dir * (h * np.cos(th) - z * np.sin(th)) + axis_dir * along
            return np.array([xy[0], xy[1], top + h * np.sin(th) + z * np.cos(th)])

        # one clock for both hands: the SLOWEST carry governs, so neither hand runs ahead
        c = 1.0
        for _h, mv in assign:
            g = q[int(mv[2])]
            if hasattr(self, "_carry_slow_for"):
                c = max(c, self._carry_slow_for((float(g[0]), float(g[1])), axis=ax))
        n = 10
        seg = 7.0 * c / n
        streams, folds = {}, []
        for (hand, mv), g, d in zip(assign, G, D):
            ga, gb = mv[0], mv[1]
            near = np.hypot(q[:, 0] - g[0], q[:, 1] - g[1]) < _FOLD_PIN
            local_top = float(np.percentile(q[near, 2], 90.0)) if near.sum() > 10 else top
            z_pinch = max(top - 0.3 - _GRASP_DZ, local_top - 0.8)
            lift = max(0.0, local_top - top - 0.8)
            poses = [(2.0, g[0], g[1], top + 11.0, self.GRIP_OPEN),
                     (2.0, g[0], g[1], z_pinch, self.GRIP_OPEN),
                     (2.5, g[0], g[1], z_pinch, self.GRIP_CLOSE)]
            end = rot(g, np.pi)
            for k in range(1, n + 1):
                th = np.pi * k / n
                u = k / n
                p = rot(g, th) + (u * u * (3.0 - 2.0 * u)) * (d - end)
                z = float(p[2]) + lift
                if k == n:
                    z = top + 2.0 + lift
                poses.append((seg, float(p[0]), float(p[1]), z, self.GRIP_CLOSE))
            _back = float(_os.environ.get("TWIN_RELEASE_BACK_CM", "5"))
            _bv = (d[:2] - g[:2]) / max(float(np.linalg.norm(d[:2] - g[:2])), 1e-6) * _back
            _rz = top + float(_os.environ.get("TWIN_RELEASE_Z_CM", "2")) + lift
            _up = float(_os.environ.get("TWIN_RELEASE_UP_CM", "0"))
            poses += [(1.5, d[0], d[1], _rz, self.GRIP_OPEN)]
            if _up > 0:
                poses += [(2.0, d[0], d[1], _rz + _up, self.GRIP_OPEN)]
            poses += [(1.5, d[0] - _bv[0], d[1] - _bv[1], _rz + _up, self.GRIP_OPEN),
                      (2.0, d[0] - _bv[0], d[1] - _bv[1], top + 14.0, self.GRIP_OPEN)]
            streams[hand["name"]] = poses
            folds.append({"axis": ax, "line": float(mid[ax]), "grab": g[:2].copy(), "drop": d[:2].copy(),
                          "start": None, "idx": None, "two_point": _os.environ.get("TWIN_FOLD_TWO_POINT", "0") == "1", "hand": hand,
                          "pin_cm": float(_os.environ.get("TWIN_FOLD_BAND_CM", _FOLD_PIN)),
                "span_cm": float(_os.environ.get("TWIN_FOLD_SPAN", "14"))})
            self._fold_stack = getattr(self, "_fold_stack", [])
            self._fold_stack.append({"axis": ax, "line": float(mid[ax]),
                                     "grab": g[:2].copy(), "drop": d[:2].copy()})
        sep0 = float(np.linalg.norm(G[0][:2] - G[1][:2]))
        sepN = float(np.linalg.norm(np.array(streams[assign[0][0]["name"]][-2][1:3])
                                    - np.array(streams[assign[1][0]["name"]][-2][1:3])))
        print(f"[twin-stock] coordinated fold: crease at {'x' if ax == 0 else 'y'}={mid[ax]:.0f}cm, "
              f"travel {travel:.0f}cm, both hands on one clock (carry x{c:.1f}); "
              f"grasp separation {sep0:.0f}cm -> {sepN:.0f}cm", flush=True)
        return streams, folds

    def _hold_poses(self, hand, label, particle, duration):
        """One hand PRESSES AND STAYS while the other folds.

        Both hands pulling the near edge across a free sheet does not fold it - the whole towel
        slid and rolled into a 40x4cm sausage, because nothing held the far half down while the
        near half came over. A human folds a towel with one hand on the cloth and one hand moving;
        this is that hand. The grasp machinery does the rest: a shut gripper that does not move
        pins the particles under it, which is exactly a hold-down.
        """
        q = self._qs()
        top = self.TABLE_TOP
        g = q[int(particle)].astype(float)
        near = np.hypot(q[:, 0] - g[0], q[:, 1] - g[1]) < _FOLD_PIN
        local_top = float(np.percentile(q[near, 2], 90.0)) if near.sum() > 10 else top
        z_press = max(top - 0.3 - _GRASP_DZ, local_top - 0.8)
        poses = [(2.0, g[0], g[1], top + 11.0, self.GRIP_OPEN),
                 (2.0, g[0], g[1], z_press, self.GRIP_OPEN),
                 (2.5, g[0], g[1], z_press, self.GRIP_CLOSE),
                 (max(1.0, float(duration)), g[0], g[1], z_press, self.GRIP_CLOSE),
                 (1.5, g[0], g[1], z_press, self.GRIP_OPEN),
                 (2.0, g[0], g[1], top + 14.0, self.GRIP_OPEN)]
        fold = {"axis": 0, "line": float(g[0]), "grab": g[:2].copy(), "drop": g[:2].copy(),
                "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False, "hand": hand}
        print(f"[twin-stock] hold: {hand['name']} hand presses '{label}' at "
              f"({g[0]:.0f},{g[1]:.0f}) for {duration:.0f}s while the other hand folds", flush=True)
        return poses, fold

    def _label_move_poses_multi(self, step):
        """A planner step executed the way the planner scored it: every pinch in the step happens
        AT THE SAME TIME, one hand each.

        fold_env has always rolled a multi-pinch FoldAction out simultaneously (one displacement
        field); with a single arm the twin had to serialise it into two carries, so what we
        executed was never what the search ranked. Each grab goes to the arm whose base is nearer
        it, and if that would put both grabs on one arm the leftmost grab is handed to the left.
        """
        hands = self._hands
        if not hands:
            a, b, pa, pb = step[0]
            return self._label_move_poses(a, b, pa, pb)
        q = self._qs()
        # a move that goes nowhere is a HOLD: that hand presses there while the other one folds
        holds = [mv for mv in step
                 if float(np.linalg.norm(q[int(mv[3])][:2] - q[int(mv[2])][:2])) < 2.0]
        if holds and len(step) - len(holds) >= 1:
            carries = [mv for mv in step if mv not in holds]
            left = next(h for h in hands if h["name"] == "left")
            right = next(h for h in hands if h["name"] == "right")
            cg = q[int(carries[0][2])][:2].astype(float)
            cost = lambda h: float(np.linalg.norm(cg - np.asarray(h["base_cm"][:2], float)))
            carry_hand = left if cost(left) <= cost(right) else right
            hold_hand = right if carry_hand is left else left
            cp, cf = self._label_move_arc(carries[0][0], carries[0][1], carries[0][2], carries[0][3],
                                          hand=carry_hand)
            span = sum(float(x[0]) for x in cp)
            hp, hf = self._hold_poses(hold_hand, holds[0][0], holds[0][2], span - 8.0)
            self._fold = {"hands": [cf, hf], "axis": cf["axis"], "line": cf["line"],
                          "grab": cf["grab"], "drop": cf["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            return {carry_hand["name"]: cp, hold_hand["name"]: hp}
        if len(step) < 2:
            # one pinch: give it to the NEARER arm (the legacy path would always use hand 0, which
            # on the rig is the right arm even when the grab sits in the left arm's workspace),
            # and keep the dict path so the idle hand is padded onto the same time base.
            a, b, pa, pb = step[0]
            g = q[int(pa)][:2].astype(float)
            h = min(hands, key=lambda H: float(np.linalg.norm(g - np.asarray(H["base_cm"][:2], float))))
            poses, fold = self._label_move_arc(a, b, pa, pb, hand=h)
            self._fold = {"hands": [fold], "axis": fold["axis"], "line": fold["line"],
                          "grab": fold["grab"], "drop": fold["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            return {h["name"]: poses}
        left = next(h for h in hands if h["name"] == "left")
        right = next(h for h in hands if h["name"] == "right")
        m0, m1 = step[0], step[-1]
        g0, g1 = q[int(m0[2])][:2].astype(float), q[int(m1[2])][:2].astype(float)
        cost = lambda h, g: float(np.linalg.norm(g - np.asarray(h["base_cm"][:2], float)))
        # STACKED GRABS -> carry + hold. After a half fold the planner's two labels (the top and
        # bottom layer's corners) lie on the same spot, and two hands on one point is not a
        # two-hand carry: the tips met at 1.6 cm and the far arm reached 76 cm across the table
        # (IK lag 16 cm, jaws closed off the fabric). One pinch on the corner STACK takes both
        # layers; the other hand presses the stationary half next to the fold line so the
        # bundle does not slide - the hold-and-fold that a person does.
        _stack_cm = float(_os.environ.get("TWIN_STACK_GRAB_CM", "6"))
        if float(np.linalg.norm(g0 - g1)) < _stack_cm:
            # EDGE-SPANNING carry. The planner's two labels (top and bottom layer corners) sit on
            # one spot after a half fold; one hand takes that corner stack and the other takes
            # the OTHER END of the same edge, and both carry along the same travel vector on one
            # clock. (Carry + hold-down was tried: the holding hand sat on the half the flap
            # lands on, so the flap never reached its corner - v16 step 2 fold goal 1/6.)
            d0 = q[int(m0[3])][:2].astype(float)
            trav = d0 - g0; tn = float(np.linalg.norm(trav)) or 1.0
            e = np.array([-trav[1], trav[0]]) / tn                      # along the carried edge
            lo_, hi_ = self._cloth_bbox()
            c_ = np.array([(lo_[0] + hi_[0]) / 2.0, (lo_[1] + hi_[1]) / 2.0])
            if float(e @ (c_ - g0)) < 0:
                e = -e                                                   # toward the cloth, not away
            rel = q[:, :2] - g0
            on_edge = np.abs(rel @ (trav / tn)) < 3.0                     # within 3 cm of the edge line
            ext = float(np.percentile(rel[on_edge] @ e, 97.0)) if on_edge.sum() > 10 else 15.0
            frac = float(_os.environ.get("TWIN_EDGE_SPAN_FRAC", "0.8"))
            gB = g0 + e * ext * frac; dB = d0 + e * ext * frac
            pB = int(np.argmin(np.linalg.norm(q[:, :2] - gB, axis=1)))
            pBd = int(np.argmin(np.linalg.norm(q[:, :2] - dB, axis=1)))
            handA = left if cost(left, g0) <= cost(right, g0) else right
            handB = right if handA is left else left
            carry = 1.0
            if hasattr(self, "_carry_slow_for"):
                ax0 = 0 if abs(trav[0]) >= abs(trav[1]) else 1
                carry = max(self._carry_slow_for((float(g0[0]), float(g0[1])), axis=ax0),
                            self._carry_slow_for((float(gB[0]), float(gB[1])), axis=ax0))
            if _os.environ.get("TWIN_STACK_CARRY"):
                carry = float(_os.environ["TWIN_STACK_CARRY"])
            if _os.environ.get("TWIN_STACK_ONE_HAND", "1") == "1":
                # one hand on the corner stack with a slightly wider pinch (the folded edge is
                # 4 sheets thick, a jaw holds a few cm of it); the other hand stays home
                _band = float(_os.environ.get("TWIN_STACK_BAND_CM", "6"))
                _old = _os.environ.get("TWIN_FOLD_BAND_CM")
                _os.environ["TWIN_FOLD_BAND_CM"] = str(_band)
                try:
                    pa_, fa = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=handA, carry=carry)
                finally:
                    if _old is None:
                        del _os.environ["TWIN_FOLD_BAND_CM"]
                    else:
                        _os.environ["TWIN_FOLD_BAND_CM"] = _old
                self._fold = {"hands": [fa], "axis": fa["axis"], "line": fa["line"], "grab": fa["grab"], "drop": fa["drop"],
                              "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
                print(f"[twin-stock] stacked grabs ({np.linalg.norm(g0 - g1):.1f}cm apart): {handA['name']} hand alone takes the "
                      f"corner stack at ({g0[0]:.0f},{g0[1]:.0f}) with a {_band:.0f}cm pinch, carry x{carry:.1f}", flush=True)
                return {handA["name"]: pa_}
            pa_, fa = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=handA, carry=carry)
            pb_, fb = self._label_move_arc(f"{m0[0]}+edge", f"{m0[1]}+edge", pB, pBd, hand=handB, carry=carry)
            self._fold = {"hands": [fa, fb], "axis": fa["axis"], "line": fa["line"],
                          "grab": fa["grab"], "drop": fa["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            print(f"[twin-stock] stacked grabs ({np.linalg.norm(g0 - g1):.1f}cm apart): {handA['name']} hand takes the corner "
                  f"stack at ({g0[0]:.0f},{g0[1]:.0f}), {handB['name']} hand the edge {ext * frac:.0f}cm along at ({gB[0]:.0f},{gB[1]:.0f}); "
                  f"edge {ext:.0f}cm", flush=True)
            return {handA["name"]: pa_, handB["name"]: pb_}
        if len(step) < 2:
            # one pinch: give it to the NEARER arm (the legacy path would always use hand 0, which
            # on the rig is the right arm even when the grab sits in the left arm's workspace),
            # and keep the dict path so the idle hand is padded onto the same time base.
            a, b, pa, pb = step[0]
            g = q[int(pa)][:2].astype(float)
            h = min(hands, key=lambda H: float(np.linalg.norm(g - np.asarray(H["base_cm"][:2], float))))
            poses, fold = self._label_move_arc(a, b, pa, pb, hand=h)
            self._fold = {"hands": [fold], "axis": fold["axis"], "line": fold["line"],
                          "grab": fold["grab"], "drop": fold["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            return {h["name"]: poses}
        left = next(h for h in hands if h["name"] == "left")
        right = next(h for h in hands if h["name"] == "right")
        m0, m1 = step[0], step[-1]
        g0, g1 = q[int(m0[2])][:2].astype(float), q[int(m1[2])][:2].astype(float)
        cost = lambda h, g: float(np.linalg.norm(g - np.asarray(h["base_cm"][:2], float)))
        # STACKED GRABS -> carry + hold. After a half fold the planner's two labels (the top and
        # bottom layer's corners) lie on the same spot, and two hands on one point is not a
        # two-hand carry: the tips met at 1.6 cm and the far arm reached 76 cm across the table
        # (IK lag 16 cm, jaws closed off the fabric). One pinch on the corner STACK takes both
        # layers; the other hand presses the stationary half next to the fold line so the
        # bundle does not slide - the hold-and-fold that a person does.
        _stack_cm = float(_os.environ.get("TWIN_STACK_GRAB_CM", "6"))
        if float(np.linalg.norm(g0 - g1)) < _stack_cm:
            d0 = q[int(m0[3])][:2].astype(float)
            carry_hand = left if cost(left, g0) <= cost(right, g0) else right
            hold_hand = right if carry_hand is left else left
            trav = d0 - g0; tn = float(np.linalg.norm(trav)) or 1.0
            lo_, hi_ = self._cloth_bbox()
            c_ = np.array([(lo_[0] + hi_[0]) / 2.0, (lo_[1] + hi_[1]) / 2.0])
            # press point: the cloth centre pushed 6 cm along the travel (onto the half that stays)
            press = c_ + trav / tn * 6.0
            pi = int(np.argmin(np.linalg.norm(q[:, :2] - press, axis=1)))
            cp, cf = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=carry_hand)
            span = sum(float(x[0]) for x in cp)
            hp, hf = self._hold_poses(hold_hand, "stationary half", pi, span - 8.0)
            self._fold = {"hands": [cf, hf], "axis": cf["axis"], "line": cf["line"],
                          "grab": cf["grab"], "drop": cf["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            print(f"[twin-stock] stacked grabs ({np.linalg.norm(g0 - g1):.1f}cm apart): {carry_hand['name']} hand carries the "
                  f"corner stack, {hold_hand['name']} hand holds at ({press[0]:.0f},{press[1]:.0f})", flush=True)
            return {carry_hand["name"]: cp, hold_hand["name"]: hp}
        # Pick the pairing with the shorter total reach rather than ordering the grabs by x:
        # two grabs that differ mainly in Y (a rotated garment) make an x-order a coin flip, and
        # the losing half sends the arms across each other.
        # minimise the LONGER reach of the two (the sum let one arm take an 80 cm grab so the
        # other could take a 34 cm one; the long reach is what lags and crosses) ...
        direct = max(cost(left, g0), cost(right, g1))
        swapped = max(cost(left, g1), cost(right, g0))
        # ... and KEEP EACH HAND WHERE IT IS between steps. Assigning each step independently made
        # the two hands swap rows between the gate fold's two steps and cross in transit (tips
        # 0.7 cm, 123 overlapping samples, s6). Continuity is measured from the CURRENT tips and
        # only breaks the tie when both options are within reach.
        _reach = float(_os.environ.get("TWIN_MAX_REACH_CM", "82"))
        _tipL, _tipR = self._ee_tip(left)[0][:2], self._ee_tip(right)[0][:2]
        tcost = lambda t, g: float(np.linalg.norm(np.asarray(g, float) - np.asarray(t, float)))
        direct_ok = max(cost(left, g0), cost(right, g1)) < _reach
        swapped_ok = max(cost(left, g1), cost(right, g0)) < _reach
        if direct_ok and swapped_ok:
            direct = max(tcost(_tipL, g0), tcost(_tipR, g1))
            swapped = max(tcost(_tipL, g1), tcost(_tipR, g0))
        elif direct_ok != swapped_ok:
            direct, swapped = (0.0, 1.0) if direct_ok else (1.0, 0.0)
        assign = [(left, m0), (right, m1)] if direct <= swapped else [(left, m1), (right, m0)]
        _sep = float(_os.environ.get("TWIN_MIN_HAND_SEP_CM", "18"))
        d0_, d1_ = q[int(m0[3])][:2].astype(float), q[int(m1[3])][:2].astype(float)

        def _cross(p1, p2, p3, p4):
            # do segments p1-p2 and p3-p4 cross (2D)?
            def orient(a, b, c):
                return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
            return (o1 * o2 < 0) and (o3 * o4 < 0)
        if _cross(g0, d0_, g1, d1_):
            # CROSSING carries (a shirt's two sides-in diagonals): done together, the two pinned
            # flaps pass through each other and the arms cross - VBD went NaN 8 s into it
            # (shirt-only test). One after the other, each on its own nearer hand.
            hA = left if cost(left, g0) <= cost(right, g0) else right
            hB = right if hA is left else left
            pa_, fa = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=hA)
            self._fold = {"hands": [fa], "axis": fa["axis"], "line": fa["line"], "grab": fa["grab"], "drop": fa["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            self._pending_gen = getattr(self, "_pending_gen", None) or []
            self._pending_gen.append((f"{m1[0]}->{m1[1]} (second pinch, {hB['name']} hand)", lambda mv=m1, h=hB: self._serial_second(mv, h)))
            print(f"[twin-stock] the two carries CROSS: {hA['name']} hand first, then {hB['name']} hand (re-grounded after the first)", flush=True)
            return {hA["name"]: pa_}
        # NO CROSS-REACH: a hand whose DROP lies past the centre line toward the other arm's base
        # (both hands carrying the right edge all the way to the left, towelB half_x in s4) ends
        # with its forearm over the other arm - 64 of 1256 samples overlapping. That edge is a
        # one-arm job for the arm on the drop side, two pinches in turn.
        _cr = float(_os.environ.get("TWIN_CROSS_REACH_CM", "12"))
        _mid = 0.5 * (float(left["base_cm"][0]) + float(right["base_cm"][0]))
        _crossing = any((h is right and float(q[int(mv[3])][0]) < _mid - _cr) or (h is left and float(q[int(mv[3])][0]) > _mid + _cr)
                        for h, mv in assign)
        if _crossing:
            dxs = [float(q[int(m[3])][0]) for m in (m0, m1)]
            hA = left if np.mean(dxs) < _mid else right
            pa_, fa = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=hA)
            self._fold = {"hands": [fa], "axis": fa["axis"], "line": fa["line"], "grab": fa["grab"], "drop": fa["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            self._pending_gen = getattr(self, "_pending_gen", None) or []
            self._pending_gen.append((f"{m1[0]}->{m1[1]} (second pinch, {hA['name']} hand)", lambda mv=m1, h=hA: self._serial_second(mv, h)))
            print(f"[twin-stock] drops at x={dxs[0]:.0f},{dxs[1]:.0f} would make an arm reach across the other: "
                  f"{hA['name']} hand does both moves in turn (second pinch re-grounded after the first)", flush=True)
            return {hA["name"]: pa_}
        if min(float(np.linalg.norm(g0 - g1)), float(np.linalg.norm(d0_ - d1_))) < _sep:
            # the two grabs (or the two drops) are closer than two grippers side by side: one
            # hand does both moves in turn - two hands there would pass through each other
            hA = left if cost(left, g0) <= cost(right, g0) else right
            pa_, fa = self._label_move_arc(m0[0], m0[1], m0[2], m0[3], hand=hA)
            self._fold = {"hands": [fa], "axis": fa["axis"], "line": fa["line"], "grab": fa["grab"], "drop": fa["drop"],
                          "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
            self._pending_gen = getattr(self, "_pending_gen", None) or []
            self._pending_gen.append((f"{m1[0]}->{m1[1]} (second pinch, {hA['name']} hand)", lambda mv=m1, h=hA: self._serial_second(mv, h)))
            print(f"[twin-stock] grabs {np.linalg.norm(g0 - g1):.0f}cm apart (< {_sep:.0f}cm hand separation): "
                  f"{hA['name']} hand does both moves in turn (second pinch re-grounded after the first)", flush=True)
            return {hA["name"]: pa_}
        got = None
        if _os.environ.get("TWIN_COORD", "rigid") == "rigid":
            got = self._coordinated_arc(assign)
        if got is not None:
            streams, folds = got
        else:
            # TWIN_COORD=independent, or a pair that is not carrying one edge: each hand flies
            # its own arc, but still on ONE clock - the slowest carry governs both, so neither
            # hand lets go while the other is still moving.
            carry = 1.0
            for h, (a, b, pa, pb) in assign:
                g = q[int(pa)]
                if hasattr(self, "_carry_slow_for"):
                    ax0 = 0 if abs(float(q[int(pb)][0] - g[0])) >= abs(float(q[int(pb)][1] - g[1])) else 1
                    carry = max(carry, self._carry_slow_for((float(g[0]), float(g[1])), axis=ax0))
            streams, folds = {}, []
            for h, (a, b, pa, pb) in assign:
                poses, fold = self._label_move_arc(a, b, pa, pb, hand=h, carry=carry)
                streams[h["name"]] = poses
                folds.append(fold)
        self._fold = {"hands": folds, "axis": folds[0]["axis"], "line": folds[0]["line"],
                      "grab": folds[0]["grab"], "drop": folds[0]["drop"],
                      "start": None, "idx": None, "pin_cm": _FOLD_PIN, "two_point": False}
        return streams

    def _stack_poses(self):
        """Pick the folded ACTIVE garment up by its centre and lay it on top of another garment.

        Same kinematic carry as the box pack (_update_grasp pins what the jaws shut on). The
        target is the OTHER garment's bundle as it lies NOW: its 3/97 footprint centre and its
        top surface, so the release height follows however tall that bundle actually is.
        """
        gi = self._garment_index(_STACK_ON)
        qo = self._q_of(gi)
        olo, ohi = np.percentile(qo, 3.0, axis=0), np.percentile(qo, 97.0, axis=0)
        tx, ty = 0.5 * (olo[0] + ohi[0]), 0.5 * (olo[1] + ohi[1])
        ttop = float(np.percentile(qo[:, 2], 97.0))
        lo, hi = self._cloth_bbox()
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        top = self.TABLE_TOP
        qz = self._qs()[:, 2]
        my_top, my_bot = float(np.percentile(qz, 97.0)), float(np.percentile(qz, 3.0))
        thick = max(my_top - my_bot, 1.0)             # how tall MY bundle is
        # Pinch at the bundle's TOP, as the fold grasp does. The first version pinched at table
        # level, so the pinned bundle rode ABOVE the fingertips; its release height then added the
        # thickness a second time and the wad fell ~7cm onto the other bundle (2026-09-11: "it
        # merges with the other one"). With the pin at the top the bundle hangs BELOW the tip.
        # TWIN_STACK_PINCH=top (default; the sim's contact-friendly pinch) | table (a real
        # parallel jaw must close near the table to hold the whole bundle, as the box pack did)
        if __import__("os").environ.get("TWIN_STACK_PINCH", "top") == "table":
            z_grasp = top - _GRASP_DZ
        else:
            z_grasp = max(top - _GRASP_DZ, my_top - 0.8)
        bottom_off = my_bot - z_grasp                 # bundle bottom relative to the tip when pinned
        z_rel = ttop + 1.0 - bottom_off               # bottom lands 1cm over the target's top
        self._stack_target = {"gi": gi, "label": self._garments[gi]["label"], "xy": (tx, ty),
                              "half": (0.5 * (ohi[0] - olo[0]), 0.5 * (ohi[1] - olo[1])), "top": ttop}
        print(f"[twin-stock] stack: carry my {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}x{thick:.1f}cm "
              f"bundle from ({cx:.0f},{cy:.0f}) onto '{self._garments[gi]['label']}' "
              f"{ohi[0] - olo[0]:.0f}x{ohi[1] - olo[1]:.0f}cm at ({tx:.0f},{ty:.0f}), "
              f"top {ttop - top:.1f}cm above the table; pinch at {z_grasp - top:+.1f}cm, "
              f"release with the bottom {z_rel + bottom_off - ttop:.1f}cm over it", flush=True)
        return [
            (2.0, cx, cy, top + 11.0, self.GRIP_OPEN),
            (2.0, cx, cy, z_grasp, self.GRIP_OPEN),
            (2.0, cx, cy, z_grasp, self.GRIP_HARD),
            (3.0, cx, cy, ttop + thick + 12.0, self.GRIP_HARD),   # clear the target's top
            (4.0, tx, ty, ttop + thick + 12.0, self.GRIP_HARD),
            (3.0, tx, ty, z_rel + 4.0, self.GRIP_HARD),
            (3.0, tx, ty, z_rel, self.GRIP_HARD),                 # slow final approach
            (2.0, tx, ty, z_rel, self.GRIP_OPEN),
            (3.0, tx, ty, z_rel + 25.0, self.GRIP_OPEN),
        ]

    def _switch_garment(self, gi, names):
        """Make garment gi the ACTIVE one and forget every per-garment cache."""
        g = self._garments[gi]
        self._active = gi
        for a in ("_flat_q", "_sheet", "_sleeve_ids", "_body_idx", "_body_cx", "_body_half_w",
                  "_sleeve_set", "_fold_only", "_last_span_cm"):
            if hasattr(self, a):
                delattr(self, a)
        self._fold_stack = []
        self._fold = None
        self._pin_idx = None
        self._plan_step = 0
        self._shirt_step = 0
        _PLAN[:] = list(names)                     # _plan_poses reads the module-level list
        self._plan_names = names
        lo, hi = self._cloth_bbox()
        print(f"[twin-stock] === active garment -> '{g['label']}' "
              f"({hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm): " + " > ".join(names), flush=True)

    def _garment_fold_stage(self, gi, names):
        """One fold of garment gi's plan (switching to it first when it is not active)."""
        if self._active != gi or getattr(self, "_plan_names", None) is not names:
            self._switch_garment(gi, names)
        return self._plan_poses()

    def _extend_plan(self):
        """Queue the next phase when the current pose list runs out."""
        # Physical folds only shrink the bundle 10-15% per pass and misfire often, so it takes
        # several passes to bring it near the box's 24x25cm interior. Kept frictional on purpose.
        if _os.environ.get("TWIN_NO_PLAN", "0") == "1":
            # settle-and-exit: build the scene, let the cloth rest, record, leave. Without this a
            # run with no plan fell through to the stock three-panel shirt choreography.
            stages = []
        elif _LABEL_MOVES:
            if getattr(self, "_hands", None):
                # one stage per PLANNER STEP: both pinches of a two-handed step run together
                stages = []
                for i, st in enumerate(_LABEL_STEPS):
                    if st[0][0] == "task":
                        _t = st[0][1]
                        # "pack[:dx,dy]" and "pick_place:block[:dx,dy[,z_on]]": the planner's placement
                        # inside the box, cm from the box centre (z_on: height of what is already under it)
                        _parts = _t.split(":")
                        _numpart = next((p_ for p_ in _parts[1:] if "," in p_), None)
                        _nums = [float(v) for v in _numpart.split(",")] if _numpart else []
                        _args = dict(zip(("dx", "dy", "z_on"), _nums))
                        _args["rot90"] = "rot90" in _parts
                        _oname = next((p_ for p_ in _parts[1:] if "," not in p_ and p_ != "rot90"), None)
                        _args["name"] = _oname
                        def _with(fn, _args=_args, _gname=_oname):
                            def _run():
                                self._task_args = _args
                                if _gname and _gname in [g["label"] for g in (self._garments or [])] and fn is not self._pick_place_poses:
                                    self._activate_garment(_gname)
                                return fn()
                            return _run
                        if _parts[0] == "pack":
                            stages.append((f"task {i + 1}: pack {_oname or 'towel'} into the box {_args}", _with(self._pack_poses_dual)))
                        elif _parts[0] == "aside":
                            stages.append((f"task {i + 1}: set {_oname} aside", _with(self._aside_poses)))
                        elif _parts[0] == "pick_place":
                            stages.append((f"task {i + 1}: {_t}", _with(self._pick_place_poses)))
                        else:
                            print(f"[twin-stock] unknown task step '{_t}' skipped", flush=True)
                        continue
                    if _os.environ.get("TWIN_ONE_HAND") and self._hands:
                        # SINGLE-ARM BASELINE: the same plan, every pinch in turn on one hand
                        for k, mv in enumerate(st):
                            stages.append((f"label move {i + 1}{'abcd'[k]} (one hand): {mv[0]}->{mv[1]}",
                                           (lambda mv=mv: self._label_move_one(mv))))
                        continue
                    _gn = _LABEL_STEP_G[i] if "_LABEL_STEP_G" in globals() and i < len(_LABEL_STEP_G) else None
                    def _lbl(st=st, _gn=_gn):
                        if _gn:
                            self._activate_garment(_gn)
                        # a carry stage's pins must not survive into a fold: after the set-aside
                        # the left jaws closing on towelA's corner pinned the WHOLE towel (3486
                        # particles) and the "fold" dragged it 40 cm across the table (Task-2 v1)
                        self._packing_dual = None; self._rigid_hold = None
                        return self._label_move_poses_multi(st)
                    stages.append(((f"{_gn}: " if _gn else "") + f"label move {i + 1}: " + " + ".join(f"{a}->{b}" for a, b, _, _ in st), _lbl))
            else:
                stages = [(f"label move {i + 1}: {a}->{b}", (lambda pa=pa, pb=pb, a=a, b=b: self._label_move_poses(a, b, pa, pb)))
                          for i, (a, b, pa, pb) in enumerate(_LABEL_MOVES)]
        elif _PLANS and getattr(self, "_garments", None):
            # several garments, one run: every fold of every plan, garment by garment
            stages = []
            for _sel, _names in _PLANS:
                _gi = self._garment_index(_sel)
                _lab = self._garments[_gi]["label"]
                for _k, _n in enumerate(_names):
                    stages.append((f"{_lab}: fold {_k + 1} {_n}",
                                   (lambda gi=_gi, names=_names: self._garment_fold_stage(gi, names))))
        elif _UNFOLD_STACK and not _PLAN:
            # a garment a PERSON folded: undo the fit's reconstructed folds, last first. The
            # count is fixed at load time - the stack shrinks as records pop, and counting it
            # live ended the plan after the first unfold.
            if not hasattr(self, "_unfold_total"):
                self._unfold_total = len(getattr(self, "_fold_stack", []))
            stages = [(f"unfold {i + 1}", self._unfold_poses) for i in range(self._unfold_total)]
        elif _PLAN:
            # a sequence fold_search.py validated headless, replayed here by the arm
            stages = [(f"fold {i + 1} {n}", self._plan_poses) for i, n in enumerate(_PLAN)]
            if _os.environ.get("TWIN_UNFOLD", "0") == "1":
                stages += [(f"unfold {i + 1}", self._unfold_poses) for i in range(len(_PLAN))]
        elif _FOLD_STYLE == "shirt":
            # left panel in, right panel in, hem to collar -> a folded rectangle in 3 moves
            stages = [("fold 1 left panel", self._shirt_folds),
                      ("fold 2 right panel", self._shirt_folds),
                      ("fold 3 in half", self._shirt_folds)]
            if _os.environ.get("TWIN_UNFOLD", "1") == "1":
                stages += [("unfold 1", self._unfold_poses), ("unfold 2", self._unfold_poses),
                           ("unfold 3", self._unfold_poses)]
        else:
            stages = [("fold 2", self._fold_poses), ("fold 3", self._fold_poses),
                      ("unfold 1", self._unfold_poses), ("unfold 2", self._unfold_poses)]
        if _os.environ.get("TWIN_PACK", "0") == "1":
            stages.append(("pack into box", self._pack_poses))
        if _STACK_ON and len(getattr(self, "_garments", [])) > 1:
            stages.append((f"stack on {_STACK_ON}", self._stack_poses))
        stages = tuple(stages)
        i = getattr(self, "_plan_stage", 0)
        if i >= len(stages) and not getattr(self, "_pending_gen", None):
            if not getattr(self, "_plan_done_logged", False):
                self._plan_done_logged = True   # the last fold is otherwise never measured
                lo, hi = self._cloth_bbox()
                print(f"[twin-stock] plan complete: cloth "
                      f"{hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm", flush=True)
                if getattr(self, "_tip_min_cm", None) is not None:
                    print(f"[twin-stock] closest the two tips came: {self._tip_min_cm:.1f}cm "
                          f"at t={self._tip_min_t:.1f}s", flush=True)
                if getattr(self, "_arm_min", None):
                    print(f"[twin-stock] closest the two ARMS came (link surfaces): {self._arm_min[0]:.1f}cm at t={self._arm_min[1]:.1f}s; "
                          f"{self._arm_hits} of {self._arm_samples} samples overlapping", flush=True)
                if getattr(self, "_task1", None) and getattr(self, "_towel", None) and _LABELS_FILE:
                    self._task1_audit("FINAL")
                    try:
                        _al = getattr(self, "_task1_log", [])
                        _fe = next((e for e in reversed(_al) if "label move" in e["tag"]), None)
                        _ln = next((e for e in _al if "FINAL" in e["tag"]), None)
                        if _fe and _ln:
                            print(f"[twin-stock] TASK 1 result: fold stage left the fold goal at {_fe['fold_sat'][0]}/{_fe['fold_sat'][1]} "
                                  f"(t={_fe['t']:.0f}s); final in(towel,box)={'T' if _ln['facts']['in(towel,box)'] else 'F'} "
                                  f"in(block,box)={'T' if _ln['facts']['in(block,box)'] else 'F'}; plan {len(_LABEL_STEPS)} actions, "
                                  f"{self.sim_time:.0f}s", flush=True)
                        if _REC_USD:
                            __import__("json").dump(_al, open(_REC_USD[:-4] + "_task1_audit.json", "w"), indent=1)
                    except Exception as _e:
                        print(f"[twin-stock] task1 summary skipped: {_e}", flush=True)
                if False:
                    try:
                        from fold_predicates import SheetPredicates, task1_state, task1_score
                        _lb = __import__("json").load(open(_LABELS_FILE))["labels"]
                        _P = SheetPredicates(_lb, spacing_cm=float(_os.environ.get("TWIN_TOWEL_SPACING_CM", "0.5")),
                                             thickness_cm=0.6, eps_cm=3.0,
                                             asset_layers=2 if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2" else 1)
                        _t1 = self._task1
                        _box = {"center": _t1["box_center"], "half": _t1["box_half"], "floor": _t1["box_floor"], "rim": _t1["box_rim"]}
                        _bq = self.state_0.body_q.numpy()[_t1["block_body"]]
                        _st = task1_state(_P, self._qs()[:_t1.get("n_towel", self._towel["nx"] * self._towel["ny"])], _box,
                                          _bq[:3], _t1["block_dims"], fold_goal=_os.environ.get("TWIN_GOAL", "half_y"))
                        _n, _tot, _facts = task1_score(_st)
                        print(f"[twin-stock] TASK 1 audit: {_n}/{_tot} " + ", ".join(f"{k}={'T' if v else 'F'}" for k, v in _facts)
                              + f"; towel in box {_st['towel_in_box_frac']:.0%}; {_P.describe(_st)}", flush=True)
                    except Exception as _e:
                        print(f"[twin-stock] task1 audit skipped: {_e}", flush=True)
                if getattr(self, "_towel", None) and _LABELS_FILE:
                    # the SAME symbolic state the planner searched over, on the executed result
                    try:
                        from fold_predicates import SheetPredicates, GOALS
                        _lb = __import__("json").load(open(_LABELS_FILE))["labels"]
                        _P = SheetPredicates(_lb, spacing_cm=float(_os.environ.get("TWIN_TOWEL_SPACING_CM", "0.5")),
                                             thickness_cm=0.6, eps_cm=3.0,
                                             asset_layers=2 if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2" else 1)
                        _st = _P.state(self._qs()[:self._towel["nx"] * self._towel["ny"]])
                        _g = _os.environ.get("TWIN_GOAL", "quarter")
                        print(f"[twin-stock] symbolic state: {_P.describe(_st)}  -> goal '{_g}' "
                              f"{_P.score(_st, _g)[0]}/{_P.score(_st, _g)[1]}", flush=True)
                    except Exception as _e:
                        print(f"[twin-stock] symbolic audit skipped: {_e}", flush=True)
                if _FINAL_STATE:
                    _qall = self.state_0.particle_q.numpy()
                    np.save(_FINAL_STATE, _qall.astype(np.float32))
                    print(f"[twin-stock] final scene state ({len(_qall)} particles) -> "
                          f"{_FINAL_STATE}  (next run: TWIN_LOAD_STATE=this, same TWIN_ANCHOR)",
                          flush=True)
            return False
        if getattr(self, "_pending_gen", None):
            # the second pinch of a step the arms do in turn: generated NOW, from the cloth as the
            # first pinch left it (built up front, it aimed at where the corner used to be)
            name, gen = self._pending_gen.pop(0)
        else:
            name, gen = stages[i]
            self._plan_stage = i + 1
        # `gen is self._pack_poses` is always False - attribute access builds a NEW bound
        # method each time - which left the pack flag off, so the carry never engaged and the
        # stale fold aid kept hold of the cloth. Compare the stage name instead.
        self._packing = name.startswith("pack") or name.startswith("stack")
        if self._packing:
            self._fold = None                      # fold aid must let go before the carry
            if _FINAL_STATE and name.startswith("stack"):
                _pre = _FINAL_STATE[:-4] + ".prestack.npy" if _FINAL_STATE.endswith(".npy") else _FINAL_STATE + ".prestack.npy"
                np.save(_pre, self.state_0.particle_q.numpy().astype(np.float32))
                print(f"[twin-stock] scene state before the stack -> {_pre}  (rerun the stack alone: "
                      f"TWIN_LOAD_STATE=this TWIN_PLANS='<id>:-' TWIN_GARMENT=<id> TWIN_STACK_ON=...)", flush=True)
        for _fr in getattr(self, "_fold_report", []):
            if "err_settled" not in _fr:
                try:
                    _qq = self._q_of(_fr["gi"])
                    _fr["err_settled"] = float(np.linalg.norm(_qq[_fr["pa"]][:2] - np.asarray(_fr["target"], float)))
                    _fr["err_to_pb"] = float(np.linalg.norm(_qq[_fr["pa"]][:2] - _qq[_fr["pb"]][:2]))
                    print(f"[twin-stock] fold {_fr['label']}: settled {_fr['err_settled']:.1f}cm from the target point, "
                          f"{_fr['err_to_pb']:.1f}cm from the label it should meet (released at {_fr['err_release']:.1f}cm)", flush=True)
                except Exception:
                    _fr["err_settled"] = None
        if getattr(self, "_task1", None) and getattr(self, "_phase_name", None):
            self._task1_audit(f"after '{self._phase_name}'")
        self._append_poses(gen())
        lo, hi = self._cloth_bbox()
        self._phase_name = name
        print(f"[twin-stock] t={self.sim_time:.0f}s phase '{name}' queued "
              f"(cloth {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm)", flush=True)
        return True

    def _arm_clearance(self):
        """Approximate minimum distance between the two arms' links: the six arm bodies before
        each wrist as a chain of segments with an 8 cm radius, plus the tip. A metric, not a
        constraint - printed at the end next to the tip distance."""
        hands = getattr(self, "_hands", None)
        if not hands or len(hands) < 2:
            return None
        bq = self.state_0.body_q.numpy()
        chains = []
        for h in hands:
            ee = int(h["ee_body"]); pts = [bq[b][:3] for b in range(max(0, ee - 5), ee + 1)]
            pts.append(self._ee_tip(h)[0]); chains.append(np.asarray(pts, float))
        def seg_dist(p1, p2, q1, q2):
            # closest distance between two segments (sampled, 12 points each - cheap and enough)
            a = p1[None] + (p2 - p1)[None] * np.linspace(0, 1, 12)[:, None]
            b = q1[None] + (q2 - q1)[None] * np.linspace(0, 1, 12)[:, None]
            return float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min())
        best = 1e9
        A, B = chains
        for i in range(1, len(A)):               # skip the base link segment (static, always apart)
            for j in range(1, len(B)):
                best = min(best, seg_dist(A[i - 1], A[i], B[j - 1], B[j]))
        return best - 8.0                         # minus two 4 cm link radii

    def _task1_audit(self, tag):
        """The Task-1 symbolic state from geometry: fold predicates + in(x, box). Printed after
        every stage and at the end; 'folded' is a fact the fold stage establishes, and whether
        the pack preserved it is reported, not assumed."""
        if getattr(self, "_towel_specs", None):
            return self._task2_audit(tag)
        try:
            from fold_predicates import SheetPredicates, task1_state, task1_score
            _lb = __import__("json").load(open(_LABELS_FILE))["labels"]
            _P = SheetPredicates(_lb, spacing_cm=float(_os.environ.get("TWIN_TOWEL_SPACING_CM", "0.5")),
                                 thickness_cm=float(_os.environ.get("TWIN_LAYER_THICK_CM", "0.6")), eps_cm=3.0,
                                 asset_layers=2 if _os.environ.get("TWIN_TOWEL_LAYERS", "1") == "2" else 1)
            _t1 = self._task1
            _box = {"center": _t1["box_center"], "half": _t1["box_half"], "floor": _t1["box_floor"], "rim": _t1["box_rim"]}
            _bq = self.state_0.body_q.numpy()[_t1["block_body"]]
            _n_t = _t1.get("n_towel", self._towel["nx"] * self._towel["ny"])
            _st = task1_state(_P, self._qs()[:_n_t], _box, _bq[:3], _t1["block_dims"],
                              fold_goal=_os.environ.get("TWIN_GOAL", "quarter"))
            _n, _tot, _facts = task1_score(_st)
            _sat, _totf = _P.score(_st, _os.environ.get("TWIN_GOAL", "quarter"))
            self._task1_log = getattr(self, "_task1_log", [])
            self._task1_log.append({"tag": tag, "t": float(self.sim_time), "facts": dict(_facts), "fold_sat": [_sat, _totf],
                                    "towel_in_box_frac": _st["towel_in_box_frac"], "block": [float(v) for v in _bq[:3]]})
            print(f"[twin-stock] TASK 1 audit {tag} (t={self.sim_time:.0f}s): {_n}/{_tot} "
                  + ", ".join(f"{k}={'T' if v else 'F'}" for k, v in _facts)
                  + f"; fold goal {_sat}/{_totf}; towel in box {_st['towel_in_box_frac']:.0%}; block at "
                  f"({_bq[0]:.0f},{_bq[1]:.0f},{_bq[2]:.0f}); {_P.describe(_st)}", flush=True)
            return _n, _tot
        except Exception as _e:
            print(f"[twin-stock] task1 audit skipped: {_e}", flush=True)
            return None

    # ---- teleoperation -------------------------------------------------------
    # The stock example's "closed" is activation 0.1 = 0.4cm per finger, i.e. an 8mm gap
    # between the fingertips - measured, with the fingers tracking the command exactly. That
    # is too slack to pinch a single cloth layer, so teleop closes all the way by default.
    # TELEOP_GRIP_CM is the per-finger target when shut (0 = fingertips together).
    TELEOP_GRIP_CM = float(_os.environ.get("TELEOP_GRIP_CM", "0.0"))
    TELEOP_LIN_CM_S = float(_os.environ.get("TELEOP_LIN_CM_S", "12"))
    TELEOP_ANG_DEG_S = float(_os.environ.get("TELEOP_ANG_DEG_S", "45"))

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product of two (x, y, z, w) quaternions."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    # ---- demonstration recording --------------------------------------------
    def _record_reset(self):
        self._rec = {"t": [], "ee_pos": [], "ee_quat": [], "grip": [],
                     "joint_q": [], "cloth_t": [], "cloth_q": [], "n": 0}

    def _record_step(self, target):
        """Append one frame of the demonstration. Called only while recording."""
        r = self._rec
        r["t"].append(float(self.sim_time))
        r["ee_pos"].append(np.asarray(target[0:3], np.float32))
        r["ee_quat"].append(np.asarray(target[3:7], np.float32))
        r["grip"].append(float(target[7]))
        r["joint_q"].append(self.state_0.joint_q.numpy().copy())
        if r["n"] % _CLOTH_EVERY == 0:                 # GPU readback: keep it sparse
            q = self.state_0.particle_q.numpy()[::_CLOTH_STRIDE]
            r["cloth_t"].append(float(self.sim_time))
            r["cloth_q"].append(q.astype(np.float32))
        r["n"] += 1
        if r["n"] % 60 == 0:                           # ~1s heartbeat so the operator sees it live
            p0 = r["ee_pos"][0]
            moved = float(np.abs(np.asarray(r["ee_pos"][-1]) - p0).sum())
            print(f"[record] {r['n']} frames  ee=({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})cm"
                  f"  moved {moved:.0f}cm  grip={'shut' if target[7] <= 0.01 else 'open'}", flush=True)

    def _record_save(self):
        """Write the episode as one npz plus a meta.json describing it."""
        import json as _j

        r = getattr(self, "_rec", None)
        if not r or not r["t"]:
            print("[record] nothing captured", flush=True)
            return None
        ep = _os.path.join(_DATA_DIR, f"ep_{int(self.sim_time * 1000):09d}")
        _os.makedirs(ep, exist_ok=True)
        np.savez_compressed(
            _os.path.join(ep, "episode.npz"),
            t=np.array(r["t"], np.float32),
            ee_pos=np.stack(r["ee_pos"]), ee_quat=np.stack(r["ee_quat"]),
            grip=np.array(r["grip"], np.float32),
            joint_q=np.stack(r["joint_q"]),
            cloth_t=np.array(r["cloth_t"], np.float32),
            cloth_q=np.stack(r["cloth_q"]) if r["cloth_q"] else np.zeros((0, 0, 3), np.float32),
        )
        meta = {
            "units": "cm, radians, seconds; quaternions (x,y,z,w)",
            "fps": self.fps, "frames": len(r["t"]),
            "cloth_every_frames": _CLOTH_EVERY, "cloth_particle_stride": _CLOTH_STRIDE,
            "cloth_particles_recorded": int(r["cloth_q"][0].shape[0]) if r["cloth_q"] else 0,
            "robot": ("ur5e + robotiq 2f-85 (sim, menagerie) - EE poses transfer" if _ARM == "ur5e" else
                      "franka fr3 (sim); real arm is a UR5e - EE poses transfer, joints do not"),
            "action_space": "absolute EE target pose + gripper activation (0=shut, 0.8=open)",
            "table_top_z_cm": self.TABLE_TOP,
            "env": {k: _os.environ.get(k) for k in
                    ("TWIN_TS", "TWIN_PLACE", "TWIN_FOLD_STYLE", "TWIN_GRASP_DZ_CM",
                     "TELEOP_GRIP_CM", "TELEOP_LIN_CM_S", "TELEOP_ANG_DEG_S")},
        }
        with open(_os.path.join(ep, "meta.json"), "w") as f:
            _j.dump(meta, f, indent=2)
        print(f"[record] saved {len(r['t'])} frames + {len(r['cloth_t'])} cloth samples -> {ep}",
              flush=True)
        self._record_reset()
        return ep

    def _teleop_target(self):
        """Integrate keyboard input into an EE target the existing IK can follow.

        Motion is a velocity command scaled by the frame time, so it is frame-rate
        independent and the same command shape the real arm takes (an EE twist).
        """
        v = self.viewer
        t = getattr(self, "_tele", None)
        if t is None:
            tip, _ = self._ee_tip()
            t = self._tele = {"p": np.array(tip, float), "yaw": 0.0,
                              "grip": self.GRIP_OPEN, "space": False,
                              "home": np.array(tip, float)}
            print("[teleop] j/l = -x/+x   i/k = -y/+y   u/o = up/down   n/m = yaw\n"
                  "[teleop] g = toggle gripper   z = slow   h = home\n"
                  "[teleop] 1 = start/stop recording a demonstration\n"
                  "[teleop] (WASD/QE/arrows stay with the viewer camera)", flush=True)

        dt = self.frame_dt
        scale = 0.25 if v.is_key_down("z") else 1.0
        lin = self.TELEOP_LIN_CM_S * dt * scale
        ang = np.radians(self.TELEOP_ANG_DEG_S) * dt * scale

        d = np.zeros(3)
        # NB: the viewer already owns WASD/QE/arrows for camera movement, so teleop uses
        # a separate cluster - pressing 'd' would fly the camera, not the arm.
        d[0] += float(v.is_key_down("l")) - float(v.is_key_down("j"))
        d[1] += float(v.is_key_down("k")) - float(v.is_key_down("i"))
        d[2] += float(v.is_key_down("u")) - float(v.is_key_down("o"))
        n = float(np.linalg.norm(d))
        if n > 0.0:
            t["p"] = t["p"] + d / n * lin
        t["yaw"] += (float(v.is_key_down("n")) - float(v.is_key_down("m"))) * ang

        shut = bool(v.is_key_down("g"))
        if shut and not t["space"]:                       # edge-triggered toggle
            grip_shut = self.TELEOP_GRIP_CM / 4.0         # activation -> cm is x4
            closed = t["grip"] <= grip_shut + 1e-3
            t["grip"] = self.GRIP_OPEN if closed else grip_shut
            print(f"[teleop] gripper {'open' if closed else 'closed'} "
                  f"({'—' if closed else f'{2 * self.TELEOP_GRIP_CM:.1f}cm gap'})", flush=True)
        t["space"] = shut

        rec_key = bool(v.is_key_down("1"))
        if rec_key and not t.get("rec_key", False):
            if getattr(self, "_recording", False):
                self._recording = False
                self._record_save()
            else:
                self._record_reset()
                self._recording = True
                print("[record] recording - press 1 again to save", flush=True)
        t["rec_key"] = rec_key

        if v.is_key_down("h"):
            t["p"], t["yaw"] = t["home"].copy(), 0.0

        t["n"] = t.get("n", 0) + 1
        if t["n"] % 120 == 0:                     # ~2s: commanded vs actual finger opening
            fq = self.state_0.joint_q.numpy()[-2:]
            print(f"[teleop] grip cmd {t['grip'] * 4.0:.2f}cm/finger  actual "
                  f"{fq[0]:.2f}/{fq[1]:.2f}cm  gap {fq[0] + fq[1]:.2f}cm", flush=True)

        # keep the tool above the table and inside the arm's comfortable reach
        t["p"][2] = max(float(t["p"][2]), self.TABLE_TOP - _GRASP_DZ)
        t["p"] = np.array(self._clamp_reach(*t["p"]), float)

        half = t["yaw"] / 2.0
        q = self._quat_mul(np.array([0.0, 0.0, np.sin(half), np.cos(half)]),
                           np.array(self.DOWN_QUAT))
        target = np.array([t["p"][0], t["p"][1], t["p"][2], q[0], q[1], q[2], q[3], t["grip"]],
                          dtype=np.float32)
        if getattr(self, "_recording", False):
            self._record_step(target)
        return target

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        if _MODE == "teleop":
            self.target = self._teleop_target()
            self._ik_to_target(state_in)
            return

        if _MODE == "replay":
            # Re-run a recorded demonstration: the actions are absolute EE targets, so
            # replaying them is the same control path teleop drove by hand.
            ep = getattr(self, "_replay", None)
            if ep is None:
                path = _os.environ["TWIN_EPISODE"]
                if _os.path.isdir(path):
                    path = _os.path.join(path, "episode.npz")
                d = np.load(path)
                ep = self._replay = {
                    "t": d["t"],
                    "a": np.concatenate([d["ee_pos"], d["ee_quat"], d["grip"][:, None]], axis=1),
                    "t0": float(self.sim_time),
                }
                print(f"[replay] {len(ep['t'])} frames from {path}", flush=True)
            i = int(np.searchsorted(ep["t"], ep["t"][0] + (self.sim_time - ep["t0"])))
            if i >= len(ep["t"]):
                self.target_joint_qd.zero_()
                return
            self.target = ep["a"][i].astype(np.float32)
            self._ik_to_target(state_in)
            return

        if _LOAD_STATE and not getattr(self, "_state_loaded", False) and self.sim_time >= 2.0:
            # Applied after the cloth has settled onto the table, so the loaded configuration
            # starts from rest rather than fighting the drop-in from spawn height.
            self._state_loaded = True
            _q = np.load(_LOAD_STATE).astype(np.float32)
            _shape0 = _q.shape
            _full = self.state_0.particle_q.numpy()
            if _q.shape == _full[self._gsl].shape:      # the active garment's state (planner contract)
                self._put_q(_q)
                self.state_0.particle_qd.zero_()
            elif _q.shape == _full.shape:               # EVERY garment: a previous run's TWIN_FINAL_STATE
                self.state_0.particle_q.assign(_q)
                self.state_0.particle_qd.zero_()
                print(f"[twin-stock] loaded the FULL scene state ({len(_q)} particles, "
                      f"{len(getattr(self, '_garments', []))} garments)", flush=True)
                _q = _q[self._gsl]
            else:
                _q = None
            if _q is not None:
                lo, hi = self._cloth_bbox()
                print(f"[twin-stock] loaded start state {_LOAD_STATE} "
                      f"({hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm)", flush=True)
                if _UNFOLD_STACK:
                    import json as _j
                    _st = _j.load(open(_UNFOLD_STACK))
                    self._fold_stack = []
                    for r in _st["stack"]:
                        rec = {"axis": int(r["axis"]), "line": float(r["line"]),
                               "grab": np.array(r["grab"], float),
                               "drop": np.array(r["drop"], float)}
                        if r.get("kind") == "sleeve":
                            rec["sleeve_particle"] = int(r["sleeve_particle"])
                            rec["sleeve_home"] = np.array(r["sleeve_home"], float)
                            rec["sleeve_set"] = np.array(r["sleeve_set"], int)
                        if r.get("flap_set"):
                            rec["flap_set"] = np.array(r["flap_set"], int)
                        self._fold_stack.append(rec)
                    print(f"[twin-stock] unfold stack from {_UNFOLD_STACK}: "
                          f"{' > '.join(_st.get('sequence', []))} ({len(self._fold_stack)} "
                          f"fold(s)) - undoing them last-first", flush=True)
            else:
                print(f"[twin-stock] state {_LOAD_STATE} has shape {_shape0}, expected "
                      f"{self.state_0.particle_q.numpy()[self._gsl].shape} (active garment) or "
                      f"{self.state_0.particle_q.numpy().shape} (whole scene) - ignored", flush=True)

        if not getattr(self, "_nan_reported", False) and (int(self.sim_time * 10) % 5 == 0):
            _qq = self.state_0.particle_q.numpy()
            if not np.isfinite(_qq).all():
                self._nan_reported = True
                print(f"[twin-stock] CLOTH DIVERGED (NaN) at t={self.sim_time:.1f}s during "
                      f"'{getattr(self, '_phase_name', '?')}' - the recording freezes here; "
                      f"raise TWIN_CARRY_SLOPE/TWIN_CARRY_SLOW or the VBD iterations", flush=True)
        if _DUMP_STATE and self.sim_time >= 4.0:
            _qall = self.state_0.particle_q.numpy()
            q = _qall[self._gsl]                         # the planner plans ONE garment
            np.save(_DUMP_STATE, q.astype(np.float32))
            lo, hi = self._cloth_bbox()
            print(f"[twin-stock] observed state -> {_DUMP_STATE} "
                  f"({hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm, {len(q)} particles)", flush=True)
            if len(getattr(self, "_garments", [])) > 1:
                _allp = _DUMP_STATE[:-4] + ".all.npy" if _DUMP_STATE.endswith(".npy") else _DUMP_STATE + ".all.npy"
                np.save(_allp, _qall.astype(np.float32))
                print(f"[twin-stock] whole scene ({len(_qall)} particles, {len(self._garments)} "
                      f"garments) -> {_allp}", flush=True)
            # The planner has never known where the ARM is, so it proposed grasps the robot
            # cannot reach and nothing downstream objected: the sim base is stepped _CLOSER
            # toward the garment and _clamp_reach silently pulls any far waypoint to MAX_REACH,
            # so an out-of-reach grasp just became a different grasp. Hand the planner the real
            # base and reach so it can rule those folds out instead.
            _b = getattr(self, "base_cm", None)
            if _b is not None:
                import json as _j
                _true = np.asarray(_b, float) - _FWD * _CLOSER - _LEFT * _CLOSER_LEFT  # undo the sim step-in
                with open(_DUMP_STATE + ".robot.json", "w") as _fh:
                    # The planner must only accept grasps THIS twin can execute: the arm clamps
                    # waypoints to MAX_REACH from its (stepped-in) base, i.e. MAX_REACH+_CLOSER
                    # from the real base. Handing it the hardware's 105cm (a wrist-lean reach the
                    # twin never uses) let a plan through whose fold drop sat 89cm out; the twin
                    # released it 15cm short and the fold never completed (2026-09-03).
                    _j.dump({"base_cm": [float(v) for v in _true],
                             "reach_cm": min(_REACH_LIMIT * 100.0, self.MAX_REACH + _CLOSER),
                             "fit_scale": getattr(self, "_fit_scale", [1.0, 1.0]),
                             "garments": [{k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                           for k, v in g.items()} for g in getattr(self, "_garments", [])],
                             "active": int(getattr(self, "_active", 0)),
                             "anchor": _ANCHOR,
                             "fwd": [float(v) for v in _FWD],
                             "closer_cm": _CLOSER}, _fh, indent=2)
                print(f"[twin-stock] robot frame -> {_DUMP_STATE}.robot.json "
                      f"(true base {_true[0]:.0f},{_true[1]:.0f},{_true[2]:.0f}cm, "
                      f"reach {_REACH_LIMIT * 100:.0f}cm)", flush=True)
            raise SystemExit(0)

        # After the key poses sequence ends, queue the next phase; hold still once done
        if self._clock() >= self.robot_key_poses_time[-1]:
            if not self._extend_plan():
                if _EXPORT_EE and not getattr(self, "_exported", False):
                    self._exported = True
                    self._export_ee(_EXPORT_EE)
                if self._usd_rec is not None and not getattr(self, "_usd_saved", False):
                    self._usd_saved = True
                    self._usd_rec.save()
                self.target_joint_qd.zero_()
                # TWIN_EXIT_ON_DONE=1 (default): the frame budget is a TRUNCATION guard, not a
                # target - after "plan complete" + export + save the run just idled (an hour of
                # wall clock on the 12.9k-particle two-garment scene, 2026-09-11).
                if _os.environ.get("TWIN_EXIT_ON_DONE", "1") == "1" and _MODE != "teleop":
                    if not getattr(self, "_done_at", None):
                        self._done_at = self.sim_time + 8.0          # let the cloth settle on record
                    elif self.sim_time >= self._done_at:
                        if self._usd_rec is not None:
                            self._usd_rec.save()
                        if _FINAL_STATE:                      # the SETTLED scene, not the release instant
                            np.save(_FINAL_STATE, self.state_0.particle_q.numpy().astype(np.float32))
                        print(f"[twin-stock] plan complete and exported - exiting at t={self.sim_time:.0f}s",
                              flush=True)
                        raise SystemExit(0)
                return

        # The setpoint used to be a STEP function: searchsorted picked the next key pose and the
        # IK chased that constant target for the whole interval, so the tip sprinted at it,
        # settled, then got yanked to the next one. The cloth is welded to the tip, so the
        # fabric was yanked too - that stop-start is why the robot fold looked robotic next to
        # fold_env, where the pinch is a continuous function of time. Interpolate instead.
        self._settle_gate()
        _tc = self._clock()
        self.target = self._target_at(self.targets, self.robot_key_poses_time, _tc)
        _hands = getattr(self, "_hands", None)
        if _hands:
            # Both hands are solved against the SAME state and their joint velocities summed, so
            # neither assign can wipe the other's command.
            _hands[0]["target"] = self.target
            _cmd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
            for _h in _hands:
                if _h is not _hands[0]:
                    _h["target"] = self._target_at(_h["targets"], _h["times"], _tc)
                self._ik_to_target(state_in, hand=_h, accum=_cmd)
            self.target_joint_qd.assign(_cmd)
        else:
            self._ik_to_target(state_in)

    def _clock(self):
        """The stage clock: sim time minus the time the settle gate has held it."""
        return self.sim_time - getattr(self, "_clock_hold", 0.0)

    def _settle_gate(self):
        """SETTLE BEFORE CLOSING. The jaws used to shut on the clock, and the IK was still 2-16 cm
        short of the pinch point when they did (measured: right tip err 16 -> 7 -> 3 cm over the
        approach of fold step 2), so the pinch landed off the fabric and only a 20 cm band hid it.
        Here the stage clock PAUSES at the entry of a closing segment, for every hand, until each
        closing tip is within TWIN_GRASP_TOL_CM of its grasp point - capped at TWIN_GRASP_SETTLE_MAX
        s per grasp so a point out of reach stalls nothing. The pause counts in the plan time."""
        hands = getattr(self, "_hands", None)
        if not hands:
            return
        cap = float(_os.environ.get("TWIN_GRASP_SETTLE_MAX", "6.0"))
        self._clock_hold = getattr(self, "_clock_hold", 0.0)
        self._settle = getattr(self, "_settle", {})
        t = self._clock()
        for h in hands:
            times, targets = h.get("times"), h.get("targets")
            if times is None or targets is None or len(targets) < 2:
                continue
            i = int(np.searchsorted(times, t)); i = min(i, len(targets) - 1)
            closing = (i > 0 and float(targets[i][-1]) <= self.GRIP_CLOSE + 1e-3
                       and float(targets[i - 1][-1]) > self.GRIP_CLOSE + 1e-3)
            # ARRIVE BEFORE LETTING GO. The jaws used to open on the clock wherever the tip had
            # got to, and at long reach that was 8-10 cm short of the drop - the carried corner
            # was simply put down in the wrong place (exec_s11: far_left->far_right released
            # 10.6 cm out, settled 8.1 cm from the label it should have met, while a short carry
            # on the same towel settled 0.3 cm). Gate the OPEN the same way as the close.
            opening = (i > 0 and float(targets[i][-1]) > self.GRIP_CLOSE + 1e-3
                       and float(targets[i - 1][-1]) <= self.GRIP_CLOSE + 1e-3)
            key = (h["name"], i)
            if not (closing or opening) or t - float(times[i - 1]) > 0.12:
                continue                      # not at the entry of a grip change: nothing to gate
            tol = (float(_os.environ.get("TWIN_GRASP_TOL_CM", "0.6")) if closing
                   else float(_os.environ.get("TWIN_RELEASE_TOL_CM", "1.0")))
            used = self._settle.get(key, 0.0)
            tip, _ = self._ee_tip(h)
            err = float(np.linalg.norm(tip - np.asarray(targets[i][:3], float)))
            if err > tol and used < cap:
                # hold the clock just before the close so the target stays the open approach pose
                self._clock_hold += self.frame_dt
                self._settle[key] = used + self.frame_dt
                if used == 0.0:
                    print(f"[twin-stock] settle: {h['name']} tip {err:.1f}cm from the "
                          f"{'grasp' if closing else 'drop'} point - holding the {'close' if closing else 'release'}", flush=True)
            elif key not in self._settle.get("_done", set()):
                self._settle.setdefault("_done", set()).add(key)
                print(f"[twin-stock] {'grasp' if closing else 'arrive'}: {h['name']} tip {err:.1f}cm from the "
                      f"{'grasp' if closing else 'drop'} point after {used:.1f}s settle"
                      + ((" (CAP, " + ("closing" if closing else "releasing") + " anyway)") if err > tol else ""), flush=True)

    def _target_at(self, targets, times, t):
        """The setpoint of one hand at sim time t."""
        i = int(np.searchsorted(times, t))
        i = min(i, len(targets) - 1)
        t1 = float(times[i])
        t0 = float(times[i - 1]) if i > 0 else 0.0
        tgt = np.array(targets[i], dtype=np.float32)
        # ONLY while the jaws are already shut. Interpolating the APPROACH makes the arm lag its
        # own setpoint, so the tip is still short of the grasp point when the jaws close and the
        # pinch takes 39 particles instead of 420 - a miss. The approach keeps the step target
        # that reliably plants the tip; the carry, where smoothness is what shows, is continuous.
        _shut = float(targets[i][-1]) <= self.GRIP_CLOSE + 1e-3
        _was_shut = i > 0 and float(targets[i - 1][-1]) <= self.GRIP_CLOSE + 1e-3
        if i > 0 and t1 > t0 and _shut and _was_shut:
            u = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
            # Position only. Every pose carries the same DOWN_QUAT, and the gripper command is
            # a STATE, not a path - ramping it would shift the frame the jaws count as shut and
            # move the grasp, which is the one thing not to disturb.
            tgt[:3] = np.asarray(targets[i - 1][:3]) + (tgt[:3] - np.asarray(targets[i - 1][:3])) * u
        return tgt

    def _ik_to_target(self, state_in: State, hand=None, accum=None):
        """Drive ONE arm toward its target. Shared by the scripted path and teleop.

        `hand` picks the arm on a bimanual rig (None = the legacy single-arm attributes, which
        hand 0 also uses). `accum` is a joint-velocity vector this ADDS into instead of assigning,
        so two hands solved in the same step cannot overwrite each other - the caller assigns once.
        """
        include_rotation = True
        ee_off = (hand or {}).get("pinch", self.endeffector_offset)
        ee_id = (hand or {}).get("ee_body", self.endeffector_id)
        ee_delta = (hand or {}).get("ee_delta", self.ee_delta)
        J_flat = (hand or {}).get("J_flat", self.J_flat)
        tgt = np.asarray(hand["target"] if hand is not None else self.target, dtype=np.float32)

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                ee_off,
                ee_id,
                self.bodies_per_world,
                wp.transform(*tgt[:7]),
            ],
            outputs=[ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
            hand=hand,
        )
        J = J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = ee_delta.numpy()[0]

        if hand is not None:
            return self._ik_arm_solve(hand, J, delta_target, state_in, accum)
        # A diverged cloth propagates into the arm: the pinned patch drags the gripper target,
        # the Jacobian goes non-finite and pinv raises "SVD did not converge", which used to
        # kill the whole run mid-fold. Hold still and say so instead - the diverged state is
        # then still on screen and in the log to look at.
        if not (np.isfinite(J).all() and np.isfinite(delta_target).all()):
            if not getattr(self, "_ik_nan_warned", False):
                self._ik_nan_warned = True
                _who = (hand or {}).get("name", "arm")
                _badJ = np.where(~np.isfinite(J).all(axis=0))[0]
                print(f"[twin-stock] IK input non-finite on the {_who} hand - holding still. "
                      f"J finite {bool(np.isfinite(J).all())} (bad dof columns {_badJ.tolist()[:8]}), "
                      f"ee delta {np.asarray(delta_target).tolist()}, target {np.asarray(tgt).tolist()}",
                      flush=True)
            self.target_joint_qd.zero_()
            return
        # DAMPED least squares, not a bare pseudo-inverse. At the far drop of a fold (78cm from
        # the base, the reach clamp) the arm is at full stretch, J is near-singular, pinv(J)
        # explodes and the joint velocities with it: links at 3-10 m/s, the elbow through the
        # table (every recording from a folded start, 2026-09-03, spiking at the same t~19s).
        # Damping 10% of the largest singular value kills the singular directions and leaves
        # normal tracking unchanged.
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)
        lam = float(__import__("os").environ.get("TWIN_IK_DAMP", "0.1")) * float(sv.max()) if sv.size else 0.0
        J_inv = (Vt.T * (sv / (sv * sv + lam * lam))) @ U.T

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        _na = getattr(self, "_n_arm_dof", None)
        if _na is None or len(q) == J.shape[1]:
            delta_q_null = K_null * (q_des - q)
        else:
            # joint coords (22) != dofs (20) with the 2F-85's ball joints: null-space pull on the
            # arm joints only (their coords and dofs coincide, they come first)
            delta_q_null = np.zeros(J.shape[1], dtype=np.float32)
            delta_q_null[:_na] = K_null * (q_des[:_na] - q[:_na])

        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control (finger positions in cm)
        # and never ask the arm joints for more than a real UR5e moves (pi rad/s); this is the
        # backstop the damping alone does not give at a hard singularity
        qd_max = float(__import__("os").environ.get("TWIN_JOINT_QD_MAX", "2.5"))
        if _na is not None:
            delta_q[:_na] = np.clip(delta_q[:_na], -qd_max, qd_max)
            delta_q[_na:] = 0.0
            # 2F-85 fingers: drive the 4-bar KINEMATICALLY from the activation (0.8 open ... 0
            # shut, the Franka convention every pose uses). driver + spring_link follow the
            # angle, the follower (pad) counter-rotates, couplers/loop-closures stay put - FK-
            # checked: pad bodies 13.6cm apart open, 5.1cm shut, pads at the 22.3cm pinch point.
            # The pin grasp still holds the cloth; this is what the user SEES (2026-09-16).
            _a = float(np.clip((0.8 - float(self.target[-1])) / 0.8, 0.0, 1.0)) * 0.8
            for _j, _c, _sgn in getattr(self, "_ur_finger_dofs", ()):
                delta_q[_j] = float(np.clip((_sgn * _a - q[_c]) * 8.0, -4.0, 4.0))
        else:
            delta_q[:-2] = np.clip(delta_q[:-2], -qd_max, qd_max)
            delta_q[-2] = self.target[-1] * 4.0 - q[-2]
            delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    # Radius of cloth carried by the closed gripper. This has to cover the WHOLE folded
    # bundle: pinning a small ball inside a multi-layer pile drags those particles while the
    # rest stays on the table, which stretches the sheet into strands and destroys the FPS.
    FPS_HINT = 60.0        # frames per second the control loop runs at
    PIN_RADIUS = float(_os.environ.get("TWIN_PIN_RADIUS_CM", "32"))

    def _ik_arm_solve(self, hand, J, delta_target, state_in, accum):
        """Solve ONE arm on its own 6x6 Jacobian block.

        The tape returns a Jacobian over every dof in the model, and the columns belonging to the
        OTHER arm come back non-finite (autodiff through a branch this end effector does not
        depend on). Checking the whole matrix made the guard hold both arms still for an entire
        run - the two-UR5e rig folded nothing and only said "IK input non-finite" once. Those
        columns are not ours to use anyway: restricting the solve to this arm's own dofs removes
        the cross-arm coupling completely and makes the inverse a 6x6 instead of 6x40.
        """
        tgt = np.asarray(hand["target"], dtype=np.float32)
        ad = hand["arm_dofs"]
        ac = hand.get("arm_coords", ad)
        Jh = np.asarray(J[:, ad], dtype=np.float64)
        if not (np.isfinite(Jh).all() and np.isfinite(delta_target).all()):
            if not getattr(self, "_ik_nan_warned", False):
                self._ik_nan_warned = True
                _jq = state_in.joint_q.numpy()
                print(f"[twin-stock] IK input non-finite on the {hand['name']} arm - holding it "
                      f"still (joint_q finite {bool(np.isfinite(_jq).all())}, "
                      f"body_q finite {bool(np.isfinite(state_in.body_q.numpy()).all())})", flush=True)
            if accum is None:                 # hold STILL, do not leave the last command standing
                self.target_joint_qd.zero_()
            return
        U, sv, Vt = np.linalg.svd(Jh, full_matrices=False)
        lam = float(_os.environ.get("TWIN_IK_DAMP", "0.1")) * (float(sv.max()) if sv.size else 0.0)
        J_inv = (Vt.T * (sv / (sv * sv + lam * lam))) @ U.T
        q = state_in.joint_q.numpy()
        null = np.eye(len(ad)) - J_inv @ Jh
        home = np.asarray(self.initial_pose, float)[ac]
        # TWIN_IK_GAIN: Cartesian servo gain (1/s). At 1.0 the tip closes its error with a ~1 s
        # time constant, and near the far reach (heavy damping) 3 cm took 6 s to fall to 1 cm -
        # the jaws closed off the fabric. 3.0 converges a 3 cm error in ~1.5 s; still 5% per frame.
        _gain = float(_os.environ.get("TWIN_IK_GAIN", "1.0"))
        dq = J_inv @ (_gain * np.asarray(delta_target, float)) + null @ (1.0 * (home - np.asarray(q, float)[ac]))
        qd_max = float(_os.environ.get("TWIN_JOINT_QD_MAX", "2.5"))
        cmd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        cmd[ad] = np.clip(dq, -qd_max, qd_max)
        kind, fds = hand.get("finger_drive", ("prismatic", [(d, d) for d in hand["finger_dofs"]]))
        if kind == "2f85":
            # Robotiq 2F-85: drive the 4-bar kinematically from the activation (driver and
            # spring_link follow the angle, the follower counter-rotates), per hand.
            # Velocity is WRITTEN by dof, position is READ by coordinate - never the same index.
            a = float(np.clip((0.8 - float(tgt[-1])) / 0.8, 0.0, 1.0)) * 0.8
            for dof, coord, sgn in fds:
                cmd[dof] = float(np.clip((sgn * a - q[coord]) * 8.0, -4.0, 4.0))
        else:
            # ALOHA's jaws are PRISMATIC (0.2 .. 3.7cm here, tied by an equality constraint)
            opn = float(np.clip(float(tgt[-1]) / self.GRIP_OPEN, 0.0, 1.0))
            want = 0.2 + 3.5 * opn
            for dof, coord in fds:
                cmd[dof] = float(np.clip((want - q[coord]) * 8.0, -20.0, 20.0))
        if accum is not None:
            accum += cmd
            return
        self.target_joint_qd.assign(cmd)

    def _ee_tip(self, hand=None):
        """World transform of a gripper tip: (position, rotation matrix). `hand` picks the arm."""
        b = self.state_0.body_q.numpy()[(hand or {}).get("ee_body", self.endeffector_id)]
        p, q = np.asarray(b[:3], float), np.asarray(b[3:7], float)
        x, y, z, w = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return p + R @ np.array([float(v) for v in (hand or {}).get("pinch", self.endeffector_offset).p]), R

    def _update_fold(self):
        """Carry the outer half across the fold line while the gripper is shut.

        Frictional folding plateaus once the pile is a few layers thick, so in 'assist' mode the
        half beyond the fold line is pinned and mirrored across it, tracking how far the gripper
        has travelled from its grasp point to its drop point. The robot still drives the timing.
        """
        f = getattr(self, "_fold", None)
        if f is None or self.target is None or _FOLD_MODE == "physical":
            return
        if _FOLD_MODE == "grasp":
            if f.get("hands"):                     # a two-handed move: one pinch per arm
                for _hf in f["hands"]:
                    self._update_fold_grasp(_hf, hand=_hf.get("hand"))
            else:
                self._update_fold_grasp(f)
            return
        if _FOLD_MODE != "assist":
            return
        if float(self.target[-1]) > self.GRIP_CLOSE + 1e-3:          # jaws open -> let go
            # `start is None` means the carry has not begun: an unfold arrives with its particle
            # set preset and its first two poses OPEN, so releasing here would wipe that set and
            # send the close back to side-selection (which grabs the whole bundle).
            if f["idx"] is not None and f["start"] is not None:
                qd = self._qds()
                qd[f["idx"]] = 0.0
                self._put_qd(qd)
                f["idx"] = None
            return

        tip, _ = self._ee_tip()
        q = self._qs()
        a, line = f["axis"], f["line"]
        if f["idx"] is None:
            side = np.sign(f["grab"][a] - line) or 1.0
            idx = np.where((q[:, a] - line) * side > 0.0)[0]
            if len(idx) == 0:
                return
            f["idx"], f["start"] = idx, q[idx].copy()
            if self._fold_stack:                      # remember WHICH half this fold moved
                self._fold_stack[-1]["idx"] = idx
            print(f"[twin-stock] fold carries {len(idx)} particles across the line", flush=True)
        elif f["start"] is None:
            # unfold: the set is preset from the fold being reversed. Re-selecting by side is
            # wrong here - after a fold the WHOLE bundle lies on one side of the line, so a
            # side test grabs all 6436 particles and flips the bundle instead of unfolding it.
            f["start"] = q[f["idx"]].copy()
            print(f"[twin-stock] unfold returns {len(f['idx'])} particles", flush=True)

        span = f["drop"] - f["grab"]
        denom = float(span @ span)
        t = 0.0 if denom < 1e-6 else float(((tip[:2] - f["grab"]) @ span) / denom)
        t = min(max(t, 0.0), 1.0)

        start = f["start"]
        target = start.copy()
        target[:, a] = 2.0 * line - start[:, a]                       # mirror across the fold line
        moved = start + (target - start) * t
        moved[:, 2] = start[:, 2] + np.abs(target[:, a] - start[:, a]) * 0.28 * np.sin(np.pi * t)
        q[f["idx"]] = moved
        self._put_q(q)
        qd = self._qds()
        qd[f["idx"]] = 0.0
        self._put_qd(qd)

    def _pin_flags(self, idx, held):
        """KINEMATIC pin: clear ParticleFlags.ACTIVE on the held particles so VBD skips them
        (they are fixed anchors for their neighbours' constraints) and restore it on release.
        A position write alone is compliant: the solver pulls an 80-particle pinch back toward
        the fabric every substep, and a 2 cm pinch on the two-layer towel rose 4 cm while the
        tip rose 11 (v14) - the band of 441 particles only held because of its count."""
        if _os.environ.get("TWIN_PIN_KINEMATIC", "1") != "1" or idx is None or len(idx) == 0:
            return
        fl = self.model.particle_flags.numpy()
        # idx is LOCAL to the active garment; the flags array is the whole scene's. Without the
        # offset the second and third garment's pins froze particles of the FIRST garment and
        # their own pinch stayed soft - towelA's corners slid 6-20 cm out of the grip (exec_p2/p3).
        _g0 = getattr(self, "_pin_flag_base", {})
        key = id(idx)
        if held:
            base = int(self._gsl.start or 0)
            gidx = np.asarray(idx) + base
            self._pin_flag_live = getattr(self, "_pin_flag_live", [])
            self._pin_flag_live.append((np.asarray(idx).copy(), gidx))
            fl[gidx] &= ~int(newton.ParticleFlags.ACTIVE)
        else:
            # release by the indices recorded at pin time (the active garment may have changed)
            live = getattr(self, "_pin_flag_live", [])
            hit = next((k for k, (li, gi) in enumerate(live) if len(li) == len(idx) and np.array_equal(li, np.asarray(idx))), None)
            gidx = live.pop(hit)[1] if hit is not None else np.asarray(idx) + int(self._gsl.start or 0)
            fl[gidx] |= int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(fl)

    def _update_fold_grasp(self, f, hand=None):
        """Fold by PINCHING: the fingers hold a few cm of fabric and carry it, nothing else.

        The half beyond the fold line is never prescribed. It follows because it is attached to
        the pinched patch through the cloth itself, so gravity, bending and self-collision all
        act on it during the carry - it hangs, sags, lags and its sleeves flop, and the crease
        forms where the material actually bends rather than along a line we chose.

        The attachment is kinematic (the patch rides the tip) because a parallel jaw on fabric
        slips: TWIN_FOLD=physical loses the cloth and plateaus. Kinematic GRIP, dynamic CLOTH.
        """
        _tgt = hand["target"] if hand is not None else self.target
        if _tgt is None:
            return
        shut = float(_tgt[-1]) <= self.GRIP_CLOSE + 1e-3
        idx = f.get("pin_idx")
        if _os.environ.get("TWIN_DEBUG_GRIP", "0") == "1":
            # per hand: every grip-state transition, with what the pin was holding at the time
            _k = "_grip_was_" + str(id(f))
            if getattr(self, _k, None) != shut:
                setattr(self, _k, shut)
                _tipz = float(self._ee_tip(hand)[0][2])
                print(f"[grip] t={self.sim_time:.2f}s {(hand or {}).get('name', 'arm')}: target grip "
                      f"{float(_tgt[-1]):.2f} -> {'SHUT' if shut else 'OPEN'}; pin holds "
                      f"{0 if idx is None else len(idx)}; tip z {_tipz:.1f}", flush=True)
        if not shut:
            if idx is not None:                       # release limp, never with stored energy
                qd = self._qds()
                qd[idx] = 0.0
                self._put_qd(qd)
                self._pin_flags(idx, False)
                f["pin_idx"] = None
                if f.get("pa") is not None and _os.environ.get("TWIN_FOLD_REPORT", "1") == "1":
                    # FOLD PRECISION at the moment of release: where the carried label actually is
                    # against where the plan wanted it. The settled residual is printed by the
                    # stage audit; this one separates "the carry missed" from "it sprang back".
                    _q = self._qs()
                    _err = float(np.linalg.norm(_q[f["pa"]][:2] - np.asarray(f["target_xy"], float)))
                    _tp = self._ee_tip(hand)[0]
                    print(f"[twin-stock] fold {f['label']}: at release label is {float(np.linalg.norm(_q[f['pa']] - _tp)):.1f}cm from the tip", flush=True)
                    f["err_release"] = _err
                    self._fold_report = getattr(self, "_fold_report", [])
                    self._fold_report.append({"label": f["label"], "pa": int(f["pa"]), "pb": int(f["pb"]),
                                              "target": [float(v) for v in f["target_xy"]], "err_release": _err,
                                              "gi": int(getattr(self, "_active", 0))})
                    print(f"[twin-stock] fold {f['label']}: at release the carried label is {_err:.1f}cm from its target", flush=True)
            return

        tip, R = self._ee_tip(hand)
        q = self._qs()
        if idx is None:
            r = float(f.get("pin_cm", _FOLD_PIN))
            if f.get("two_point") and _FOLD_HOLDS > 1:
                # three 4cm holds spread along the fold edge, carried together
                other = 1 - int(f["axis"])
                # Spread the holds across the WIDTH OF THE EDGE being folded, not a fixed +-9cm.
                # On a ~40cm collar edge three holds at +-9cm grip only the middle 18cm, so the
                # centre gets carried while the ends stay put and the fabric gathers into a
                # ruffle at the gripper. Five holds over ~70% of the edge carry it as a line.
                span = float(f.get("span_cm", _FOLD_SPAN))
                # THREE holds, kept wide. Five across a 40cm edge drove too much cloth into
                # the stack and the fold diverged; three at the same span still carry the
                # edge as a line instead of gathering it at one point.
                n_holds = _FOLD_HOLDS
                sel = np.zeros(len(q), dtype=bool)
                for o in np.linspace(-span, span, n_holds):
                    c = tip.copy()
                    c[other] += o
                    sel |= np.linalg.norm(q - c, axis=1) < _FOLD_PIN
                idx = np.where(sel)[0]
                if len(idx) > _FOLD_HOLD_MAX:      # keep the closest; never hold a whole half
                    d = np.linalg.norm(q[idx] - tip, axis=1)
                    idx = idx[np.argsort(d)[:_FOLD_HOLD_MAX]]
            elif r > _FOLD_PIN:
                # A wide grip is wide along the EDGE only - still narrow across it and in
                # height. A sphere of the same radius swallowed 3252-4671 particles (up to 73%
                # of the shirt), which stops being a grasp and becomes the kinematic mirror.
                ax = int(f["axis"])
                idx = np.where((np.abs(q[:, 1 - ax] - tip[1 - ax]) < r)
                               & (np.abs(q[:, ax] - tip[ax]) < _FOLD_PIN)
                               & (np.abs(q[:, 2] - tip[2]) < _FOLD_PIN))[0]
            else:
                idx = np.where(np.linalg.norm(q - tip, axis=1) < r)[0]
            only = f.get("only")
            if only is not None and len(only):
                idx = np.intersect1d(idx, only)        # peel one layer, not the stack
            hold_max = _FOLD_HOLD_MAX if only is None else min(_FOLD_HOLD_MAX, 150)
            if len(idx) > hold_max:                    # peeling a flap: a lighter pinch
                d = np.linalg.norm(q[idx] - tip, axis=1)
                idx = idx[np.argsort(d)[:hold_max]]
            if len(idx) == 0:
                return                                # missed the cloth; try again next step
            f["pin_idx"] = idx
            f["pin_local"] = (q[idx] - tip) @ R       # offsets in the tip frame
            # A near-miss is silent otherwise: holding a couple of particles still "folds",
            # the extent barely moves, and the plan looks like it executed. Measured: a real
            # pinch takes 70-480 particles depending on how many layers are under the tip.
            # 50, not 20: a repeat of the same plan caught 2 then 21 particles at the same
            # waypoint - both were misses that barely moved the cloth, but 21 slipped past a
            # threshold of 20. A working pinch on this garment takes 70+.
            warn = "  <-- MISSED, too little fabric" if len(idx) < 50 else ""
            print(f"[twin-stock] pinch holds {len(idx)} particles "
                  f"({r:.0f}cm patch){warn}", flush=True)
            self._pin_flags(idx, True)
            if f.get("pa") is not None:
                _in = bool(np.isin(int(f["pa"]), idx))
                print(f"[twin-stock] fold {f['label']}: label particle {'IN' if _in else 'NOT in'} the pinch, "
                      f"{float(np.linalg.norm(q[f['pa']] - tip)):.1f}cm from the tip (tip z {tip[2]:.1f}, label z {q[f['pa']][2]:.1f})", flush=True)
        # FIRM hold - the grip keeps what it closed on. I tried driving these by VELOCITY
        # instead (TWIN_GRIP_GAIN < 1) to stop the cloth stretching, and it broke the grasp:
        # 0.35 barely folded (width 55->41 where the firm hold reached 21), 0.7 and 0.92 DROPPED
        # the shirt mid-carry. Stretching is handled by TWIN_FOLD_HOLD_MAX and the adaptive
        # carry speed, NOT by weakening the hold. Compliance stays available but off.
        want = tip + f["pin_local"] @ R.T
        want[:, 2] = np.maximum(want[:, 2], self.TABLE_TOP + 0.3)     # the table is solid: no pinned particle under its surface
        if _GRIP_GAIN >= 1.0:
            # POSITION drive, RATE-LIMITED. A bare teleport (q = want) on a 4-layer stack threw
            # particles into finger/stack contact in one frame and VBD went NaN; a velocity drive
            # (gain < 1) is undone by those same contacts, so the fingers rose and the cloth
            # stayed on the table (2026-09-03, half+sleeve start). Setting positions but moving
            # them at most _GRIP_VMAX per second keeps the grip firm without the jump.
            d = want - q[idx]
            n = np.linalg.norm(d, axis=1, keepdims=True)
            cap = _GRIP_VMAX / self.FPS_HINT
            q[idx] = q[idx] + np.where(n > cap, d / np.maximum(n, 1e-6) * cap, d)
            self._put_q(q)
            qd = self._qds()
            qd[idx] = 0.0
        else:
            v = (want - q[idx]) * (self.FPS_HINT * _GRIP_GAIN)
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(speed > _GRIP_VMAX, v / np.maximum(speed, 1e-6) * _GRIP_VMAX, v)
            qd = self._qds()
            qd[idx] = v
        self._put_qd(qd)

    def _update_grasp(self):
        """Carry the cloth the gripper closed on, during the pack phase only.

        The folds stay purely frictional (stock behaviour). But a parallel-jaw pinch on a
        MULTI-LAYER bundle physically will not hold it against gravity - measured here as the
        arm arriving at the box empty every time - so the carry uses a kinematic attachment:
        the particles inside PIN_RADIUS when the fingers shut ride the tip until release.
        """
        if not getattr(self, "_packing", False) or self.target is None:
            return
        shut = float(self.target[-1]) <= self.GRIP_CLOSE + 1e-3
        idx = getattr(self, "_pin_idx", None)
        if shut:
            tip, R = self._ee_tip()
            q = self._qs()
            if idx is None:
                idx = np.where(np.linalg.norm(q - tip, axis=1) < self.PIN_RADIUS)[0]
                if len(idx) == 0:
                    return
                self._pin_idx = idx
                self._pin_local = (q[idx] - tip) @ R          # offsets in the tip frame
                print(f"[twin-stock] grasp holds {len(idx)} particles", flush=True)
            q[idx] = tip + self._pin_local @ R.T
            self._put_q(q)
            qd = self._qds()
            qd[idx] = 0.0
            self._put_qd(qd)
        elif idx is not None:
            qd = self._qds()
            qd[idx] = 0.0                                     # release limp, never with stored energy
            self._put_qd(qd)
            self._pin_idx = None
            self._audit_at = self.sim_time + (6.0 if getattr(self, "_stack_target", None) else 3.0)
            print(f"[twin-stock] released over the "
                  f"{'bundle' if getattr(self, '_stack_target', None) else 'box'}", flush=True)

    def _audit_pack(self):
        """After the bundle settles, count what actually ended up in the box."""
        at = getattr(self, "_audit_at", None)
        if at is None or self.sim_time < at:
            return
        self._audit_at = None
        st = getattr(self, "_stack_target", None)
        if st is not None:                             # stacked on another garment, not boxed
            q = self._qs()
            hx, hy = st["half"]
            bx, by = st["xy"]
            on = ((np.abs(q[:, 0] - bx) <= hx + 3.0) & (np.abs(q[:, 1] - by) <= hy + 3.0)
                  & (q[:, 2] >= st["top"] - 1.0))
            # interpenetration: my particles below the OTHER bundle's local top (2cm cells)
            qo = self._q_of(st["gi"])
            cell = {}
            for x, y, z in qo:
                k = (int(np.floor(x / 2.0)), int(np.floor(y / 2.0)))
                cell[k] = max(cell.get(k, -1e9), float(z))
            sink = np.array([cell.get((int(np.floor(x / 2.0)), int(np.floor(y / 2.0))), -1e9) - z
                             for x, y, z in q])
            inside = sink > 0.0
            print(f"[twin-stock] AUDIT stacked on '{st['label']}': {int(on.sum())}/{len(q)} "
                  f"({100.0 * on.sum() / len(q):.0f}%) of the garment lies on that bundle; "
                  f"{int(inside.sum())} particles ({100.0 * inside.mean():.0f}%) sit below its local "
                  f"top surface, deepest {float(sink[inside].max()) if inside.any() else 0.0:.1f}cm",
                  flush=True)
            return
        bx, by = getattr(self, "box_center_cm", (0.0, 0.0))
        hx, hy = getattr(self, "box_half_cm", (13.0, 13.0))
        rim = getattr(self, "box_rim_z_cm", 32.0)
        floor = getattr(self, "box_floor_z_cm", self.TABLE_TOP)
        q = self._qs()
        inside = ((np.abs(q[:, 0] - bx) <= hx) & (np.abs(q[:, 1] - by) <= hy)
                  & (q[:, 2] >= floor - 1.0) & (q[:, 2] <= rim + 2.0))
        n = len(q)
        below = int((q[:, 2] < floor - 1.0).sum())
        print(f"[twin-stock] AUDIT in-box {inside.sum()}/{n} "
              f"({100.0 * inside.sum() / n:.0f}%), through-floor {below}, "
              f"outside {n - int(inside.sum()) - below}", flush=True)

    def _nan_probe(self, where):
        """TWIN_DEBUG_NAN=1: say where the arm state first goes non-finite. Host reads are only
        safe OUTSIDE the captured graph, which is why this is called from step(), never simulate()."""
        if _os.environ.get("TWIN_DEBUG_NAN", "0") != "1" or getattr(self, "_nan_found", False):
            return
        q = self.state_0.joint_q.numpy(); b = self.state_0.body_q.numpy()
        okq, okb = bool(np.isfinite(q).all()), bool(np.isfinite(b).all())
        print(f"[nan-probe] {where}: joint_q {'ok' if okq else 'BAD ' + str(np.where(~np.isfinite(q))[0].tolist()[:6])}, "
              f"body_q {'ok' if okb else 'BAD bodies ' + str(sorted(set(np.where(~np.isfinite(b))[0].tolist()))[:6])}"
              f"  (state_0 is buffer {id(self.state_0) % 1000})", flush=True)
        if not (okq and okb):
            self._nan_found = True

    def step(self):
        _fr = getattr(self, "_frame_no", 0); self._frame_no = _fr + 1
        self.generate_control_joint_qd(self.state_0)
        if _os.environ.get("TWIN_DEBUG_NAN", "0") == "1" and not getattr(self, "_nan_found", False):
            # per-arm command size and tip error EVERY frame until the first non-finite state,
            # so the blow-up is caught with the command that caused it
            _c = self.target_joint_qd.numpy()
            _parts = []
            for _h in (getattr(self, "_hands", None) or []):
                _ad = _h["arm_dofs"]
                _tip, _ = self._ee_tip(_h)
                _err = float(np.linalg.norm(np.asarray(_h["target"][:3], float) - _tip))
                _parts.append(f"{_h['name']}: |cmd| {float(np.abs(_c[_ad]).max()):.2f} rad/s, tip err {_err:.1f}cm")
            if _fr % 15 == 0 or _fr < 3:
                print(f"[nan-probe] frame {_fr} t={self.sim_time:.1f}s  " + " | ".join(_parts), flush=True)
            self._last_cmd_line = f"frame {_fr} t={self.sim_time:.1f}s  " + " | ".join(_parts)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        if _os.environ.get("TWIN_DEBUG_NAN", "0") == "1" and not getattr(self, "_nan_found", False):
            q = self.state_0.joint_q.numpy()
            if _fr < 7:
                _qd = self.state_0.joint_qd.numpy()
                _rep = " | ".join(f"{_h['name']} max|qd| arm {float(np.abs(_qd[_h['arm_dofs']]).max()):.3g} "
                                  f"fingers {float(np.abs(_qd[_h['finger_dofs']]).max()) if _h.get('finger_dofs') else 0:.3g}"
                                  for _h in (getattr(self, "_hands", None) or []))
                print(f"[nan-probe] frame {_fr} after sim: {_rep}", flush=True)
            if not np.isfinite(q).all():
                print(f"[nan-probe] FIRST NON-FINITE after frame {_fr} (t={self.sim_time:.2f}s); last command: "
                      f"{getattr(self, '_last_cmd_line', '?')}; bad coords {np.where(~np.isfinite(q))[0].tolist()[:8]}", flush=True)
                self._nan_found = True

        if getattr(self, "_hands", None) and len(self._hands) == 2 and int(self.sim_time * 60) % 15 == 0:
            _c = self._arm_clearance()
            if _c is not None:
                self._arm_min = getattr(self, "_arm_min", (1e9, 0.0))
                if _c < self._arm_min[0]:
                    self._arm_min = (_c, float(self.sim_time))
                self._arm_hits = getattr(self, "_arm_hits", 0) + (1 if _c < 0.0 else 0)
                self._arm_samples = getattr(self, "_arm_samples", 0) + 1
        if getattr(self, "_rigid_hold", None):
            self._update_rigid_hold()
        if getattr(self, "_rw", None):
            if (getattr(self, "_rigid_hold", None) or {}).get("held"):
                wp.synchronize()        # separates a hang in the cloth/arm step from one in the rigid world (stack dump)
            self._step_rigid_world()
        if getattr(self, "_packing_dual", None):
            self._update_pack_dual()
        if _MODE == "teleop":
            pass                       # teleop grasps frictionally; no fold/carry aids
        elif getattr(self, "_packing", False):
            if _os.environ.get("TWIN_GRASP", "attach") == "attach":
                self._update_grasp()
            self._audit_pack()
        else:
            self._update_fold()

        if self._usd_rec is not None:
            self._usd_rec.sample(self.state_0, self.model.particle_count)
        _hs = getattr(self, "_hands", None)
        if _hs and len(_hs) > 1:
            # the two arms' tip separation: with no arm-arm collision modelled, this is the number
            # that says whether a bimanual run kept the arms apart
            _d = float(np.linalg.norm(self._ee_tip(_hs[0])[0] - self._ee_tip(_hs[1])[0]))
            if _d < getattr(self, "_tip_min_cm", 1e9):
                self._tip_min_cm, self._tip_min_t = _d, self.sim_time
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                # NB: no .numpy() anywhere in here - simulate() is captured into a CUDA graph and
                # a device-to-host copy during capture aborts the run ("operation would make the
                # legacy stream depend on a capturing blocking stream").
                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth and not _SKIP_CLOTH_STEP:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            elif _SKIP_CLOTH_STEP:
                # diagnostic only: the cloth stands still, so a robot blow-up is the robot step's
                self.state_1.particle_q.assign(self.state_0.particle_q)
                self.state_1.particle_qd.assign(self.state_0.particle_qd)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters for visualization
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap model shape and particle data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale
        self.model.particle_radius = self.viz_particle_radius

        try:
            self.viewer.show_triangles = False        # our lime overlay is the only cloth draw
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.viz_state)
            if not hasattr(self, "_lime_tri"):
                self._lime_tri = wp.array(self.model.tri_indices.numpy().reshape(-1).astype(np.int32), dtype=wp.int32)
            self.viewer.log_mesh("shirt_lime", self.viz_state.particle_q, self._lime_tri,
                                 color=wp.vec3(0.66, 0.74, 0.36), backface_culling=False)
            # Render the table box manually at meter scale
            self.viewer.log_shapes(
                "/table",
                newton.GeoType.BOX,
                self.table_viz_scale,
                self.table_viz_xform,
                self.table_viz_color,
            )
            self.viewer.end_frame()
        finally:
            # Restore simulation shape and particle data
            self.model.shape_transform = self.sim_shape_transform
            self.model.shape_scale = self.sim_shape_scale
            self.model.particle_radius = self.sim_particle_radius

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    # 10500 frames = 175s of sim, which silently truncates long plans: the run EXITS CLEANLY
    # mid-fold with no error and no "plan complete", which looked like a crash for hours.
    # Budget per move instead - a pinched fold plus settle runs 25-40s of sim, and unfolds
    # double the move count.
    _moves = max(1, len(_PLAN) * (2 if _os.environ.get("TWIN_UNFOLD", "0") == "1" else 1))
    if _PLANS:
        _moves = sum(len(n_) for _, n_ in _PLANS) + (1 if _STACK_ON else 0)
    if _LABEL_MOVES:
        _moves = len(_LABEL_MOVES) + (1 if _STACK_ON else 0)
    if _UNFOLD_STACK and not _PLAN:
        try:
            _moves = max(1, len(__import__("json").load(open(_UNFOLD_STACK))["stack"]))
        except Exception:
            pass
    parser.set_defaults(num_frames=200000 if _MODE == "teleop" else 3000 + _moves * 3600)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
