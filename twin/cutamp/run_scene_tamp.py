"""SCENE-TAMP step 1: build the cuTAMP environment from the REAL perceived scene —
every movable at its true relative pose (including the green cloth stacked ON the
shirt) instead of the artificial spread-out staging of run_sam3d_pack.

Why: the plan must come FROM the planner (user mandate). With real initial poses,
ordering emerges geometrically — a skeleton that grasps a covered object first dies
via CollisionFreeGrasp against whatever rests on it, exactly how rigid-first emerged
in the mixed pack. This script validates that plumbing on scene robot_20260729_131708:
  movables = 2 black blocks + green cloth (stacked on the shirt slab)
  goal     = blocks packed in the real box, cloth set aside on a staging zone
The shirt is a STATIC obstacle for now — it becomes a movable once the Fold operator
lands (step 2: mid-plan collision-sphere swap; step 3: fold-count via competing
Fold-N skeletons).

Run (isaaclab env):  python run_scene_tamp.py --disable_visualizer
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sam3d_pack import (_slab_spheres, _rest_z, _build_config,        # noqa: E402
                            default_constraint_to_mult, default_constraint_to_tol)

from curobo.types.base import TensorDeviceType                              # noqa: E402
from curobo.geom.types import Cuboid                                        # noqa: E402
from cutamp.tamp_domain import HandEmpty, On                                # noqa: E402
from cutamp.utils.shapes import MultiSphere                                 # noqa: E402
from cutamp.envs import TAMPEnvironment                                     # noqa: E402
from cutamp.envs.utils import create_walls_for_cuboid, unit_quat            # noqa: E402
_STAGING = "staging_zone"
# register the staging surface in BOTH dicts (gotcha: unknown surface name ->
# tolerance 0 + weight 0 -> 0/1024 forever)
from cutamp.task_planning.constraints import StablePlacement                # noqa: E402
default_constraint_to_tol[StablePlacement.type][f"{_STAGING}_in_xy"] = 1e-3
default_constraint_to_tol[StablePlacement.type][f"{_STAGING}_support"] = 1e-2
default_constraint_to_mult[StablePlacement.type][f"{_STAGING}_in_xy"] = 2.0
default_constraint_to_mult[StablePlacement.type][f"{_STAGING}_support"] = 2.0


def load_real_scene_env(scene_dir, box_cx=0.42, wall_h=0.05,
                        tensor_args=TensorDeviceType()):
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    objs = {o["id"]: o for o in layout["objects"]}

    def dims(o):
        di = o["depth_info"]
        return float(di["physical_width_m"]), float(di["physical_height_m"])

    def color(o):
        return [int(round(255 * c)) for c in o.get("display_color", [0.6, 0.6, 0.6])]

    def px_center(o):
        x0, y0, x1, y1 = o["box_px"]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    box_o = next(o for o in layout["objects"] if "box" in o["label"].lower())
    shirt_o = next(o for o in layout["objects"] if "shirt" in o["label"].lower())
    cloth_o = next(o for o in layout["objects"] if "cloth" in o["label"].lower())
    blocks = [o for o in layout["objects"] if "block" in o["label"].lower()
              or "rectangular" in o["label"].lower()]

    # map image-plane layout -> table plane, preserving the REAL relative arrangement
    # (photo: box left, blocks beside it, shirt right, cloth ON the shirt). Scale px ->
    # metres so the whole cluster spans ~0.55 m in front of the robot at box_cx.
    cxs = [px_center(o)[0] for o in layout["objects"]]
    scale = 0.55 / max(1.0, (max(cxs) - min(cxs)))
    bx_px = px_center(box_o)

    def table_xy(o):
        px, py = px_center(o)
        # UNIFORM px->m scale both axes (a per-axis fudge distorted the layout enough
        # to put a block inside the shirt slab -> cuTAMP initial-state collision)
        return (box_cx + (py - bx_px[1]) * scale,
                0.12 - (px - bx_px[0]) * scale)

    table = Cuboid(name="table", dims=[1.1, 1.5, 0.02],
                   pose=[0.15, 0.0, -0.011, *unit_quat], color=[235, 196, 145])
    bw, bh = dims(box_o)
    bx, by = table_xy(box_o)
    box_floor = Cuboid(name="box_floor", dims=[bw * 0.85, bh * 0.85, 0.01],
                       pose=[bx, by, 0.005, *unit_quat], color=color(box_o))
    walls = create_walls_for_cuboid(box_floor, wall_height=wall_h,
                                    wall_thickness=0.012, wall_color=color(box_o))
    # staging on FREE table (x < shirt's near edge): overlapping the shirt's initial
    # footprint put the hovering shirt INSIDE this static slab -> movable_to_world 0/1024
    staging = Cuboid(name=_STAGING, dims=[0.26, 0.30, 0.01],
                     pose=[0.16, -0.36, 0.005, *unit_quat], color=[210, 210, 215])

    # STEP 2: garments are FOLDABLE movables — flat slab spheres at their REAL poses
    # (cloth stacked ON the shirt), folded bundle spheres registered from the twin's
    # measured fold curves. The Fold operator swaps geometry mid-plan.
    from cutamp.fold_registry import register_foldable, clear as clear_foldables
    clear_foldables()
    sw, sh = dims(shirt_o)
    cap = float(shirt_o.get("physical_size_m", 0.45))
    sw, sh = min(sw, cap), min(sh, cap)
    sx, sy = table_xy(shirt_o)

    movables = []
    foldables = []
    # shirt: flat slab at its real pose; folded bundle = the robot-executed garment fold
    # measured in the twin (24x29x12cm, /tmp/fold_hb6 chain)
    # PLANNING MARGIN: cuTAMP's collision constraints require strict separation — resting
    # contact reads as collision (0/1024 from t=0). Hover supports by ~7mm: physically a
    # no-op, geometrically preserves the occlusion (cloth still blocks shirt grasps).
    GAP = 0.007
    sh_sph = _slab_spheres(sw, sh, 0.015, 0.02, tensor_args)
    shirt = MultiSphere(name="shirt", spheres=sh_sph,
                        pose=[sx, sy, _rest_z(sh_sph) + GAP, *unit_quat], color=color(shirt_o))
    movables.append(shirt); foldables.append(shirt)
    register_foldable("shirt", _slab_spheres(0.24, 0.29, 0.12, 0.02, tensor_args))

    # cloth stacked ON the shirt — its real configuration; folded = analytic 2-fold
    # (37x29 flat -> ~19x15x3; pin-sweep measurement pending)
    cw, ch = dims(cloth_o)
    cl_sph = _slab_spheres(cw, ch, 0.012, 0.02, tensor_args)
    cx, cy = table_xy(cloth_o)
    # stack the cloth ABOVE the shirt with clearance computed from ACTUAL sphere extents
    # (mixed slab/handle z-conventions left the sphere surfaces interpenetrating)
    shirt_top = float(shirt.pose[2] + (sh_sph[:, 2] + sh_sph[:, 3]).max())
    cloth_bot_off = float((cl_sph[:, 2] - cl_sph[:, 3]).min())
    cloth_z = shirt_top + GAP - cloth_bot_off
    cloth = MultiSphere(name="green_cloth", spheres=cl_sph,
                        pose=[cx, cy, cloth_z, *unit_quat],
                        color=color(cloth_o))
    movables.append(cloth); foldables.append(cloth)
    register_foldable("green_cloth", _slab_spheres(0.19, 0.15, 0.03, 0.015, tensor_args))
    for i, o in enumerate(blocks):
        w, h = dims(o)
        sph = _slab_spheres(w, min(h, 0.04), 0.03, 0.015, tensor_args)
        x, y = table_xy(o)
        # mapping is approximate — if a block lands inside the shirt slab or box
        # footprint (+margin), project it just outside the nearest edge
        for ox, oy, ow, oh in ((sx, sy, sw, sh), (bx, by, bw, bh)):
            mx, my = ow / 2 + 0.05, oh / 2 + 0.05
            if abs(x - ox) < mx and abs(y - oy) < my:
                if mx - abs(x - ox) < my - abs(y - oy):
                    x = ox + (mx if x >= ox else -mx)
                else:
                    y = oy + (my if y >= oy else -my)
        movables.append(MultiSphere(name=f"block_{i+1}", spheres=sph,
                                    pose=[x, y, _rest_z(sph) + GAP, *unit_quat], color=color(o)))

    # goal: blocks + folded cloth INTO the box; folded shirt to staging (its measured
    # 24x29 bundle exceeds the box interior — honest capacity result from the twin)
    goal_state = frozenset({HandEmpty.ground(),
                            On.ground("green_cloth", box_floor.name),
                            On.ground("shirt", _STAGING),
                            *[On.ground(m.name, box_floor.name) for m in movables
                              if m.name.startswith("block")]})
    plain_movables = [m for m in movables if m not in foldables]
    env = TAMPEnvironment(
        name="real_scene_tamp",
        movables=movables,
        statics=[table, box_floor, *walls, staging],
        type_to_objects={"Movable": plain_movables, "Foldable": foldables,
                         "Surface": [table, box_floor, staging]},
        goal_state=goal_state,
    )
    print(f"[scene_tamp] movables={[m.name for m in plain_movables]} foldables="
          f"{[m.name for m in foldables]} | shirt flat ({sx:.2f},{sy:.2f}) "
          f"{sw*100:.0f}x{sh*100:.0f}cm | cloth ON shirt ({cx:.2f},{cy:.2f}) | "
          f"box ({bx:.2f},{by:.2f}) {bw*100:.0f}x{bh*100:.0f}cm", flush=True)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default="/home/aoloo/sam-3d-objects/outputs/robot_20260729_131708")
    ap.add_argument("--box_x", type=float, default=0.42)
    ap.add_argument("--wall_h", type=float, default=0.05)
    ap.add_argument("--num_particles", type=int, default=1024)
    ap.add_argument("--num_opt_steps", type=int, default=1000)
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--disable_visualizer", action="store_true")
    ap.add_argument("--experiment_id", default=None)
    ap.add_argument("--fold_area", type=float, default=0.26)   # keeps _build_config happy
    ap.add_argument("--interior_scale", type=float, default=1.0)
    ap.add_argument("--obj_scale", type=float, default=1.0)
    ap.add_argument("--export", default=None, help="write winning placements JSON for Newton playback")
    args = ap.parse_args()

    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import setup_logging
    setup_logging()

    captured = {}
    if args.export:
        from cutamp.utils import visualizer as _viz
        _orig = _viz.MockVisualizer.log_mat4x4

        def _capture(self, name, mat4x4):
            if name.startswith("world/"):
                captured[name.split("world/")[-1]] = (
                    mat4x4.tolist() if hasattr(mat4x4, "tolist") else mat4x4)
            return _orig(self, name, mat4x4)
        _viz.MockVisualizer.log_mat4x4 = _capture
        args.disable_visualizer = True

    env = load_real_scene_env(args.scene_dir, box_cx=args.box_x, wall_h=args.wall_h)
    config = _build_config(args, curobo_plan=False, warmup_motion_gen=False, exp_logging=False)
    # ORDERINGS DECIDED BY PHYSICS: the explored-state dedup collapses action orderings
    # that reach the same SYMBOLIC state (Fold(shirt);Fold(cloth) == Fold(cloth);Fold(shirt))
    # — but geometrically only cloth-first works (the cloth lies ON the shirt). Disable the
    # dedup so order variants reach the continuous layer, which kills the infeasible ones.
    from dataclasses import replace as _dc_replace
    config = _dc_replace(config, explored_state_check=False)
    best, num_sat = run_cutamp(env, config,
                               CostReducer(default_constraint_to_mult.copy()),
                               ConstraintChecker(default_constraint_to_tol.copy()))
    print(f"[scene_tamp] satisfying: {num_sat}/{args.num_particles}", flush=True)
    if args.export:
        from run_sam3d_pack import _mat_to_pos_quat
        out = {"scene_dir": args.scene_dir, "num_satisfying": int(num_sat),
               "initial": {m.name: list(m.pose) for m in env.movables},
               "statics": {s.name: {"pose": list(s.pose), "dims": list(s.dims)}
                           for s in env.statics if hasattr(s, "dims")},
               "placements": []}
        for m in env.movables:
            if m.name in captured:
                pos, quat = _mat_to_pos_quat(captured[m.name])
                out["placements"].append({"name": m.name, "pos": pos, "quat_wxyz": quat})
        with open(args.export, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[scene_tamp] exported {len(out['placements'])} placements -> {args.export}", flush=True)


if __name__ == "__main__":
    main()
