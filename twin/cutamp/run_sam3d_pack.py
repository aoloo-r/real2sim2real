#!/usr/bin/env python
"""
cuTAMP env stub: pack real reconstructed objects into the real reconstructed box.

This is the first bridge from our real2sim reconstruction to cuTAMP's differentiable
GPU TAMP planner. It ingests OUR geometry (not a toy Tetris scene):

  * Container      = the real SAM3D "cardboard box" (footprint + colour read from
                     the scene_layout.json of a captured RGBD scene).
  * Folded garment = a rigid/compressible PROXY for a folded shirt. cuTAMP has no
                     deformable pathway, so we follow the standard packing-TAMP
                     approximation: fold in Newton, freeze the result to a bounding
                     slab. Footprint is derived from the real flat shirt size times
                     the Newton-measured fold ratio (~26% of flat area, i.e. ~0.51x
                     per linear dimension for a 5-grasp garment fold).
  * Rigid item     = a small reconstructed rigid object (stand-in cube) so we exercise
                     the MIXED rigid + deformable-proxy packing case.

The planner then searches plan skeletons (which object first, what order), samples
thousands of GPU particles, and runs differentiable optimisation to place every item
inside the box collision-free and reachable. Run with --disable_visualizer on headless.

Run (isaaclab env, which has cuRobo + cuTAMP installed):
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  /home/aoloo/miniforge3/envs/isaaclab/bin/python run_sam3d_pack.py \
      --scene_dir /home/aoloo/sam-3d-objects/outputs/robot_20260715_192119 \
      --disable_visualizer
"""
import argparse
import json
import math
import os

import torch
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import create_walls_for_cuboid, unit_quat
from cutamp.scripts.run_cutamp import cutamp_demo
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import HandEmpty, On
from cutamp.task_planning.constraints import StablePlacement
from cutamp.utils.shapes import MultiSphere

# cuTAMP keys StablePlacement tolerances AND optimizer weights by the SUPPORT SURFACE
# NAME (goal_support, floor_support, ...). Our container surface is "box_floor", so we
# must register it in both dicts or the support constraint gets tolerance 0.0 (impossible)
# and weight 0 (never optimized). This is the framework's documented extension point.
_SURFACE = "box_floor"
default_constraint_to_tol[StablePlacement.type][f"{_SURFACE}_in_xy"] = 1e-3
default_constraint_to_tol[StablePlacement.type][f"{_SURFACE}_support"] = 1e-2
# Match the DEFAULT (untuned) weight: support=2.0, no in_xy weight. The tetris-TUNED
# weights (support 6.28) over-weight placement and collapse the place-IK (pos_err->0).
default_constraint_to_mult[StablePlacement.type][f"{_SURFACE}_support"] = 2.0

# A FOLDED garment is a compact bundle (like a shirt on a store shelf), NOT a wide flat
# sheet -- that's what the Newton 5-grasp fold actually produces and what packs well.
# We conserve the folded VOLUME: footprint area ~= 26% of flat (Newton-measured
# FOLDED FOOTPRINT log) at a bundle thickness, then make the footprint compact/square
# so it is graspable top-down. See project_cutamp memory.
FOLD_AREA_RATIO = 0.26   # folded footprint area / flat area (Newton-measured)
FOLD_BUNDLE_H = 0.05     # folded bundle thickness (m)
FOLD_MAX_ASPECT = 1.4    # keep the bundle roughly square (graspable), not a long strip


def _load_scene(scene_dir):
    """Read real box + shirt footprint and colours from a SAM3D scene_layout.json."""
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    box, shirt = None, None
    for o in layout["objects"]:
        label = o.get("depth_info", {}).get("label", "").lower()
        entry = {
            "w": o["depth_info"]["physical_width_m"],
            "h": o["depth_info"]["physical_height_m"],
            "color": [int(round(255 * c)) for c in o.get("display_color", [0.6, 0.6, 0.6])],
        }
        if "box" in label:
            box = entry
        elif "shirt" in label or "cloth" in label or "garment" in label or "t-shirt" in label:
            shirt = entry
    if box is None:
        raise ValueError(f"No 'box' object found in {scene_dir}/scene_layout.json")
    if shirt is None:  # fall back to a default garment if scene has no cloth
        shirt = {"w": 0.33, "h": 0.26, "color": [190, 215, 95]}
    return box, shirt


def _slab_spheres(dx, dy, dz, r, tensor_args):
    """Decompose a dx*dy*dz slab (centred at origin) into a grid of collision spheres,
    plus two small grasp-handle spheres above the centre for a top-down (4-DOF) grasp.
    Returns (n,4) [x,y,z,radius]; poses are lifted at env-build time so the slab rests
    on the surface (mirrors the Tetris block convention)."""
    def axis(length):
        n = max(1, int(round(length / (2 * r))))
        if n == 1:
            return [0.0]
        half = length / 2 - r
        return [(-half + 2 * half * i / (n - 1)) for i in range(n)]

    xs, ys, zs = axis(dx), axis(dy), axis(dz)
    sph = [[x, y, z, r] for x in xs for y in ys for z in zs]
    top = max(zs)
    # Tall grasp stalk: gripper must clear a WIDE flat footprint, so the grasp point
    # sits well above the slab. Scale stalk height with footprint so the fingers
    # don't collide with the wings.
    stalk = max(0.06, 0.35 * max(dx, dy))
    sph.append([0.0, 0.0, top + stalk * 0.66, 0.01])  # grasp handle spheres
    sph.append([0.0, 0.0, top + stalk, 0.01])
    out = tensor_args.to_device(sph)
    # CRITICAL: the 4-DOF grasp sampler grasps a MultiSphere at object-frame z=0.
    # Shift so the TOP handle sphere sits at z=0 (the grasp point) with the whole
    # body hanging below -- exactly the Tetris convention. Otherwise the grasp lands
    # inside the slab and every grasp collides (robot_to_movables fails).
    out[:, 2] -= out[-1, 2].item()
    return out


def _rest_z(spheres):
    """z offset so the lowest sphere just touches the surface (z=0)."""
    return -(spheres[:, 2] - spheres[:, 3]).min().item() + 1e-2


def _start_poses(box_cx, box_hh, obj_hw, n):
    """Distinct staging spots on the table around the box: two x-columns either side of
    the box, spaced by the object footprint so staged objects DON'T overlap each other,
    and offset in y so they clear the box interior (box spans y in [-box_hh, box_hh]).
    Overlapping start poses are an unrecoverable collision, so spacing is essential."""
    spacing = max(0.24, 2 * obj_hw + 0.06)
    y0 = box_hh + obj_hw + 0.05
    ys = [y0, -y0, y0 + spacing, -(y0 + spacing), y0 + 2 * spacing, -(y0 + 2 * spacing)]
    ys = [y for y in ys if abs(y) <= 0.55]
    xs = [box_cx - 0.16, box_cx + 0.16]
    spots = [[round(x, 3), round(y, 3)] for y in ys for x in xs]
    if n > len(spots):
        print(f"[sam3d_pack] WARNING: only {len(spots)} non-overlapping staging spots "
              f"available for {n} objects; capping.", flush=True)
    return spots


def _register_box_tilt(box_cx, box_cy, floor_top_z, iw, ih, tilt_deg):
    """Register a TILTED support plane for the 'box_floor' surface so cuTAMP plans placements
    flush on it (see cutamp/tilt_registry). R leans the floor by tilt_deg about the x-axis;
    origin = floor-top centre; bounds = box-local footprint. Returns the cuboid quat (wxyz)
    so the collision geometry is tilted to match the support plane."""
    import numpy as np
    import torch
    from cutamp.tilt_registry import SURFACE_TILT
    th = math.radians(tilt_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])   # lean about +x
    origin = np.array([box_cx, box_cy, floor_top_z])
    T4 = np.eye(4); T4[:3, :3] = R; T4[:3, 3] = origin
    SURFACE_TILT["box_floor"] = {
        "T4": torch.tensor(T4, dtype=torch.float32),
        "R": torch.tensor(R, dtype=torch.float32),
        "origin": torch.tensor(origin, dtype=torch.float32),
        "bounds": (-iw / 2, iw / 2, -ih / 2, ih / 2),
        "floor_z": 0.0,
    }
    # cuboid quat (wxyz) from R for the tilted collision geometry
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w) if w > 1e-6 else 0.0
    return [w, x, 0.0, 0.0]


def load_sam3d_pack_env(scene_dir, n_shirts=2, add_rigid=True, box_cx=0.42,
                        wall_h=0.05, interior_scale=1.0, obj_scale=1.0,
                        fold_area_ratio=FOLD_AREA_RATIO, tilt_deg=0.0,
                        bundle_wdh=None, tensor_args=TensorDeviceType()):
    box, shirt = _load_scene(scene_dir)

    # --- Table ---
    table = Cuboid(name="table", dims=[1.1, 1.5, 0.02],
                   pose=[0.15, 0.0, -0.011, *unit_quat], color=[235, 196, 145])

    # --- Container = real reconstructed box, as a goal floor + 4 walls ---
    # Interior footprint = real box footprint (wall thickness folded into the margin).
    # box_cx / wall_h / interior_scale are exposed because a realistically LARGE box
    # (0.30m) at 0.45m pushes far-edge placements past panda's dexterous reach over the
    # walls -- the same overstretch limit seen in the Newton fold. Bring it into reach.
    box_cy = 0.0
    interior_w, interior_h = box["w"] * interior_scale, box["h"] * interior_scale
    if tilt_deg > 0:
        # TILT-AWARE: register a tilted support plane and orient the floor collider to match.
        # Walls dropped (create_walls needs a unit quat; the tilted support+collision floor is
        # what the placement rests on). cuTAMP now plans placements flush on the tilted plane.
        floor_quat = _register_box_tilt(box_cx, box_cy, 0.01, interior_w, interior_h, tilt_deg)
        box_floor = Cuboid(name="box_floor", dims=[interior_w, interior_h, 0.01],
                           pose=[box_cx, box_cy, 0.005, *floor_quat], color=box["color"])
        walls = []
    else:
        box_floor = Cuboid(name="box_floor",
                           dims=[interior_w, interior_h, 0.01],
                           pose=[box_cx, box_cy, 0.005, *unit_quat],
                           color=box["color"])
        walls = create_walls_for_cuboid(box_floor, wall_height=wall_h,
                                        wall_thickness=0.012, wall_color=box["color"])

    # --- Folded-garment proxies (deformable frozen to a compact bundle) ---
    # Conserve folded footprint AREA (= 26% of flat) but make it roughly square/compact.
    import math
    if bundle_wdh is not None:
        # MEASURED folded-shirt proxy (from the Newton physical fold), possibly with a
        # documented compression allowance — overrides the area-ratio estimate
        fw, fh, bundle_h = bundle_wdh
    else:
        fold_area = fold_area_ratio * shirt["w"] * shirt["h"]
        aspect = min(FOLD_MAX_ASPECT, shirt["w"] / shirt["h"])
        fh = math.sqrt(fold_area / aspect) * obj_scale
        fw = aspect * fh
        bundle_h = FOLD_BUNDLE_H
    r = 0.02                              # collision-sphere radius for the bundle
    shirt_sph = _slab_spheres(fw, fh, bundle_h, r, tensor_args)
    shirt_z = _rest_z(shirt_sph)

    # spread the movables around the box, within reach, not overlapping each other
    start_poses = _start_poses(box_cx, interior_h / 2, fw / 2, n_shirts + (1 if add_rigid else 0))
    movables = []
    for i in range(n_shirts):
        sx, sy = start_poses[i % len(start_poses)]
        movables.append(MultiSphere(
            name=f"shirt_{i+1}", spheres=shirt_sph.clone(),
            pose=[sx, sy, shirt_z, *unit_quat], color=shirt["color"]))

    # --- One rigid reconstructed item (stand-in cube) for the mixed case ---
    if add_rigid:
        rc = 0.06
        cube_sph = _slab_spheres(rc, rc, rc, rc / 2, tensor_args)
        cube_z = _rest_z(cube_sph)
        sx, sy = start_poses[n_shirts % len(start_poses)]
        movables.append(MultiSphere(
            name="rigid_1", spheres=cube_sph,
            pose=[sx, sy, cube_z, *unit_quat], color=[150, 150, 160]))

    # --- Goal: every movable inside the box, hand empty ---
    goal_state = frozenset({HandEmpty.ground(),
                            *[On.ground(m.name, box_floor.name) for m in movables]})

    env = TAMPEnvironment(
        name=f"sam3d_pack_{len(movables)}",
        movables=movables,
        statics=[table, box_floor, *walls],
        type_to_objects={"Movable": movables, "Surface": [table, box_floor]},
        goal_state=goal_state,
    )
    # geometry meta for the Newton stability replay (metres, panda z-up frame)
    env.meta = {
        "box": {"cx": box_cx, "cy": box_cy, "interior_w": interior_w, "interior_h": interior_h,
                "wall_h": wall_h, "floor_top_z": 0.01, "color": box["color"], "tilt_deg": tilt_deg},
        "bundle": {"w": fw, "h": fh, "thickness": bundle_h, "color": shirt["color"]},
        "rigid": {"size": 0.06, "color": [150, 150, 160]} if add_rigid else None,
    }
    print(f"[sam3d_pack] box interior {interior_w*100:.0f}x{interior_h*100:.0f}cm "
          f"colour{box['color']} | folded bundle {fw*100:.0f}x{fh*100:.0f}x{bundle_h*100:.0f}cm "
          f"(={fold_area_ratio*100:.0f}% of flat area) | movables: "
          f"{[m.name for m in movables]}", flush=True)
    return env


def _build_config(args, curobo_plan=False, warmup_motion_gen=False, exp_logging=True):
    config = TAMPConfiguration(
        num_particles=args.num_particles,
        robot=args.robot,
        grasp_dof=4,
        approach="optimization",
        num_opt_steps=args.num_opt_steps,
        num_initial_plans=30,
        curobo_plan=curobo_plan,
        warmup_motion_gen=warmup_motion_gen,
        enable_visualizer=not args.disable_visualizer,
        enable_experiment_logging=exp_logging,
        movable_activation_distance=getattr(args, "gap", 0.0),
    )
    validate_tamp_config(config)
    return config


def adaptive_pack(args):
    """Adaptive object-count outer loop: keep adding folded garments until cuTAMP can no
    longer pack them into the reconstructed box. Quantifies the fold-count <-> pack-count
    tradeoff: better folds (smaller footprint / lower --fold_area) => more garments fit.

    Stop rule: increase N until a solve yields < --min_satisfy satisfying particles.
    Reports the max N that packs (the box capacity for this fold quality)."""
    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import setup_logging

    setup_logging()
    print(f"\n[adaptive] scene={os.path.basename(args.scene_dir)} fold_area={args.fold_area:.0%} "
          f"box_x={args.box_x} wall_h={args.wall_h} min_satisfy={args.min_satisfy} "
          f"max_n={args.max_n}\n", flush=True)

    best_n, history = 0, []
    for n in range(1, args.max_n + 1):
        env = load_sam3d_pack_env(
            args.scene_dir, n_shirts=n, add_rigid=False, box_cx=args.box_x,
            wall_h=args.wall_h, interior_scale=args.interior_scale,
            obj_scale=args.obj_scale, fold_area_ratio=args.fold_area)
        config = _build_config(args, curobo_plan=False, warmup_motion_gen=False, exp_logging=False)
        cost_reducer = CostReducer(default_constraint_to_mult.copy())
        constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
        _, num_sat = run_cutamp(env, config, cost_reducer, constraint_checker)
        fits = num_sat >= args.min_satisfy
        history.append((n, num_sat, fits))
        print(f"[adaptive] N={n}: {num_sat}/{args.num_particles} satisfying -> "
              f"{'FITS' if fits else 'DOES NOT FIT'}", flush=True)
        if fits:
            best_n = n
        else:
            break  # first N that fails to pack -> box is full

    print(f"\n[adaptive] ===== RESULT: box holds {best_n} folded garment(s) "
          f"at {args.fold_area:.0%}-of-flat fold quality =====", flush=True)
    for n, num_sat, fits in history:
        print(f"[adaptive]   N={n}: {num_sat} satisfying  {'ok' if fits else 'FAIL'}", flush=True)
    return best_n


def _mat_to_pos_quat(M):
    """4x4 (list/np) -> (pos xyz, quat wxyz). roma/curobo use standard homogeneous."""
    import numpy as np
    M = np.asarray(M, dtype=float).reshape(4, 4)
    pos = M[:3, 3].tolist()
    R = M[:3, :3]
    # rotation matrix -> quaternion (wxyz), numerically stable branch
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return pos, [w, x, y, z]


def export_placements(args):
    """Solve one N-garment pack and export cuTAMP's WINNING final placement of each
    object (in the box) + the box/bundle geometry to JSON, for the Newton stability
    replay. Captures placements by monkeypatching the (no-op) MockVisualizer.log_mat4x4,
    which cuTAMP calls with the best-particle final pose of each object."""
    import json as _json
    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import setup_logging
    from cutamp.utils import visualizer as _viz

    setup_logging()
    captured = {}
    _orig = _viz.MockVisualizer.log_mat4x4

    def _capture(self, name, mat4x4):
        if name.startswith("world/"):
            obj = name.split("world/")[-1]
            m = mat4x4.tolist() if hasattr(mat4x4, "tolist") else mat4x4
            captured[obj] = m  # last write wins = final placement of the best particle
        return _orig(self, name, mat4x4)

    _viz.MockVisualizer.log_mat4x4 = _capture
    try:
        env = load_sam3d_pack_env(
            args.scene_dir, n_shirts=args.n_shirts, add_rigid=not args.no_rigid,
            box_cx=args.box_x, wall_h=args.wall_h, interior_scale=args.interior_scale,
            obj_scale=args.obj_scale, fold_area_ratio=args.fold_area, tilt_deg=args.tilt_deg,
            bundle_wdh=tuple(float(v) for v in args.bundle_wdh.split(",")) if args.bundle_wdh else None)
        # force viz off (so MockVisualizer is used and captured), keep logging off
        args.disable_visualizer = True
        config = _build_config(args, curobo_plan=False, warmup_motion_gen=False, exp_logging=False)
        cost_reducer = CostReducer(default_constraint_to_mult.copy())
        constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
        _, num_sat = run_cutamp(env, config, cost_reducer, constraint_checker)
    finally:
        _viz.MockVisualizer.log_mat4x4 = _orig

    movable_names = {m.name for m in env.movables}
    placements = []
    for name in [m.name for m in env.movables]:
        if name not in captured:
            print(f"[export] WARNING: no placement captured for {name}", flush=True)
            continue
        pos, quat = _mat_to_pos_quat(captured[name])
        placements.append({"name": name, "pos": pos, "quat_wxyz": quat})

    out = {
        "scene_dir": args.scene_dir, "n": args.n_shirts,
        "num_satisfying": int(num_sat), "found_solution": num_sat > 0,
        "meta": env.meta, "placements": placements,
    }
    with open(args.export_placements, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"\n[export] {num_sat} satisfying | wrote {len(placements)} placements -> "
          f"{args.export_placements}", flush=True)
    for p in placements:
        print(f"[export]   {p['name']}: pos={[round(v,3) for v in p['pos']]} "
              f"quat={[round(v,3) for v in p['quat_wxyz']]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="cuTAMP pack of real SAM3D reconstructed objects")
    ap.add_argument("--scene_dir", default="/home/aoloo/sam-3d-objects/outputs/robot_20260715_192119")
    ap.add_argument("--n_shirts", type=int, default=2)
    ap.add_argument("--no_rigid", action="store_true", help="omit the rigid cube (garments only)")
    ap.add_argument("--box_x", type=float, default=0.42, help="box center x (m) in panda frame")
    ap.add_argument("--wall_h", type=float, default=0.05, help="box wall height (m)")
    ap.add_argument("--interior_scale", type=float, default=1.0, help="scale real box footprint")
    ap.add_argument("--obj_scale", type=float, default=1.0, help="scale the folded bundle footprint")
    ap.add_argument("--num_particles", type=int, default=1024)
    ap.add_argument("--num_opt_steps", type=int, default=1000)
    ap.add_argument("--robot", default="panda", choices=["panda", "ur5"])
    ap.add_argument("--motion_plan", action="store_true", help="also plan full cuRobo motions for the winner")
    ap.add_argument("--disable_visualizer", action="store_true")
    ap.add_argument("--experiment_id", default=None)
    # Adaptive object-count outer loop
    ap.add_argument("--adaptive", action="store_true",
                    help="run the adaptive object-count loop: add garments until the box is full")
    ap.add_argument("--max_n", type=int, default=8, help="[adaptive] max garments to try")
    ap.add_argument("--min_satisfy", type=int, default=1,
                    help="[adaptive] min satisfying particles to count N as fitting")
    ap.add_argument("--fold_area", type=float, default=FOLD_AREA_RATIO,
                    help="folded footprint as fraction of flat shirt area (lower = better fold => more fit)")
    ap.add_argument("--export_placements", default=None,
                    help="solve N-garment pack and write winning placements+geometry JSON here (for Newton replay)")
    ap.add_argument("--tilt_deg", type=float, default=0.0,
                    help="tilt the box_floor support plane by this many deg -> cuTAMP plans placements flush on it")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="movable_activation_distance: enforce this gap (m) between packed objects")
    ap.add_argument("--bundle_wdh", default=None,
                    help="folded-shirt proxy dims 'w,d,h' in metres (e.g. '0.29,0.27,0.07' = measured "
                         "physical fold); overrides --fold_area")
    args = ap.parse_args()

    if args.export_placements:
        export_placements(args)
        return
    if args.adaptive:
        adaptive_pack(args)
        return

    env = load_sam3d_pack_env(args.scene_dir, n_shirts=args.n_shirts, add_rigid=not args.no_rigid,
                              box_cx=args.box_x, wall_h=args.wall_h, interior_scale=args.interior_scale,
                              obj_scale=args.obj_scale, fold_area_ratio=args.fold_area, tilt_deg=args.tilt_deg,
                              bundle_wdh=tuple(float(v) for v in args.bundle_wdh.split(",")) if args.bundle_wdh else None)
    print(env, flush=True)
    cutamp_demo(env, _build_config(args, curobo_plan=args.motion_plan, warmup_motion_gen=args.motion_plan),
                experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()
