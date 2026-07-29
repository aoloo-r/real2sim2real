"""FOLD-COUNT PLANNER: cuTAMP decides how many folds the shirt needs so that it AND the
other objects all fit the real reconstructed box — including simple vertical stacking.

Hierarchy (v0, honest about its approximations):
  1. The Newton twin MEASURES the fold-count -> bundle-size curve (physical-grasp multi
     folds; pass --measured 'N:w,d,h' entries in cm from those runs, or use the built-in
     analytic fallback calibrated to the measured garment fold).
  2. For each candidate fold count N, cuTAMP packs the FLOOR LAYER (bundle_N + rigid
     items) into the real box — the full 1024-particle differentiable search.
  3. STACKING is a deterministic post-layer: a rigid item can ride ON the bundle if its
     footprint fits inside the bundle top and the combined height clears the walls +
     gripper. (cuTAMP's StablePlacement only supports fixed surfaces; a differentiable
     movable-on-movable stack cost is the documented upgrade path.)
  4. Recommendation = the SMALLEST N that packs everything (fewer folds = fewer grasps =
     less fold-quality risk), preferring flat layouts over stacks at equal N.
Newton settle replay of the winner (incl. the stack) is the validation step, as usual.

Run (isaaclab env):
  python plan_fold_count.py --measured "2:34,25,6" --measured "3:24,26,9" \
      --measured "4:18,20,14" --cube 6 --disable_visualizer
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sam3d_pack import (load_sam3d_pack_env, _build_config,          # noqa: E402
                            default_constraint_to_mult, default_constraint_to_tol)


def analytic_bundle(flat_w_cm, flat_h_cm, n_folds, base_h_cm=2.0):
    """Fallback fold model: each half-fold halves the longer span and doubles height,
    with the measured ~15% bunching overhead per fold (folds are never crisp)."""
    w, h, t = flat_w_cm, flat_h_cm, base_h_cm
    for _ in range(n_folds):
        if w >= h:
            w = w / 2.0 * 1.15
        else:
            h = h / 2.0 * 1.15
        t *= 1.9
    return w, h, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", default="/home/aoloo/sam-3d-objects/outputs/robot_20260715_192119")
    ap.add_argument("--measured", action="append", default=[],
                    help="measured bundle per fold count, 'N:w,d,h' in cm (repeatable)")
    ap.add_argument("--flat", default="48,40", help="flat garment w,h (cm) for the analytic fallback")
    ap.add_argument("--folds", default="2,3,4", help="candidate fold counts")
    ap.add_argument("--cube", type=float, default=6.0, help="rigid cube edge (cm); 0 = no rigid item")
    ap.add_argument("--wall_h", type=float, default=0.05)
    ap.add_argument("--box_x", type=float, default=0.42)
    ap.add_argument("--clearance_h", type=float, default=18.0,
                    help="max stack height (cm): wall top + gripper clearance")
    ap.add_argument("--min_satisfy", type=int, default=20)
    ap.add_argument("--num_particles", type=int, default=1024)
    ap.add_argument("--num_opt_steps", type=int, default=1000)
    ap.add_argument("--interior_scale", type=float, default=1.0)
    ap.add_argument("--obj_scale", type=float, default=1.0)
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--disable_visualizer", action="store_true")
    ap.add_argument("--experiment_id", default=None)
    ap.add_argument("--fold_area", type=float, default=0.26)   # unused; keeps _build_config happy
    args = ap.parse_args()

    from cutamp.algorithm import run_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import setup_logging
    setup_logging()

    measured = {}
    for spec in args.measured:
        n, dims = spec.split(":")
        measured[int(n)] = tuple(float(v) for v in dims.split(","))
    flat_w, flat_h = (float(v) for v in args.flat.split(","))
    candidates = [int(v) for v in args.folds.split(",")]

    def solve(bundle_wdh_m, add_rigid):
        env = load_sam3d_pack_env(
            args.scene_dir, n_shirts=1, add_rigid=add_rigid, box_cx=args.box_x,
            wall_h=args.wall_h, interior_scale=args.interior_scale,
            obj_scale=args.obj_scale, bundle_wdh=bundle_wdh_m)
        config = _build_config(args, curobo_plan=False, warmup_motion_gen=False, exp_logging=False)
        _, num_sat = run_cutamp(env, config,
                                CostReducer(default_constraint_to_mult.copy()),
                                ConstraintChecker(default_constraint_to_tol.copy()))
        return num_sat

    rows = []
    for n in candidates:
        w, d, t = measured.get(n) or analytic_bundle(flat_w, flat_h, n)
        src = "measured" if n in measured else "analytic"
        wdh = (w / 100.0, d / 100.0, t / 100.0)
        flat_sat = solve(wdh, add_rigid=args.cube > 0) if args.cube > 0 else None
        flat_ok = flat_sat is not None and flat_sat >= args.min_satisfy
        # stacked variant: bundle alone on the floor; cube rides on the bundle if it fits
        stack_ok, stack_sat = False, None
        if args.cube > 0 and not flat_ok:
            stack_sat = solve(wdh, add_rigid=False)
            cube_fits_top = args.cube <= min(w, d) - 2.0          # 1cm margin each side
            height_ok = t + args.cube <= args.clearance_h
            stack_ok = stack_sat >= args.min_satisfy and cube_fits_top and height_ok
        solo_sat = flat_sat if args.cube == 0 else None
        rows.append((n, src, (w, d, t), flat_sat, flat_ok, stack_sat, stack_ok))
        print(f"[plan] N={n} ({src}) bundle {w:.0f}x{d:.0f}x{t:.0f}cm | "
              f"flat(bundle+cube): {flat_sat} sat -> {'OK' if flat_ok else 'no'} | "
              f"stacked(cube on bundle): {stack_sat} sat -> {'OK' if stack_ok else 'no'}",
              flush=True)

    print("\n[plan] ===== FOLD-COUNT RECOMMENDATION =====", flush=True)
    pick = None
    for n, src, dims, fs, fo, ss, so in rows:
        verdict = "flat pack" if fo else ("stacked pack" if so else "does not fit")
        print(f"[plan]   {n} folds ({src}, {dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}cm): {verdict}", flush=True)
        if pick is None and (fo or so):
            pick = (n, "flat" if fo else "stacked")
    if pick:
        print(f"[plan] RECOMMEND: {pick[0]} folds, {pick[1]} layout "
              f"(fewest folds that packs everything; validate with newton_pack_settle)", flush=True)
    else:
        print("[plan] RECOMMEND: no candidate packs — better folding or a bigger box", flush=True)


if __name__ == "__main__":
    main()
