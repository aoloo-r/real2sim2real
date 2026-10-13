"""Record a Newton cloth fold as a time-sampled USD for Isaac Sim / Isaac Lab playback.

Isaac Lab's Newton integration has no cloth support yet, so the garment cannot be SIMULATED
there - but a fold already computed by Newton can be PLAYED there. This runs a fold sequence
headless in fold_env, samples the cloth mesh every N frames, and writes a USD stage whose
mesh points are time-sampled. Open it in Isaac Sim (or any USD viewer) and press play: the
same tuned-shirt fold, RTX-rendered.

    python fold_to_usd.py --start obs.npy --plan half_collar_dn,half_right_in,half_collar_dn \
        --out fold.usd [--cloth shirt] [--every 2]
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="", help="observed .npy state to fold from")
    ap.add_argument("--plan", default="half_collar_dn,half_right_in,half_collar_dn")
    ap.add_argument("--cloth", default="shirt")
    ap.add_argument("--out", default="/tmp/fold.usd")
    ap.add_argument("--every", type=int, default=2, help="record every Nth sim frame")
    a = ap.parse_args()

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    from fold_env import FoldEnv
    from fold_search import ACTIONS

    env = FoldEnv(n_worlds=1, view=False, cloth=a.cloth)
    if a.start.endswith(".npy"):
        env.reset(q=np.load(a.start).astype(np.float32), randomize=False)
    else:
        env.reset()
    env.measure_flat()

    faces = env.model.tri_indices.numpy().reshape(-1, 3)
    n_pts = env.per_world

    stage = Usd.Stage.CreateNew(a.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)          # the sim runs in centimetres
    fps = env.FPS / a.every
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/garment")
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.83, 0.90, 0.28)])   # the lime the twin renders
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    pts_attr = mesh.GetPointsAttr()

    # the table under the garment, sized off the settled cloth footprint
    q0 = env.particles()[0]
    lo, hi = q0.min(0), q0.max(0)
    table = UsdGeom.Cube.Define(stage, "/World/table")
    table.CreateSizeAttr(1.0)
    tx = UsdGeom.Xformable(table)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    tz = float(np.percentile(q0[:, 2], 2.0)) - 7.6
    tx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, tz))
    tx.AddScaleOp().Set(Gf.Vec3f(120.0, 120.0, 15.0))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])

    light = UsdLux.DistantLight.Define(stage, "/World/sun")
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(400.0)

    frame = {"i": 0, "t": 0}

    def record(e):
        if frame["i"] % a.every == 0:
            q = e.state_0.particle_q.numpy()[:n_pts]
            pts_attr.Set([Gf.Vec3f(*map(float, v)) for v in q], Usd.TimeCode(frame["t"]))
            frame["t"] += 1
        frame["i"] += 1

    env.frame_cb = record

    plan = [p.strip() for p in a.plan.split(",") if p.strip()]
    for i, name in enumerate(plan):
        print(f"[usd] fold {i + 1}/{len(plan)}: {name}", flush=True)
        env.step(ACTIONS[name])
        print(f"[usd]   extent now {env.extents()[0][0]:.0f}x{env.extents()[0][1]:.0f}cm, "
              f"{frame['t']} frames recorded", flush=True)

    stage.SetEndTimeCode(max(frame["t"] - 1, 1))
    stage.GetRootLayer().Save()
    print(f"[usd] wrote {a.out}: {frame['t']} frames at {fps:.0f}fps, "
          f"{n_pts} points, {len(faces)} tris", flush=True)
    print(f"[usd] open in Isaac Sim and press PLAY", flush=True)


if __name__ == "__main__":
    main()
