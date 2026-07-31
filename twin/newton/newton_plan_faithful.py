"""PLAN-FAITHFUL playback of a cuTAMP plan, rendered in cuTAMP's own frame from the
exported plan JSON (statics + initial + placements) — no frame-mapping guesswork.

Each object is drawn as the box proxy cuTAMP actually planned with. In the plan's action
order: rigids lift and drop into the box; garments FOLD in place (flat footprint box
shrinks to the compact bundle box = the Fold operator's geometry swap), then the bundle
is carried to cuTAMP's placement. A PREVIEW of the plan before any real execution.

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_plan_faithful.py --plan /tmp/santa_plan.json \
      --scene_dir .../outputs/robot_20260731_143933 \
      --order "rigid_2,fold:santa_shirt,santa_shirt,rigid_1,fold:santa_pants,santa_pants" \
      --bundles "santa_shirt:12,13,9;santa_pants:12,13,10" \
      --flat "santa_shirt:40,39;santa_pants:31,31"
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL, ViewerNull

GARMENT_KW = ("shirt", "cloth", "pants", "trouser", "legging", "garment", "towel", "fabric", "tee")


def parse_dims(spec):
    out = {}
    for s in filter(None, spec.split(";")):
        nm, d = s.split(":"); out[nm] = tuple(float(v) / 100.0 for v in d.split(","))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--bundles", default="")
    ap.add_argument("--flat", default="", help="name:w,h;... flat garment footprint (cm)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=900.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    initial = plan["initial"]
    place = {p["name"]: p for p in plan["placements"]}
    statics = plan["statics"]
    bundles = parse_dims(args.bundles)
    flats = parse_dims(args.flat)

    layout = json.load(open(os.path.join(args.scene_dir, "scene_layout.json")))
    col_of = {}
    for o in layout["objects"]:
        col_of[o["label"].lower().replace(" ", "_")] = o.get("display_color", [0.6, 0.6, 0.65])

    wp.init()
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

    # statics from the plan (box floor + walls); skip the huge table + staging for a clean box view
    for sn, s in statics.items():
        if sn in ("table", "staging_zone"):
            continue
        p, d = s["pose"], s["dims"]
        builder.add_shape_box(-1, wp.transform(wp.vec3(p[0], p[1], p[2]), wp.quat_identity()),
                              hx=d[0] / 2, hy=d[1] / 2, hz=d[2] / 2, cfg=cfg,
                              color=wp.vec3(0.72, 0.60, 0.42))

    # each movable: a body with TWO shapes (flat + bundle); we scale whichever is active by
    # moving the inactive one far below. Simpler: one box body we resize by re-placing verts is
    # not supported, so use a body per state and toggle visibility via z-parking.
    PARK = -1.0
    body_ids, bundle_ids, start = {}, {}, {}
    for nm, pose in initial.items():
        is_g = any(k in nm for k in GARMENT_KW)
        col = wp.vec3(*col_of.get(nm, [0.6, 0.6, 0.65]))
        if is_g:
            fw, fh = flats.get(nm, (0.38, 0.38))
            bw, bd, bh = bundles.get(nm, (0.12, 0.13, 0.09))
            b = builder.add_body(xform=wp.transform(wp.vec3(*pose[:3]), wp.quat_identity()), label=nm)
            builder.add_shape_box(b, hx=fw / 2, hy=fh / 2, hz=0.008, cfg=cfg, color=col)
            bb = builder.add_body(xform=wp.transform(wp.vec3(pose[0], pose[1], PARK), wp.quat_identity()),
                                  label=f"{nm}_bundle")
            builder.add_shape_box(bb, hx=bw / 2, hy=bd / 2, hz=bh / 2, cfg=cfg, color=col)
            bundle_ids[nm] = bb
        else:
            b = builder.add_body(xform=wp.transform(wp.vec3(*pose[:3]), wp.quat_identity()), label=nm)
            builder.add_shape_box(b, hx=0.045, hy=0.025, hz=0.015, cfg=cfg, color=col)
        body_ids[nm] = b
        start[nm] = np.array(pose[:3])

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    bq = state.body_q.numpy()

    allpos = np.array([initial[n][:3] for n in initial])
    cx, cy = allpos[:, 0].mean(), allpos[:, 1].mean()
    if args.viewer == "gl":
        viewer = ViewerGL(); viewer.set_model(model)
        viewer.set_camera(wp.vec3(cx - 0.02, cy - 0.42, 0.40), -44, 90)
    else:
        viewer = ViewerNull(num_frames=10**9); viewer.set_model(model)

    def setb(bid, pos):
        bq[bid, :3] = pos
        state.body_q = wp.array(bq, dtype=wp.transform)

    def render(t):
        viewer.begin_frame(t); viewer.log_state(state); viewer.end_frame()

    fps, LIFT = 60.0, 0.14
    t = 0.0
    render(t)
    for action in args.order.split(","):
        if action.startswith("fold:"):
            nm = action.split(":", 1)[1]
            p = start[nm].copy()
            bz = bundles.get(nm, (0, 0, 0.09))[2] / 2 + 0.005
            n = int(1.0 * fps)
            for k in range(n):                          # flat sinks, bundle rises in place
                s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
                bq[body_ids[nm], 2] = p[2] + (PARK - p[2]) * s
                bq[bundle_ids[nm], :3] = [p[0], p[1], PARK + (bz - PARK) * s]
                state.body_q = wp.array(bq, dtype=wp.transform)
                t += 1 / fps; render(t)
            body_ids[nm] = bundle_ids[nm]               # now move the bundle
            start[nm] = np.array([p[0], p[1], bz])
            print(f"[faithful] folded {nm}", flush=True)
            for _ in range(int(0.3 * fps)):
                t += 1 / fps; render(t)
            continue
        nm = action
        if nm not in place:
            continue
        p0 = start[nm].copy(); p1 = np.array(place[nm]["pos"])
        hi0 = p0 + [0, 0, LIFT]; hi1 = np.array([p1[0], p1[1], p0[2] + LIFT])
        for a, bb_, dur in ((p0, hi0, 0.5), (hi0, hi1, 1.4), (hi1, p1, 0.6)):
            n = int(dur * fps)
            for k in range(n):
                s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
                setb(body_ids[nm], a + (bb_ - a) * s)
                t += 1 / fps; render(t)
        start[nm] = p1
        print(f"[faithful] placed {nm} at {p1.round(3)}", flush=True)
        for _ in range(int(0.4 * fps)):
            t += 1 / fps; render(t)

    print("[faithful] plan complete — all in box", flush=True)
    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[faithful] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        t += 1 / fps; render(t)


if __name__ == "__main__":
    main()
