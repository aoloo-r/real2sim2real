#!/usr/bin/env python
"""
Newton rigid-body STABILITY REPLAY of a cuTAMP pack plan.

cuTAMP decides placements with a STATIC feasibility check (collision-free + reachable +
resting-on-floor) -- it does NOT simulate physics. This script closes that gap: it reads
the exported placements JSON, drops N rigid folded-garment bundles into the reconstructed
box at cuTAMP's chosen (x, y, yaw), settles them under gravity with contact, and reports
whether the pack is physically STABLE -- i.e. every bundle stays inside the box, none
ejects or topples. This is exactly the validation the research flagged as missing
(cuTAMP's stability is a sampler check, not a rollout), and it matters most for a tilted
or tightly packed box.

Runs at cm-scale with gravity -981 (Newton convention, matches newton_fold.py). Uses
SolverXPBD for rigid contact. Launch with a VISIBLE GL viewer on DISPLAY=:1.

  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  /home/aoloo/miniforge3/envs/newton-spike/bin/python newton_pack_settle.py \
      --plan /tmp/pack_placements.json --viewer gl
"""
import argparse
import json
import math

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverXPBD
from newton.viewer import ViewerGL, ViewerNull

M2CM = 100.0


def yaw_of(quat_wxyz):
    """cuTAMP 4-DOF placements rotate about z only -> recover the yaw angle."""
    w, x, y, z = quat_wxyz
    return 2.0 * math.atan2(z, w)


def load_real_box_cm(scene_dir, capture_dir):
    """Load the REAL reconstructed box mesh (SAM3D) in cm, with its true reconstructed
    tilt baked in (via sam3d_scene: icp_pose + camera->base extrinsics). Calibrated to the
    depth-measured size and snapped so its base rests at z=0. Returns dict with verts (cm),
    faces (flat int32), open-top faces, centroid xy, z range, rim-opening center, colour."""
    import sys
    nd = "/home/aoloo/real2sim2real/twin/newton"
    if nd not in sys.path:
        sys.path.insert(0, nd)
    import sam3d_scene
    sc = sam3d_scene.load_scene(scene_dir, capture_dir)
    box = next((o for o in sc["objects"] if "box" in o["label"].lower()), None)
    if box is None:
        raise SystemExit(f"[settle] no box object in {scene_dir}")
    v = box["verts"].astype(np.float64)
    c = v.mean(0)
    fp = (v[:, :2].max(0) - v[:, :2].min(0)).max()
    meas = max([m for m in box["measured_m"] if m > 0] or [fp])
    v = (c + (meas / fp) * (v - c)) * 100.0          # scale to measured size, -> cm
    v[:, 2] += -v[:, 2].min()                        # base rests on table z=0
    faces = box["faces"].reshape(-1, 3).astype(np.int32)
    zt, zb = v[:, 2].max(), v[:, 2].min()
    keep = v[faces].mean(1)[:, 2] < zb + 0.82 * (zt - zb)   # open the top (drop-in)
    open_faces = faces[keep].reshape(-1).astype(np.int32)
    rim = v[v[:, 2] > zt - 0.30 * (zt - zb)]         # opening rim

    # --- box ORIENTATION frame via PCA: the two largest-variance axes span the opening
    # plane, the smallest-variance axis is the box up-normal (tilted as reconstructed). ---
    c3 = v.mean(0)
    cov = np.cov((v - c3).T)
    evals, evecs = np.linalg.eigh(cov)               # ascending eigenvalues
    e3 = evecs[:, 0]                                  # smallest variance = box normal
    if e3[2] < 0:
        e3 = -e3                                      # point "up" (out of the opening)
    e1 = evecs[:, 2]                                  # largest in-plane axis
    e1 = e1 - (e1 @ e3) * e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)                             # right-handed
    R = np.column_stack([e1, e2, e3])
    if np.linalg.det(R) < 0:
        e2 = -e2
        R = np.column_stack([e1, e2, e3])
    tilt_deg = math.degrees(math.acos(min(1.0, abs(e3[2]))))
    proj = v @ e3
    floor_center = c3 + (proj.min() - c3 @ e3) * e3   # box interior floor centre
    return {
        "verts": v.astype(np.float32), "open_faces": open_faces,
        "cxy": v[:, :2].mean(0), "bbox": [v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()],
        "zt": float(zt), "rim_xy": rim[:, :2].mean(0), "rim_top_z": float(rim[:, 2].mean()),
        "R": R, "e3": e3, "floor_center": floor_center, "centroid": c3, "tilt_deg": tilt_deg,
        "color": box["color"],
    }


def mat3_to_wp_quat(R):
    """3x3 rotation matrix -> wp.quat (x,y,z,w), numerically stable."""
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
    return wp.quat(float(x), float(y), float(z), float(w))


def quat_tilt_deg(q_xyzw):
    """Angle (deg) between the body's local +z and world +z, from a wxyz-or-xyzw quat.
    Newton body_q stores transforms as (px,py,pz, qx,qy,qz,qw)."""
    x, y, z, w = q_xyzw
    # world-z component of the body's local z-axis (rotation matrix column 3, entry zz)
    zz = 1.0 - 2.0 * (x * x + y * y)
    zz = max(-1.0, min(1.0, zz))
    return math.degrees(math.acos(zz))


def main():
    ap = argparse.ArgumentParser(description="Newton stability replay of a cuTAMP pack plan")
    ap.add_argument("--plan", required=True, help="placements JSON from run_sam3d_pack.py --export_placements")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--cam", default="40,-42,34,-38,90", help="x,y,z,pitch,yaw (cm-scale), aimed at box")
    ap.add_argument("--sim_seconds", type=float, default=3.0, help="settle duration")
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--drop_gap", type=float, default=1.0, help="cm above floor to spawn bundles")
    ap.add_argument("--density", type=float, default=200.0)
    ap.add_argument("--mu", type=float, default=0.7, help="friction")
    ap.add_argument("--out_margin", type=float, default=2.0, help="cm slack for in-box test")
    ap.add_argument("--tilt_thresh", type=float, default=30.0, help="deg tilt = toppled")
    ap.add_argument("--hold", type=float, default=600.0)
    ap.add_argument("--screenshot", default=None)
    # real reconstructed (tilted) box mesh as the container
    ap.add_argument("--box_mesh", action="store_true",
                    help="use the REAL reconstructed box MESH (with its true ~tilt) as the container")
    ap.add_argument("--tilt_aware", action="store_true",
                    help="place cuTAMP's arrangement at the box's real orientation (flush on the tilted floor)")
    ap.add_argument("--native_tilt", action="store_true",
                    help="plan already carries tilted placements (cuTAMP planned with tilt); build a tilted "
                         "synthetic box (meta.box.tilt_deg) and use the full exported orientations directly")
    ap.add_argument("--scene_dir", default="/home/aoloo/sam-3d-objects/outputs/robot_20260715_192119")
    ap.add_argument("--capture_dir", default="/home/aoloo/sam-3d-objects/captures/robot_20260715_192119")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    box = plan["meta"]["box"]
    bundle = plan["meta"]["bundle"]
    placements = plan["placements"]
    if not placements:
        raise SystemExit("[settle] no placements in plan -- nothing to replay")

    bx, by = box["cx"] * M2CM, box["cy"] * M2CM
    iw, ih = box["interior_w"] * M2CM, box["interior_h"] * M2CM
    wall_h = box["wall_h"] * M2CM
    floor_top = box["floor_top_z"] * M2CM
    bw, bh, bt = bundle["w"] * M2CM, bundle["h"] * M2CM, bundle["thickness"] * M2CM
    bcol = wp.vec3(*[c / 255.0 for c in bundle["color"]])
    boxcol = wp.vec3(*[c / 255.0 for c in box["color"]])

    print(f"[settle] plan n={plan.get('n')} sat={plan.get('num_satisfying')} | box interior "
          f"{iw:.0f}x{ih:.0f}cm walls {wall_h:.0f}cm | bundle {bw:.0f}x{bh:.0f}x{bt:.0f}cm | "
          f"{len(placements)} bundles", flush=True)

    builder = newton.ModelBuilder(gravity=-981.0)
    wall_t = 1.2  # cm
    stat_cfg = builder.ShapeConfig(density=0.0, mu=args.mu, ke=3.0e4, kd=1.0e2)

    # cuTAMP planned relative to its axis-aligned proxy box centre (ccx, ccy); we re-anchor
    # each bundle's box-relative offset onto the container we actually build below.
    ccx, ccy = box["cx"] * M2CM, box["cy"] * M2CM

    if args.native_tilt:
        # --- SYNTHETIC box tilted by meta.box.tilt_deg (matches the plane cuTAMP planned on);
        #     bundles use cuTAMP's FULL exported (tilted) orientation directly ---
        tdeg = float(box.get("tilt_deg", 0.0))
        th = math.radians(tdeg)
        cth, sth = math.cos(th), math.sin(th)
        Rb = np.array([[1.0, 0.0, 0.0], [0.0, cth, -sth], [0.0, sth, cth]])   # lean about +x
        boxq = mat3_to_wp_quat(Rb)
        ctr = np.array([bx, by, floor_top])
        def _tx(local):
            wpos = ctr + Rb @ np.array(local)
            return wp.transform((float(wpos[0]), float(wpos[1]), float(wpos[2])), boxq)
        # tilted FLOOR only (no walls): cuTAMP planned on an open tilted plane, so match it here;
        # on a 19deg floor friction (tan19=0.34 < mu 0.7) holds the flush-placed bundles.
        builder.add_shape_box(-1, _tx([0, 0, -0.5]), hx=iw / 2 + wall_t, hy=ih / 2 + wall_t, hz=0.5,
                              cfg=stat_cfg, color=boxcol)
        base_x, base_y = bx, by
        spawn_top = floor_top + wall_h + 3.0
        verdict_box = {"mode": "native", "e3": Rb[:, 2], "iw": iw, "ih": ih, "ctr": ctr}
        rbox = None
        print(f"[settle] NATIVE-TILT box: synthetic box tilted {tdeg:.0f}deg, using cuTAMP's tilted "
              f"placements directly", flush=True)
    elif args.box_mesh:
        # --- REAL reconstructed box MESH (true tilt), static open-top collision mesh ---
        rb = load_real_box_cm(args.scene_dir, args.capture_dir)
        boxcol = wp.vec3(*[c for c in rb["color"]])
        cmesh = newton.Mesh(rb["verts"], rb["open_faces"])
        # SDF-based collision: rigid-vs-raw-triangle-mesh contact tunnels/ejects on a noisy
        # reconstructed mesh; an SDF gives robust, stable contact against the tilted walls.
        cmesh.build_sdf(target_voxel_size=0.6)
        builder.add_shape_mesh(-1, wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                               mesh=cmesh, cfg=stat_cfg, color=boxcol)
        base_x, base_y = float(rb["cxy"][0]), float(rb["cxy"][1])   # anchor at box CENTROID
        spawn_top = rb["zt"] + 4.0                                  # drop in from above the box top
        verdict_box = {"mode": "mesh", "bbox": rb["bbox"], "zt": rb["zt"], "e3": rb["e3"]}
        rbox = rb
        print(f"[settle] REAL box mesh: {len(rb['open_faces'])//3} open-top faces, "
              f"tilt {rb['tilt_deg']:.0f}deg (as reconstructed), centroid@({base_x:.0f},{base_y:.0f}) "
              f"top_z={rb['zt']:.0f}cm{' | TILT-AWARE placement' if args.tilt_aware else ''}", flush=True)
    else:
        rbox = None
        # --- synthetic axis-aligned box: floor slab + 4 walls ---
        builder.add_shape_box(-1, wp.transform((bx, by, floor_top - 0.5), wp.quat_identity()),
                              hx=iw / 2 + wall_t, hy=ih / 2 + wall_t, hz=0.5, cfg=stat_cfg, color=boxcol)
        for ox, oy, hx, hy in [
            (0.0, ih / 2 + wall_t / 2, iw / 2 + wall_t, wall_t / 2),
            (0.0, -(ih / 2 + wall_t / 2), iw / 2 + wall_t, wall_t / 2),
            (iw / 2 + wall_t / 2, 0.0, wall_t / 2, ih / 2),
            (-(iw / 2 + wall_t / 2), 0.0, wall_t / 2, ih / 2),
        ]:
            builder.add_shape_box(-1, wp.transform((bx + ox, by + oy, floor_top + wall_h / 2), wp.quat_identity()),
                                  hx=hx, hy=hy, hz=wall_h / 2, cfg=stat_cfg, color=boxcol)
        base_x, base_y = bx, by
        spawn_top = floor_top + bt / 2
        verdict_box = {"mode": "synthetic"}

    # --- dynamic bundles at cuTAMP's (x, y, yaw) placements, re-anchored to the container ---
    bcfg = builder.ShapeConfig(density=args.density, mu=args.mu, restitution=0.0, ke=2.0e4, kd=1.0e2)
    init_xy, names = [], []
    tilt_aware = args.tilt_aware and rbox is not None
    for i, p in enumerate(placements):
        ox, oy = p["pos"][0] * M2CM - ccx, p["pos"][1] * M2CM - ccy   # offset from cuTAMP box centre
        yaw = yaw_of(p["quat_wxyz"])
        if args.native_tilt:
            # cuTAMP already planned the tilted pose -> use its full orientation, and place the
            # bundle FLUSH on the tilted floor at its cuTAMP xy (minimal drop = stable settle)
            e3 = verdict_box["e3"]; ctr = verdict_box["ctr"]
            px, py = p["pos"][0] * M2CM, p["pos"][1] * M2CM
            zf = ctr[2] - (e3[0] * (px - ctr[0]) + e3[1] * (py - ctr[1])) / e3[2]   # tilted-floor z at (px,py)
            center = np.array([px, py, zf]) + e3 * (bt / 2 + args.drop_gap)
            px, py, pz = float(center[0]), float(center[1]), float(center[2])
            qw, qx, qy, qz = p["quat_wxyz"]
            q = wp.quat(float(qx), float(qy), float(qz), float(qw))
        elif tilt_aware:
            # TILT-AWARE: treat cuTAMP's flat arrangement as box-LOCAL and place it at the box's
            # real orientation R, so each bundle rests FLUSH on the tilted floor (not edge-on).
            R = rbox["R"]
            cz, sz = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            Rworld = R @ Rz
            local = np.array([ox, oy, bt / 2 + args.drop_gap])       # floor-plane offset + height along normal
            wpos = rbox["floor_center"] + R @ local
            px, py, pz = float(wpos[0]), float(wpos[1]), float(wpos[2])
            q = mat3_to_wp_quat(Rworld)
        else:
            px, py = base_x + ox, base_y + oy
            pz = spawn_top + args.drop_gap
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        b = builder.add_body(xform=wp.transform((px, py, pz), q))
        # per-object dims: rigid items use their own cube size from meta, garments the bundle dims
        if p["name"].startswith("rigid") and plan["meta"].get("rigid"):
            rs = float(plan["meta"]["rigid"]["size"]) * M2CM
            rc = plan["meta"]["rigid"].get("color", [150, 150, 160])
            builder.add_shape_box(b, hx=rs / 2, hy=rs / 2, hz=rs / 2, cfg=bcfg,
                                  color=wp.vec3(*[c / 255.0 for c in rc]))
        else:
            builder.add_shape_box(b, hx=bw / 2, hy=bh / 2, hz=bt / 2, cfg=bcfg, color=bcol)
        builder.add_joint_free(b)
        init_xy.append((px, py))
        names.append(p["name"])

    builder.add_ground_plane()
    model = builder.finalize()

    solver = SolverXPBD(model)
    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.5)
    contacts = collision_pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    if args.viewer == "gl":
        viewer = ViewerGL()
    else:
        viewer = ViewerNull(num_frames=10 ** 9)
    viewer.set_model(model)

    fps = 60
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / args.substeps
    n_settle = int(args.sim_seconds * fps)
    n_hold = int(args.hold * fps)

    if args.viewer == "gl":
        if args.box_mesh:   # auto-frame the real box wherever it landed in the base frame
            cx, cy, cz, cp, cyaw = base_x, base_y - 48.0, 42.0, -38.0, 90.0
        else:
            cx, cy, cz, cp, cyaw = (float(v) for v in args.cam.split(","))
        viewer.set_camera(wp.vec3(cx, cy, cz), cp, cyaw)

    # state swap via a small dict (avoids nonlocal churn)
    _st = {"a": state_0, "b": state_1}

    def step_once(t):
        for _ in range(args.substeps):
            _st["a"].clear_forces()
            collision_pipeline.collide(_st["a"], contacts)
            solver.step(_st["a"], _st["b"], control, contacts, sim_dt)
            _st["a"], _st["b"] = _st["b"], _st["a"]
        if viewer is not None:
            viewer.begin_frame(t)
            viewer.log_state(_st["a"])
            viewer.end_frame()

    print(f"[settle] settling {args.sim_seconds:.1f}s ({n_settle} frames)...", flush=True)
    t = 0.0
    for f in range(n_settle):
        step_once(t)
        t += frame_dt
        if args.viewer == "gl" and not viewer.is_running():
            break

    # --- stability verdict ---
    bq = _st["a"].body_q.numpy()   # (num_bodies, 7): px,py,pz, qx,qy,qz,qw
    print(f"\n[settle] ===== STABILITY VERDICT =====", flush=True)
    all_ok = True
    m = args.out_margin
    for i, name in enumerate(names):
        px, py, pz = bq[i, 0], bq[i, 1], bq[i, 2]
        x, y, z, w = bq[i, 3], bq[i, 4], bq[i, 5], bq[i, 6]
        moved = math.hypot(px - init_xy[i][0], py - init_xy[i][1])
        if verdict_box["mode"] == "mesh":
            bb = verdict_box["bbox"]
            in_box = (bb[0] - m) <= px <= (bb[1] + m) and (bb[2] - m) <= py <= (bb[3] + m)
            z_ok = -1.0 <= pz <= (verdict_box["zt"] + bt)   # not through table, not launched out
        elif verdict_box["mode"] == "native":
            in_box = abs(px - bx) <= (iw / 2 + wall_h + m) and abs(py - by) <= (ih / 2 + wall_h + m)
            z_ok = (floor_top - 1.0) <= pz <= (floor_top + wall_h + bt + 2)
        else:
            in_box = abs(px - bx) <= (iw / 2 + m) and abs(py - by) <= (ih / 2 + m)
            z_ok = (floor_top - 1.0) <= pz <= (floor_top + wall_h + bt)
        if tilt_aware or verdict_box["mode"] == "native":
            # bundle SHOULD rest at the box tilt -> measure deviation of its up-axis from the
            # box normal (flush error), not from world-vertical
            up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
            tilt = math.degrees(math.acos(min(1.0, abs(up @ verdict_box["e3"]))))
            tlabel = "flush"
        else:
            tilt = quat_tilt_deg((x, y, z, w))
            tlabel = "upright"
        upright = tilt <= args.tilt_thresh
        ok = in_box and upright and z_ok
        all_ok = all_ok and ok
        print(f"[settle]   {name:10s}: xy_drift={moved:5.1f}cm {tlabel}_err={tilt:5.1f}deg z={pz:5.1f}cm | "
              f"{'IN-BOX' if in_box else 'OUT!'} {'ok' if upright else 'TILTED!'} "
              f"{'z-ok' if z_ok else 'Z-BAD!'} -> {'STABLE' if ok else 'UNSTABLE'}", flush=True)
    print(f"[settle] ===== PACK IS {'STABLE (all bundles settled in-box, upright)' if all_ok else 'UNSTABLE (see above)'} =====\n", flush=True)

    if args.screenshot and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
        print(f"[settle] shot -> {args.screenshot}", flush=True)

    # hold so the user can inspect the settled pack
    for f in range(n_hold):
        if args.viewer == "gl" and not viewer.is_running():
            break
        viewer.begin_frame(t)
        viewer.log_state(_st["a"])
        viewer.end_frame()
        t += frame_dt


if __name__ == "__main__":
    main()
