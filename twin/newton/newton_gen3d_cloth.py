"""Load a GENERATED garment mesh (Meshy/Rodin GLB from perception/gen3d_cloth.py)
as a Newton VBD cloth and drape-test it — the QA gate before swapping it into the
fold pipeline.

Generation recovers garment TOPOLOGY (sleeves, front/back — what SAM3D loses);
the real capture stays the metric ground truth: the mesh is rescaled to the
depth-measured flat width and dropped onto the table to check VBD stability.

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_gen3d_cloth.py --mesh .../gen3d_shirt/shirt_gen.glb \
      --flat_w 0.48 --screenshot /tmp/gen3d_drape.png
Accepts .glb/.obj/.usd (usd path used for stand-in validation with the example shirt).
"""
import argparse, os

import numpy as np
import warp as wp

import newton

TABLE_Z = 20.0  # cm, matches newton_fold


def load_mesh(path):
    """-> verts (N,3) float, faces (M,3) int, vcolors (N,3) float in 0..1 or None."""
    if path.lower().endswith(".usd"):
        from pxr import Usd
        import newton.usd as nu
        stage = Usd.Stage.Open(path)
        prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
        m = nu.get_mesh(prim)
        return np.array(m.vertices, float), np.array(m.indices).reshape(-1, 3), None
    import trimesh
    tm = trimesh.load(path, force="mesh")
    vcol = None
    vis = getattr(tm, "visual", None)
    if vis is not None and getattr(vis, "kind", None) == "texture":
        try:
            vcol = np.asarray(vis.to_color().vertex_colors, float)[:, :3] / 255.0
        except Exception:
            vcol = None
    elif vis is not None and getattr(vis, "vertex_colors", None) is not None:
        vcol = np.asarray(vis.vertex_colors, float)[:, :3] / 255.0
    return np.asarray(tm.vertices, float), np.asarray(tm.faces, int), vcol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--flat_w", type=float, default=0.48,
                    help="depth-measured flat garment width (m) — the metric calibration")
    ap.add_argument("--decimate", type=int, default=0,
                    help="0=off (default). Local quadric decimation of a CLOSED garment leaves "
                         "sliver tris that NaN VBD — prefer the service's target_polycount remesh")
    ap.add_argument("--settle", type=float, default=6.0, help="drape seconds")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=600.0)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()

    v, f, vcol = load_mesh(args.mesh)
    if args.decimate and len(f) > args.decimate:
        import trimesh
        from scipy.spatial import cKDTree
        tm = trimesh.Trimesh(v, f, process=False)
        dm = tm.simplify_quadric_decimation(face_count=args.decimate)
        dm.update_faces(dm.nondegenerate_faces())      # decimation leaves sliver tris VBD rejects
        if vcol is not None:
            vcol = vcol[cKDTree(v).query(np.asarray(dm.vertices))[1]]
        v, f = np.asarray(dm.vertices, float), np.asarray(dm.faces, int)
    # calibrate: longest xy extent -> measured flat width, cm scale, base at table
    v = v - v.mean(0)
    ext = v.max(0) - v.min(0)
    s = (args.flat_w * 100.0) / max(ext[0], ext[1])
    v = v * s
    v[:, 2] -= v[:, 2].min() - (TABLE_Z + 1.0)
    print(f"[gen3d] {os.path.basename(args.mesh)}: {len(v)} verts {len(f)} tris, "
          f"scaled x{s:.3f} -> {(v.max(0)-v.min(0)).round(1)} cm", flush=True)

    b = newton.ModelBuilder(gravity=-981.0, up_axis=newton.Axis.Z)
    b.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                     vertices=[wp.vec3(*p) for p in v], indices=f.flatten().tolist(),
                     density=0.02, tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.0e-5,
                     edge_ke=5.0, edge_kd=1.0e-2,
                     particle_radius=0.8)
    b.add_ground_plane()
    b.color(include_bending=True)
    model = b.finalize()
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 0.5
    model.soft_contact_margin = 0.8
    # table = the ground plane at z offset handled by dropping to TABLE_Z above ground:
    # keep it simple — drape onto the ground plane, garment starts 1cm above it
    solver = newton.solvers.SolverVBD(
        model, iterations=10, particle_enable_self_contact=True,
        particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=0.5,
        particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=64)
    viewer = None
    if args.viewer == "gl":
        from newton.viewer import ViewerGL
        viewer = ViewerGL()
        viewer.set_model(model)
    st = [model.state(), model.state()]
    contacts = model.collide(st[0])
    dt, sub = 1.0 / 60.0, 10
    steps = int(args.settle / dt)
    tri_flat = f.flatten().astype(np.int32)
    col = tuple(vcol.mean(0)) if vcol is not None else (0.85, 0.8, 0.3)
    for i in range(steps):
        for _ in range(sub):
            st[0].clear_forces()
            contacts = model.collide(st[0])
            solver.step(st[0], st[1], None, contacts, dt / sub)
            st = [st[1], st[0]]
        if viewer is not None:
            viewer.begin_frame(i * dt)
            viewer.log_mesh("/gen_cloth", wp.array(st[0].particle_q.numpy(), dtype=wp.vec3),
                            wp.array(tri_flat, dtype=wp.int32), color=col)
            viewer.end_frame()
        if i % 60 == 0:
            q = st[0].particle_q.numpy()
            if not np.isfinite(q).all():
                raise SystemExit(f"[gen3d] NaN at t={i*dt:.1f}s — mesh not VBD-stable")
            print(f"[gen3d] t={i*dt:.1f}s z=[{q[:,2].min():.1f},{q[:,2].max():.1f}]cm", flush=True)
    q = st[0].particle_q.numpy()
    e = q.max(0) - q.min(0)
    print(f"[gen3d] DRAPED footprint {e[0]:.0f} x {e[1]:.0f} x {e[2]:.0f} cm — "
          f"{'OK' if np.isfinite(q).all() else 'FAIL'}", flush=True)
    if args.screenshot and viewer is not None:
        from PIL import Image
        frame = viewer.get_frame().numpy()
        Image.fromarray(frame).save(args.screenshot)
        print(f"[gen3d] shot -> {args.screenshot}", flush=True)
    if viewer is not None:
        import time
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < args.hold:
            viewer.begin_frame(0.0)
            viewer.log_mesh("/gen_cloth", wp.array(q, dtype=wp.vec3),
                            wp.array(tri_flat, dtype=wp.int32), color=col)
            viewer.end_frame()


if __name__ == "__main__":
    main()
