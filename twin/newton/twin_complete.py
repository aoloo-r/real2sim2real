"""COMPLETE manipulable twin — every real object, one coupled sim (Newton 1.5 VBD rigid+cloth):
  - shirt  = REAL garment (garment_clean.obj) as foldable VBD cloth (lime)
  - eraser = REAL SAM3D mesh as DYNAMIC rigid body resting ON the shirt
  - cup    = REAL SAM3D mesh as DYNAMIC rigid body (teal, upright)
  - box    = REAL SAM3D VIVO box mesh, static container
Real layout; settle, then wide-band fold of the shirt UNDER the resting eraser physics.
Pattern = example_vbd_soft_rigid_mix_contact (SolverVBD couples rigids + cloth). Meters.
"""
import numpy as np, trimesh, warp as wp, newton
from newton import ModelBuilder, CollisionPipeline
from newton.solvers import SolverVBD
from newton.viewer import ViewerGL

D = "/home/aoloo/sam-3d-objects/outputs/robot_20260817_183305_full"
SC = "/home/aoloo/real2sim2real/twin/newton"
Z = 0.0                                   # table top (m)
BOX_XY, CUP_XY, ERA_XY = (-0.40, 0.03), (-0.26, -0.19), (-0.03, -0.05)

def load_real(idx, size_m, mode):
    m = trimesh.load(f"{D}/object_{idx}/mesh.obj", process=False, force="mesh")
    c = np.load(f"{D}/object_{idx}/vertex_colors.npy").astype(np.float64).reshape(-1, 3)
    if c.max() > 1.5: c /= 255.0
    V = np.asarray(m.vertices, float); V -= V.mean(0)
    V *= size_m / max(V.max(0) - V.min(0))
    cc = V - V.mean(0); _, _, vt = np.linalg.svd(cc, full_matrices=False); Vr = cc @ vt.T
    ext = Vr.max(0) - Vr.min(0); order = np.argsort(ext)
    Vr = Vr[:, [order[0], order[1], order[2]]] if mode == "up" else Vr[:, [order[2], order[1], order[0]]]
    Vr[:, 2] -= Vr[:, 2].min()
    return Vr.astype(np.float32), np.asarray(m.faces, np.int32), tuple(np.median(c, axis=0))

wp.init()
b = ModelBuilder(gravity=(0.0, 0.0, -9.81))

# --- REAL shirt as foldable cloth (garment_clean, cm->m) ---
g = trimesh.load(f"{SC}/garment_template_40cm.obj", process=False, force="mesh")
GV = np.asarray(g.vertices, float) * 0.01
GF = np.asarray(g.faces, np.int32)
GV[:, 0] -= GV[:, 0].mean(); GV[:, 1] -= GV[:, 1].mean(); GV[:, 2] += (Z + 0.01) - GV[:, 2].min()   # dressed shirt collapses flat during settle
b.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                 vertices=[wp.vec3(*p) for p in GV], indices=GF.reshape(-1).tolist(),
                 density=0.08, tri_ke=1e5, tri_ka=1e5, tri_kd=1e2, edge_ke=50.0, edge_kd=5e-1,
                 particle_radius=0.008)
print(f"[twin] shirt cloth {len(GV)}v", flush=True)

# --- REAL rigids ---
rcfg = ModelBuilder.ShapeConfig(density=100.0, ke=1e3, kd=1e-1, mu=0.8)
def add_rigid(idx, size, mode, xy, z, label):
    V, F, col = load_real(idx, size, mode)
    mesh = newton.Mesh(V, F.reshape(-1), is_solid=False, color=col, maxhullvert=64)
    body = b.add_body(xform=wp.transform(wp.vec3(xy[0], xy[1], z), wp.quat_identity()), label=label)
    b.add_shape_mesh(body=body, mesh=mesh, cfg=rcfg)
    print(f"[twin] rigid '{label}' {len(V)}v col={np.round(col,2)}", flush=True)
    return body
# era removed per user — fold the shirt unobstructed
# era = add_rigid(2, 0.10, "flat", ERA_XY, Z + 0.025, "eraser")     # dropped onto the shirt
cup = add_rigid(1, 0.13, "up",  CUP_XY, Z + 0.01, "cup")
# box: static (container)
BV, BF, bcol = load_real(0, 0.27, "flat")
b.add_shape_mesh(body=-1, xform=wp.transform(wp.vec3(*BOX_XY, Z), wp.quat_identity()),
                 mesh=newton.Mesh(BV, BF.reshape(-1), is_solid=False, color=bcol, maxhullvert=64),
                 cfg=ModelBuilder.ShapeConfig(mu=0.8))
# table
b.add_shape_box(-1, xform=wp.transform(wp.vec3(-0.15, -0.05, -0.05), wp.quat_identity()),
                hx=0.55, hy=0.45, hz=0.05, cfg=ModelBuilder.ShapeConfig(mu=0.8),
                color=(0.82, 0.82, 0.84))
b.color()
model = b.finalize()
solver = SolverVBD(model, iterations=15, rigid_body_particle_contact_buffer_size=128,
                   rigid_body_contact_buffer_size=512,
                   particle_enable_self_contact=True,
                   particle_self_contact_radius=0.004, particle_self_contact_margin=0.005)
cp = CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.01)
contacts = cp.contacts(); s0, s1 = model.state(), model.state()
_gtri = wp.array(GF.reshape(-1), dtype=wp.int32)
newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

@wp.kernel
def _pin(idx: wp.array(dtype=wp.int32), tgt: wp.array(dtype=wp.vec3),
         pq: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3)):
    t = wp.tid(); p = idx[t]; pq[p] = tgt[t]; qd[p] = wp.vec3(0.0)
@wp.kernel
def _vg(qd: wp.array(dtype=wp.vec3)):
    i = wp.tid(); v = qd[i]
    if not (wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])): qd[i] = wp.vec3(0.0)
    elif wp.length(v) > 8.0: qd[i] = wp.normalize(v) * 8.0

p0 = s0.particle_q.numpy()
xthr = np.percentile(p0[:, 0], 65)
edge = np.where(p0[:, 0] >= xthr)[0]; ex0 = p0[edge].copy()
gidx = wp.array(edge.astype(np.int32), dtype=wp.int32)
print(f"[twin] fold band {len(edge)} particles; running settle->fold", flush=True)

viewer = ViewerGL(); viewer.set_model(model)
viewer.set_camera(wp.vec3(-0.12, 0.7, 0.55), -38.0, -90.0)
fps = 60; sub = 8; dt = 1.0 / fps / sub; nan = -1
for f in range(2600):
    tgt = None
    if 700 <= f < 1600:                          # fold after everything settles
        a = (f - 700) / 900.0
        t = ex0.copy()
        t[:, 0] = ex0[:, 0] * (1.0 - 1.8 * a)
        t[:, 2] = ex0[:, 2] + np.sin(a * np.pi) * 0.05
        tgt = wp.array([wp.vec3(*p) for p in t], dtype=wp.vec3)
    cp.collide(s0, contacts)
    for _ in range(sub):
        s0.clear_forces()
        viewer.apply_forces(s0)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
        if tgt is not None:
            wp.launch(_pin, dim=len(edge), inputs=[gidx, tgt, s0.particle_q, s0.particle_qd])
        wp.launch(_vg, dim=model.particle_count, inputs=[s0.particle_qd])
    if f % 25 == 0:
        pq = s0.particle_q.numpy()
        if not np.isfinite(pq).all(): nan = f; print("NaN", f, flush=True); break
    viewer.begin_frame(f / fps)
    viewer.log_state(s0)
    _pq=s0.particle_q.numpy().astype(np.float32); _pq[:,2]+=0.002
    viewer.log_mesh("shirt_lime", wp.array(_pq, dtype=wp.vec3), _gtri,
                    color=wp.vec3(0.66,0.74,0.36), backface_culling=False)
    viewer.end_frame()
print(f"[twin] DONE f={f} nan={nan}", flush=True)
while viewer.is_running():
    viewer.begin_frame(f / fps); viewer.log_state(s0)
    _pq=s0.particle_q.numpy().astype(np.float32); _pq[:,2]+=0.002
    viewer.log_mesh("shirt_lime", wp.array(_pq, dtype=wp.vec3), _gtri,
                    color=wp.vec3(0.66,0.74,0.36), backface_culling=False)
    viewer.end_frame()
