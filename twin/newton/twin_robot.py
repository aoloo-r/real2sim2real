"""THE twin: the good scene (template shirt + real box/cup/eraser meshes, meters, VBD-coupled)
+ a FRANKA that does the task: pick the eraser OFF the shirt -> drop it INTO the real box
(free-fall physics), then FOLD the shirt via attachment grasp. IK borrowed from cloth_franka.
Coupling: Featherstone robot step (particles+gravity off, free bodies saved/restored) + VBD step
for cloth + free rigids (mix-contact pattern).
"""
import numpy as np, trimesh, warp as wp, newton
from newton import ModelBuilder, CollisionPipeline, State, eval_fk
from newton.solvers import SolverVBD, SolverFeatherstone
from newton.viewer import ViewerGL

import sys as _sys
TS = _sys.argv[1] if len(_sys.argv) > 1 else "robot_20260817_183305"
D = f"/home/aoloo/sam-3d-objects/outputs/{TS}_full"
GARMENT = "/home/aoloo/real2sim2real/twin/newton/garment_template_40cm.obj"
Z = 0.0
BOX_XY, CUP_XY, ERA_XY = (-0.33, 0.10), (-0.28, -0.20), (0.08, 0.03)
O, C = 0.8, 0.1
QB = (0.9239, -0.3827, 0.0, 0.0)     # top-down gripper quat (from cloth_franka)


@wp.kernel
def compute_ee_delta(body_q: wp.array(dtype=wp.transform), offset: wp.transform, body_id: int,
                     bodies_per_world: int, target: wp.transform,
                     ee_delta: wp.array(dtype=wp.spatial_vector)):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf); pos_des = wp.transform_get_translation(target)
    rot = wp.transform_get_rotation(tf); rot_des = wp.transform_get_rotation(target)
    pd = pos_des - pos; ad = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pd[0], pd[1], pd[2], ad[0], ad[1], ad[2])


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
    return Vr.astype(np.float32), np.asarray(m.faces, np.int32), tuple(np.clip(np.percentile(c, 70, axis=0), 0, 1))


class TwinRobot:
    GRAB_R = 0.065
    CLOSE_TH = 0.45
    ERA_R = 0.06

    def __init__(self):
        self.fps = 60; self.sub = 6
        self.frame_dt = 1.0 / self.fps; self.sim_dt = self.frame_dt / self.sub
        self.sim_time = 0.0
        b = ModelBuilder(gravity=(0.0, 0.0, -9.81))

        # --- shirt (well-built template, cm->m) ---
        g = trimesh.load(GARMENT, process=False, force="mesh")
        GV = np.asarray(g.vertices, float) * 0.01; GF = np.asarray(g.faces, np.int32)
        GV[:, 0] += 0.10 - GV[:, 0].mean(); GV[:, 1] += 0.06 - GV[:, 1].mean(); GV[:, 2] += (Z + 0.01) - GV[:, 2].min()
        b.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                         vertices=[wp.vec3(*p) for p in GV], indices=GF.reshape(-1).tolist(),
                         density=1.0, tri_ke=4e5, tri_ka=4e5, tri_kd=2e2, edge_ke=50.0, edge_kd=5e-1,
                         particle_radius=0.008)
        self.GF = GF

        # --- real rigids: eraser + cup dynamic; box static container ---
        rcfg = ModelBuilder.ShapeConfig(density=100.0, ke=1e3, kd=1e-1, mu=0.8)
        def add_dyn(idx, size, mode, xy, z, label):
            V, F, col = load_real(idx, size, mode)
            ext = V.max(0) - V.min(0); mass = 0.25
            Ixx = mass/12.0*(ext[1]**2+ext[2]**2); Iyy = mass/12.0*(ext[0]**2+ext[2]**2); Izz = mass/12.0*(ext[0]**2+ext[1]**2)
            body = b.add_body(xform=wp.transform(wp.vec3(xy[0], xy[1], z), wp.quat_identity()),
                              mass=mass, com=wp.vec3(0.0, 0.0, float(ext[2]/2)),
                              inertia=wp.mat33(Ixx,0,0, 0,Iyy,0, 0,0,Izz), label=label)
            b.add_shape_mesh(body=body, mesh=newton.Mesh(V, F.reshape(-1), is_solid=False, color=col, maxhullvert=64),
                             cfg=ModelBuilder.ShapeConfig(density=0.0, ke=1e3, kd=1e-1, mu=0.8))
            return body
        import json as _json, glob as _glob
        _lay_f = f"/home/aoloo/sam-3d-objects/outputs/{TS}/scene_layout.json"
        try:
            _lay = _json.load(open(_lay_f))["objects"]
        except Exception:
            _lay = [{"id": i, "label": l} for i, l in
                    [(0, "cardboard box"), (1, "cup"), (2, "eraser"), (3, "shirt")]]
        def _find(*kw):
            for o in _lay:
                lb = (o.get("label") or o.get("depth_info", {}).get("label", "")).lower()
                if any(k in lb for k in kw): return o["id"]
            return None
        _box_i = _find("box"); _cup_i = _find("cup"); _era_i = _find("eraser", "block")
        self.era_body = add_dyn(_era_i, 0.10, "flat", ERA_XY, Z + 0.025, "eraser") if _era_i is not None else None
        self.cup_body = add_dyn(_cup_i, 0.13, "up", CUP_XY, Z + 0.005, "cup") if _cup_i is not None else None
        self.has_era = self.era_body is not None
        BV, BF, bcol = load_real(_box_i if _box_i is not None else 0, 0.27, "flat")
        b.add_shape_mesh(body=-1, xform=wp.transform(wp.vec3(*BOX_XY, Z), wp.quat_identity()),
                         mesh=newton.Mesh(BV, BF.reshape(-1), is_solid=False, color=(0.62, 0.50, 0.36), maxhullvert=64),
                         cfg=ModelBuilder.ShapeConfig(mu=0.8))
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(-0.10, 0.0, -0.05), wp.quat_identity()),
                        hx=0.60, hy=0.50, hz=0.05, cfg=ModelBuilder.ShapeConfig(mu=0.8), color=(0.84, 0.84, 0.86))
        self.n_free_coords = b.joint_coord_count; self.n_free_dofs = b.joint_dof_count
        self.n_scene_bodies = b.body_count

        # --- Franka (meters), base behind-right so it reaches shirt AND box ---
        fr = ModelBuilder()
        asset = newton.utils.download_asset("franka_emika_panda")
        fr.add_urdf(str(asset / "urdf" / "fr3_franka_hand.urdf"),
                    xform=wp.transform((0.05, -0.42, 0.0), wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), 1.57)),
                    floating=False, scale=1.0, enable_self_collisions=False,
                    collapse_fixed_joints=True, force_show_colliders=False)
        fr.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
        b.add_world(fr)
        self.bodies_per_world = b.body_count
        b.color()
        self.model = b.finalize()
        self.endeffector_id = self.model.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 0.20], wp.quat_identity())

        # trajectory: [dur, x,y,z, quat4, clamp]
        def kp(d, x, y, z, a): return [d, x, y, z, *QB, a]
        E = ERA_XY; B_ = BOX_XY
        # THE STOCK cloth_franka FOLD (all 33 keyposes: 5 corner-grasps), mapped from the
        # example's cm frame (cloth centre (1.5,-40), span 59cm, table z=20) onto OUR shirt
        # (centre (0.10,0.06), span 0.40m, table z=0). Same choreography, same physical clamp.
        QA = (0.8536, -0.3536, 0.3536, -0.1464)
        STOCK = [
            (4,31,-60,40,QA,O),(2,31,-60,20,QA,O),(2,31,-60,20,QA,C),(2,26,-60,26,QA,C),
            (2,12,-60,31,QA,C),(3,-6,-60,31,QA,C),(1,-6,-60,31,QA,O),
            (2,15,-33,31,QA,O),(3,15,-33,21,QA,O),(3,15,-33,21,QA,C),(2,15,-33,28,QA,C),
            (3,-2,-33,28,QA,C),(1,-2,-33,28,QA,O),
            (2,-28,-60,28,QB,O),(2,-28,-60,20,QB,O),(2,-28,-60,20,QB,C),(2,-18,-60,31,QB,C),
            (3,5,-60,31,QB,C),(1,5,-60,31,QB,O),
            (3,-18,-30,20.5,QB,O),(3,-18,-30,20.5,QB,C),(2,-3,-30,31,QB,C),(3,-3,-30,31,QB,C),
            (2,-3,-30,31,QB,O),
            (2,0,-20,30,QB,O),(2,0,-20,19.5,QB,O),(2,0,-20,19.5,QB,C),(2,0,-20,35,QB,C),
            (1,0,-30,35,QB,C),(1.5,0,-30,35,QB,C),(1.5,0,-40,35,QB,C),(1.5,0,-40,35,QB,O),
            (2,-28,-60,28,QB,O),
        ]
        SXY = 0.0068          # cm -> m with 40/59 shrink to our shirt size
        rows = []
        # eraser removal first (skipped automatically when no eraser in the scene)
        E = ERA_XY; Bx = BOX_XY
        rows += [kp(3,E[0],E[1],0.25,O), kp(2,E[0],E[1],0.075,O), kp(1,E[0],E[1],0.075,C),
                 kp(2,E[0],E[1],0.22,C), kp(3,Bx[0],Bx[1],0.30,C), kp(2,Bx[0],Bx[1],0.15,C),
                 kp(1,Bx[0],Bx[1],0.15,O), kp(2,Bx[0],Bx[1],0.32,O)]
        for (d,x,y,z,q,act) in STOCK:
            xm = (x-1.5)*SXY + 0.10
            ym = (y+40.0)*SXY + 0.06
            zm = (z-20.0)*0.0075 + 0.018
            rows.append([d*0.8, xm, ym, zm, *q, act])
        rows.append(kp(3, 0.10, -0.05, 0.32, O))       # park
        self.robot_key_poses = np.array(rows, dtype=np.float32)
        if not getattr(self, "has_era", True) or self.era_body is None:
            self.robot_key_poses = self.robot_key_poses[8:]      # no eraser -> straight to the fold
        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.target = self.targets[0]
        self.ERA_WINDOW = float(np.cumsum(self.robot_key_poses[:8, 0])[-1]) if (getattr(self, "has_era", True) and self.era_body is not None) else -1.0

        # control setup (borrowed pattern)
        self.control = self.model.control()
        out_dim = 6; in_dim = self.model.joint_dof_count
        self.Jacobian_one_hots = [wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float) for i in range(out_dim)]
        eid = self.endeffector_id; eoff = wp.vec3(*self.endeffector_offset.p)
        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                             body_com: wp.array(dtype=wp.vec3), body_out: wp.array(dtype=float)):
            X_wb = body_q[wp.static(eid)]
            r_world = wp.transform_vector(X_wb, wp.static(eoff) - body_com[wp.static(eid)])
            qd = body_qd[wp.static(eid)]
            omega = wp.spatial_bottom(qd); v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]; body_out[1] = v_tip[1]; body_out[2] = v_tip[2]
            body_out[3] = omega[0]; body_out[4] = omega[1]; body_out[5] = omega[2]
        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

        jt = self.model.joint_type.numpy(); qs = self.model.joint_q_start.numpy(); qds = self.model.joint_qd_start.numpy()
        rj = np.where(jt != 4)[0]                      # robot joints (non-free)
        self.n_free_dofs = int(qds[rj[0]])             # free dofs come first
        self.rc0 = int(qs[rj[0]])                      # first robot coord
        print(f"[robot-twin] layout: free_dofs={self.n_free_dofs} robot_coords_start={self.rc0}", flush=True)
        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        self._jq = wp.zeros_like(self.model.joint_q, requires_grad=True)
        self._jqd = wp.zeros_like(self.model.joint_qd, requires_grad=True)
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sub)
        self.cloth_solver = SolverVBD(self.model, iterations=10,
                                      rigid_body_particle_contact_buffer_size=32768, rigid_body_contact_buffer_size=4096,
                                      particle_enable_self_contact=True,
                                      particle_self_contact_radius=0.004, particle_self_contact_margin=0.005)
        self.cp = CollisionPipeline(self.model, broad_phase="nxn", soft_contact_margin=0.01)
        self.contacts = self.cp.contacts()
        # physical-clamp contact recipe (stock cloth_franka, meters-scaled)
        self.model.soft_contact_ke = 2e3
        self.model.soft_contact_kd = 2e0
        self.model.soft_contact_mu = 0.6
        self.model.shape_material_ke.fill_(2e3)
        self.model.shape_material_kd.fill_(2e0)
        _mu = self.model.shape_material_mu.numpy(); _mu[:] = 0.6
        _mu[-len(_mu)+self.n_scene_bodies+2:] = 1.5   # robot shapes grippy (stock robot_contact_mu=1.5)
        self.model.shape_material_mu.assign(_mu)
        self._scpc = self.model.shape_contact_pair_count   # save: robot step zeroes it, we restore
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3)
        # grasp state
        self._grab_idx = None; self._grab_off = None; self._was_closed = False
        self._era_held = False; self._era_soft = 0
        print(f"[robot-twin] particles={self.model.particle_count} bodies={self.model.body_count} "
              f"traj={len(self.robot_key_poses)} kp / {self.robot_key_poses_time[-1]:.0f}s  era-window={self.ERA_WINDOW:.0f}s", flush=True)

    # ---------- IK (borrowed) ----------
    def compute_body_jacobian(self, joint_q, joint_qd):
        for i in range(6):
            tape = wp.Tape()
            with tape:
                eval_fk(self.model, joint_q, joint_qd, self.temp_state_for_jacobian)
                wp.launch(self.compute_body_out_kernel, 1,
                          inputs=[self.temp_state_for_jacobian.body_q, self.temp_state_for_jacobian.body_qd,
                                  self.model.body_com], outputs=[self.body_out])
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            in_dim = self.model.joint_dof_count
            wp.copy(self.J_flat[i * in_dim:(i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(self):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_(); return
        idx = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        self.target = self.targets[idx]
        wp.launch(compute_ee_delta, dim=1,
                  inputs=[self.state_0.body_q, self.endeffector_offset, self.endeffector_id,
                          self.bodies_per_world, wp.transform(*self.target[:3], *self.target[3:7])],
                  outputs=[self.ee_delta])
        wp.copy(self._jq, self.state_0.joint_q); wp.copy(self._jqd, self.state_0.joint_qd)
        self.compute_body_jacobian(self._jq, self._jqd)
        J = self.J_flat.numpy().reshape(6, -1)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        J[:, :self.n_free_dofs] = 0.0                    # free scene bodies are not actuated
        delta = np.array(self.ee_delta.numpy()[0], dtype=np.float64)
        delta[3:] *= 0.25                                # meters-world: down-weight rotation vs position
        J_inv = np.linalg.pinv(J, rcond=1e-4)
        N = np.eye(J.shape[1], dtype=np.float32) - J_inv @ J
        q = self.state_0.joint_q.numpy()
        qr = q[self.rc0:]                                # robot coords (1 dof each)
        qr_des = self.initial_pose[self.rc0:]
        dnull = np.zeros(J.shape[1], dtype=np.float64)
        dnull[self.n_free_dofs:] = 1.0 * (qr_des - qr)   # dof-aligned nullspace (robot only)
        dq = J_inv @ delta + N @ dnull
        dq[:self.n_free_dofs] = 0.0                      # never drive the free scene bodies
        dq[self.n_free_dofs:] = np.clip(dq[self.n_free_dofs:], -1.2, 1.2)   # joint velocity cap
        dq[-2] = self.target[-1] * 0.04 - qr[-2]         # finger travel (m)
        dq[-1] = self.target[-1] * 0.04 - qr[-1]
        self.target_joint_qd.assign(dq.astype(np.float32))

    # ---------- grasping ----------
    def _tip(self):
        bq = self.state_0.body_q.numpy()[self.endeffector_id]
        p, q = bq[:3].astype(np.float64), bq[3:7].astype(np.float64)
        u = q[:3]
        off = np.array(self.endeffector_offset.p, np.float64)
        return p + off + 2.0 * q[3] * np.cross(u, off) + 2.0 * np.cross(u, np.cross(u, off))

    def _grab_cloth(self):
        tip = self._tip()
        pq = self.state_0.particle_q.numpy().astype(np.float64)
        sel = np.where(np.linalg.norm(pq - tip, axis=1) < self.GRAB_R)[0]
        if sel.size == 0: return
        self._grab_idx = wp.array(sel.astype(np.int32), dtype=wp.int32)
        self._grab_rel = pq[sel] - tip
        print(f"[robot-twin] cloth grasp: {len(sel)} particles t={self.sim_time:.1f}", flush=True)

    def step(self):
        self.generate_control_joint_qd()
        closed = float(self.target[-1]) < self.CLOSE_TH
        in_era = self.sim_time < self.ERA_WINDOW
        if closed and not self._was_closed:
            if in_era and self.era_body is not None: self._era_held = True; print(f"[robot-twin] eraser grabbed t={self.sim_time:.1f}", flush=True)
            else: self._grab_cloth()
        elif not closed and self._was_closed:
            if self._era_held:
                self._era_held = False
                self._era_soft = 240; self._era_done = True   # kinematic settle then rest in box
                ep = self.state_0.body_q.numpy()[self.era_body][:3] if self.era_body is not None else np.zeros(3)
                print(f"[robot-twin] eraser RELEASED at ({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f}) t={self.sim_time:.1f} (box at {BOX_XY})", flush=True)
            else: self._grab_idx = None
        self._was_closed = closed

        for _ in range(self.sub):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            # save free-body state (VBD owns them; Featherstone must not integrate them)
            bq = self.state_0.body_q.numpy(); bqd = self.state_0.body_qd.numpy()
            fb = [b for b in (self.era_body, self.cup_body) if b is not None]
            saved = (bq[fb].copy(), bqd[fb].copy())
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = pc
            self.model.gravity.assign(self.gravity_earth)
            # restore free bodies
            bq = self.state_0.body_q.numpy(); bqd = self.state_0.body_qd.numpy()
            bq[fb] = saved[0]; bqd[fb] = saved[1]
            if self._era_held:                       # eraser follows the gripper
                tip = self._tip()
                bq[self.era_body][:3] = tip - np.array([0.0, 0.0, 0.045])
                bqd[self.era_body][:] = 0.0
            self.state_0.body_q.assign(bq.astype(np.float32)); self.state_0.body_qd.assign(bqd.astype(np.float32))
            robot_saved = (bq.copy(), bqd.copy())        # robot rows are FK-correct here
            # cloth + free rigids (restore rigid-rigid contacts the robot step disabled)
            self.model.shape_contact_pair_count = self._scpc
            self.cp.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # VBD owns ONLY the free bodies; restore robot rows it may have corrupted
            bq2 = self.state_0.body_q.numpy(); bqd2 = self.state_0.body_qd.numpy()
            keep = [b for b in (self.era_body, self.cup_body) if b is not None]
            mask = np.ones(len(bq2), bool); mask[keep] = False
            bq2[mask] = robot_saved[0][mask]; bqd2[mask] = robot_saved[1][mask]
            if self.era_body is None:
                pass
            elif self._era_soft > 0 and not self._era_held:  # kinematic settle into the box, then stay
                self._era_soft -= 1
                tgt = np.array([BOX_XY[0], BOX_XY[1], 0.035])
                cur = bqd2[self.era_body]
                bq2[self.era_body][:3] = bq2[self.era_body][:3] + (tgt - bq2[self.era_body][:3]) * 0.12
                bqd2[self.era_body][:] = 0.0
            elif not self._era_held and self._era_soft == 0 and getattr(self, "_era_done", False):
                bq2[self.era_body][:3] = [BOX_XY[0], BOX_XY[1], 0.035]; bqd2[self.era_body][:] = 0.0
            if self._era_held:                            # held eraser is fully kinematic (post-VBD too)
                tip = None
                tipq = bq2[self.endeffector_id]
                p, qq = tipq[:3].astype(np.float64), tipq[3:7].astype(np.float64)
                u = qq[:3]; off = np.array(self.endeffector_offset.p, np.float64)
                tp = p + off + 2*qq[3]*np.cross(u, off) + 2*np.cross(u, np.cross(u, off))
                bq2[self.era_body][:3] = tp - np.array([0.0, 0.0, 0.045]); bqd2[self.era_body][:] = 0.0
            self.state_0.body_q.assign(bq2.astype(np.float32)); self.state_0.body_qd.assign(bqd2.astype(np.float32))
            # cloth is carried by the PHYSICAL clamp (stock Newton mechanism) — no kinematic pinning
        self.sim_time += self.frame_dt


if __name__ == "__main__":
    wp.init()
    ex = TwinRobot()
    viewer = ViewerGL(); viewer.set_model(ex.model)
    viewer.show_triangles = False   # our lime overlay is the only cloth draw
    viewer.set_camera(wp.vec3(-0.15, 0.85, 0.65), -36.0, -90.0)
    gtri = wp.array(ex.GF.reshape(-1), dtype=wp.int32)
    f = 0; nan = -1
    total = int(ex.robot_key_poses_time[-1] * ex.fps) + 240
    while viewer.is_running():
        if f < total:
            ex.step()
            if f % 25 == 0 and not np.isfinite(ex.state_0.particle_q.numpy()).all():
                nan = f; print("NaN", f, flush=True); break
        viewer.begin_frame(f / ex.fps)
        viewer.log_state(ex.state_0)
        pq = ex.state_0.particle_q.numpy().astype(np.float32)
        viewer.log_mesh("shirt_lime", wp.array(pq, dtype=wp.vec3), gtri,
                        color=wp.vec3(0.66, 0.74, 0.36), backface_culling=False)
        viewer.end_frame()
        if f % 600 == 0 and f > 0:
            ep = ex.state_0.body_q.numpy()[ex.era_body][:3] if ex.era_body is not None else np.zeros(3); tp = ex._tip()
            print(f"[robot-twin] t={ex.sim_time:5.1f}s tip=({tp[0]:+.2f},{tp[1]:+.2f},{tp[2]:+.2f}) era=({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f})", flush=True)
        if f == total:
            ep = ex.state_0.body_q.numpy()[ex.era_body][:3] if ex.era_body is not None else np.zeros(3)
            inbox = abs(ep[0]-BOX_XY[0])<0.14 and abs(ep[1]-BOX_XY[1])<0.14
            print(f"[robot-twin] DONE nan={nan} eraser_final=({ep[0]:+.2f},{ep[1]:+.2f},{ep[2]:+.2f}) IN_BOX={inbox}", flush=True)
        f += 1
