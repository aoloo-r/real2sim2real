"""Newton twin — robot FOLDS the captured cloth.

A Franka arm (Featherstone) folds the real captured fabric (VBD cloth, textured with the
capture photo) by pinching a corner, lifting, carrying it diagonally over the cloth, and
releasing — VBD particle self-contact handles the stacked layers, intersection-free.

Built on the shipped cloth_franka example (coupled Featherstone+VBD, diff-IK velocity
control, cm-scale sim / m-scale viz). Two adaptations:
  * cloth = our measured, TEXTURED add_cloth_grid (not the shirt USD); rendered via log_mesh
    with the fabric image (show_triangles=False so log_state doesn't double-draw it).
  * GRASP = pin the corner particle(s) to the gripper tip while closed (a flat-grid corner
    pinch by friction alone is tuning-fragile; the pin is load-bearing, the arm does the
    motion). Release restores the particles to free.

Run (newton-spike env):
  .../python twin/newton/newton_fold.py --scene_dir <out> [--capture_dir <cap>] --viewer gl
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.viewer import ViewerGL, ViewerNull

CLOTH_KW = ("fabric", "cloth", "towel", "shirt", "blanket", "napkin", "garment", "sheet", "scarf", "rag")
DOWN_Q = (0.9239, -0.3827, 0.0, 0.0)   # gripper pointing down-ish (qw,qx,qy,qz), from the demo
OPEN, CLOSE = 0.8, 0.1


# ----------------------------- warp kernels (from the example + pin) -----------------------------
@wp.kernel
def scale_positions(src: wp.array(dtype=wp.vec3), scale: float, dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array(dtype=wp.transform), scale: float, dst: wp.array(dtype=wp.transform)):
    i = wp.tid()
    p = wp.transform_get_translation(src[i]); q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(body_q: wp.array(dtype=wp.transform), offset: wp.transform, body_id: int,
                     bodies_per_world: int, target: wp.transform,
                     ee_delta: wp.array(dtype=wp.spatial_vector)):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf); pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf); rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2],
                                           ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def compute_tip(body_q: wp.array(dtype=wp.transform), ee_id: int, ee_off: wp.vec3,
                out: wp.array(dtype=wp.vec3)):
    out[0] = wp.transform_point(body_q[ee_id], ee_off)


@wp.kernel
def pin_to_ee(body_q: wp.array(dtype=wp.transform), ee_id: int, ee_off: wp.vec3,
              pin_idx: wp.array(dtype=wp.int32), pin_off: wp.array(dtype=wp.vec3),
              particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    tip = wp.transform_point(body_q[ee_id], ee_off)
    pidx = pin_idx[i]
    particle_q[pidx] = tip + pin_off[i]
    particle_qd[pidx] = wp.vec3(0.0, 0.0, 0.0)


def load_cloth_spec(scene_dir):
    import json
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    objs = layout.get("objects", [])
    o = next((o for o in objs if any(k in str(o.get("label", "")).lower() for k in CLOTH_KW)), objs[0])
    di = o.get("depth_info") or {}
    w = float(di.get("physical_width_m") or o.get("physical_size_m") or 0.35)
    h = float(di.get("physical_height_m") or w)
    img = layout.get("image_size_px", [640, 480])
    box = o.get("box_px") or [0, 0, img[0], img[1]]
    return w, h, str(o.get("label", "cloth")), box, img


class Fold:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_substeps = 10
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viz_scale = 0.01
        self.pins = args.pins
        self.fold_mode = args.fold_mode
        self.folds = args.folds
        self.next_fold = 0

        # cloth footprint + texture
        w, h, label, box, img = load_cloth_spec(args.scene_dir)
        cap = args.capture_dir or args.scene_dir.replace("/outputs/", "/captures/")
        self.rgb_path = os.path.join(cap, "rgb.png")
        self.use_tex = (not args.no_texture) and os.path.exists(self.rgb_path)
        S = 100.0
        cell = 2.0                                   # cm
        self.dim_x = max(2, round(w * S / cell)); self.dim_y = max(2, round(h * S / cell))
        cloth_w = self.dim_x * cell; cloth_h = self.dim_y * cell
        # cloth centred on the table in front of the robot
        self.cloth_cx, self.cloth_cy, self.cloth_top = 0.0, -50.0, 21.0
        self.cell = cell
        xh, yh = cloth_w / 2.0, cloth_h / 2.0
        self.init_bbox = [self.cloth_cx - xh, self.cloth_cx + xh, self.cloth_cy - yh, self.cloth_cy + yh]
        print(f"[FOLD] '{label}' {w*100:.0f}x{h*100:.0f}cm -> grid {self.dim_x}x{self.dim_y}, "
              f"mode={self.fold_mode} folds={self.folds}", flush=True)

        self.scene = ModelBuilder(gravity=-981.0)

        # robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count

        # table
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_hx, self.table_hy, self.table_hz = 40.0, 40.0, 10.0
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(-1, wp.transform(self.table_pos_cm, wp.quat_identity()),
                                 hx=self.table_hx, hy=self.table_hy, hz=self.table_hz)

        # cloth grid (centred), resting just above the table top
        self.scene.add_cloth_grid(
            pos=wp.vec3(self.cloth_cx - 0.5 * cloth_w, self.cloth_cy - 0.5 * cloth_h, self.cloth_top),
            rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x, dim_y=self.dim_y, cell_x=cell, cell_y=cell, mass=0.1,
            tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6, edge_ke=5.0, edge_kd=1.0e-2,
            particle_radius=0.8)
        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # hide table from auto shape rendering (GL bakes prim dims, ignores scale) -> draw manually
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # contact material
        self.model.soft_contact_ke = 1e4; self.model.soft_contact_kd = 1e-2; self.model.soft_contact_mu = 0.25
        ke = self.model.shape_material_ke.numpy(); kd = self.model.shape_material_kd.numpy(); mu = self.model.shape_material_mu.numpy()
        ke[...] = 5e4; kd[...] = 1e-3; mu[...] = 1.5
        self.model.shape_material_ke = wp.array(ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.8)
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.model.edge_rest_angle.zero_()
        self.cloth_solver = SolverVBD(
            self.model, iterations=8, integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5, particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16, particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1)

        # ---- grasp / pin state ----
        self.orig_inv_mass = self.model.particle_inv_mass.numpy().copy()
        self.grasp_active = False
        self.prev_grip = OPEN
        self.pin_idx = None; self.pin_off = None
        self.tip_buf = wp.zeros(1, dtype=wp.vec3)
        self.ee_off_vec = wp.vec3(0.0, 0.0, 22.0)

        # ---- texture / render set-up ----
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False          # we draw the cloth ourselves (textured)
        self.viewer.show_particles = False
        self.tri_flat = wp.array(self.model.tri_indices.numpy().reshape(-1).astype(np.int32), dtype=wp.int32)
        pq0 = self.state_0.particle_q.numpy()
        nx = (pq0[:, 0] - pq0[:, 0].min()) / max(1e-6, np.ptp(pq0[:, 0]))
        ny = (pq0[:, 1] - pq0[:, 1].min()) / max(1e-6, np.ptp(pq0[:, 1]))
        bx0, by0, bx1, by1 = box
        uu = (bx0 + nx * (bx1 - bx0)) / img[0]; vv = (by0 + (1.0 - ny) * (by1 - by0)) / img[1]
        self.uvs = wp.array(np.stack([uu, vv], 1).astype(np.float32), dtype=wp.vec2)
        self.tex_img = None
        if self.use_tex:
            from newton._src.utils.texture import load_texture
            self.tex_img = load_texture(self.rgb_path)
        # table viz (meters)
        self.table_viz_xform = wp.array([wp.transform(
            (float(self.table_pos_cm[0]) * self.viz_scale, float(self.table_pos_cm[1]) * self.viz_scale,
             float(self.table_pos_cm[2]) * self.viz_scale), wp.quat_identity())], dtype=wp.transform)
        self.table_viz_scale = (self.table_hx * self.viz_scale, self.table_hy * self.viz_scale, self.table_hz * self.viz_scale)
        self.table_viz_color = wp.array([wp.vec3(0.55, 0.55, 0.58)], dtype=wp.vec3)

        # meter-scale shape data for the robot (the example's two-path swap)
        self.sim_shape_transform = self.model.shape_transform; self.sim_shape_scale = self.model.shape_scale
        xf = self.model.shape_transform.numpy().copy(); xf[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xf, dtype=wp.transform, device=self.model.device)
        sc = self.model.shape_scale.numpy().copy(); sc *= self.viz_scale
        self.viz_shape_scale = wp.array(sc, dtype=wp.vec3, device=self.model.device)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy(); xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                scc = shapes.scales.numpy(); scc *= self.viz_scale
                shapes.scales = wp.array(scc, dtype=wp.vec3, device=shapes.device)

        self.viz_state = self.model.state()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    # ----------------------------- robot setup + keyframes -----------------------------
    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
                         xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
                         floating=False, scale=100, enable_self_collisions=False,
                         collapse_fixed_joints=True, force_show_colliders=False)
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        q = DOWN_Q
        if self.fold_mode == "corner":
            cx, cy = 19.0, -35.0      # pick corner (near-right)
            fx, fy = -19.0, -65.0     # fold target (far-left corner)
            self.robot_key_poses = np.array([
                [3.0,  cx, cy, 42.0, *q, OPEN],    # approach above corner
                [2.0,  cx, cy, 21.5, *q, OPEN],    # descend to corner
                [1.5,  cx, cy, 21.5, *q, CLOSE],   # CLOSE -> grasp (pin)
                [2.5,  cx, cy, 42.0, *q, CLOSE],   # lift
                [4.0,  fx, fy, 42.0, *q, CLOSE],   # carry diagonally over the cloth
                [2.0,  fx, fy, 25.0, *q, CLOSE],   # lower onto far corner
                [1.5,  fx, fy, 25.0, *q, OPEN],    # OPEN -> release
                [2.0,  fx, fy, 44.0, *q, OPEN],    # retreat up
                [2.5, -45.0, -50.0, 44.0, *q, OPEN],  # clear away
            ], dtype=np.float32)
            self.fold_specs = []
        else:
            # multi / half: N alternating half-folds (y, x, y, x, ...) down to a small piece.
            # Each fold grasps the cloth's CURRENT max edge along the axis and brings it onto the
            # opposite edge (fold line = mid). bbox shrinks toward the (xmin,ymin) corner; layers double.
            nf = 1 if self.fold_mode == "half" else self.folds
            bbox = list(self.init_bbox)               # [x0, x1, y0, y1]
            poses, specs = [], []
            ZHI, ZLIFT, ZAPEX = 42.0, 39.0, 41.0
            for i in range(nf):
                axis = "y" if i % 2 == 0 else "x"
                x0, x1, y0, y1 = bbox
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                zg = 21.5 + 1.3 * i                   # grasp/land rise as the stack thickens
                zland = 23.0 + 1.3 * i
                if axis == "y":
                    g = (cx, y1); t = (cx, y0); mid = (cx, (y0 + y1) / 2.0)
                    bbox[3] = (y0 + y1) / 2.0
                else:
                    g = (x1, cy); t = (x0, cy); mid = ((x0 + x1) / 2.0, cy)
                    bbox[1] = (x0 + x1) / 2.0
                poses += [
                    [1.5, g[0], g[1], ZHI,   *q, OPEN],    # approach above grasp edge
                    [1.2, g[0], g[1], zg,    *q, OPEN],    # descend
                    [1.0, g[0], g[1], zg,    *q, CLOSE],   # grasp edge (pin band)
                    [1.5, g[0], g[1], ZLIFT, *q, CLOSE],   # peel/lift
                    [1.5, mid[0], mid[1], ZAPEX, *q, CLOSE],  # arc over the fold line
                    [1.5, t[0], t[1], ZLIFT, *q, CLOSE],   # carry over the far edge
                    [1.2, t[0], t[1], zland, *q, CLOSE],   # lay down
                    [1.0, t[0], t[1], zland, *q, OPEN],    # release
                    [1.2, t[0], t[1], ZHI,   *q, OPEN],    # retreat up
                ]
                specs.append({"axis": axis})
            poses += [[1.5, -45.0, -50.0, ZHI, *q, OPEN]]   # clear away
            self.robot_key_poses = np.array(poses, dtype=np.float32)
            self.fold_specs = specs
        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.target = self.targets[0]
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def set_up_control(self):
        self.control = self.model.control()
        out_dim = 6
        self.Jacobian_one_hots = [wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
                                  for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                             body_com: wp.array(dtype=wp.vec3), body_out: wp.array(dtype=float)):
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            X_wb = body_q[ee_id]
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd); v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]; body_out[1] = v_tip[1]; body_out[2] = v_tip[2]
            body_out[3] = omega[0]; body_out[4] = omega[1]; body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * self.model.joint_dof_count, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def compute_body_jacobian(self, joint_q, joint_qd):
        joint_q.requires_grad = True; joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(self.compute_body_out_kernel, 1,
                      inputs=[self.temp_state_for_jacobian.body_q, self.temp_state_for_jacobian.body_qd,
                              self.model.body_com],
                      outputs=[self.body_out])
        for i in range(6):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim:(i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(self, state_in):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_(); return
        interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        self.target = self.targets[interval]
        wp.launch(compute_ee_delta, dim=1,
                  inputs=[state_in.body_q, self.endeffector_offset, self.endeffector_id,
                          self.bodies_per_world, wp.transform(*self.target[:7])],
                  outputs=[self.ee_delta])
        self.compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)
        N = np.eye(J.shape[1], dtype=np.float32) - J_inv @ J
        q = state_in.joint_q.numpy()
        q_des = q.copy(); q_des[1:] = self.initial_pose[1:]
        delta_q = J_inv @ delta_target + N @ (1.0 * (q_des - q))
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]
        self.target_joint_qd.assign(delta_q)

    # ----------------------------- grasp (pin) -----------------------------
    def update_grasp(self):
        grip = float(self.target[-1])
        if self.prev_grip > 0.4 and grip <= 0.4:        # close -> grasp
            self._grasp()
        elif self.prev_grip <= 0.4 and grip > 0.4:      # open -> release
            self._release()
        self.prev_grip = grip

    def _grasp(self):
        wp.launch(compute_tip, 1, inputs=[self.state_0.body_q, self.endeffector_id, self.ee_off_vec],
                  outputs=[self.tip_buf])
        tip = self.tip_buf.numpy()[0]
        pq = self.state_0.particle_q.numpy()
        if self.fold_mode == "corner":
            d = np.linalg.norm(pq - tip, axis=1)
            idx = np.argsort(d)[:self.pins]
            what = f"corner ({self.pins} particles {idx.tolist()})"
        else:
            # pin the whole CURRENT max edge along this fold's axis (across all layers), so the
            # edge translates rigidly with the gripper and stays straight + full width
            axis = self.fold_specs[min(self.next_fold, len(self.fold_specs) - 1)]["axis"]
            self.next_fold += 1
            col = 1 if axis == "y" else 0
            idx = np.where(pq[:, col] >= pq[:, col].max() - 3.0)[0]
            what = f"fold {self.next_fold} {axis}-edge ({len(idx)} particles)"
        self.pin_idx = wp.array(idx.astype(np.int32), dtype=wp.int32)
        self.pin_off = wp.array((pq[idx] - tip).astype(np.float32), dtype=wp.vec3)
        im = self.orig_inv_mass.copy(); im[idx] = 0.0
        self.model.particle_inv_mass.assign(im)
        self.grasp_active = True
        print(f"[FOLD] GRASP {what} at t={self.sim_time:.1f}s", flush=True)

    def _release(self):
        self.model.particle_inv_mass.assign(self.orig_inv_mass)
        self.grasp_active = False
        print(f"[FOLD] RELEASE at t={self.sim_time:.1f}s", flush=True)

    # ----------------------------- sim -----------------------------
    def step(self):
        self.generate_control_joint_qd(self.state_0)
        self.update_grasp()
        self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # robot
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = pc
            self.model.gravity.assign(self.gravity_earth)
            # pin the grasped corner to the NEW ee pose (state_1.body_q), into the cloth input (state_0)
            if self.grasp_active and self.pin_idx is not None:
                wp.launch(pin_to_ee, dim=len(self.pin_idx),
                          inputs=[self.state_1.body_q, self.endeffector_id, self.ee_off_vec,
                                  self.pin_idx, self.pin_off],
                          outputs=[self.state_0.particle_q, self.state_0.particle_qd])
            # cloth
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return
        wp.launch(scale_positions, dim=self.model.particle_count,
                  inputs=[self.state_0.particle_q, self.viz_scale], outputs=[self.viz_state.particle_q])
        if self.model.body_count > 0:
            wp.launch(scale_body_transforms, dim=self.model.body_count,
                      inputs=[self.state_0.body_q, self.viz_scale], outputs=[self.viz_state.body_q])
        self.model.shape_transform = self.viz_shape_transform; self.model.shape_scale = self.viz_shape_scale
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)       # robot only (triangles hidden)
        self.viewer.log_shapes("/table", newton.GeoType.BOX, self.table_viz_scale,
                               self.table_viz_xform, self.table_viz_color)
        # textured cloth (meter scale)
        if self.use_tex:
            self.viewer.log_mesh("cloth", self.viz_state.particle_q, self.tri_flat, uvs=self.uvs,
                                 texture=self.tex_img, color=(1.0, 1.0, 1.0), backface_culling=False)
            o = getattr(self.viewer, "objects", {}).get("cloth")
            if o is not None:
                r, m, c, _ = o.material; o.material = (r, m, c, 1.0)
        else:
            self.viewer.log_mesh("cloth", self.viz_state.particle_q, self.tri_flat,
                                 color=(0.3, 0.28, 0.32), backface_culling=False)
        self.viewer.end_frame()
        self.model.shape_transform = self.sim_shape_transform; self.model.shape_scale = self.sim_shape_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", default=None)
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--fold_mode", default="multi", choices=["multi", "half", "corner"])
    ap.add_argument("--folds", type=int, default=3, help="number of alternating half-folds (multi mode)")
    ap.add_argument("--pins", type=int, default=2, help="corner particles pinned (corner mode only)")
    ap.add_argument("--no_texture", action="store_true")
    ap.add_argument("--hold", type=float, default=1800.0)
    ap.add_argument("--screenshot", default=None, help="PNG glob: save frames at fold milestones")
    ap.add_argument("--cam", default="0.25,-0.92,0.52,-36,120", help="x,y,z,pitch,yaw")
    args = ap.parse_args()

    wp.init()
    if args.viewer == "gl":
        viewer = ViewerGL()
    else:
        viewer = ViewerNull(num_frames=10 ** 9)

    ex = Fold(viewer, args)
    if args.viewer == "gl":                          # AFTER set_model() (which auto-frames the scene)
        cx, cy, cz, cp, cyaw = (float(v) for v in args.cam.split(","))
        viewer.set_camera(wp.vec3(cx, cy, cz), cp, cyaw)
    n_frames = int(ex.robot_key_poses_time[-1] * ex.fps) + 30
    shots = {}
    if args.screenshot:
        base = args.screenshot.replace(".png", "")
        # one shot just after each fold's release pose, plus the final result
        for i, _ in enumerate(ex.fold_specs):
            rel = i * 9 + 7                       # release pose index within fold i
            shots[int(ex.robot_key_poses_time[rel] * ex.fps)] = f"{base}_fold{i+1}.png"
        shots[n_frames - 1] = f"{base}_done.png"

    print(f"[FOLD] running {n_frames} frames ({ex.robot_key_poses_time[-1]:.0f}s schedule)...", flush=True)
    t0 = time.time()
    for f in range(n_frames):
        ex.step(); ex.render()
        if f in shots and args.viewer == "gl":
            from PIL import Image
            Image.fromarray(viewer.get_frame().numpy()).save(shots[f])
            print(f"[FOLD] shot -> {shots[f]}", flush=True)
        if args.viewer == "gl" and not viewer.is_running():
            break
    zmin = float(ex.state_0.particle_q.numpy()[:, 2].min())
    print(f"[FOLD] done in {time.time()-t0:.0f}s; lowest particle z={zmin:.2f}cm "
          f"({'NaN!' if not np.isfinite(zmin) else 'ok'})", flush=True)

    if args.viewer == "gl" and args.hold > 0:
        print(f"[FOLD] holding GUI {args.hold:.0f}s", flush=True)
        t_end = time.time() + args.hold
        while time.time() < t_end and viewer.is_running():
            ex.render()


if __name__ == "__main__":
    main()
