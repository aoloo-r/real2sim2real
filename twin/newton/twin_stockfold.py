# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import os as _os
import numpy as np
import warp as wp

# ROBOT_SIDE=left (default) | right — which side of the default camera the arm stands on.
# left  = base x=+50, yaw pi, key poses mirrored (arm faces the shirt+box; user-approved view)
# right = stock pose, base x=-50, unmirrored
_SIDE = 1.0 if _os.environ.get('ROBOT_SIDE', 'left') == 'left' else -1.0

# TWIN_PLACE=measured (default) -> the robot stands where the REAL robot stood.
# The capture's extrinsics give the camera pose in the real robot's base frame and the
# scene layout gives each object in the camera frame, so base->shirt and base->box are
# MEASURED from the capture, not chosen. TWIN_PLACE=stock keeps the example's own layout.
_PLACE = _os.environ.get('TWIN_PLACE', 'measured')
_TS_DEFAULT = _os.environ.get('TWIN_TS', 'robot_20260819_123718')
# TWIN_VIEW=front (default) | shoulder | capture
_VIEW = _os.environ.get('TWIN_VIEW', 'front')
# The measured standoff (68.6cm to the shirt centre) puts the two far corner grasps at
# 87cm, past the fr3's 85.5cm envelope -> the arm over-stretches. Stepping the base this
# far forward along its own facing axis brings the worst grasp to 78cm. The shirt and the
# box do not move, so the measured layout the capture gives us is preserved.
_CLOSER = float(_os.environ.get('TWIN_CLOSER_CM', '10'))
# TWIN_FOLD=assist (default) | physical. Purely frictional folds plateau: measured 54 -> 49 ->
# 45 -> 46 -> 45 -> 44cm over six passes, because after two folds the stack is ~2cm of layered
# fabric and a parallel jaw at table level only catches the top layer. 'assist' pins the outer
# half and mirrors it across the fold line (the iter9 wide-band recipe), which truly halves it.
_FOLD_MODE = _os.environ.get('TWIN_FOLD', 'assist')
# How much lower than the authored height the jaws go when closing on cloth. The stock poses
# grasp exactly at table level (z=20) and the first corner grasp visibly misses.
# 1.0 catches the corner but closes THROUGH the cloth and wedges it between the jaws, so the
# sheet rides the gripper instead of dropping on release; 0.0 misses the first grasp entirely.
_GRASP_DZ = float(_os.environ.get('TWIN_GRASP_DZ_CM', '0.5'))


def _measured_from_capture(ts):
    """Return the capture geometry in the real robot's base frame, cm.

    Keys are (forward, left, up) triples relative to the robot base flange:
      shirt / box : object centroid, cam : camera position, cam_dir : optical +z.
    """
    import json as _j
    import numpy as _np

    ex = _j.load(open(f"/home/aoloo/sam-3d-objects/captures/{ts}/extrinsics.json"))["transforms"]
    lay = _j.load(open(f"/home/aoloo/sam-3d-objects/outputs/{ts}/scene_layout.json"))["objects"]
    T = _np.array(ex["ur5e_base_link"]["T_base_cam"], float)   # camera pose in the robot base frame

    def _find(sub):
        return next(o for o in lay
                    if sub in (o.get("depth_info", {}).get("label") or o.get("label", "")).lower())

    def _to_base(o):
        p = _np.array(list(o["depth_info"]["position_cam"]) + [1.0], float)
        return (T @ p)[:3] * 100.0

    return {
        "shirt": _to_base(_find("shirt")),
        "box": _to_base(_find("box")),
        "cam": T[:3, 3] * 100.0,
        "cam_dir": T[:3, 2],
    }


# World convention for the measured layout: the arm faces -y (so the viewer looks down
# the same axis the capture did), its left is +x, up is +z.
_FWD = np.array([0.0, -1.0, 0.0])
_LEFT = np.array([1.0, 0.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])
_SHIRT_XY = np.array([0.05, -50.3])      # stock shirt bbox centre (cm), measured from the asset
_TABLE_TOP_Z = 20.0


def _place_world(ts):
    """Map the measured base-frame geometry into stock world coordinates (cm)."""
    m = _measured_from_capture(ts)
    f2, l2 = _FWD[:2], _LEFT[:2]
    base_xy = _SHIRT_XY - f2 * m["shirt"][0] - l2 * m["shirt"][1]
    base_z = _TABLE_TOP_Z - m["shirt"][2]                      # table sits this far below the flange
    base = np.array([base_xy[0], base_xy[1], base_z])
    to_world = lambda v: base + _FWD * v[0] + _LEFT * v[1] + _UP * v[2]
    d = m["cam_dir"]
    box_world, cam_world = to_world(m["box"]), to_world(m["cam"])   # anchored to the measurement
    base = base + _FWD * _CLOSER                                    # step in so the arm stops stretching
    cam_world = cam_world + _FWD * _CLOSER
    return {
        "base": base,
        "standoff": float(m["shirt"][0] - _CLOSER),
        "box": box_world,
        "cam": cam_world,
        "cam_dir": _FWD * d[0] + _LEFT * d[1] + _UP * d[2],
        "raw": m,
    }

from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        #       self-contact
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e1

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 5e1
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-2

        self.bending_ke = 5
        self.bending_kd = 5e-1

        self.scene = ModelBuilder(gravity=(0.0, 0.0, -981.0))

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table (cm scale)
        # measured layout puts the real box further to the robot's left -> wider table
        self.table_hx_cm = 70.0 if _PLACE == "measured" else 40.0
        self.table_hy_cm = 48.0 if _PLACE == "measured" else 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(*((5.0, -58.0, 10.0) if _PLACE == "measured" else (0.0, -50.0, 10.0)))
        if _PLACE == "measured":
            _pb = _place_world(_TS_DEFAULT)["base"]
            self.scene.add_shape_box(
                -1,
                xform=wp.transform((float(_pb[0]), float(_pb[1]), float(_pb[2]) / 2.0), wp.quat_identity()),
                hx=11.0,
                hy=11.0,
                hz=float(_pb[2]) / 2.0,
            )

        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            xform=wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # --- OUR REAL reconstructed box (cm units), placed clear of the fold path ---
        import os as _os, json as _json, trimesh as _tm
        _TS = _TS_DEFAULT
        _D = f"/home/aoloo/sam-3d-objects/outputs/{_TS}_full"
        try:
            _lay = _json.load(open(f"/home/aoloo/sam-3d-objects/outputs/{_TS}/scene_layout.json"))["objects"]
            _bi = next(o["id"] for o in _lay
                       if "box" in (o.get("depth_info", {}).get("label") or o.get("label", "")).lower())
            _bsz = next(o["depth_info"]["physical_size_m"] for o in _lay if o["id"] == _bi) * 100.0
            _bm = _tm.load(f"{_D}/object_{_bi}/mesh.obj", process=False, force="mesh")
            _bv = np.asarray(_bm.vertices, float); _bv -= _bv.mean(0)
            _bv *= _bsz / max(_bv.max(0) - _bv.min(0))
            _c = _bv - _bv.mean(0); _, _, _vt = np.linalg.svd(_c, full_matrices=False); _bv = _c @ _vt.T
            _e = _bv.max(0) - _bv.min(0); _o = np.argsort(_e)
            _bv = _bv[:, [_o[2], _o[1], _o[0]]]; _bv[:, 2] -= _bv[:, 2].min()

            # Single view: SAM3D squashes the axis it never saw, giving a 26 x 16.4 x 9.7cm
            # tray that nothing fits into. The capture measured the mask itself - width 0.26m
            # and height 0.2523m - so restore the second footprint axis to that measurement and
            # carry the same factor into the height, which the same foreshortening compressed.
            _dep = next(o["depth_info"] for o in _lay if o["id"] == _bi)
            _want = float(_dep.get("physical_height_m", _dep["physical_size_m"])) * 100.0
            _e = _bv.max(0) - _bv.min(0)
            _grow = _want / _e[1]
            _bv[:, 1] *= _grow
            _bv[:, 2] *= _grow
            _bv[:, 2] -= _bv[:, 2].min()

            # The box's yaw here is PCA-derived, not measured (the capture's icp rotation is
            # unreliable), so orienting its long axis along the bundle's costs no fidelity and
            # lets the folded shirt drop in lengthwise.
            _yaw = np.radians(float(_os.environ.get("TWIN_BOX_YAW_DEG", "90")))
            _cs, _sn = np.cos(_yaw), np.sin(_yaw)
            _bv[:, :2] = _bv[:, :2] @ np.array([[_cs, _sn], [-_sn, _cs]])
            print(f"[twin-stock] box rebuilt to {_bv.max(0)[0] - _bv.min(0)[0]:.0f}x"
                  f"{_bv.max(0)[1] - _bv.min(0)[1]:.0f}x{_bv.max(0)[2] - _bv.min(0)[2]:.0f}cm "
                  f"(measured mask {_bsz:.0f}x{_want:.0f}cm)", flush=True)

            # SAM3D closes the top of the box (its mesh is watertight) so it renders as a solid
            # block. The real box is open: SLICE the cap off on the lip plane - dropping whole
            # faces instead leaves a sawtooth rim - then give the shell real wall thickness with
            # an inward-offset copy. The offset matters: a mirrored copy of the SAME faces
            # z-fights and renders as speckle.
            _lip = (_bv[:, 2].max()) * 0.80
            _shell = _tm.Trimesh(vertices=_bv, faces=np.asarray(_bm.faces), process=False)
            _shell = _tm.intersections.slice_mesh_plane(
                _shell, plane_normal=[0.0, 0.0, -1.0], plane_origin=[0.0, 0.0, _lip], cap=False)
            if len(_shell.faces) > 3000:            # 23k box faces against 12.7k of cloth tanks contact cost
                try:
                    _shell = _shell.simplify_quadric_decimation(face_count=3000)
                except Exception:
                    pass

            _bv = np.asarray(_shell.vertices, float)
            _bf = np.asarray(_shell.faces, np.int32)
            _ctr = _bv[:, :2].mean(axis=0)
            _inner = _bv.copy()
            _r = _inner[:, :2] - _ctr
            _inner[:, :2] -= _r / np.maximum(np.linalg.norm(_r, axis=1, keepdims=True), 1e-6) * 0.5
            _inner[:, 2] += 0.5                                   # 5mm walls and floor
            _bv = np.vstack([_bv, _inner])
            _bf = np.vstack([_bf, _bf[:, ::-1] + len(_shell.vertices)])

            # The reconstruction's underside is open once sliced, so the box reads as hollow -
            # you see the table through it. Lay a floor across the footprint just above its base.
            _fz = _bv[:, 2].min() + 0.6
            _fx0, _fx1 = _bv[:, 0].min() + 0.5, _bv[:, 0].max() - 0.5
            _fy0, _fy1 = _bv[:, 1].min() + 0.5, _bv[:, 1].max() - 0.5
            _n0 = len(_bv)
            _bv = np.vstack([_bv, [[_fx0, _fy0, _fz], [_fx1, _fy0, _fz],
                                   [_fx1, _fy1, _fz], [_fx0, _fy1, _fz]]])
            # up-facing only: adding the reversed pair as well makes the floor z-fight and
            # the interior renders black
            _bf = np.vstack([_bf, [[_n0, _n0 + 1, _n0 + 2], [_n0, _n0 + 2, _n0 + 3]]])

            if _PLACE == "measured":
                _bxy = _place_world(_TS)["box"][:2]      # 80cm out, 36 deg to the robot's left
                _clear = float(_os.environ.get("TWIN_BOX_CLEAR_CM", "4"))
                _dir = _bxy - _SHIRT_XY
                _bxy = _bxy + _dir / np.linalg.norm(_dir) * _clear
                _bv[:, 0] += float(_bxy[0]); _bv[:, 1] += float(_bxy[1])
            else:
                _bv[:, 0] += -25.0 * _SIDE; _bv[:, 1] += -75.0
            _bv[:, 2] += 20.0

            self.box_center_cm = (float(_bv[:, 0].mean()), float(_bv[:, 1].mean()))
            # The mesh floor is a single zero-thickness quad and the cloth punches straight
            # through it on release. Back it with a real solid pad across the interior.
            _ix = (_bv[:, 0].max() - _bv[:, 0].min()) / 2.0 - 1.0
            _iy = (_bv[:, 1].max() - _bv[:, 1].min()) / 2.0 - 1.0
            self.box_floor_z_cm = float(_bv[:, 2].min()) + 2.4
            _cx0, _cy0 = float(_bv[:, 0].mean()), float(_bv[:, 1].mean())
            _z0, _z1 = float(_bv[:, 2].min()), float(_bv[:, 2].max())
            _wt = 1.2                                   # wall thickness
            _wh = (_z1 - _z0) / 2.0
            # Solid collision cage: floor pad + four walls. These are PRIMITIVES, and the GL
            # viewer bakes primitive dimensions at set_model time in cm while the scene renders
            # in metres, so they draw 100x oversized (a wall of yellow) unless hidden - same
            # reason the table is hidden and drawn manually. They are collision-only anyway;
            # the reconstructed shell remains the visual.
            self.cage_shape_idx = []
            for _t in ((_cx0, _cy0, _z0 + 1.2, _ix, _iy, 1.2),
                       (_cx0 - _ix - _wt, _cy0, _z0 + _wh, _wt, _iy, _wh),
                       (_cx0 + _ix + _wt, _cy0, _z0 + _wh, _wt, _iy, _wh),
                       (_cx0, _cy0 - _iy - _wt, _z0 + _wh, _ix + 2 * _wt, _wt, _wh),
                       (_cx0, _cy0 + _iy + _wt, _z0 + _wh, _ix + 2 * _wt, _wt, _wh)):
                self.cage_shape_idx.append(self.scene.shape_count)
                self.scene.add_shape_box(
                    -1,
                    xform=wp.transform((_t[0], _t[1], _t[2]), wp.quat_identity()),
                    hx=_t[3], hy=_t[4], hz=_t[5],
                    cfg=ModelBuilder.ShapeConfig(mu=0.8),
                )
            self.box_half_cm = ((_bv[:, 0].max() - _bv[:, 0].min()) / 2.0,
                                (_bv[:, 1].max() - _bv[:, 1].min()) / 2.0)
            self.box_rim_z_cm = float(_bv[:, 2].max())
            self.scene.add_shape_mesh(body=-1, mesh=newton.Mesh(_bv.astype(np.float32),
                _bf.reshape(-1).astype(np.int32), is_solid=False, color=(0.62, 0.50, 0.36)),
                cfg=ModelBuilder.ShapeConfig(mu=0.8))
            print(f"[twin-stock] real box '{_TS}' {_bsz:.0f}cm at "
                  f"({self.box_center_cm[0]:.0f},{self.box_center_cm[1]:.0f}) rim z="
                  f"{self.box_rim_z_cm:.0f} opened ({len(_bf)} faces) place={_PLACE}", flush=True)
        except Exception as _ex:
            print("[twin-stock] box skipped:", _ex, flush=True)

        # add the T-shirt
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.0, 70.0, 30.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.02,
                scale=1.0,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        for _ci in getattr(self, "cage_shape_idx", []):
            flags[_ci] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
            )

        self.viewer.set_model(self.model)
        if _PLACE == "measured":
            _pl = _place_world(_TS_DEFAULT)
            _view = _VIEW
            if _view == "capture":
                # exactly what the capture camera saw (the arm fills the frame, as it did)
                _cam, _tgt = _pl["cam"], _pl["cam"] + _pl["cam_dir"] * 90.0
            elif _view == "shoulder":
                # third person from behind the arm: keeps the capture's left-right layout
                _cam = _pl["base"] - _FWD * 145.0 + _UP * 125.0
                _tgt = np.array([_SHIRT_XY[0], _SHIRT_XY[1], _TABLE_TOP_Z + 4.0])
            else:
                # front view: from across the table, looking back at the robot
                _cam = _pl["base"] + _FWD * 225.0 + _UP * 95.0
                _tgt = _pl["base"] + _FWD * 60.0 + _UP * 18.0
            _cp = _cam * self.viz_scale                            # cm -> m for the viewer
            _ct = _tgt * self.viz_scale
            self.viewer.camera.pos = self.viewer.camera._as_vec3(
                wp.vec3(float(_cp[0]), float(_cp[1]), float(_cp[2]))
            )
            self.viewer.camera.look_at(wp.vec3(float(_ct[0]), float(_ct[1]), float(_ct[2])))
        else:
            self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale particle radii from cm to meters for visualization
        self.sim_particle_radius = self.model.particle_radius
        if self.model.particle_radius is not None:
            radius_np = self.model.particle_radius.numpy().copy()
            radius_np *= self.viz_scale
            self.viz_particle_radius = wp.array(radius_np, dtype=wp.float32, device=self.model.device)
        else:
            self.viz_particle_radius = None

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(self.model.gravity.shape[0], dtype=wp.vec3, device=self.model.device)
        # gravity in cm/s²
        self.gravity_earth = wp.full(
            self.model.gravity.shape[0], wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3, device=self.model.device
        )

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_out: wp.array[float],
        ):
            # body_qd is COM-referenced (linear velocity at body COM, world
            # frame).  Compute EE tip velocity in world frame, consistent with
            # compute_ee_delta which measures the tip position as
            # transform_point(body_q, ee_offset).
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            X_wb = body_q[ee_id]
            # Vector from COM to EE tip, rotated to world frame
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        if _PLACE == "measured":
            _b = _place_world(_TS_DEFAULT)["base"]
            _base_xform = wp.transform(
                (float(_b[0]), float(_b[1]), float(_b[2])),
                # forward = -y, i.e. the arm looks down the same axis the capture camera did
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0),
            )
            _pw = _place_world(_TS_DEFAULT)
            self.base_cm = np.asarray(_b, float)
            print(f"[twin-stock] robot base ({_b[0]:.0f},{_b[1]:.0f},{_b[2]:.0f})cm facing -y | "
                  f"standoff {_pw['standoff']:.0f}cm (measured {_pw['standoff'] + _CLOSER:.0f} "
                  f"- {_CLOSER:.0f} step-in)", flush=True)
        else:
            _base_xform = wp.transform(
                (50.0 * _SIDE, -50.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi) if _SIDE > 0 else wp.quat_identity(),
            )
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=_base_xform,
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # descend to working height before approaching the cloth
                [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top left
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # bottom left
                [2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top right
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        if _SIDE > 0 and _PLACE != "measured":   # mirror the stock choreography to the other side
            self.robot_key_poses[:, 1] *= -1.0   # x
            self.robot_key_poses[:, 5] *= -1.0   # qy
            self.robot_key_poses[:, 6] *= -1.0   # qz
        if _GRASP_DZ:
            _z = self.robot_key_poses[:, 3]
            _z[_z <= 21.0] -= _GRASP_DZ            # grasp heights only; lifts/drags untouched
            print(f"[twin-stock] grasp height lowered by {_GRASP_DZ:.1f}cm "
                  f"({int((_z <= 21.0 - _GRASP_DZ).sum())} poses)", flush=True)

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                22.0,
            ],
            wp.quat_identity(),
        )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel,
                1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                ],
                outputs=[self.body_out],
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    # ---- phases that run after the stock 5-corner fold -------------------------
    # Targets are generated from the cloth's live bounding box rather than hardcoded,
    # so each fold halves whatever the previous one actually produced.
    GRIP_OPEN = 0.8
    GRIP_CLOSE = 0.1
    GRIP_HARD = 0.0          # fingers fully shut, for carrying rather than dragging
    DOWN_QUAT = (0.9239, -0.3827, 0.0, 0.0)   # (qx, qy, qz, qw), same as the stock fold poses
    TABLE_TOP = 20.0
    MAX_REACH = 78.0                          # keeps the arm off its 85.5cm limit

    def _cloth_bbox(self):
        """Robust extent of the cloth bulk.

        min/max is useless here: a handful of stray particles (caught on a finger, flung by a
        contact) stretch the box to the full sheet size long after the bulk has folded, which
        puts generated grasp points in empty table.
        """
        q = self.state_0.particle_q.numpy()
        return np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)

    def _clamp_reach(self, x, y, z):
        base = getattr(self, "base_cm", None)
        if base is None:
            return x, y, z
        v = np.array([x, y, z], float) - base
        r = float(np.linalg.norm(v))
        if r <= self.MAX_REACH:
            return x, y, z
        p = base + v * (self.MAX_REACH / r)
        return float(p[0]), float(p[1]), float(p[2])

    def _append_poses(self, poses):
        rows = []
        for d, x, y, z, g in poses:
            x, y, z = self._clamp_reach(x, y, z)
            rows.append([d, x, y, z, *self.DOWN_QUAT, g])
        arr = np.array(rows, dtype=np.float32)
        self.targets = np.vstack([self.targets, arr[:, 1:]])
        self.transition_duration = np.concatenate([self.transition_duration, arr[:, 0]])
        self.robot_key_poses_time = np.cumsum(self.transition_duration)

    def _fold_poses(self):
        """Halve the bundle across its longer axis."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        sx, sy = hi[0] - lo[0], hi[1] - lo[1]
        top = self.TABLE_TOP
        if sy >= sx:                                   # carry the +y edge over to -y
            grab, drop = (cx, hi[1] - 0.03 * sy), (cx, lo[1] + 0.18 * sy)
            axis, line = 1, cy
        else:
            grab, drop = (hi[0] - 0.03 * sx, cy), (lo[0] + 0.18 * sx, cy)
            axis, line = 0, cx
        # the half beyond the fold line is what gets carried over onto the other half
        self._fold = {"axis": axis, "line": float(line), "grab": np.array(grab, float),
                      "drop": np.array(drop, float), "start": None, "idx": None}
        # Follow the stock fold's motion: pinch at table level, then DRAG low (~11cm up).
        # A high lifting arc just slips the fabric out of the fingers.
        _lerp = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        return [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_OPEN),  # press in so the jaws straddle it
            (2.5, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_CLOSE),
            (2.0, *_lerp(0.15), top + 4.0, self.GRIP_CLOSE),
            (2.5, *_lerp(0.5), top + 8.0, self.GRIP_CLOSE),       # drag low: height sheds the grip
            (3.0, drop[0], drop[1], top + 8.0, self.GRIP_CLOSE),
            (1.0, drop[0], drop[1], top + 8.0, self.GRIP_OPEN),
            (1.5, drop[0], drop[1], top + 20.0, self.GRIP_OPEN),
        ]

    def _pack_poses(self):
        """Pinch the folded bundle and drop it into the real box."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        top = self.TABLE_TOP
        bx, by = getattr(self, "box_center_cm", (cx, cy))
        rim = getattr(self, "box_rim_z_cm", top + 10.0)
        floor = getattr(self, "box_floor_z_cm", top + 1.0)
        # Grip the bundle at its centre so it hangs balanced, lift, carry over the rim and
        # release into the box. The attachment grasp in _update_grasp is what holds it.
        return [
            (2.0, cx, cy, top + 11.0, self.GRIP_OPEN),
            (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_OPEN),
            (2.0, cx, cy, top - _GRASP_DZ, self.GRIP_HARD),
            (3.0, cx, cy, top + 20.0, self.GRIP_HARD),
            (4.0, bx, by, rim + 18.0, self.GRIP_HARD),
            (3.0, bx, by, floor + 7.0, self.GRIP_HARD),   # reach down INTO the box
            (2.0, bx, by, floor + 7.0, self.GRIP_OPEN),   # short drop, no punch-through
            (2.0, bx, by, rim + 26.0, self.GRIP_OPEN),
        ]

    def _extend_plan(self):
        """Queue the next phase when the current pose list runs out."""
        # Physical folds only shrink the bundle 10-15% per pass and misfire often, so it takes
        # several passes to bring it near the box's 24x25cm interior. Kept frictional on purpose.
        # assisted folds genuinely halve, so three passes take 54cm -> ~27 -> ~14
        stages = (("fold 2", self._fold_poses), ("fold 3", self._fold_poses),
                  ("fold 4", self._fold_poses), ("pack into box", self._pack_poses))
        i = getattr(self, "_plan_stage", 0)
        if i >= len(stages):
            return False
        name, gen = stages[i]
        self._plan_stage = i + 1
        # `gen is self._pack_poses` is always False - attribute access builds a NEW bound
        # method each time - which left the pack flag off, so the carry never engaged and the
        # stale fold aid kept hold of the cloth. Compare the stage name instead.
        self._packing = name.startswith("pack")
        if self._packing:
            self._fold = None                      # fold aid must let go before the carry
        self._append_poses(gen())
        lo, hi = self._cloth_bbox()
        print(f"[twin-stock] t={self.sim_time:.0f}s phase '{name}' queued "
              f"(cloth {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm)", flush=True)
        return True

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        # After the key poses sequence ends, queue the next phase; hold still once done
        if self.sim_time >= self.robot_key_poses_time[-1]:
            if not self._extend_plan():
                self.target_joint_qd.zero_()
                return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        include_rotation = True

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control (finger positions in cm)
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    # Radius of cloth carried by the closed gripper. This has to cover the WHOLE folded
    # bundle: pinning a small ball inside a multi-layer pile drags those particles while the
    # rest stays on the table, which stretches the sheet into strands and destroys the FPS.
    PIN_RADIUS = float(_os.environ.get("TWIN_PIN_RADIUS_CM", "32"))

    def _ee_tip(self):
        """World transform of the gripper tip: (position, rotation matrix)."""
        b = self.state_0.body_q.numpy()[self.endeffector_id]
        p, q = np.asarray(b[:3], float), np.asarray(b[3:7], float)
        x, y, z, w = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return p + R @ np.array([0.0, 0.0, 22.0]), R

    def _update_fold(self):
        """Carry the outer half across the fold line while the gripper is shut.

        Frictional folding plateaus once the pile is a few layers thick, so in 'assist' mode the
        half beyond the fold line is pinned and mirrored across it, tracking how far the gripper
        has travelled from its grasp point to its drop point. The robot still drives the timing.
        """
        f = getattr(self, "_fold", None)
        if _FOLD_MODE != "assist" or f is None or self.target is None:
            return
        if float(self.target[-1]) > self.GRIP_CLOSE + 1e-3:          # jaws open -> let go
            if f["idx"] is not None:
                qd = self.state_0.particle_qd.numpy()
                qd[f["idx"]] = 0.0
                self.state_0.particle_qd.assign(qd)
                f["idx"] = None
            return

        tip, _ = self._ee_tip()
        q = self.state_0.particle_q.numpy()
        a, line = f["axis"], f["line"]
        if f["idx"] is None:
            side = np.sign(f["grab"][a] - line) or 1.0
            idx = np.where((q[:, a] - line) * side > 0.0)[0]
            if len(idx) == 0:
                return
            f["idx"], f["start"] = idx, q[idx].copy()
            print(f"[twin-stock] fold carries {len(idx)} particles across the line", flush=True)

        span = f["drop"] - f["grab"]
        denom = float(span @ span)
        t = 0.0 if denom < 1e-6 else float(((tip[:2] - f["grab"]) @ span) / denom)
        t = min(max(t, 0.0), 1.0)

        start = f["start"]
        target = start.copy()
        target[:, a] = 2.0 * line - start[:, a]                       # mirror across the fold line
        moved = start + (target - start) * t
        moved[:, 2] = start[:, 2] + np.abs(target[:, a] - start[:, a]) * 0.28 * np.sin(np.pi * t)
        q[f["idx"]] = moved
        self.state_0.particle_q.assign(q)
        qd = self.state_0.particle_qd.numpy()
        qd[f["idx"]] = 0.0
        self.state_0.particle_qd.assign(qd)

    def _update_grasp(self):
        """Carry the cloth the gripper closed on, during the pack phase only.

        The folds stay purely frictional (stock behaviour). But a parallel-jaw pinch on a
        MULTI-LAYER bundle physically will not hold it against gravity - measured here as the
        arm arriving at the box empty every time - so the carry uses a kinematic attachment:
        the particles inside PIN_RADIUS when the fingers shut ride the tip until release.
        """
        if not getattr(self, "_packing", False) or self.target is None:
            return
        shut = float(self.target[-1]) <= self.GRIP_CLOSE + 1e-3
        idx = getattr(self, "_pin_idx", None)
        if shut:
            tip, R = self._ee_tip()
            q = self.state_0.particle_q.numpy()
            if idx is None:
                idx = np.where(np.linalg.norm(q - tip, axis=1) < self.PIN_RADIUS)[0]
                if len(idx) == 0:
                    return
                self._pin_idx = idx
                self._pin_local = (q[idx] - tip) @ R          # offsets in the tip frame
                print(f"[twin-stock] grasp holds {len(idx)} particles", flush=True)
            q[idx] = tip + self._pin_local @ R.T
            self.state_0.particle_q.assign(q)
            qd = self.state_0.particle_qd.numpy()
            qd[idx] = 0.0
            self.state_0.particle_qd.assign(qd)
        elif idx is not None:
            qd = self.state_0.particle_qd.numpy()
            qd[idx] = 0.0                                     # release limp, never with stored energy
            self.state_0.particle_qd.assign(qd)
            self._pin_idx = None
            self._audit_at = self.sim_time + 3.0
            print(f"[twin-stock] released over the box", flush=True)

    def _audit_pack(self):
        """After the bundle settles, count what actually ended up in the box."""
        at = getattr(self, "_audit_at", None)
        if at is None or self.sim_time < at:
            return
        self._audit_at = None
        bx, by = getattr(self, "box_center_cm", (0.0, 0.0))
        hx, hy = getattr(self, "box_half_cm", (13.0, 13.0))
        rim = getattr(self, "box_rim_z_cm", 32.0)
        floor = getattr(self, "box_floor_z_cm", self.TABLE_TOP)
        q = self.state_0.particle_q.numpy()
        inside = ((np.abs(q[:, 0] - bx) <= hx) & (np.abs(q[:, 1] - by) <= hy)
                  & (q[:, 2] >= floor - 1.0) & (q[:, 2] <= rim + 2.0))
        n = len(q)
        below = int((q[:, 2] < floor - 1.0).sum())
        print(f"[twin-stock] AUDIT in-box {inside.sum()}/{n} "
              f"({100.0 * inside.sum() / n:.0f}%), through-floor {below}, "
              f"outside {n - int(inside.sum()) - below}", flush=True)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        if getattr(self, "_packing", False):
            if _os.environ.get("TWIN_GRASP", "attach") == "attach":
                self._update_grasp()
            self._audit_pack()
        else:
            self._update_fold()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters for visualization
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap model shape and particle data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale
        self.model.particle_radius = self.viz_particle_radius

        try:
            self.viewer.show_triangles = False        # our lime overlay is the only cloth draw
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.viz_state)
            if not hasattr(self, "_lime_tri"):
                self._lime_tri = wp.array(self.model.tri_indices.numpy().reshape(-1).astype(np.int32), dtype=wp.int32)
            self.viewer.log_mesh("shirt_lime", self.viz_state.particle_q, self._lime_tri,
                                 color=wp.vec3(0.66, 0.74, 0.36), backface_culling=False)
            # Render the table box manually at meter scale
            self.viewer.log_shapes(
                "/table",
                newton.GeoType.BOX,
                self.table_viz_scale,
                self.table_viz_xform,
                self.table_viz_color,
            )
            self.viewer.end_frame()
        finally:
            # Restore simulation shape and particle data
            self.model.shape_transform = self.sim_shape_transform
            self.model.shape_scale = self.sim_shape_scale
            self.model.particle_radius = self.sim_particle_radius

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=10500)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
