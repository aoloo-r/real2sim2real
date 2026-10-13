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
# TWIN_FOLD=grasp (default) | assist | physical.
#   grasp   - the gripper pinches a small patch and CARRIES it; the rest of the garment is left
#             to VBD, so the flap drapes, sags and its sleeves lag. This is the only mode in
#             which the shirt behaves like a shirt.
#   assist  - the whole half beyond the fold line is position-prescribed to its mirror image and
#             its velocity zeroed every step, so during the carry that half is not simulated at
#             all. It also mirrors x only and leaves z alone, so the flap lands COPLANAR with the
#             stationary half instead of on top of it - four layers at two heights, resolved
#             afterwards by self-collision. Clean numbers, but it is an animated reflection.
#   physical- friction alone. Honest, but folds plateau: 54 -> 49 -> 45 -> 46 -> 45 -> 44cm over
#             six passes, because a parallel jaw at table level only catches the top layer.
_FOLD_MODE = _os.environ.get('TWIN_FOLD', 'grasp')
# Radius of the pinch in 'grasp' mode. A real fingertip grip on cloth catches a few cm of fabric;
# PIN_RADIUS (32cm, for the pack carry) would swallow the whole bundle and reproduce 'assist'.
_FOLD_PIN = float(_os.environ.get('TWIN_FOLD_PIN_CM', '4'))
# Length folds (collar down / hem up) carry the WHOLE upper or lower half, not a narrow strip.
# A 4cm pinch at one point drags that half from a single spot: the ends lag (54cm -> 41cm
# instead of 27cm), and pulling the corners one at a time to fix it TWISTS the flap into a
# lump. A wider hold spreads the pull along the edge so the half comes over flat, which is
# what two hands do. Side folds keep the narrow pinch - the strip follows fine.
_FOLD_PIN_WIDE = float(_os.environ.get('TWIN_FOLD_PIN_WIDE_CM', '11'))
# Length folds carry a SHEET (26x27cm), not a strip, and one 4cm patch cannot carry a sheet
# flat: it splays, and all the strain runs through that patch until the solve diverges. Three
# separate small holds spread across the fold edge carry it evenly - what two hands do. Each
# hold is still a normal 4cm pinch; they ride the gripper together.
_FOLD_SPAN = float(_os.environ.get('TWIN_FOLD_SPAN_CM', '9'))
# Hard cap on how much fabric a grasp may hold. Pinned particles are POSITION-FORCED, so no
# material stiffness resists them: drag a large set rigidly through a multi-layer stack and the
# triangles bridging held and unheld regions stretch until the mesh visibly SHREDS. Each hold
# catches every layer under it, so on a thick bundle three holds reached 1780 particles (28% of
# the shirt) and tore it. Keep the closest ones and let the rest follow through the cloth.
_FOLD_HOLD_MAX = int(_os.environ.get('TWIN_FOLD_HOLD_MAX', '420'))
# Compliance of the grip. GAIN 1.0 would close the gap to the gripper in a single frame, i.e.
# a hard pin; lower lets the cloth pull back. VMAX caps how fast held fabric can be dragged.
_GRIP_GAIN = float(_os.environ.get('TWIN_GRIP_GAIN', '1.0'))   # 1.0 = firm hold (default)
_GRIP_VMAX = float(_os.environ.get('TWIN_GRIP_VMAX_CM_S', '45'))
# MAXIMUM multiplier on the fold CARRY segments. Applied ADAPTIVELY: what destabilises the
# solve is dragging the pinch through many layers, so a single-layer fold runs at stock speed
# and only a thick stack crawls. A flat 2.5x made every fold slow and still diverged on a
# 24x56 four-layer bundle; a flat 4.5x fixed that and made the whole fold tedious to watch.
_CARRY_SLOW = float(_os.environ.get('TWIN_CARRY_SLOW', '4.5'))
# Closing the loop between the twin and the planner.
#   TWIN_DUMP_STATE=<f.npy>  write the settled cloth state and exit, so the planner can be run
#                            on the state the ROBOT actually observes.
#   TWIN_PLAN_FILE=<f.txt>   read the plan from a file the planner wrote, instead of me pasting
#                            a TWIN_PLAN string copied out of a search log by hand.
_DUMP_STATE = _os.environ.get('TWIN_DUMP_STATE', '')
_PLAN_FILE = _os.environ.get('TWIN_PLAN_FILE', '')
#   TWIN_LOAD_STATE=<f.npy>   start from a SAVED garment state instead of a flat shirt, so the
#                             arm can be given a half-folded / rotated / displaced shirt - the
#                             states fold_search.py --make-starts writes and plans against.
_LOAD_STATE = _os.environ.get('TWIN_LOAD_STATE', '')
# N ROBOTS IN ONE SCENE. Each world is a complete environment - its own arm, its own table, its
# own shirt - laid out side by side so several folds can be watched at once. Worlds are
# independent: nothing couples them, they simply share one Model so one solver step advances
# all of them. TWIN_WORLD_GAP is the spacing between environments.
_WORLDS = int(_os.environ.get('TWIN_WORLDS', '1'))
_WORLD_GAP = float(_os.environ.get('TWIN_WORLD_GAP_CM', '260'))
# Per-world plans: TWIN_PLANS=planA.txt:planB.txt... one per world (falls back to TWIN_PLAN).
_PLANS = [f for f in _os.environ.get('TWIN_PLANS', '').split(':') if f]
# Per-world start states, same form.
_STATES = [f for f in _os.environ.get('TWIN_LOAD_STATES', '').split(':') if f]
if _PLAN_FILE:
    with open(_PLAN_FILE) as _fh:
        _PLAN = [a.strip() for a in _fh.read().replace('\n', ',').split(',') if a.strip()]
# How much lower than the authored height the jaws go when closing on cloth. The stock poses
# grasp exactly at table level (z=20) and the first corner grasp visibly misses.
# 1.0 catches the corner but closes THROUGH the cloth and wedges it between the jaws, so the
# sheet rides the gripper instead of dropping on release; 0.0 misses the first grasp entirely.
_GRASP_DZ = float(_os.environ.get('TWIN_GRASP_DZ_CM', '0.5'))
# TWIN_FOLD_STYLE=shirt folds the FLAT sheet in three deterministic moves (left panel, right
# panel, then in half), skipping the stock 5-corner choreography which crumples first and needs
# extra halvings afterwards. 'stock' keeps the original behaviour.
_FOLD_STYLE = _os.environ.get('TWIN_FOLD_STYLE', 'shirt')
# TWIN_PLAN=a,b,c executes a fold sequence found by fold_search.py with the real arm, e.g.
# TWIN_PLAN=half_right_in,panel_left_in,half_collar_dn. The names and their (axis, frac, side)
# are the SAME action set fold_env rolls out, and a fold is a mirror across a line in both, so
# a plan the search validated headless transfers here without reinterpretation.
_PLAN = [a for a in _os.environ.get('TWIN_PLAN', '').split(',') if a]
# (axis, frac, side, grab_at). grab_at is WHERE ALONG THE FOLD EDGE the gripper takes hold,
# as a fraction of the other axis. 0.5 grabs the middle and the far ends of the edge lag and
# drag - a "half" fold of a 55cm shirt landed at 38cm instead of 27.5cm. 0.75/0.25 grab the
# upper and lower parts, so a side fold done as two moves completes where one move does not.
_PLAN_ACTIONS = {
    "half_hem_up":    (1, 0.5, +1, 0.5),
    "half_collar_dn": (1, 0.5, -1, 0.5),
    "half_right_in":  (0, 0.5, +1, 0.5),
    "panel_left_in":  (0, 1.0 / 3.0, -1, 0.5),
    "panel_right_in": (0, 2.0 / 3.0, +1, 0.5),
    "right_in_top":   (0, 0.5, +1, 0.75),
    "right_in_bot":   (0, 0.5, +1, 0.25),
    "left_panel_top": (0, 1.0 / 3.0, -1, 0.75),
    "left_panel_bot": (0, 1.0 / 3.0, -1, 0.25),
    # Same treatment for the LENGTH folds: a single mid grab took 54cm -> 41cm where a
    # complete halving is 27cm, because the left and right ends of that edge lag exactly as
    # the top and bottom did on the side folds.
    "collar_dn_l":    (1, 0.5, -1, 0.25),
    "collar_dn_r":    (1, 0.5, -1, 0.75),
    "hem_up_l":       (1, 0.5, +1, 0.25),
    "hem_up_r":       (1, 0.5, +1, 0.75),
}
# TWIN_MODE=auto runs the scripted fold/unfold/pack; teleop hands the arm to the keyboard.
_MODE = _os.environ.get('TWIN_MODE', 'auto')
# Where teleop demonstrations are written, and how often the cloth state is sampled.
# The arm/gripper track is recorded every frame (cheap); cloth particles are a GPU
# readback, so they go at a lower rate.
_DATA_DIR = _os.environ.get('TWIN_DATA_DIR', _os.path.expanduser('~/real2sim2real/data/teleop'))
_CLOTH_EVERY = int(_os.environ.get('TWIN_CLOTH_EVERY', '6'))      # frames: 6 -> 10Hz at 60fps
_CLOTH_STRIDE = int(_os.environ.get('TWIN_CLOTH_STRIDE', '4'))    # keep every Nth particle


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

        self.n_worlds = max(1, _WORLDS)
        self.world_dx = [i * _WORLD_GAP for i in range(self.n_worlds)]
        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)          # measured once, for the dof counts
            for i in range(self.n_worlds):
                # each environment shifted along x so they stand side by side
                # Offset inside the builder rather than via add_world(xform=...): passing an
                # xform made the arms vanish from the render entirely, while a single world
                # drew correctly.
                _w = ModelBuilder()
                self.create_articulation(_w, dx=self.world_dx[i])
                self.scene.add_world(_w)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count
            if self.n_worlds > 1:
                print(f"[twin-parallel] {self.n_worlds} environments, "
                      f"{_WORLD_GAP:.0f}cm apart", flush=True)

        # add a table (cm scale)
        # measured layout puts the real box further to the robot's left -> wider table
        self.table_hx_cm = 70.0 if _PLACE == "measured" else 40.0
        self.table_hy_cm = 48.0 if _PLACE == "measured" else 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(*((5.0, -58.0, 10.0) if _PLACE == "measured" else (0.0, -50.0, 10.0)))
        # One pedestal and one table PER ENVIRONMENT - otherwise every world past the first
        # has a robot and a shirt but nothing to work on, and the cloth falls to the ground.
        for _dx in self.world_dx:
            if _PLACE == "measured":
                _pb = _place_world(_TS_DEFAULT)["base"]
                self.scene.add_shape_box(
                    -1,
                    xform=wp.transform((float(_pb[0]) + _dx, float(_pb[1]), float(_pb[2]) / 2.0),
                                       wp.quat_identity()),
                    hx=11.0,
                    hy=11.0,
                    hz=float(_pb[2]) / 2.0,
                )
            if _dx == self.world_dx[0]:
                self.table_shape_idx = self.scene.shape_count
            self.table_shape_idx_all = getattr(self, "table_shape_idx_all", [])
            self.table_shape_idx_all.append(self.scene.shape_count)
            self.scene.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3(self.table_pos_cm[0] + _dx, self.table_pos_cm[1], self.table_pos_cm[2]),
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
            # One reconstructed box per environment, shifted with its world.
            for _dx in self.world_dx:
                _bvw = _bv.copy()
                _bvw[:, 0] += _dx
                self.scene.add_shape_mesh(body=-1, mesh=newton.Mesh(_bvw.astype(np.float32),
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

        for _w in range(self.n_worlds if self.add_cloth else 0):
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(self.world_dx[_w], 70.0, 30.0),
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

        if self.add_cloth:
            self.scene.color()          # once, after every world's cloth is in

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        if self.n_worlds > 1:
            print(f"[twin-parallel] built {self.model.body_count} bodies, "
                  f"{self.model.shape_count} shapes, {self.model.particle_count} particles "
                  f"across {self.n_worlds} environments", flush=True)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        for _ti in getattr(self, "table_shape_idx_all", [self.table_shape_idx]):
            flags[_ti] &= ~int(newton.ShapeFlags.VISIBLE)
        for _ci in getattr(self, "cage_shape_idx", []):
            flags[_ci] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        (float(self.table_pos_cm[0]) + _dx) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
                for _dx in self.world_dx        # one table per environment
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)] * self.n_worlds,
                                       dtype=wp.vec3)

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
        if self.n_worlds > 1:
            # The viewer AUTO-COMPUTES a grid offset per world and adds it on top of whatever
            # the Model already contains. Our environments are already laid out in x inside the
            # Model, so that extra offset pushed every arm and table off-screen, while the cloth
            # - drawn manually with log_mesh, which bypasses world offsets - stayed put. That is
            # why three shirts appeared with no robots.
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
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

        # One kernel per environment. The end-effector index is baked in with wp.static, so a
        # single kernel can only ever read world 0's arm - which is why only the first robot
        # moved. N small kernels is the cheapest way to give every world its own.
        self.ee_ids = [self.endeffector_id + w * self.bodies_per_world
                       for w in range(self.n_worlds)]

        def _make_out_kernel(ee_index):
            @wp.kernel
            def _body_out_w(
                body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector],
                body_com: wp.array[wp.vec3],
                body_out: wp.array[float],
            ):
                ee_id = wp.static(ee_index)
                ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
                X_wb = body_q[ee_id]
                r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
                qd = body_qd[ee_id]
                omega = wp.spatial_bottom(qd)
                v_tip = wp.spatial_top(qd) + wp.cross(omega, r_world)
                body_out[0] = v_tip[0]
                body_out[1] = v_tip[1]
                body_out[2] = v_tip[2]
                body_out[3] = omega[0]
                body_out[4] = omega[1]
                body_out[5] = omega[2]
            return _body_out_w

        self.body_out_kernels = [_make_out_kernel(i) for i in self.ee_ids]
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(max(1, self.n_worlds), dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def create_articulation(self, builder, dx=0.0):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        if _PLACE == "measured":
            _b = _place_world(_TS_DEFAULT)["base"]
            _base_xform = wp.transform(
                (float(_b[0]) + dx, float(_b[1]), float(_b[2])),
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
        if _PLAN or _FOLD_STYLE == "shirt":
            # keep only the opening "descend to working height" pose; the three garment folds
            # are generated from the flat sheet instead of the stock corner-dragging sequence
            self.robot_key_poses = self.robot_key_poses[:1]

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
        kernel=None,
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
                kernel if kernel is not None else self.compute_body_out_kernel,
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

    # ---- per-world plan execution: each environment runs its OWN candidate ----------
    def _init_worlds(self):
        """One independent plan cursor, waypoint list and fold state per environment.

        Sharing them is what made all N arms move in lockstep: same waypoints, same clock, same
        fold. A parallel TRIAL needs each world to hold its own plan and its own fold bookkeeping
        while the single GPU solver step advances them all together.
        """
        self.W = []
        for w in range(self.n_worlds):
            plan = list(_PLAN)
            if w < len(_PLANS):
                with open(_PLANS[w]) as fh:
                    plan = [a.strip() for a in fh.read().replace("\n", ",").split(",") if a.strip()]
            self.W.append({
                "plan": plan, "step": 0, "targets": np.zeros((0, 8), np.float32),
                "times": np.zeros(0, np.float32), "fold": None, "stack": [], "done": False,
                "flat_q": None, "sleeve_ids": None, "sleeve_set": None,
                "body_cx": None, "body_half_w": None,
                "target": np.array([0, -40, 45, *self.DOWN_QUAT, self.GRIP_OPEN], np.float32),
            })
        print("[twin-parallel] plans: "
              + " | ".join(f"w{w}={'>'.join(d['plan']) or '(none)'}"
                           for w, d in enumerate(self.W)), flush=True)

    def _ee_tip_w(self, w):
        b = self.state_0.body_q.numpy()[self.ee_ids[w]]
        p, q = np.asarray(b[:3], float), np.asarray(b[3:7], float)
        x, y, z, ww = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * ww), 2 * (x * z + y * ww)],
            [2 * (x * y + z * ww), 1 - 2 * (x * x + z * z), 2 * (y * z - x * ww)],
            [2 * (x * z - y * ww), 2 * (y * z + x * ww), 1 - 2 * (x * x + y * y)],
        ])
        return p + R @ np.array([0.0, 0.0, 22.0]), R

    def _append_w(self, w, poses):
        if not poses:
            return
        d = self.W[w]
        # Clamp to THIS arm's reach. Without it the gripper never arrives at an out-of-reach
        # grasp - it stops at the envelope and closes on whatever is there, which is why w0 kept
        # holding 87 particles at a point measured to have 187.
        if not hasattr(self, "_reach_warned"):
            self._reach_warned = [False] * self.n_worlds
        base = np.asarray(self.base_cm, float).copy()
        base[0] += self.world_dx[w]
        rows = []
        for t, x, y, z, g in poses:
            v = np.array([x, y, z], float) - base
            r = float(np.linalg.norm(v))
            if r > self.MAX_REACH:
                if g <= self.GRIP_CLOSE + 1e-3 and not self._reach_warned[w]:
                    self._reach_warned[w] = True
                    print(f"[twin-parallel] w{w} OUT OF REACH: grasp {r:.0f}cm from base vs "
                          f"{self.MAX_REACH:.0f}cm limit - pulled {r - self.MAX_REACH:.0f}cm "
                          f"short of the fold target", flush=True)
                x, y, z = base + v * (self.MAX_REACH / r)
            rows.append([t, x, y, z, *self.DOWN_QUAT, g])
        arr = np.array(rows, dtype=np.float32)
        d["targets"] = np.vstack([d["targets"], arr[:, 1:]]) if len(d["targets"]) else arr[:, 1:]
        base = d["times"][-1] if len(d["times"]) else self.sim_time
        d["times"] = np.concatenate([d["times"], base + np.cumsum(arr[:, 0])])

    def _extend_plan_w(self, w):
        """Queue this world's next fold. Returns False when its plan is finished."""
        d = self.W[w]
        if d["step"] >= len(d["plan"]):
            if not d["done"]:
                d["done"] = True
                lo, hi = self._cloth_bbox_w(w)
                print(f"[twin-parallel] w{w} plan complete: "
                      f"{hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm", flush=True)
            return False
        name = d["plan"][d["step"]]
        d["step"] += 1
        poses = self._plan_poses_w(w, name)
        self._append_w(w, poses)
        lo, hi = self._cloth_bbox_w(w)
        print(f"[twin-parallel] w{w} fold {d['step']}/{len(d['plan'])} {name} "
              f"(cloth {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm)", flush=True)
        return True

    def _plan_poses_w(self, w, name):
        """Waypoints for one action, computed against THIS world's shirt."""
        d = self.W[w]
        a, b = self._world_slice(w)
        q = self.state_0.particle_q.numpy()[a:b]
        if d["flat_q"] is None:
            d["flat_q"] = q.copy()
            y = d["flat_q"][:, 1]
            edges = np.linspace(y.min(), y.max(), 11)
            widths = [float(np.ptp(d["flat_q"][(y >= lo) & (y < hi), 0]))
                      if ((y >= lo) & (y < hi)).sum() else 0.0
                      for lo, hi in zip(edges[:-1], edges[1:])]
            widths = np.asarray(widths)
            k = int(widths.argmax())
            band = np.where((y >= edges[k]) & (y < edges[k + 1]))[0]
            d["sleeve_ids"] = (int(band[d["flat_q"][band, 0].argmin()]),
                               int(band[d["flat_q"][band, 0].argmax()]))
            body_w = float(np.median([x for x in widths if x < 0.75 * widths.max()]))
            d["body_half_w"] = 0.5 * body_w
            d["body_cx"] = float(np.median(d["flat_q"][:, 0]))

        # Did the PREVIOUS fold actually move the half it was meant to? A strong grasp with no
        # extent change means the pinch carried the top layer and left the rest behind - the
        # measurement that separates that from a weak grasp or an unreachable target.
        prev = d.get("prev_fold")
        if prev is not None:
            ax_p, line_p, side_p, idx_p = prev
            crossed = int((((q[idx_p, ax_p] - line_p) * side_p) < 0.0).sum())
            print(f"[twin-parallel] w{w} previous fold moved {crossed}/{len(idx_p)} "
                  f"({100.0 * crossed / max(len(idx_p), 1):.0f}%) of the intended half",
                  flush=True)

        lo, hi = self._cloth_bbox_w(w)
        if name in ("sleeve_l_in", "sleeve_r_in"):
            tip = q[d["sleeve_ids"][0] if name == "sleeve_l_in" else d["sleeve_ids"][1]]
            line = d["body_cx"] if name == "sleeve_r_in" else d["body_cx"] - d["body_half_w"]
            grab = (float(tip[0]), float(tip[1]))
            drop = ((float(2.0 * line - tip[0]), float(tip[1])) if name == "sleeve_r_in"
                    else (float(line), float(tip[1])))
            axis = 0
        else:
            axis, frac, side, grab_at = _PLAN_ACTIONS[name]
            other = 1 - axis
            line = lo[axis] + (hi[axis] - lo[axis]) * frac
            gy = lo[other] + (hi[other] - lo[other]) * grab_at
            band = np.abs(q[:, other] - gy) < 4.0
            if int(band.sum()) > 20:
                col = q[band, axis]
                edge = float(np.percentile(col, 97.0 if side > 0 else 3.0))
            else:
                edge = hi[axis] if side > 0 else lo[axis]
            # Put the grasp WHERE THERE IS FABRIC TO HOLD. A fixed 3cm inside the edge lands on
            # a thin rim on some states; the hold then catches ~87 particles and the carry pulls
            # the shirt into a strand. Step inward until the hold has real cloth in it.
            floor_z = float(np.percentile(q[:, 2], 2.0))
            g = [0.0, 0.0]
            g[other] = float(gy)
            best = (float(edge - 3.0 * side), -1)
            # The grasp must stay on the MOVING side of the fold line. The inward search ran
            # to 23cm, but a third-panel is only ~18cm wide, so it walked past the line and
            # gripped the half meant to stay put - which is why the "moving" set came out at
            # 4218 particles (66% of the shirt) and the panel barely folded.
            # Start from the MIDDLE of the piece being moved, not its edge. A third-panel is
            # ~18cm wide and its outer edge is thin - a 4cm pinch there holds 61 particles.
            # Halfway between edge and fold line is where the panel actually has fabric, and
            # mirroring from there folds it just the same.
            _span_to_line = abs(float(edge) - float(line))
            _mid_in = max(3.0, 0.5 * _span_to_line)
            _max_in = max(3.0, _span_to_line - 4.0)
            _try = sorted(np.arange(3.0, min(24.0, _max_in) + 0.1, 2.0),
                          key=lambda v: abs(v - _mid_in))     # nearest the middle first
            for inset in _try:
                gx_try = float(edge - inset * side)
                dd = (q[:, axis] - gx_try) ** 2 + (q[:, other] - gy) ** 2
                n_here = int(((dd < _FOLD_PIN ** 2) & (q[:, 2] < floor_z + _FOLD_PIN)).sum())
                if n_here > best[1]:
                    best = (gx_try, n_here)
                if n_here >= 130:
                    break
            g[axis] = best[0]
            if best[1] < 130:
                print(f"[twin-parallel] w{w} {name}: thin edge, best hold {best[1]} particles",
                      flush=True)
            d2 = (q[:, 0] - g[0]) ** 2 + (q[:, 1] - g[1]) ** 2
            near = q[int(np.argmin(d2))]
            grab = (float(near[0]), float(near[1]))
            drop = list(grab)
            drop[axis] = float(2.0 * line - grab[axis])
            drop = tuple(drop)

        top = self.TABLE_TOP
        d["fold"] = {"axis": axis, "line": float(line), "grab": np.array(grab, float),
                     "drop": np.array(drop, float), "pin_idx": None, "pin_local": None,
                     # Spread holds for EVERY fold. Measured: a length fold with three
                     # holds moved 85% of its half; width/panel folds with a single pinch
                     # moved 9-14%. A panel is thousands of particles - one 4cm grip
                     # cannot carry it.
                     "two_point": name not in ("sleeve_l_in", "sleeve_r_in"),
                     # Spread holds for PANEL and HALF folds, a single pinch for SLEEVES.
                     # Measured both ways: panels 9% -> 24% of their half moved with the
                     # spread, but sleeves 54% -> 6%, because a spread of +-35% of the
                     # garment width puts the outer holds on the TORSO, so the arm drags
                     # the body instead of carrying the sleeve.
                     "span_cm": float(np.clip(0.35 * (hi[1 - axis] - lo[1 - axis]), 6.0, 22.0))}
        # WHERE TO HOLD THE PIECE. Fixed +-35% offsets put the outer holds past the fabric on a
        # shirt that narrows at collar and hem, so only the middle one carried and panels moved
        # 25% of their cloth. Scan along the fold edge, keep only offsets with cloth under them,
        # and spread as widely as the piece allows. For a sleeve, stay INSIDE the sleeve.
        _ea = 1 - axis
        _only = None
        if name in ("sleeve_l_in", "sleeve_r_in"):
            fq0, hw0, cx0_ = d["flat_q"], d["body_half_w"], d["body_cx"]
            _only = (np.where(fq0[:, 0] > cx0_ + hw0)[0] if name == "sleeve_r_in"
                     else np.where(fq0[:, 0] < cx0_ - hw0)[0])
            _reach = 0.40 * float(np.ptp(q[_only, _ea]))
        else:
            _reach = 0.45 * float(hi[_ea] - lo[_ea])
        _fl = float(np.percentile(q[:, 2], 2.0))
        _keep = np.zeros(len(q), bool)
        if _only is not None:
            _keep[_only] = True
        _good = []
        for _o in np.linspace(-_reach, _reach, 13):
            _c = np.array(grab, float)
            _c[_ea] += _o
            _m = (((q[:, 0] - _c[0]) ** 2 + (q[:, 1] - _c[1]) ** 2) < _FOLD_PIN ** 2)
            _m &= q[:, 2] < _fl + _FOLD_PIN
            if _only is not None:
                _m &= _keep
            if int(_m.sum()) >= 40:
                _good.append(float(_o))
        if len(_good) >= 3:
            _offs = [_good[0], _good[len(_good) // 2], _good[-1]]
        elif _good:
            _offs = [_good[len(_good) // 2]]
        else:
            _offs = [0.0]
        d["fold"]["hold_offsets"] = _offs
        d["fold"]["hold_axis"] = _ea
        d["fold"]["only"] = _only
        d["fold"]["two_point"] = True
        print(f"[twin-parallel] w{w} {name}: {len(_offs)} hold(s) at "
              f"{[round(o) for o in _offs]}cm along the edge", flush=True)

        _sgn = float(np.sign(grab[axis] - line)) or 1.0
        if name in ("sleeve_l_in", "sleeve_r_in"):
            # A sleeve fold moves the SLEEVE, not the whole half. Scoring it against everything
            # past the fold line (3218 particles) measured a target it never aimed at, and the
            # figure swung 54%/6%/3% across runs while the fold itself barely changed.
            fq, hw, cx0 = d["flat_q"], d["body_half_w"], d["body_cx"]
            _mv = (np.where(fq[:, 0] > cx0 + hw)[0] if name == "sleeve_r_in"
                   else np.where(fq[:, 0] < cx0 - hw)[0])
        else:
            _mv = np.where(((q[:, axis] - line) * _sgn) > 0.0)[0]
        d["prev_fold"] = (axis, float(line), _sgn, _mv)
        _col = (np.abs(q[:, 0] - grab[0]) < 4.0) & (np.abs(q[:, 1] - grab[1]) < 4.0)
        _layers = (1 + int(np.ptp(q[_col, 2]) / 1.6)) if int(_col.sum()) else 0
        print(f"[twin-parallel] w{w} {name}: moving half {len(_mv)} particles, "
              f"~{_layers} layer(s) under the grasp", flush=True)

        travel = float(np.hypot(drop[0] - grab[0], drop[1] - grab[1]))
        rise = float(np.clip(0.10 * travel, 4.0, 8.0))
        apex = float(np.clip(0.16 * travel, 6.0, 12.0))
        # Adaptive carry, as the single-world twin does: scale the drag with how much fabric is
        # actually under the grasp. A fixed factor dragged a marginal 87-particle hold at full
        # speed and pulled the shirt into a strand.
        _qw = self.state_0.particle_q.numpy()[a:b]
        _floor = float(np.percentile(_qw[:, 2], 2.0))
        _offs = (np.linspace(-d["fold"]["span_cm"], d["fold"]["span_cm"], 3)
                 if axis == 1 else [0.0])
        _n = 0
        for _o in _offs:
            _gx = grab[0] + (_o if axis == 1 else 0.0)
            _gy = grab[1] + (0.0 if axis == 1 else _o)
            _dd = (_qw[:, 0] - _gx) ** 2 + (_qw[:, 1] - _gy) ** 2
            _n += int(((_dd < _FOLD_PIN ** 2) & (_qw[:, 2] < _floor + _FOLD_PIN)).sum())
        _cap = _CARRY_SLOW * (2.0 if axis == 1 else 1.0)
        c = float(np.clip(1.0 + (_n - 80) / 120.0, 1.0, _cap))
        print(f"[twin-parallel] w{w} {name}: carry x{c:.1f} ({_n} under the grasp)", flush=True)
        _l = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        return [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_OPEN),
            (2.5, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_CLOSE),
            (2.0 * c, *_l(0.2), top + rise, self.GRIP_CLOSE),
            (2.5 * c, *_l(0.5), top + apex, self.GRIP_CLOSE),
            (2.5 * c, *_l(0.8), top + rise, self.GRIP_CLOSE),
            (2.0 * c, drop[0], drop[1], top + 2.0, self.GRIP_CLOSE),
            (1.5 * c, drop[0], drop[1], top + 2.0, self.GRIP_OPEN),
            (1.5, drop[0], drop[1], top + 20.0, self.GRIP_OPEN),
        ]

    def _update_fold_grasp_w(self, w):
        """This world's pinch: hold fabric from ITS shirt only, ride ITS gripper."""
        d = self.W[w]
        f = d["fold"]
        if f is None:
            return
        shut = float(d["target"][-1]) <= self.GRIP_CLOSE + 1e-3
        a, b = self._world_slice(w)
        qd_all = self.state_0.particle_qd.numpy()
        if not shut:
            if f["pin_idx"] is not None:
                qd_all[f["pin_idx"]] = 0.0
                self.state_0.particle_qd.assign(qd_all)
                f["pin_idx"] = None
            return
        tip, R = self._ee_tip_w(w)
        q_all = self.state_0.particle_q.numpy()
        q = q_all[a:b]
        if f["pin_idx"] is None:
            if f.get("two_point"):
                # Holds at the offsets chosen when the fold was planned - each one verified to
                # have cloth under it, and for a sleeve restricted to the sleeve itself so the
                # outer holds cannot grab torso.
                ea = int(f.get("hold_axis", 1 - int(f["axis"])))
                offs = f.get("hold_offsets") or [0.0]
                sel = np.zeros(len(q), dtype=bool)
                for o in offs:
                    c = tip.copy(); c[ea] += float(o)
                    sel |= np.linalg.norm(q - c, axis=1) < _FOLD_PIN
                only = f.get("only")
                if only is not None and len(only):
                    keep = np.zeros(len(q), bool)
                    keep[only] = True
                    sel &= keep
                idx = np.where(sel)[0]
            else:
                idx = np.where(np.linalg.norm(q - tip, axis=1) < _FOLD_PIN)[0]
            if len(idx) > _FOLD_HOLD_MAX:
                dist = np.linalg.norm(q[idx] - tip, axis=1)
                idx = idx[np.argsort(dist)[:_FOLD_HOLD_MAX]]
            if len(idx) == 0:
                return
            f["pin_idx"] = idx + a                       # absolute particle indices
            f["pin_local"] = (q[idx] - tip) @ R
            warn = "  <-- MISSED" if len(idx) < 50 else ""
            print(f"[twin-parallel] w{w} pinch holds {len(idx)} particles{warn}", flush=True)
        q_all[f["pin_idx"]] = tip + f["pin_local"] @ R.T
        self.state_0.particle_q.assign(q_all)
        qd_all[f["pin_idx"]] = 0.0
        self.state_0.particle_qd.assign(qd_all)

    def _world_slice(self, w):
        """Particle index range for world w's shirt.

        Every world holds one identical garment, added in order, so the particle array is N
        contiguous blocks. Without this the fold logic measured ALL the shirts as one garment
        and reached for a grasp point in the empty space between two environments (measured:
        grab x=226 with the shirts at x=0 and x=260).
        """
        per = self.model.particle_count // self.n_worlds
        return w * per, (w + 1) * per

    def _cloth_bbox_w(self, w):
        a, b = self._world_slice(w)
        q = self.state_0.particle_q.numpy()[a:b]
        return np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)

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
        if not poses:                       # e.g. unfold with nothing left on the stack
            return
        rows = []
        for d, x, y, z, g in poses:
            x, y, z = self._clamp_reach(x, y, z)
            rows.append([d, x, y, z, *self.DOWN_QUAT, g])
        arr = np.array(rows, dtype=np.float32)
        self.targets = np.vstack([self.targets, arr[:, 1:]])
        self.transition_duration = np.concatenate([self.transition_duration, arr[:, 0]])
        self.robot_key_poses_time = np.cumsum(self.transition_duration)

    def _shirt_folds(self):
        """Three-move garment fold on the flat sheet: left panel in, right panel in, in half.

        Lines come from the sheet's ORIGINAL bbox, captured once, so the panels are true
        thirds rather than thirds of whatever the previous fold produced.
        """
        if getattr(self, "_sheet", None) is None:
            lo, hi = self._cloth_bbox()
            self._sheet = (lo.copy(), hi.copy())
            print(f"[twin-stock] flat sheet {hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm", flush=True)
        lo, hi = self._sheet
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        w = hi[0] - lo[0]
        i = getattr(self, "_shirt_step", 0)
        self._shirt_step = i + 1
        if i == 0:                                     # left panel over onto the body
            axis, line = 0, lo[0] + w / 3.0
            grab, drop = (lo[0] + 3.0, cy), (2.0 * line - (lo[0] + 3.0), cy)
        elif i == 1:                                   # right panel over onto the body
            axis, line = 0, hi[0] - w / 3.0
            grab, drop = (hi[0] - 3.0, cy), (2.0 * line - (hi[0] - 3.0), cy)
        else:                                          # hem up to the collar
            axis, line = 1, cy
            grab, drop = (cx, hi[1] - 3.0), (cx, 2.0 * line - (hi[1] - 3.0))
        return self._mirror_fold_poses(axis, float(line), grab, drop)

    def _sleeve_poses(self, name):
        """Pick the actual sleeve tip up and lay it across the torso."""
        ids = self._sleeve_idx()
        if ids is None:
            print("[twin-stock] no sleeve landmarks; skipping", flush=True)
            return []
        q = self.state_0.particle_q.numpy()
        tip = q[ids[0] if name == "sleeve_l_in" else ids[1]]
        lo, hi = self._cloth_bbox()
        cx = 0.5 * (lo[0] + hi[0])
        # Fold at the SHOULDER SEAM, not the garment midline. Mirroring the tip across the
        # midline dragged it 39-64cm - the whole width of the shirt - when the sleeve itself is
        # only ~17cm long, so everything between tip and shoulder stretched instead of folding.
        # Fold about the SHOULDER SEAM. The midline is beyond what a sleeve can physically
        # reach: the left sleeve runs from the seam at x=-21 to its tip at x=-35, so it is 14cm
        # long and even folded flat its tip only reaches x=-7 - asking it to reach the midline
        # at x=0 stretches the fabric 7cm or drags the shoulder with it. FULL and HALF are
        # therefore measured against the SLEEVE'S OWN LENGTH, not the garment's width.
        c0 = getattr(self, "_body_cx", cx)
        hw = getattr(self, "_body_half_w", 0.25 * (hi[0] - lo[0]))
        # RIGHT: fold about the garment MIDLINE and mirror fully - the full sleeve fold.
        # LEFT: fold about its own SHOULDER SEAM and stop there - only this one is halved,
        # because carrying it to the midline is 35cm for a 14cm sleeve (even folded flat its
        # tip only reaches x=-7), which stretches the fabric instead of folding it.
        line = c0 if name == "sleeve_r_in" else c0 - hw
        grab = (float(tip[0]), float(tip[1]))
        # RIGHT sleeve: FULL fold - a complete 180 deg about the seam, so the tip travels twice
        # the sleeve's length and the sleeve lies flat across the body. That is the most a
        # sleeve can fold without stretching. LEFT sleeve: HALF - the tip comes to rest ON the
        # seam, i.e. one sleeve-length of travel, folding it up onto the shoulder.
        if name == "sleeve_r_in":
            drop = (float(2.0 * line - tip[0]), float(tip[1]))
        else:
            drop = (float(line), float(tip[1]))
        print(f"[twin-stock] {name}: tip ({grab[0]:.0f},{grab[1]:.0f}) -> "
              f"({drop[0]:.0f},{drop[1]:.0f})  shoulder x={line:.0f}  "
              f"travel {abs(drop[0]-grab[0]):.0f}cm", flush=True)
        poses = self._mirror_fold_poses(0, float(line), grab, drop)
        self._fold["two_point"] = False                 # a sleeve is small; one pinch is right
        self._fold["pin_cm"] = _FOLD_PIN
        # Remember WHICH sleeve and where its tip started. Undoing a sleeve fold is just
        # "carry that particle back", and we know it by index - far more reliable than hunting
        # for a free edge, which on a sleeve fold found the garment's own far edge (2296
        # top-layer particles) and folded the shirt further instead of opening it.
        sid = ids[0] if name == "sleeve_l_in" else ids[1]
        if self._fold_stack:
            self._fold_stack[-1]["sleeve_particle"] = int(sid)
            self._fold_stack[-1]["sleeve_set"] = self._sleeve_set[name]
            self._fold_stack[-1]["sleeve_home"] = (float(tip[0]), float(tip[1]))
        return poses

    def _sleeve_idx(self):
        """Particle indices of the two sleeve tips, from the garment's own geometry.

        Every fold so far is GEOMETRIC - a fold line at a fraction of the bbox and a grasp at
        whatever fabric sits on that edge. Nothing targets the sleeves, which is why asking for
        a sleeve fold picked up whatever happened to be there. fold_env has tracked these
        landmarks all along (it measures sleeve overhang with them); the twin never used them.
        Indices are fixed, so they still identify the sleeves after any number of folds.
        """
        if getattr(self, "_sleeve_ids", None) is not None:
            return self._sleeve_ids
        q = getattr(self, "_flat_q", None)
        if q is None:
            return None
        y = q[:, 1]
        edges = np.linspace(y.min(), y.max(), 11)
        widths = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (y >= a) & (y < b)
            widths.append(float(np.ptp(q[m, 0])) if m.sum() else 0.0)
        widths = np.asarray(widths)
        k = int(widths.argmax())                       # widest slice = the sleeve band
        band = np.where((y >= edges[k]) & (y < edges[k + 1]))[0]
        self._sleeve_ids = (int(band[q[band, 0].argmin()]), int(band[q[band, 0].argmax()]))
        # Half-width of the TORSO: the body is the narrow part, the sleeves are what sticks out
        # past it. The shoulder seam sits at this distance from the centre, and that - not the
        # garment midline - is where a sleeve folds.
        body_w = float(np.median([w for w in widths if w < 0.75 * widths.max()]))
        self._body_half_w = 0.5 * body_w
        self._body_cx = float(np.median(q[:, 0]))      # centre of the FLAT garment
        # The two sleeves as PARTICLE SETS, taken on the flat garment: everything outside the
        # torso span. Unfolding must lift only the piece that was folded - a pinch at the
        # sleeve tip catches the torso lying underneath as well (measured 360-420 particles
        # against 57-126 for the fold) and drags the whole bundle instead of peeling.
        self._sleeve_set = {
            "sleeve_l_in": np.where(q[:, 0] < self._body_cx - self._body_half_w)[0],
            "sleeve_r_in": np.where(q[:, 0] > self._body_cx + self._body_half_w)[0],
        }
        print(f"[twin-stock] sleeves identified: particles {self._sleeve_ids}, "
              f"torso half-width {self._body_half_w:.0f}cm", flush=True)
        return self._sleeve_ids

    def _plan_poses(self):
        """Execute the next action of TWIN_PLAN with the arm.

        The fold line comes from the CURRENT bbox, exactly as `fold_env.step` computes it, so
        the geometry matches the rollout the search scored. The half on `side` is the one that
        moves; the arm grabs it 3cm inside its outer edge and carries that point to its mirror.
        """
        i = getattr(self, "_plan_step", 0)
        self._plan_step = i + 1
        if getattr(self, "_flat_q", None) is None:
            self._flat_q = self.state_0.particle_q.numpy().copy()   # the garment laid flat
        name = _PLAN[i]
        if name in ("sleeve_l_in", "sleeve_r_in"):
            return self._sleeve_poses(name)
        axis, frac, side, grab_at = _PLAN_ACTIONS[name]
        lo, hi = self._cloth_bbox()
        line = lo[axis] + (hi[axis] - lo[axis]) * frac
        other = 1 - axis
        # The edge AT THE SLICE WE GRAB, not the garment's global bounding box - a shirt
        # narrows toward the collar, so a high grab computed from the global box reaches past
        # the shoulder into empty table (measured: 8 particles held).
        _gy = lo[other] + (hi[other] - lo[other]) * grab_at
        _q = self.state_0.particle_q.numpy()
        _band = np.abs(_q[:, other] - _gy) < 4.0
        if int(_band.sum()) > 20:
            # PERCENTILE, not max/min: a single stray particle flung out by a contact defines
            # the edge otherwise, and the grasp 3cm inside it lands in empty table. That is
            # what dropped the pinch from 74-137 particles to 18-57. Same 3/97 the bbox uses.
            _col = _q[_band, axis]
            edge = float(np.percentile(_col, 97.0 if side > 0 else 3.0))
        else:
            edge = hi[axis] if side > 0 else lo[axis]
        grab = [0.0, 0.0]
        grab[axis] = float(edge - 3.0 * side)          # 3cm in from the edge, as _shirt_folds does
        grab[other] = float(_gy)
        # Snap onto real fabric. The grab point is computed from edge GEOMETRY and the arm then
        # descends there blind - if it lands a centimetre off the cloth the jaws take nothing,
        # which is why the same plan gave 21x27 one run and 25x33 the next (fold 2 held 184
        # particles vs 71). fold_env has always snapped to the nearest particle; the twin did
        # not, so rollout and robot disagreed for a reason that had nothing to do with physics.
        _d2 = (_q[:, 0] - grab[0]) ** 2 + (_q[:, 1] - grab[1]) ** 2
        _near = _q[int(np.argmin(_d2))]
        if float(np.hypot(_near[0] - grab[0], _near[1] - grab[1])) > 0.5:
            print(f"[twin-stock] grab snapped {np.hypot(_near[0]-grab[0], _near[1]-grab[1]):.1f}cm "
                  f"onto fabric", flush=True)
        grab[0], grab[1] = float(_near[0]), float(_near[1])
        drop = list(grab)
        drop[axis] = float(2.0 * line - grab[axis])    # mirror across the fold line
        print(f"[twin-stock] plan {i + 1}/{len(_PLAN)} {name}: axis={axis} line={line:.1f} "
              f"grab=({grab[0]:.0f},{grab[1]:.0f}) drop=({drop[0]:.0f},{drop[1]:.0f})", flush=True)
        self._last_span_cm = float(np.clip(0.35 * (hi[other] - lo[other]), 6.0, 22.0))
        poses = self._mirror_fold_poses(axis, float(line), tuple(grab), tuple(drop))
        # A length fold carries a whole half - hold a wide band of its edge so it comes over
        # flat instead of being dragged from one spot.
        self._fold["two_point"] = (axis == 1)   # sheet folds need more than one hold
        # How wide the fold edge actually is right now, so the holds span it instead of
        # gripping a fixed patch in the middle of a much wider sheet.
        self._fold["span_cm"] = float(np.clip(0.35 * (hi[other] - lo[other]), 6.0, 22.0))
        self._fold["pin_cm"] = _FOLD_PIN   # wide band held 1182 particles through
                                           # every layer; lifting that gathers the
                                           # garment into a column, not a fold
        return poses

    def _fold_poses(self):
        """Halve the bundle across its longer axis."""
        lo, hi = self._cloth_bbox()
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        sx, sy = hi[0] - lo[0], hi[1] - lo[1]
        if sy >= sx:                                   # carry the +y edge over to -y
            grab, drop = (cx, hi[1] - 0.03 * sy), (cx, lo[1] + 0.18 * sy)
            axis, line = 1, cy
        else:
            grab, drop = (hi[0] - 0.03 * sx, cy), (lo[0] + 0.18 * sx, cy)
            axis, line = 0, cx
        return self._mirror_fold_poses(axis, float(line), grab, drop)

    def _carry_slow_for(self, grab, axis=None):
        """How much to stretch the carry, from the layers sitting under the grasp point.

        Same measurement fold_env scores as `grip`: fabric within the pinch radius near table
        height. ~80 particles is a single layer at an edge (drag it at stock speed); 400+ is a
        multi-layer bundle, where outrunning the self-collision solve sends the cloth to NaN.
        """
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            return _CARRY_SLOW
        floor = float(np.percentile(q[:, 2], 2.0))
        # Count EVERY hold. A length fold is carried by three patches spread along the edge,
        # so sampling only the middle one underestimated the load ~3x (measured 190 when the
        # fold actually held 1273) and picked x1.9 instead of x4.3 - and the cloth diverged.
        span = float(getattr(self, '_last_span_cm', _FOLD_SPAN))
        offs = np.linspace(-span, span, 3) if axis == 1 else (0.0,)
        other = 1 - (axis if axis is not None else 0)
        n = 0
        for o in offs:
            gx = grab[0] + (o if other == 0 else 0.0)
            gy = grab[1] + (o if other == 1 else 0.0)
            d2 = (q[:, 0] - gx) ** 2 + (q[:, 1] - gy) ** 2
            n += int(((d2 < _FOLD_PIN ** 2) & (q[:, 2] < floor + _FOLD_PIN)).sum())
        # Length folds carry a SHEET through a multi-layer stack and diverged four times today
        # even at the 4.5 cap, so they get their own, higher ceiling. Width folds move a strip
        # and have never needed it - keeping their cap low is what keeps the fold watchable.
        cap = _CARRY_SLOW * (2.0 if axis == 1 else 1.0)
        c = float(np.clip(1.0 + (n - 80) / 120.0, 1.0, cap))
        print(f"[twin-stock] carry x{c:.1f} ({n} particles under the grasp)", flush=True)
        return c

    def _mirror_fold_poses(self, axis, line, grab, drop):
        """Gripper waypoints for mirroring the piece beyond `line` back across it."""
        top = self.TABLE_TOP
        # the half beyond the fold line is what gets carried over onto the other half
        self._fold = {"axis": axis, "line": float(line), "grab": np.array(grab, float),
                      "drop": np.array(drop, float), "start": None, "idx": None}
        # keep the fold on a stack so it can be undone: mirroring a half across a line is its
        # own inverse, so replaying the same line with grab/drop swapped unfolds it
        self._fold_stack = getattr(self, "_fold_stack", [])
        self._fold_stack.append({"axis": axis, "line": float(line),
                                 "grab": np.array(grab, float), "drop": np.array(drop, float)})
        # Follow the stock fold's motion: pinch at table level, then DRAG low (~11cm up).
        # A high lifting arc just slips the fabric out of the fingers.
        _lerp = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        # The CARRY segments are stretched by _CARRY_SLOW. Dragging the pinned patch into a
        # multi-layer stack faster than VBD can resolve the self-collision it creates is what
        # sends the cloth to NaN - shown on the flat sheet control too, so it is the fold speed
        # and not the garment. fold_env needed 2.0s -> 5.0s; the arm needs the same treatment.
        c = self._carry_slow_for(grab, axis)
        # LIFT the edge over the fold line - do not drag it across the table. The stock path
        # stayed at table+4/+8cm, which for a ~27cm-deep flap slides the grasp sideways while
        # friction and weight pin the rest down. The kinematic grip cannot slip, so instead of
        # the flap coming over, the material between grip and cloth STRETCHES LIKE RUBBER.
        # Arc apex scales with the fold distance: a fold rotates the flap about its crease.
        travel = float(np.hypot(drop[0] - grab[0], drop[1] - grab[1]))
        # Modest: enough to peel the edge off the table, not enough to hoist the
        # bundle. 0.28/0.45 of a 50cm travel lifted to 22cm and wadded the shirt.
        rise = float(np.clip(0.10 * travel, 4.0, 8.0))
        apex = float(np.clip(0.16 * travel, 6.0, 12.0))
        return [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_OPEN),  # press in so the jaws straddle it
            (2.5, grab[0], grab[1], top - 0.3 - _GRASP_DZ, self.GRIP_CLOSE),
            (2.0 * c, *_lerp(0.2), top + rise, self.GRIP_CLOSE),   # peel the edge UP off the table
            (2.5 * c, *_lerp(0.5), top + apex, self.GRIP_CLOSE),   # over the crease, not across it
            (2.5 * c, *_lerp(0.8), top + rise, self.GRIP_CLOSE),
            (2.0 * c, drop[0], drop[1], top + 2.0, self.GRIP_CLOSE),   # lay it down flat
            (1.5 * c, drop[0], drop[1], top + 2.0, self.GRIP_OPEN),
            (1.5, drop[0], drop[1], top + 20.0, self.GRIP_OPEN),
        ]

    def _free_edge_grab(self, axis, line, drop):
        """Find the free edge of the TOP layer, on the side the flap was folded onto.

        Unfolding needs something folding does not: knowing which layer is on top and where
        its loose edge lies. Extent, landmarks and grip count all say nothing about layering,
        so this reads it off the particles directly - upper half of the bundle by height, then
        the extreme along the fold axis on the folded-onto side.
        """
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            return None
        zlo = float(np.percentile(q[:, 2], 5.0))
        zhi = float(np.percentile(q[:, 2], 95.0))
        upper = q[q[:, 2] > (zlo + zhi) / 2.0]         # top layers only
        if len(upper) < 40:
            return None
        side = 1.0 if float(drop[axis]) >= line else -1.0
        sel = upper[(upper[:, axis] - line) * side > 0.0]
        if len(sel) < 20:
            sel = upper
        p = sel[int(np.argmax(sel[:, axis] * side))]   # furthest out = the loose edge
        g = [float(p[0]), float(p[1]), float(p[2])]
        g[axis] -= 3.0 * side                          # 3cm inside the edge, as folds do
        print(f"[twin-stock] free edge at ({g[0]:.0f},{g[1]:.0f}) z={g[2]:.1f} "
              f"({g[2] - self.TABLE_TOP:+.1f} vs table) from {len(sel)} top-layer particles",
              flush=True)
        return g

    def _unfold_poses(self):
        """Undo the most recent fold: carry the folded half back across its own fold line."""
        stack = getattr(self, "_fold_stack", [])
        if not stack:
            print("[twin-stock] nothing left to unfold", flush=True)
            return []
        rec = stack.pop()
        # Where the edge ACTUALLY is, not where the fold was told to put it. Replaying the
        # stored drop point is open-loop: it was exact under the mirror model, but a pinched
        # fold lands short, so the gripper reached ~10cm past the fabric and closed on empty
        # table (measured: 28 and 30 particles, both misses).
        self._fold_only = None
        sid = rec.get("sleeve_particle")
        if sid is not None:
            # A sleeve: grab the tip particle where it now lies and take it home.
            qs = self.state_0.particle_q.numpy()
            found = [float(qs[sid, 0]), float(qs[sid, 1]), float(qs[sid, 2])]
            grab = np.array(found[:2], float)
            drop = np.array(rec["sleeve_home"], float)
            self._fold_only = rec.get("sleeve_set")     # hold sleeve fabric ONLY
            print(f"[twin-stock] unfold sleeve particle {sid}: "
                  f"({grab[0]:.0f},{grab[1]:.0f}) -> ({drop[0]:.0f},{drop[1]:.0f})  "
                  f"restricted to {len(self._fold_only)} sleeve particles", flush=True)
        else:
            found = self._free_edge_grab(rec["axis"], rec["line"], rec["drop"])
            grab = np.array(found[:2], float) if found is not None else np.array(rec["drop"], float)
            drop = rec["grab"]                         # carry it back across its own fold line
        top = self.TABLE_TOP
        # Peeling needs the jaws to close on the TOP layer only. Folding grasps at table level
        # because it is picking up a single lying sheet; a folded bundle has layers stacked
        # under the edge, and reaching table level would pinch all of them and drag the whole
        # bundle instead of opening it. Close just under the edge's own height.
        z_grab = (float(found[2]) - 0.3) if found is not None else (top - 0.3 - _GRASP_DZ)
        z_grab = max(z_grab, top - 0.3 - _GRASP_DZ)
        self._fold = {"only": getattr(self, "_fold_only", None),
                      "axis": rec["axis"], "line": rec["line"],
                      "grab": np.array(grab, float), "drop": np.array(drop, float),
                      "start": None, "idx": rec.get("idx")}
        _lerp = lambda t: (grab[0] + (drop[0] - grab[0]) * t, grab[1] + (drop[1] - grab[1]) * t)
        c = self._carry_slow_for(grab)
        return [
            (2.0, grab[0], grab[1], top + 11.0, self.GRIP_OPEN),
            (2.0, grab[0], grab[1], z_grab, self.GRIP_OPEN),
            (2.5, grab[0], grab[1], z_grab, self.GRIP_CLOSE),
            (2.0 * c, *_lerp(0.15), top + 6.0, self.GRIP_CLOSE),   # lift clear so the layer peels
            (2.5 * c, *_lerp(0.5), top + 10.0, self.GRIP_CLOSE),
            (3.0 * c, drop[0], drop[1], top + 6.0, self.GRIP_CLOSE),
            (1.5 * c, drop[0], drop[1], top + 1.0, self.GRIP_CLOSE),  # lay it down, do not drop
            (1.0, drop[0], drop[1], top + 1.0, self.GRIP_OPEN),
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
        if _PLAN:
            # a sequence fold_search.py validated headless, replayed here by the arm
            stages = [(f"fold {i + 1} {n}", self._plan_poses) for i, n in enumerate(_PLAN)]
            if _os.environ.get("TWIN_UNFOLD", "0") == "1":
                stages += [(f"unfold {i + 1}", self._unfold_poses) for i in range(len(_PLAN))]
        elif _FOLD_STYLE == "shirt":
            # left panel in, right panel in, hem to collar -> a folded rectangle in 3 moves
            stages = [("fold 1 left panel", self._shirt_folds),
                      ("fold 2 right panel", self._shirt_folds),
                      ("fold 3 in half", self._shirt_folds)]
            if _os.environ.get("TWIN_UNFOLD", "1") == "1":
                stages += [("unfold 1", self._unfold_poses), ("unfold 2", self._unfold_poses),
                           ("unfold 3", self._unfold_poses)]
        else:
            stages = [("fold 2", self._fold_poses), ("fold 3", self._fold_poses),
                      ("unfold 1", self._unfold_poses), ("unfold 2", self._unfold_poses)]
        if _os.environ.get("TWIN_PACK", "0") == "1":
            stages.append(("pack into box", self._pack_poses))
        stages = tuple(stages)
        i = getattr(self, "_plan_stage", 0)
        if i >= len(stages):
            if not getattr(self, "_plan_done_logged", False):
                self._plan_done_logged = True   # the last fold is otherwise never measured
                lo, hi = self._cloth_bbox()
                print(f"[twin-stock] plan complete: cloth "
                      f"{hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm", flush=True)
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

    # ---- teleoperation -------------------------------------------------------
    # The stock example's "closed" is activation 0.1 = 0.4cm per finger, i.e. an 8mm gap
    # between the fingertips - measured, with the fingers tracking the command exactly. That
    # is too slack to pinch a single cloth layer, so teleop closes all the way by default.
    # TELEOP_GRIP_CM is the per-finger target when shut (0 = fingertips together).
    TELEOP_GRIP_CM = float(_os.environ.get("TELEOP_GRIP_CM", "0.0"))
    TELEOP_LIN_CM_S = float(_os.environ.get("TELEOP_LIN_CM_S", "12"))
    TELEOP_ANG_DEG_S = float(_os.environ.get("TELEOP_ANG_DEG_S", "45"))

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product of two (x, y, z, w) quaternions."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    # ---- demonstration recording --------------------------------------------
    def _record_reset(self):
        self._rec = {"t": [], "ee_pos": [], "ee_quat": [], "grip": [],
                     "joint_q": [], "cloth_t": [], "cloth_q": [], "n": 0}

    def _record_step(self, target):
        """Append one frame of the demonstration. Called only while recording."""
        r = self._rec
        r["t"].append(float(self.sim_time))
        r["ee_pos"].append(np.asarray(target[0:3], np.float32))
        r["ee_quat"].append(np.asarray(target[3:7], np.float32))
        r["grip"].append(float(target[7]))
        r["joint_q"].append(self.state_0.joint_q.numpy().copy())
        if r["n"] % _CLOTH_EVERY == 0:                 # GPU readback: keep it sparse
            q = self.state_0.particle_q.numpy()[::_CLOTH_STRIDE]
            r["cloth_t"].append(float(self.sim_time))
            r["cloth_q"].append(q.astype(np.float32))
        r["n"] += 1
        if r["n"] % 60 == 0:                           # ~1s heartbeat so the operator sees it live
            p0 = r["ee_pos"][0]
            moved = float(np.abs(np.asarray(r["ee_pos"][-1]) - p0).sum())
            print(f"[record] {r['n']} frames  ee=({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})cm"
                  f"  moved {moved:.0f}cm  grip={'shut' if target[7] <= 0.01 else 'open'}", flush=True)

    def _record_save(self):
        """Write the episode as one npz plus a meta.json describing it."""
        import json as _j

        r = getattr(self, "_rec", None)
        if not r or not r["t"]:
            print("[record] nothing captured", flush=True)
            return None
        ep = _os.path.join(_DATA_DIR, f"ep_{int(self.sim_time * 1000):09d}")
        _os.makedirs(ep, exist_ok=True)
        np.savez_compressed(
            _os.path.join(ep, "episode.npz"),
            t=np.array(r["t"], np.float32),
            ee_pos=np.stack(r["ee_pos"]), ee_quat=np.stack(r["ee_quat"]),
            grip=np.array(r["grip"], np.float32),
            joint_q=np.stack(r["joint_q"]),
            cloth_t=np.array(r["cloth_t"], np.float32),
            cloth_q=np.stack(r["cloth_q"]) if r["cloth_q"] else np.zeros((0, 0, 3), np.float32),
        )
        meta = {
            "units": "cm, radians, seconds; quaternions (x,y,z,w)",
            "fps": self.fps, "frames": len(r["t"]),
            "cloth_every_frames": _CLOTH_EVERY, "cloth_particle_stride": _CLOTH_STRIDE,
            "cloth_particles_recorded": int(r["cloth_q"][0].shape[0]) if r["cloth_q"] else 0,
            "robot": "franka fr3 (sim); real arm is a UR5e - EE poses transfer, joints do not",
            "action_space": "absolute EE target pose + gripper activation (0=shut, 0.8=open)",
            "table_top_z_cm": self.TABLE_TOP,
            "env": {k: _os.environ.get(k) for k in
                    ("TWIN_TS", "TWIN_PLACE", "TWIN_FOLD_STYLE", "TWIN_GRASP_DZ_CM",
                     "TELEOP_GRIP_CM", "TELEOP_LIN_CM_S", "TELEOP_ANG_DEG_S")},
        }
        with open(_os.path.join(ep, "meta.json"), "w") as f:
            _j.dump(meta, f, indent=2)
        print(f"[record] saved {len(r['t'])} frames + {len(r['cloth_t'])} cloth samples -> {ep}",
              flush=True)
        self._record_reset()
        return ep

    def _teleop_target(self):
        """Integrate keyboard input into an EE target the existing IK can follow.

        Motion is a velocity command scaled by the frame time, so it is frame-rate
        independent and the same command shape the real arm takes (an EE twist).
        """
        v = self.viewer
        t = getattr(self, "_tele", None)
        if t is None:
            tip, _ = self._ee_tip()
            t = self._tele = {"p": np.array(tip, float), "yaw": 0.0,
                              "grip": self.GRIP_OPEN, "space": False,
                              "home": np.array(tip, float)}
            print("[teleop] j/l = -x/+x   i/k = -y/+y   u/o = up/down   n/m = yaw\n"
                  "[teleop] g = toggle gripper   z = slow   h = home\n"
                  "[teleop] 1 = start/stop recording a demonstration\n"
                  "[teleop] (WASD/QE/arrows stay with the viewer camera)", flush=True)

        dt = self.frame_dt
        scale = 0.25 if v.is_key_down("z") else 1.0
        lin = self.TELEOP_LIN_CM_S * dt * scale
        ang = np.radians(self.TELEOP_ANG_DEG_S) * dt * scale

        d = np.zeros(3)
        # NB: the viewer already owns WASD/QE/arrows for camera movement, so teleop uses
        # a separate cluster - pressing 'd' would fly the camera, not the arm.
        d[0] += float(v.is_key_down("l")) - float(v.is_key_down("j"))
        d[1] += float(v.is_key_down("k")) - float(v.is_key_down("i"))
        d[2] += float(v.is_key_down("u")) - float(v.is_key_down("o"))
        n = float(np.linalg.norm(d))
        if n > 0.0:
            t["p"] = t["p"] + d / n * lin
        t["yaw"] += (float(v.is_key_down("n")) - float(v.is_key_down("m"))) * ang

        shut = bool(v.is_key_down("g"))
        if shut and not t["space"]:                       # edge-triggered toggle
            grip_shut = self.TELEOP_GRIP_CM / 4.0         # activation -> cm is x4
            closed = t["grip"] <= grip_shut + 1e-3
            t["grip"] = self.GRIP_OPEN if closed else grip_shut
            print(f"[teleop] gripper {'open' if closed else 'closed'} "
                  f"({'—' if closed else f'{2 * self.TELEOP_GRIP_CM:.1f}cm gap'})", flush=True)
        t["space"] = shut

        rec_key = bool(v.is_key_down("1"))
        if rec_key and not t.get("rec_key", False):
            if getattr(self, "_recording", False):
                self._recording = False
                self._record_save()
            else:
                self._record_reset()
                self._recording = True
                print("[record] recording - press 1 again to save", flush=True)
        t["rec_key"] = rec_key

        if v.is_key_down("h"):
            t["p"], t["yaw"] = t["home"].copy(), 0.0

        t["n"] = t.get("n", 0) + 1
        if t["n"] % 120 == 0:                     # ~2s: commanded vs actual finger opening
            fq = self.state_0.joint_q.numpy()[-2:]
            print(f"[teleop] grip cmd {t['grip'] * 4.0:.2f}cm/finger  actual "
                  f"{fq[0]:.2f}/{fq[1]:.2f}cm  gap {fq[0] + fq[1]:.2f}cm", flush=True)

        # keep the tool above the table and inside the arm's comfortable reach
        t["p"][2] = max(float(t["p"][2]), self.TABLE_TOP - _GRASP_DZ)
        t["p"] = np.array(self._clamp_reach(*t["p"]), float)

        half = t["yaw"] / 2.0
        q = self._quat_mul(np.array([0.0, 0.0, np.sin(half), np.cos(half)]),
                           np.array(self.DOWN_QUAT))
        target = np.array([t["p"][0], t["p"][1], t["p"][2], q[0], q[1], q[2], q[3], t["grip"]],
                          dtype=np.float32)
        if getattr(self, "_recording", False):
            self._record_step(target)
        return target

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        if _MODE == "teleop":
            self.target = self._teleop_target()
            self._ik_to_target(state_in)
            return

        if _MODE == "replay":
            # Re-run a recorded demonstration: the actions are absolute EE targets, so
            # replaying them is the same control path teleop drove by hand.
            ep = getattr(self, "_replay", None)
            if ep is None:
                path = _os.environ["TWIN_EPISODE"]
                if _os.path.isdir(path):
                    path = _os.path.join(path, "episode.npz")
                d = np.load(path)
                ep = self._replay = {
                    "t": d["t"],
                    "a": np.concatenate([d["ee_pos"], d["ee_quat"], d["grip"][:, None]], axis=1),
                    "t0": float(self.sim_time),
                }
                print(f"[replay] {len(ep['t'])} frames from {path}", flush=True)
            i = int(np.searchsorted(ep["t"], ep["t"][0] + (self.sim_time - ep["t0"])))
            if i >= len(ep["t"]):
                self.target_joint_qd.zero_()
                return
            self.target = ep["a"][i].astype(np.float32)
            self._ik_to_target(state_in)
            return

        if (_STATES and self.n_worlds > 1
                and not getattr(self, "_states_loaded", False) and self.sim_time >= 2.0):
            # Give each environment the SYMBOLIC START STATE its plan was computed for. Saved
            # states are single-world and in world-0 coordinates, so each is shifted onto its
            # own environment. Without this every world starts flat and the plans are executed
            # against a state they were never planned for.
            self._states_loaded = True
            q_all = self.state_0.particle_q.numpy()
            for w, path in enumerate(_STATES[:self.n_worlds]):
                a_, b_ = self._world_slice(w)
                qs = np.load(path).astype(np.float32)
                if qs.shape[0] != (b_ - a_):
                    print(f"[twin-parallel] w{w} state has {qs.shape[0]} particles, expected "
                          f"{b_ - a_} - ignored", flush=True)
                    continue
                qs = qs.copy()
                qs[:, 0] += self.world_dx[w]
                # fold_env has NO TABLE - its cloth settles on the ground at z~0.5, while the
                # twin's table top is at z=20. Loading a rollout state straight in drops the
                # shirt 20cm through the table and it vanishes. Lift it onto the surface.
                qs[:, 2] += self.TABLE_TOP - float(np.percentile(qs[:, 2], 2.0))
                q_all[a_:b_] = qs
                print(f"[twin-parallel] w{w} start state {path.split('/')[-1]}", flush=True)
            self.state_0.particle_q.assign(q_all)
            self.state_0.particle_qd.zero_()

        if _LOAD_STATE and not getattr(self, "_state_loaded", False) and self.sim_time >= 2.0:
            # Applied after the cloth has settled onto the table, so the loaded configuration
            # starts from rest rather than fighting the drop-in from spawn height.
            self._state_loaded = True
            _q = np.load(_LOAD_STATE).astype(np.float32)
            if _q.shape == self.state_0.particle_q.numpy().shape:
                self.state_0.particle_q.assign(_q)
                self.state_0.particle_qd.zero_()
                lo, hi = self._cloth_bbox()
                print(f"[twin-stock] loaded start state {_LOAD_STATE} "
                      f"({hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm)", flush=True)
            else:
                print(f"[twin-stock] state {_LOAD_STATE} has shape {_q.shape}, expected "
                      f"{self.state_0.particle_q.numpy().shape} - ignored", flush=True)

        if _DUMP_STATE and self.sim_time >= 4.0:
            q = self.state_0.particle_q.numpy()
            np.save(_DUMP_STATE, q.astype(np.float32))
            lo, hi = self._cloth_bbox()
            print(f"[twin-stock] observed state -> {_DUMP_STATE} "
                  f"({hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}cm, {len(q)} particles)", flush=True)
            raise SystemExit(0)

        if self.n_worlds > 1:
            # Hold every arm still until the start states are in. Queuing a fold first computes
            # its grab points against a FLAT shirt, and then a half-folded one arrives
            # underneath - that is how w1 came to catch only 49 particles.
            if _STATES and not getattr(self, "_states_loaded", False):
                self.target_joint_qd.zero_()
                return
            if not hasattr(self, "W"):
                self._init_worlds()
            for w in range(self.n_worlds):
                d = self.W[w]
                while len(d["times"]) == 0 or self.sim_time >= d["times"][-1]:
                    if not self._extend_plan_w(w):
                        break
                if len(d["times"]):
                    k = int(np.searchsorted(d["times"], self.sim_time))
                    k = min(k, len(d["targets"]) - 1)
                    d["target"] = d["targets"][k]
            self._ik_all_worlds(state_in)
            return

        # After the key poses sequence ends, queue the next phase; hold still once done
        if self.sim_time >= self.robot_key_poses_time[-1]:
            if not self._extend_plan():
                self.target_joint_qd.zero_()
                return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]
        self._ik_to_target(state_in)

    def _ik_all_worlds(self, state_in: State):
        """Per-world IK: one Jacobian solve per environment, all applied in one assign.

        compute_ee_delta already indexes by world, so one launch covers every arm. The Jacobian
        does not - its end-effector index is baked into a kernel - so each world needs its own
        kernel and its own column block of the full Jacobian. This is what puts the parallel
        environments to work: before it, N arms were built and only the first one moved.
        """
        # The target is expressed in world 0's frame. Every other environment sits world_dx
        # along x, so handing them the same target asks arms 2 and 3 to reach a point 230 and
        # 460cm away - outside the 85cm envelope - and they collapse trying. Shift the target
        # into each world's own frame.
        deltas = np.empty((self.n_worlds, 6), dtype=np.float32)
        for w in range(self.n_worlds):
            if hasattr(self, "W"):
                # Per-world targets are ALREADY absolute: they come from that world's own cloth
                # bbox, which sits at its world offset. Adding world_dx again sent arms 2 and 3
                # to x=460 and x=920 - far outside the 85cm envelope - and they collapsed.
                tgt = np.array(self.W[w]["target"][:7], dtype=np.float32).copy()
            else:
                tgt = np.array(self.target[:7], dtype=np.float32).copy()
                tgt[0] += self.world_dx[w]
            wp.launch(
                compute_ee_delta,
                dim=1,
                inputs=[
                    state_in.body_q,
                    self.endeffector_offset,
                    self.ee_ids[w],
                    0,                      # ee_ids are already absolute body indices
                    wp.transform(*tgt),
                ],
                outputs=[self.ee_delta],
            )
            deltas[w] = self.ee_delta.numpy()[0]
        q_all = state_in.joint_q.numpy()
        dofs = self.dof_qd_per_world
        full = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for w in range(self.n_worlds):
            self.compute_body_jacobian(
                self.model, state_in.joint_q, state_in.joint_qd,
                include_rotation=True, kernel=self.body_out_kernels[w],
            )
            J_full = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
            J = J_full[:, w * dofs:(w + 1) * dofs]      # this arm's own columns
            d = deltas[w]
            if not (np.isfinite(J).all() and np.isfinite(d).all()):
                continue                                # diverged cloth: leave this arm still
            J_inv = np.linalg.pinv(J)
            Nn = np.eye(J.shape[1], dtype=np.float32) - J_inv @ J
            q = q_all[w * dofs:(w + 1) * dofs]
            q_des = q.copy()
            q_des[1:] = self.initial_pose[w * dofs:(w + 1) * dofs][1:]
            dq = J_inv @ d + Nn @ (q_des - q)
            grip = (self.W[w]["target"][-1] if hasattr(self, "W") else self.target[-1])
            dq[-2] = grip * 4.0 - q[-2]
            dq[-1] = grip * 4.0 - q[-1]
            full[w * dofs:(w + 1) * dofs] = dq
        self.target_joint_qd.assign(full)

    def _ik_to_target(self, state_in: State):
        """Drive the arm toward self.target. Shared by the scripted path and teleop."""
        if self.n_worlds > 1:
            self._ik_all_worlds(state_in)
            return
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
        # A diverged cloth propagates into the arm: the pinned patch drags the gripper target,
        # the Jacobian goes non-finite and pinv raises "SVD did not converge", which used to
        # kill the whole run mid-fold. Hold still and say so instead - the diverged state is
        # then still on screen and in the log to look at.
        if not (np.isfinite(J).all() and np.isfinite(delta_target).all()):
            if not getattr(self, "_ik_nan_warned", False):
                self._ik_nan_warned = True
                print("[twin-stock] IK input non-finite (cloth diverged) - holding still",
                      flush=True)
            self.target_joint_qd.zero_()
            return
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
    FPS_HINT = 60.0        # frames per second the control loop runs at
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
        if self.n_worlds > 1:
            if hasattr(self, "W"):
                for w in range(self.n_worlds):
                    self._update_fold_grasp_w(w)
            return

        """Carry the outer half across the fold line while the gripper is shut.

        Frictional folding plateaus once the pile is a few layers thick, so in 'assist' mode the
        half beyond the fold line is pinned and mirrored across it, tracking how far the gripper
        has travelled from its grasp point to its drop point. The robot still drives the timing.
        """
        f = getattr(self, "_fold", None)
        if f is None or self.target is None or _FOLD_MODE == "physical":
            return
        if _FOLD_MODE == "grasp":
            self._update_fold_grasp(f)
            return
        if _FOLD_MODE != "assist":
            return
        if float(self.target[-1]) > self.GRIP_CLOSE + 1e-3:          # jaws open -> let go
            # `start is None` means the carry has not begun: an unfold arrives with its particle
            # set preset and its first two poses OPEN, so releasing here would wipe that set and
            # send the close back to side-selection (which grabs the whole bundle).
            if f["idx"] is not None and f["start"] is not None:
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
            if self._fold_stack:                      # remember WHICH half this fold moved
                self._fold_stack[-1]["idx"] = idx
            print(f"[twin-stock] fold carries {len(idx)} particles across the line", flush=True)
        elif f["start"] is None:
            # unfold: the set is preset from the fold being reversed. Re-selecting by side is
            # wrong here - after a fold the WHOLE bundle lies on one side of the line, so a
            # side test grabs all 6436 particles and flips the bundle instead of unfolding it.
            f["start"] = q[f["idx"]].copy()
            print(f"[twin-stock] unfold returns {len(f['idx'])} particles", flush=True)

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

    def _update_fold_grasp(self, f):
        """Fold by PINCHING: the fingers hold a few cm of fabric and carry it, nothing else.

        The half beyond the fold line is never prescribed. It follows because it is attached to
        the pinched patch through the cloth itself, so gravity, bending and self-collision all
        act on it during the carry - it hangs, sags, lags and its sleeves flop, and the crease
        forms where the material actually bends rather than along a line we chose.

        The attachment is kinematic (the patch rides the tip) because a parallel jaw on fabric
        slips: TWIN_FOLD=physical loses the cloth and plateaus. Kinematic GRIP, dynamic CLOTH.
        """
        shut = float(self.target[-1]) <= self.GRIP_CLOSE + 1e-3
        idx = f.get("pin_idx")
        if not shut:
            if idx is not None:                       # release limp, never with stored energy
                qd = self.state_0.particle_qd.numpy()
                qd[idx] = 0.0
                self.state_0.particle_qd.assign(qd)
                f["pin_idx"] = None
            return

        tip, R = self._ee_tip()
        q = self.state_0.particle_q.numpy()
        if idx is None:
            r = float(f.get("pin_cm", _FOLD_PIN))
            if f.get("two_point"):
                # three 4cm holds spread along the fold edge, carried together
                other = 1 - int(f["axis"])
                # Spread the holds across the WIDTH OF THE EDGE being folded, not a fixed +-9cm.
                # On a ~40cm collar edge three holds at +-9cm grip only the middle 18cm, so the
                # centre gets carried while the ends stay put and the fabric gathers into a
                # ruffle at the gripper. Five holds over ~70% of the edge carry it as a line.
                span = float(f.get("span_cm", _FOLD_SPAN))
                # THREE holds, kept wide. Five across a 40cm edge drove too much cloth into
                # the stack and the fold diverged; three at the same span still carry the
                # edge as a line instead of gathering it at one point.
                n_holds = 3
                sel = np.zeros(len(q), dtype=bool)
                for o in np.linspace(-span, span, n_holds):
                    c = tip.copy()
                    c[other] += o
                    sel |= np.linalg.norm(q - c, axis=1) < _FOLD_PIN
                idx = np.where(sel)[0]
                if len(idx) > _FOLD_HOLD_MAX:      # keep the closest; never hold a whole half
                    d = np.linalg.norm(q[idx] - tip, axis=1)
                    idx = idx[np.argsort(d)[:_FOLD_HOLD_MAX]]
            elif r > _FOLD_PIN:
                # A wide grip is wide along the EDGE only - still narrow across it and in
                # height. A sphere of the same radius swallowed 3252-4671 particles (up to 73%
                # of the shirt), which stops being a grasp and becomes the kinematic mirror.
                ax = int(f["axis"])
                idx = np.where((np.abs(q[:, 1 - ax] - tip[1 - ax]) < r)
                               & (np.abs(q[:, ax] - tip[ax]) < _FOLD_PIN)
                               & (np.abs(q[:, 2] - tip[2]) < _FOLD_PIN))[0]
            else:
                idx = np.where(np.linalg.norm(q - tip, axis=1) < r)[0]
            only = f.get("only")
            if only is not None and len(only):
                idx = np.intersect1d(idx, only)        # peel one layer, not the stack
            if len(idx) > _FOLD_HOLD_MAX:
                d = np.linalg.norm(q[idx] - tip, axis=1)
                idx = idx[np.argsort(d)[:_FOLD_HOLD_MAX]]
            if len(idx) == 0:
                return                                # missed the cloth; try again next step
            f["pin_idx"] = idx
            f["pin_local"] = (q[idx] - tip) @ R       # offsets in the tip frame
            # A near-miss is silent otherwise: holding a couple of particles still "folds",
            # the extent barely moves, and the plan looks like it executed. Measured: a real
            # pinch takes 70-480 particles depending on how many layers are under the tip.
            # 50, not 20: a repeat of the same plan caught 2 then 21 particles at the same
            # waypoint - both were misses that barely moved the cloth, but 21 slipped past a
            # threshold of 20. A working pinch on this garment takes 70+.
            warn = "  <-- MISSED, too little fabric" if len(idx) < 50 else ""
            print(f"[twin-stock] pinch holds {len(idx)} particles "
                  f"({r:.0f}cm patch){warn}", flush=True)
        # FIRM hold - the grip keeps what it closed on. I tried driving these by VELOCITY
        # instead (TWIN_GRIP_GAIN < 1) to stop the cloth stretching, and it broke the grasp:
        # 0.35 barely folded (width 55->41 where the firm hold reached 21), 0.7 and 0.92 DROPPED
        # the shirt mid-carry. Stretching is handled by TWIN_FOLD_HOLD_MAX and the adaptive
        # carry speed, NOT by weakening the hold. Compliance stays available but off.
        want = tip + f["pin_local"] @ R.T
        if _GRIP_GAIN >= 1.0:
            q[idx] = want
            self.state_0.particle_q.assign(q)
            qd = self.state_0.particle_qd.numpy()
            qd[idx] = 0.0
        else:
            v = (want - q[idx]) * (self.FPS_HINT * _GRIP_GAIN)
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(speed > _GRIP_VMAX, v / np.maximum(speed, 1e-6) * _GRIP_VMAX, v)
            qd = self.state_0.particle_qd.numpy()
            qd[idx] = v
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

        if _MODE == "teleop":
            pass                       # teleop grasps frictionally; no fold/carry aids
        elif getattr(self, "_packing", False):
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
    # 10500 frames = 175s of sim, which silently truncates long plans: the run EXITS CLEANLY
    # mid-fold with no error and no "plan complete", which looked like a crash for hours.
    # Budget per move instead - a pinched fold plus settle runs 25-40s of sim, and unfolds
    # double the move count.
    _moves = max(1, len(_PLAN) * (2 if _os.environ.get("TWIN_UNFOLD", "0") == "1" else 1))
    parser.set_defaults(num_frames=200000 if _MODE == "teleop" else 3000 + _moves * 3600)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
