"""PLAN-DRIVEN multi-object executor: ONE coupled Newton sim (UR5e + Robotiq, both garments as
foldable VBD cloth, both erasers as rigid bodies, the real box) that EXECUTES a cuTAMP plan JSON.

The cuTAMP plan drives everything: the skeleton ORDER (fold pants, fold shirt, pick/place the
erasers, pack the bundles), each garment's initial pose, its target bundle, and every object's
final box placement. The physical fold motion is newton_fold's proven bbox-parameterized fold
primitive (grasp a side, peel, arc over the fold line, drape) — reused per garment, made active-
garment aware. Nothing about the sequence is hardcoded.

Frame: the real ur5e_base_link frame at CM scale. The robot sits at that origin; objects sit at
their real reconstructed positions (matching the RGBD). Table top z=20cm / cloth rest z=21cm match
newton_fold's keyframe z-convention so its fold primitive transfers unchanged.

Increment B (this build): assemble scene + robot, then EXECUTE each Fold action in the plan
skeleton (pants, then shirt) with a real physical fold. Rigid pick-place + bundle pack come next.

Run (newton-spike env, GUI):
  DISPLAY=:1 python newton_plan_exec.py --plan .../santa_20260731_plan.json \
     --scene_dir .../outputs/robot_20260731_143933 --capture_dir .../captures/... \
     --screenshot /tmp/plan_exec.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.viewer import ViewerGL, ViewerNull
from newton._src.utils.texture import load_texture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_fold import (masked_grid, pin_to_ee, compute_ee_delta, _dls_pinv,  # noqa: E402
                         DOWN_Q, OPEN, CLOSE)
from cloth_mesh import cloth_from_mask  # noqa: E402  realistic garment (real silhouette+depth+photo)
from ur5e_gripper import add_ur5e_gripper  # noqa: E402
from newton_twin_hydro import load_scene_objects, load_obj_mesh  # noqa: E402
import sam3d_scene  # noqa: E402

GARMENT_KW = ("shirt", "cloth", "pants", "trouser", "legging", "garment", "towel", "fabric", "tee")
M2CM = 100.0
GRAVITY = -981.0
TABLE_TOP = 20.0        # table top plane z (cm) — matches newton_fold's keyframe z-convention
CLOTH_TOP = TABLE_TOP + 1.0


@wp.kernel
def _stabilize_particle_vel(qd: wp.array(dtype=wp.vec3), vmax: float):
    """Anti-NaN guard: scrub any non-finite particle velocity to zero and cap the magnitude at vmax.
    A pin RELEASE (or a self-contact blip on stacked fold layers) can inject a huge velocity spike that
    grows to NaN over a few substeps; capping it each substep keeps the fold from blowing up while
    leaving normal-speed cloth motion untouched (vmax is far above any real fold speed)."""
    i = wp.tid()
    v = qd[i]
    finite = (v[0] == v[0]) and (v[1] == v[1]) and (v[2] == v[2])
    if not finite:
        qd[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    s = wp.length(v)
    if s > vmax:
        qd[i] = v * (vmax / s)
CELL = 2.0              # cloth grid cell (cm)
ROBOT_Z = 20.0                    # UR5e base at TABLE LEVEL (per user) — mobile-base top flush w/ table
ROBOT_BASE_XY = (-18.0, 8.0)      # OFF the table, on its mobile base beside it (mobile manipulator)


def is_garment(label):
    return any(k in label.lower() for k in GARMENT_KW)


def norm_label(s):
    return str(s).lower().replace(" ", "_")


def _qrot(qq, v):
    """Rotate vector v by quaternion qq=(x,y,z,w)."""
    x, y, z, w = qq
    t = 2.0 * np.cross([x, y, z], v)
    return np.asarray(v) + w * t + np.cross([x, y, z], t)


def load_box_mesh_cm(scene_dir, capture_dir):
    """Real reconstructed box mesh in cm, base sitting on the table top (z=TABLE_TOP)."""
    sc = sam3d_scene.load_scene(scene_dir, capture_dir)
    box = next(o for o in sc["objects"] if "box" in o["label"].lower())
    v = box["verts"].astype(np.float64)
    c = v.mean(0)
    fp = (v[:, :2].max(0) - v[:, :2].min(0)).max()
    meas = max([m for m in box["measured_m"] if m > 0] or [fp])
    v = (c + (meas / fp) * (v - c)) * M2CM
    v[:, 2] += TABLE_TOP - v[:, 2].min()
    return v.astype(np.float32), box["faces"].reshape(-1, 3).astype(np.int32), box["color"]


def garment_specs(scene_dir):
    """Per-garment footprint + mask + pixel box, keyed by normalized label (for masked_grid)."""
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    img = layout.get("image_size_px", [640, 480])
    mask_dir = scene_dir.rstrip("/") + "_sam3d_raw/masks"
    specs = {}
    for i, o in enumerate(layout.get("objects", [])):
        if not is_garment(str(o.get("label", ""))):
            continue
        di = o.get("depth_info") or {}
        w = float(di.get("physical_width_m") or o.get("physical_size_m") or 0.35)
        h = float(di.get("physical_height_m") or w)
        box_px = o.get("box_px") or [0, 0, img[0], img[1]]
        mpath = os.path.join(mask_dir, f"{o.get('id', i)}.png")
        mask = None
        if os.path.exists(mpath):
            from PIL import Image
            mask = np.array(Image.open(mpath).convert("L"))
        specs[norm_label(o.get("label", ""))] = dict(
            label=o.get("label", "cloth"), id=o.get("id", i), w=w, h=h, box_px=box_px, img=img,
            mask=mask, mask_path=mpath, color=o.get("display_color", [0.6, 0.6, 0.65]))
    return specs


def parse_skeleton(skeleton):
    """Extract the ordered high-level actions from the plan skeleton strings.
    Returns a list of (op, target) e.g. ('Fold','santa_pants'), ('Pick','rigid_1'), ('Place','rigid_1')."""
    acts = []
    for s in skeleton:
        m = re.match(r"(\w+)\(([^,\)]+)", s)
        if not m:
            continue
        op, tgt = m.group(1), m.group(2).strip()
        if op in ("Fold", "Pick", "Place"):
            acts.append((op, tgt))
    return acts


class PlanExec:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = args.substeps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.max_reach = args.reach
        self.cell = args.cell
        self.stretch_ke = args.stretch_ke
        self.bend_ke = args.bend_ke
        self.garment_max_cm = float(getattr(args, "garment_max_cm", 0.0))
        self.tri_kd = float(getattr(args, "tri_kd", 1.5e-6))
        self.edge_kd = float(getattr(args, "edge_kd", 1.0e-2))
        self.speed = max(float(getattr(args, "speed", 1.0)), 0.1)  # >1 = brisker arm motion (compress schedule)
        self.profile = bool(getattr(args, "profile", False))
        self.prof = {}
        self.prof_n = 0
        self.mask_step = getattr(args, "mask_step", 9)
        self.only_fold = getattr(args, "only_fold", None)

        plan = json.load(open(args.plan))
        self.plan = plan
        self.initial = plan["initial"]
        self.actions = parse_skeleton(plan["skeleton"])
        self.placements = {norm_label(p["name"]): p for p in plan.get("placements", [])}
        self.plan_statics = plan.get("statics", {})
        self.capture_dir = args.capture_dir

        gspecs = garment_specs(args.scene_dir)
        real = load_scene_objects(args.scene_dir, args.capture_dir, args.base_frame)
        real_by_label = {norm_label(r["label"]): r for r in real}

        wp.init()
        scene = ModelBuilder(gravity=GRAVITY)
        cfg = ModelBuilder.ShapeConfig(mu=0.6)

        # --- robot at the base-frame origin, facing +x toward the objects, on a pedestal ---
        self.robot_base = (ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], ROBOT_Z)
        self.arm = getattr(args, "arm", "ur5e")
        _phys = bool(getattr(args, "physical_grasp", False))
        # PHYSICAL clamp works for the Franka fingers OR the UR5e + Hand-E-style 2-jaw (both are
        # simple parallel grippers Newton can actually drive). The old UR5e used the 2F-85, a closed
        # 4-bar linkage Featherstone can't close -> that path falls back to the software pin.
        self.physical_grip = _phys and self.arm in ("franka", "ur5e")
        self.base_yaw_deg = float(getattr(args, "base_yaw", 0.0))
        if _phys and self.arm == "ur5e":
            # mobile base: bring the UR5e (reach ~0.85m) close enough to reach every grasp, same as
            # the franka (objects span ~0.34-0.73m in +x from the faithful origin).
            self.robot_base = (22.0, 12.0, ROBOT_Z)
        if self.arm == "franka":
            # Franka in the FAITHFUL scene (objects stay at their cuTAMP-plan REAL positions -- the
            # scene-build below is untouched). Robot swapped for the validated PHYSICAL grasp path.
            # The Franka reach is ~0.85m; the UR5e's off-table base put the objects (and the fold's far
            # edge) at ~0.98m. It's a MOBILE base, so move it closer to the object cluster (objects
            # unchanged) to bring everything comfortably in reach.
            self.robot_base = (22.0, 12.0, ROBOT_Z)
            # Faithful base is at ROBOT_BASE_XY with the objects at +x -> the franka must face +x
            # (its canonical forward), so NO 180deg flip (that's newton_fold's geometry, not ours).
            _ang = np.radians(self.base_yaw_deg)
            rq = wp.quat(0.0, 0.0, float(np.sin(_ang / 2)), float(np.cos(_ang / 2)))
            self.robot_rq = (0.0, 0.0, float(np.sin(_ang / 2)), float(np.cos(_ang / 2)))
            dof0 = scene.joint_dof_count
            asset_path = newton.utils.download_asset("franka_emika_panda")
            scene.add_urdf(str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
                           xform=wp.transform(wp.vec3(*self.robot_base), rq),
                           floating=False, scale=100, enable_self_collisions=False,
                           collapse_fixed_joints=True, force_show_colliders=False)
            scene.joint_q[dof0:dof0 + 6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
            self.n_arm = 7
            self.endeffector_id = scene.body_count - 3                  # franka hand link (before 2 fingers)
            self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())
            self.ee_off_vec = wp.vec3(0.0, 0.0, 22.0)
            self.grip_dofs = list(range(scene.joint_dof_count - 2, scene.joint_dof_count))
            self.free_loop_dofs = []
            self.grip_closed = None; self.grip_open = None
            # generalized clamp targets (fingers: ~4cm open, 0 closed; overshoot for a tight hold)
            self.g_open, self.g_close = 4.2, -1.0
            self.g_open_ref, self.g_close_ref = 4.0, 0.0
            self.bodies_per_world = scene.body_count
            print(f"[plan] arm = Franka (PHYSICAL grasp={self.physical_grip}): {self.n_arm} arm dof, "
                  f"grip dofs {self.grip_dofs}, ee {self.endeffector_id}", flush=True)
        elif _phys and self.arm == "ur5e":
            # UR5e + 2-JAW gripper == the REAL robot (UR5e + Robotiq Hand-E, a simple parallel jaw).
            # Newton drives the jaws shut ON the fabric (physical clamp, no pin) -> reuses the exact
            # on-contact grasp fix. Tool axis = wrist_3 +y; pinch point ~9cm out along the jaws.
            from ur5e_jaw_gripper import add_ur5e_jaw_gripper
            w3, dof0, n_arm, pinch_local, gdofs, op, cp = add_ur5e_jaw_gripper(
                scene, wp.transform(wp.vec3(*self.robot_base), wp.quat_identity()), scale=100.0)
            self.n_arm = n_arm                                    # 6 for UR5e
            self.endeffector_id = w3
            self.endeffector_offset = wp.transform(list(pinch_local), wp.quat_identity())
            self.ee_off_vec = wp.vec3(*pinch_local)
            self.grip_dofs = list(gdofs)
            self.free_loop_dofs = []
            self.grip_closed = None; self.grip_open = None
            # jaws: op=open slide, cp=closed slide (a few mm apart). Overshoot closed to hold.
            self.g_open_ref, self.g_close_ref = float(op), float(cp)
            self.g_open = float(op) - 0.5
            self.g_close = float(cp) + 0.8
            self.bodies_per_world = scene.body_count
            print(f"[plan] arm = UR5e + Hand-E JAW (PHYSICAL clamp): {n_arm} arm dof, grip dofs {gdofs}, "
                  f"open {op:.1f} close {cp:.1f}cm, ee {w3}", flush=True)
        else:
            ur_w3, dof0, n_arm, pinch_local = add_ur5e_gripper(
                scene, wp.transform(wp.vec3(*self.robot_base), wp.quat_identity()), scale=100.0)
            self.n_arm = n_arm
            self.endeffector_id = ur_w3
            self.endeffector_offset = wp.transform(list(pinch_local), wp.quat_identity())
            self.ee_off_vec = wp.vec3(*pinch_local)
            self.grip_dofs = list(range(dof0 + n_arm, dof0 + n_arm + 8))
            self.free_loop_dofs = list(range(dof0 + n_arm + 8, scene.joint_dof_count))
            self.grip_closed = np.array([0.781, 0.001, 0.78, -0.756, 0.781, 0.001, 0.78, -0.756])
            self.grip_open = np.zeros(8)
            self.bodies_per_world = scene.body_count
            print(f"[plan] UR5e + 2F-85: wrist3 {ur_w3}, {n_arm} arm dof, pinch {tuple(pinch_local)}", flush=True)

        # PLAN-DRIVEN positions: place every object EXACTLY where the cuTAMP plan JSON says. The plan
        # is now generated on the REAL layout (translated only for reach — see run_scene_tamp), so the
        # JSON positions already match the RGBD AND are reachable, and what executes == the JSON.
        # Real shapes (cloth_from_mask garments, real eraser meshes, real box mesh) sit at those poses.
        bf_pose = self.plan_statics.get("box_floor", {}).get("pose", [0.55, 0.30, 0.0])
        bfx, bfy = float(bf_pose[0]) * M2CM, float(bf_pose[1]) * M2CM
        self.plan_box_center = np.array(bf_pose[:2])
        plan_xy = {norm_label(k): (float(v[0]) * M2CM, float(v[1]) * M2CM) for k, v in self.initial.items()}
        allxy = np.array(list(plan_xy.values()) + [[bfx, bfy]])
        self.tcx, self.tcy = float(allxy[:, 0].mean()), float(allxy[:, 1].mean())
        # table sized to sit UNDER the objects (the robot is off it, on its mobile base). The margin
        # must cover the GARMENT FOOTPRINT, not just object centres: a garment centred at the cluster
        # edge extends ~half its width past its centre (a ~55cm shirt -> ~28cm), so a small margin lets
        # it drape off the edge (particles fall to the floor and the fold grasps air). Use a generous
        # 32cm margin so the largest garment stays fully supported on the table.
        thx = float(np.ptp(allxy[:, 0])) / 2 + 30.0
        thy = float(np.ptp(allxy[:, 1])) / 2 + 30.0
        self.tcx = 0.5 * float(allxy[:, 0].min() + allxy[:, 0].max())     # centre on the object cluster
        self.tcy = 0.5 * float(allxy[:, 1].min() + allxy[:, 1].max())
        self.table_hz = 10.0
        self.table_top = TABLE_TOP
        tbl_idx = scene.shape_count
        # VISIBLE light table (like the RGBD) so the dark erasers read against it
        scene.add_shape_box(-1, wp.transform(wp.vec3(self.tcx, self.tcy, TABLE_TOP - 10.0), wp.quat_identity()),
                            hx=thx, hy=thy, hz=10.0, cfg=cfg, color=wp.vec3(0.80, 0.80, 0.82))
        # VISIBLE mobile-manipulator base: a floor-standing platform (off the table) the UR5e mounts on
        ped_idx = scene.shape_count
        scene.add_shape_box(-1, wp.transform(wp.vec3(self.robot_base[0], self.robot_base[1], ROBOT_Z / 2),
                                             wp.quat_identity()), hx=17.0, hy=17.0, hz=ROBOT_Z / 2, cfg=cfg,
                            color=wp.vec3(0.28, 0.30, 0.34))

        # real box mesh placed so its opening centres on the plan box_floor
        bv, bf, bcol = load_box_mesh_cm(args.scene_dir, args.capture_dir)
        zt, zb = bv[:, 2].max(), bv[:, 2].min()
        rim0 = bv[bv[:, 2] > zt - 0.30 * (zt - zb)]
        bv[:, 0] += bfx - float(rim0[:, 0].mean())
        bv[:, 1] += bfy - float(rim0[:, 1].mean())
        keep = bv[bf].mean(1)[:, 2] < zb + 0.82 * (zt - zb)
        liner = newton.Mesh(bv, bf[keep].reshape(-1))
        box_liner_idx = scene.shape_count
        scene.add_shape_mesh(-1, wp.transform(wp.vec3(0.0), wp.quat_identity()), mesh=liner, cfg=cfg)
        box_viz = newton.Mesh(bv, bf.reshape(-1))
        scene.add_shape_mesh(-1, wp.transform(wp.vec3(0.0), wp.quat_identity()), mesh=box_viz,
                             cfg=ModelBuilder.ShapeConfig(has_shape_collision=False), color=wp.vec3(*bcol))
        self.box_open = (bfx, bfy, float(zt))
        self.box_hw = (float(np.ptp(bv[:, 0])) / 2.0, float(np.ptp(bv[:, 1])) / 2.0)
        self.box_floor_z = float(zb)

        # --- garments: real silhouette + depth + photo texture, at their REAL positions (shifted) ---
        self.garments = {}
        self.garment_order = []
        part0 = 0
        for nm in list(self.initial):
            if not is_garment(nm):
                continue
            key = next((k for k in gspecs if k == norm_label(nm) or k in norm_label(nm)
                        or norm_label(nm) in k), None)
            sp = gspecs.get(key) if key else None
            if sp is None:
                print(f"[plan] WARN no garment spec for '{nm}', skip", flush=True)
                continue
            v, faces, uvs, _tex = cloth_from_mask(sp["mask_path"], self.capture_dir,
                                                  step=self.mask_step, base_frame=args.base_frame)
            v = v.astype(np.float64)
            # The garment's absolute SCALE comes from back-projecting the mask with depth, but soft
            # low-texture fabric gives sparse/noisy depth -> the size can come out too large (e.g. a
            # 43x56cm reconstruction of a ~35cm child's shirt), which makes it overhang the table and
            # look oversized. If --garment_max_cm is set, uniformly rescale the garment (about its
            # centroid, preserving the real SHAPE from the mask) so its largest span == that size.
            if self.garment_max_cm > 0:
                span = float(max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2])))
                if span > 1e-3 and span > self.garment_max_cm:
                    c = v.mean(axis=0)
                    v = c + (v - c) * (self.garment_max_cm / span)
                    print(f"[plan] garment '{nm}': rescaled {span:.0f}cm -> {self.garment_max_cm:.0f}cm max span", flush=True)
            gx, gy = plan_xy[norm_label(nm)]             # place the real garment shape AT its plan pose
            v[:, 0] += gx - v[:, 0].mean()
            v[:, 1] += gy - v[:, 1].mean()
            v[:, 2] += CLOTH_TOP - v[:, 2].min()         # lie flat on the table
            faces = faces.reshape(-1, 3).astype(np.int32)
            n = len(v)
            scene.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                                 vertices=[wp.vec3(*x) for x in v], indices=faces.reshape(-1).tolist(),
                                 density=0.02, tri_ke=self.stretch_ke, tri_ka=self.stretch_ke, tri_kd=self.tri_kd,
                                 edge_ke=self.bend_ke, edge_kd=self.edge_kd, particle_radius=0.8)
            bbox = [float(v[:, 0].min()), float(v[:, 0].max()),
                    float(v[:, 1].min()), float(v[:, 1].max())]
            self.garments[norm_label(nm)] = dict(label=nm, p0=part0, n=n, uvs=uvs,
                                                 faces=faces.reshape(-1), bbox=bbox,
                                                 real_x=float(v[:, 0].mean()), real_y=float(v[:, 1].mean()))
            self.garment_order.append(norm_label(nm))
            part0 += n
            print(f"[plan] garment '{nm}': realistic mask cloth -> {n} particles "
                  f"@({v[:,0].mean():.0f},{v[:,1].mean():.0f})cm", flush=True)

        # --- erasers as rigid free bodies with their REAL reconstructed MESHES (not primitive boxes) ---
        # the plan names them rigid_1/rigid_2 but both are "black eraser"; pair plan rigid -> real
        # eraser by order and key by the PLAN name so the pack actions resolve.
        real_rigids = [r for r in real if not is_garment(r["label"]) and "box" not in r["label"].lower()]
        plan_rigids = [nm for nm in self.initial if not is_garment(nm)]
        self.rigids = {}
        self.rigid_xy = {}
        self.rigid_hxy = {}                              # each eraser's XY half-extent (cm) for box containment
        self.rigid_hz = 1.5
        for i, plan_nm in enumerate(plan_rigids):
            r = real_rigids[i] if i < len(real_rigids) else None
            if r is None:
                continue
            nm = norm_label(plan_nm)
            rx, ry = plan_xy[nm]                          # real eraser MESH placed at its plan pose
            hz = max(1.0, float(r.get("height", 0.03)) * M2CM * 0.5)
            self.rigid_hz = hz
            b = scene.add_body(xform=wp.transform(wp.vec3(rx, ry, TABLE_TOP + hz), wp.quat_identity()), label=nm)
            mp = r.get("mesh_path")
            placed = False
            if mp and os.path.exists(mp):
                try:
                    mesh, _ = load_obj_mesh(mp)             # recentered newton mesh (m)
                    mv = np.asarray(mesh.vertices)
                    ext = float(max(np.ptp(mv[:, 0]), np.ptp(mv[:, 1]), np.ptp(mv[:, 2])))
                    meas = float(max(r.get("radius", 0.05) * 2.0, r.get("height", 0.03)))
                    scale = (meas / ext) * M2CM if ext > 1e-6 else M2CM
                    scene.add_shape_mesh(b, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), mesh=mesh,
                                         cfg=cfg, color=wp.vec3(*r["color"]),
                                         scale=wp.vec3(scale, scale, scale))
                    self.rigid_hxy[nm] = (float(np.ptp(mv[:, 0])) / 2.0 * scale,
                                          float(np.ptp(mv[:, 1])) / 2.0 * scale)
                    placed = True
                except Exception as e:
                    print(f"[plan] eraser mesh '{nm}' failed ({e}); box fallback", flush=True)
            if not placed:
                scene.add_shape_box(b, hx=4.5, hy=2.5, hz=hz, cfg=cfg, color=wp.vec3(*r["color"]))
                self.rigid_hxy[nm] = (4.5, 2.5)
            self.rigids[nm] = b
            self.rigid_xy[nm] = (rx, ry)
            print(f"[plan] rigid '{nm}' (real mesh={'yes' if placed else 'no'}) @({rx:.0f},{ry:.0f})cm", flush=True)
        self.rigid_grasp = None

        scene.color(include_bending=True)
        scene.add_ground_plane(cfg=cfg)
        self.model = scene.finalize()

        flags = self.model.shape_flags.numpy()
        for si in (box_liner_idx,):                  # keep the table + mobile base VISIBLE
            flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # HIGH cloth-table friction so the un-grabbed fabric STAYS on the table while the grabbed
        # flap folds over (the crease forms and the cloth deforms — instead of the whole sheet
        # sliding rigidly). Firm normal contact so it rests flat.
        if self.physical_grip:
            # PHYSICAL grasp: finger-cloth friction 0.6 (the value that HELD the pinch through the
            # lift+fold without exploding -- 0.25 slipped, 1.6 ejected the fabric). Example pinch stiffness.
            # EXACT reference (example_cloth_franka) contact material: soft_contact_mu 0.25, robot shape mu 1.5.
            self.model.soft_contact_ke = 1e4; self.model.soft_contact_kd = 1e-2; self.model.soft_contact_mu = 0.25
            self.model.soft_contact_margin = 0.8
            mm = self.model.shape_material_mu.numpy(); mm[...] = 1.5
            self.model.shape_material_mu = wp.array(mm, dtype=self.model.shape_material_mu.dtype, device=self.model.device)
            _scm = 0.8
        else:
            self.model.soft_contact_ke = 2.0e4; self.model.soft_contact_kd = 8.0e-2
            self.model.soft_contact_mu = 1.2; self.model.soft_contact_margin = 1.0
            _scm = 1.2

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=_scm)
        self.contacts = self.collision_pipeline.contacts()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, GRAVITY), dtype=wp.vec3)

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.substeps)
        # Self-contact (keeping folded layers from interpenetrating) is the dominant per-frame cost
        # (~59% in profiling). Detecting it less often (interval>1) and with smaller buffers cuts that
        # cost a lot, freeing budget for MESH RESOLUTION + ITERATIONS that actually make the cloth FOLD
        # (crease + drape) instead of sliding as a stiff plate. Both are exposed as CLI knobs.
        _sc = int(getattr(args, "self_contact_interval", 1))
        _scbuf = int(getattr(args, "self_contact_buf", 64))
        self.cloth_solver = SolverVBD(
            self.model, iterations=args.vbd_iters, integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5, particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=_scbuf, particle_edge_contact_buffer_size=_scbuf,
            particle_conservative_bound_relaxation=0.5, particle_collision_detection_interval=_sc)
        if self.model.edge_rest_angle is not None:
            self.model.edge_rest_angle.zero_()

        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        # the cloth solver zeroes free-body (eraser) poses in its output state, so track each
        # eraser's intended world pose and re-stamp it every render frame (else they vanish).
        _bq0 = self.state_0.body_q.numpy()
        self.rigid_pose = {bid: _bq0[bid].copy() for bid in self.rigids.values()}
        self.control = self.model.control()
        self.target_joint_qd = wp.zeros_like(self.state_0.joint_qd)

        # grasp / pin state
        self.orig_inv_mass = self.model.particle_inv_mass.numpy().copy()
        self.grasp_active = False
        self.prev_grip = OPEN
        self.pin_idx = None; self.pin_off = None

        self._set_up_control()
        self._build_keyframes()
        self._setup_render()

    # ----------------------------- keyframe schedule (plan-driven) -----------------------------
    def _apply_speed(self):
        """Compress every keyframe duration by self.speed so the arm moves at a brisk, natural pace
        (the raw schedule is deliberately gentle for cloth stability, which reads as slow-motion).
        A per-keyframe floor keeps each move spread over enough frames to stay smoothly interpolated
        and gives the fabric a beat to settle after a fold/release."""
        if abs(self.speed - 1.0) < 1e-6:
            return
        d = self.key_poses[:, 0] / self.speed
        self.key_poses[:, 0] = np.maximum(d, 0.30)

    def _fold_block(self, gkey):
        """REAL garment fold — fold in quarters, the way laundry is folded. Each fold grabs a WHOLE
        edge (a coherent flap, not a corner), peels it just off the table, and HINGES it over the
        crease onto the other half in a low arc, then lays it flat. The fabric stays flat and keeps
        its shape — no corner-drag bunching. Fold 1: left half over right (hinge = vertical centre
        line). Fold 2: bottom half over top (hinge = horizontal centre line)."""
        g = self.garments[gkey]
        x0, x1, y0, y1 = g["bbox"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        q = DOWN_Q
        ZHI = 40.0
        p0, n = g["p0"], g["n"]
        qx = (cx + x1) / 2.0                          # quarter line of the final +x,+y quadrant
        folds = [
            # fold the -x half over the +x half: grasp the x=x0 edge, hinge over cx, lay at x1
            {"axis": "x", "edge": x0, "to": x1, "hinge": cx, "along": (y0, y1), "mid": cy},
            # fold the -y half over the +y half (of the now half-width garment): grasp y=y0 edge
            {"axis": "y", "edge": y0, "to": y1, "hinge": cy, "along": (cx, x1), "mid": (cx + x1) / 2.0},
            # THIRD fold (final): fold the quarter in half along x again. Stop at 3 folds — a 4th
            # over-works the garment into a shapeless wad (keeps its shape better at 3).
            {"axis": "x", "edge": cx, "to": x1, "hinge": qx, "along": (cy, y1), "mid": (cy + y1) / 2.0},
        ]
        poses, specs, adaptive = [], {}, set()
        for i, f in enumerate(folds):
            base = len(poses)
            z_edge = TABLE_TOP + 0.6 + i * 1.6       # grasp hugging the fabric edge (not above it)
            dist = abs(f["to"] - f["edge"])
            z_ap = TABLE_TOP + min(0.42 * dist, 14.0)  # hinge the flap over the crease, staying LOW
            z_land = TABLE_TOP + 1.2 + i * 1.6
            if f["axis"] == "x":
                g_xy, h_xy, t_xy = (f["edge"], f["mid"]), (f["hinge"], f["mid"]), (f["to"], f["mid"])
            else:
                g_xy, h_xy, t_xy = (f["mid"], f["edge"]), (f["mid"], f["hinge"]), (f["mid"], f["to"])
            q1 = ((f["hinge"] + f["to"]) / 2.0)          # quarter point past the crease (extra arc waypoint)
            m_xy = (q1, f["mid"]) if f["axis"] == "x" else (f["mid"], q1)
            poses += [
                [1.8, g_xy[0], g_xy[1], ZHI, *q, OPEN],         # approach above the edge
                [1.6, g_xy[0], g_xy[1], z_edge, *q, OPEN],      # descend low onto the whole edge
                [1.2, g_xy[0], g_xy[1], z_edge, *q, CLOSE],     # grasp the ENTIRE edge (band)
                [1.6, g_xy[0], g_xy[1], z_edge + 3.0, *q, CLOSE],  # peel just off the table
                [3.2, h_xy[0], h_xy[1], z_ap, *q, CLOSE],       # slow hinge UP over the crease (low arc)
                [3.0, m_xy[0], m_xy[1], z_edge + 3.0, *q, CLOSE],  # slow hinge DOWN toward the far half
                [2.2, t_xy[0], t_xy[1], z_land, *q, CLOSE],     # lay it flat on the other half
                [1.2, t_xy[0], t_xy[1], z_land, *q, OPEN],      # release
                [1.2, t_xy[0], t_xy[1], z_land, *q, OPEN],      # dwell -> the flap settles flat
                [1.5, t_xy[0], t_xy[1], ZHI, *q, OPEN],         # retreat up
            ]
            specs[base + 2] = {"kind": "edge", "axis": f["axis"], "edge": f["edge"],
                               "along": f["along"], "band": 3.5, "range": (p0, n)}
            # mark the descend + grasp keyframes ADAPTIVE so the finger PADS are driven onto the ACTUAL
            # fabric top (dig-in ctop-4), not a fixed z where only the tips touch and the pads clamp air.
            adaptive.add(base + 1)
            adaptive.add(base + 2)
        return poses, specs, adaptive

    def _in_box(self, x, y, half=(0.0, 0.0), wall=2.5, clearance=1.5):
        """Clamp an XY so the object's FULL footprint (its half-extent `half`) stays inside the box
        INTERIOR. box_hw is the OUTER half-width, so subtract the wall thickness AND the object's own
        half-extent (+ clearance): a centre that's inside but an EDGE that crosses the wall is exactly
        why erasers poked THROUGH the wall."""
        cx, cy = self.box_open[0], self.box_open[1]
        hx = max(0.5, self.box_hw[0] - wall - half[0] - clearance)
        hy = max(0.5, self.box_hw[1] - wall - half[1] - clearance)
        return cx + float(np.clip(x - cx, -hx, hx)), cy + float(np.clip(y - cy, -hy, hy))

    def _place_xy(self, key):
        """Map the cuTAMP in-box arrangement (offset from the plan box centre) onto the REAL box
        opening, clamped so the object's whole footprint stays inside. The cuTAMP offsets push objects
        out to the walls, so DAMPEN them (0.55x) toward centre -> the arrangement is preserved but
        nothing perches on / pokes through a wall."""
        half = self.rigid_hxy.get(key, (0.0, 0.0))
        p = self.placements.get(key)
        if p is not None:
            dxp = (float(p["pos"][0]) - float(self.plan_box_center[0])) * M2CM * 0.55
            dyp = (float(p["pos"][1]) - float(self.plan_box_center[1])) * M2CM * 0.55
            return self._in_box(self.box_open[0] + dxp, self.box_open[1] + dyp, half=half)
        return self._in_box(self.box_open[0], self.box_open[1], half=half)

    def _rigid_block(self, nm):
        """Pick a rigid (eraser) from its real spot and Place it at the cuTAMP placement in the box."""
        bid = self.rigids[nm]
        rx, ry = self.rigid_xy[nm]
        px, py = self._place_xy(nm)
        q = DOWN_Q
        ZHI = 42.0
        zgrab = TABLE_TOP + self.rigid_hz + 0.2            # pads AROUND the eraser body (not above it)
        zdrop = TABLE_TOP + self.rigid_hz + 1.5
        poses = [
            [1.5, rx, ry, ZHI, *q, OPEN],
            [1.2, rx, ry, zgrab, *q, OPEN],
            [0.8, rx, ry, zgrab, *q, CLOSE],               # attach
            [1.2, rx, ry, ZHI, *q, CLOSE],
            [2.0, px, py, ZHI, *q, CLOSE],                 # carry over the box
            [1.2, px, py, zdrop, *q, CLOSE],               # lower in
            [0.8, px, py, zdrop, *q, OPEN],                # release -> snap to placement
            [1.2, px, py, ZHI, *q, OPEN],
        ]
        specs = {2: {"kind": "rigid", "body": bid, "name": nm,
                     "place": (px, py, TABLE_TOP + self.rigid_hz)}}
        return poses, specs

    def _bundle_block(self, gkey):
        """Pick the folded garment bundle (pin all its particles) and pack it at the cuTAMP placement."""
        g = self.garments[gkey]
        cx = (g["bbox"][0] + g["bbox"][1]) / 2.0
        cy = (g["bbox"][2] + g["bbox"][3]) / 2.0           # fold gathers the garment near its flat centre
        # A folded garment nearly fills the box, so it must go in the box CENTRE — the cuTAMP offset
        # placement (tuned for small objects) shoves the big bundle against a wall, where it clips
        # through the thin box mesh. Centre it so it settles inside the walls, not through them.
        px, py = self.box_open[0], self.box_open[1]
        q = DOWN_Q
        ZHI = 42.0
        zgrab = TABLE_TOP + 2.0
        box_top = self.box_open[2]
        # set the bundle DOWN near the box floor rather than dropping from the rim: the fingertip
        # sits ~2cm above the pinned particles, so fingertip z = floor + ~5.5 lands the bundle bottom
        # on the floor. Hold there (pinned) to kill swing, release gently, THEN lift out.
        z_set = self.box_floor_z + 5.5
        poses = [
            [1.5, cx, cy, ZHI, *q, OPEN],
            [1.5, cx, cy, zgrab, *q, OPEN],
            [1.2, cx, cy, zgrab, *q, CLOSE],               # bundle grasp (pin all)
            [1.5, cx, cy, ZHI, *q, CLOSE],                 # peel up
            [3.0, px, py, ZHI, *q, CLOSE],                 # carry slowly over the box
            [1.8, px, py, box_top + 4.0, *q, CLOSE],       # descend to the opening
            [1.8, px, py, z_set, *q, CLOSE],               # lower DEEP, set on the box floor
            [1.0, px, py, z_set, *q, CLOSE],               # hold to damp the swing (still pinned)
            [1.0, px, py, z_set, *q, OPEN],                # release gently at the floor
            [1.5, px, py, z_set, *q, OPEN],                # dwell open -> the bundle settles flat
            [1.5, px, py, box_top + 6.0, *q, OPEN],        # lift straight out (clear of the pile)
            [1.5, px, py, ZHI, *q, OPEN],                  # retreat
        ]
        specs = {2: {"kind": "bundle", "axis": "bundle", "range": (g["p0"], g["n"])}}
        adaptive = {1, 2}
        return poses, specs, adaptive

    def _build_keyframes(self):
        """Walk the plan skeleton IN ORDER; emit the matching block for each action.
        Pick+Place of the same target -> one pick-place block (rigid) or pack block (garment)."""
        poses, self.kf_spec, self.fold_grasp_iv = [], {}, set()
        # fast-iteration debug: schedule ONLY one garment's fold (no packs) to tune fold fidelity
        if self.only_fold:
            gkey = norm_label(self.only_fold)
            block, specs, adaptive = self._fold_block(gkey)
            for li, sp in specs.items():
                self.kf_spec[li] = sp
            self.fold_grasp_iv.update(adaptive)
            poses += block
            poses += [[1.5, self.tcx, self.tcy, 44.0, *DOWN_Q, OPEN]]
            self.key_poses = np.array(poses, dtype=np.float32)
            self._apply_speed()
            self.key_times = np.cumsum(self.key_poses[:, 0])
            self.targets = self.key_poses[:, 1:]
            self.target = self.targets[0]
            self.pose_reach = np.full(len(self.targets), self.max_reach, dtype=np.float32)
            self.total_time = float(self.key_times[-1])
            print(f"[plan] ONLY_FOLD({self.only_fold}): {len(self.targets)} kf, {self.total_time:.0f}s (speed {self.speed:g}x)", flush=True)
            return
        folded, placed, i = set(), set(), 0
        acts = self.actions
        while i < len(acts):
            op, tgt = acts[i]
            gkey = norm_label(tgt)
            if op == "Fold" and gkey in self.garments and gkey not in folded:
                folded.add(gkey)
                block, specs, adaptive = self._fold_block(gkey)
                off = len(poses)
                for li, sp in specs.items():
                    self.kf_spec[off + li] = sp
                self.fold_grasp_iv.update({off + a for a in adaptive})
                poses += block
                print(f"[plan] scheduled Fold({tgt}) -> {len(block)} kf", flush=True)
                i += 1
            elif op == "Pick" and i + 1 < len(acts) and acts[i + 1][0] == "Place" \
                    and norm_label(acts[i + 1][1]) == gkey and gkey not in placed:
                placed.add(gkey)
                off = len(poses)
                if gkey in self.rigids:
                    block, specs = self._rigid_block(gkey)
                elif gkey in self.garments:
                    block, specs, adaptive = self._bundle_block(gkey)
                    self.fold_grasp_iv.update({off + a for a in adaptive})
                else:
                    i += 2; continue
                for li, sp in specs.items():
                    self.kf_spec[off + li] = sp
                poses += block
                print(f"[plan] scheduled Pick/Place({tgt}) -> {len(block)} kf", flush=True)
                i += 2
            else:
                i += 1
        # park pose at the end (reachable retreat above the table centre)
        poses += [[1.5, self.tcx, self.tcy, 44.0, *DOWN_Q, OPEN]]
        self.key_poses = np.array(poses, dtype=np.float32)
        self._apply_speed()
        self.key_times = np.cumsum(self.key_poses[:, 0])
        self.targets = self.key_poses[:, 1:]
        self.target = self.targets[0]
        self.pose_reach = np.full(len(self.targets), self.max_reach, dtype=np.float32)
        self.total_time = float(self.key_times[-1])
        print(f"[plan] schedule: {len(self.targets)} keyframes, {self.total_time:.1f}s", flush=True)

    # ----------------------------- IK control (arm-only, position) -----------------------------
    def _set_up_control(self):
        out_dim = 6
        self.jac_onehots = [wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
                            for i in range(out_dim)]

        ee_id = self.endeffector_id
        ee_off = wp.vec3(*self.endeffector_offset.p)

        @wp.kernel
        def body_out_k(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                       body_com: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
            X_wb = body_q[wp.static(ee_id)]
            r_world = wp.transform_vector(X_wb, wp.static(ee_off) - body_com[wp.static(ee_id)])
            qd = body_qd[wp.static(ee_id)]
            omega = wp.spatial_bottom(qd); v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            out[0] = v_tip[0]; out[1] = v_tip[1]; out[2] = v_tip[2]
            out[3] = omega[0]; out[4] = omega[1]; out[5] = omega[2]

        self.body_out_k = body_out_k
        self.temp_state = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * self.model.joint_dof_count, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        # nominal arm config (the tidy home pose) — the null-space bias keeps the wrist/gripper
        # near this so it stays pointed down instead of twisting into awkward angles.
        self.arm_nominal = self.model.joint_q.numpy()[:self.n_arm].copy()

    def _jacobian(self, joint_q, joint_qd):
        joint_q.requires_grad = True; joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state)
            wp.launch(self.body_out_k, 1, inputs=[self.temp_state.body_q, self.temp_state.body_qd,
                                                  self.model.body_com], outputs=[self.body_out])
        for i in range(6):
            tape.backward(grads={self.body_out: self.jac_onehots[i]})
            wp.copy(self.J_flat[i * in_dim:(i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def _exec_control(self, state_in):
        """Franka: follow the cuRobo joint trajectory + the close-hold-move gripper latch (ported from
        newton_fold's validated physical-grasp exec). The pads clamp only once the arm ARRIVES at the
        cloth, stay shut through the carry, and open at the release."""
        i = int(np.clip(np.searchsorted(self.exec_t, self.sim_time), 0, len(self.exec_t) - 1))
        qt = self.exec_q[i]; self.cur_grip = float(self.exec_g[i])
        q = state_in.joint_q.numpy()
        dq = np.zeros(self.model.joint_dof_count)
        na = self.n_arm
        g0, g1 = self.grip_dofs
        closing_cmd = self.cur_grip < 0.4
        arm_arrived = float(np.linalg.norm(qt - q[:na])) < 0.15
        if not closing_cmd:
            self._grasp_latched = False
        elif arm_arrived:
            self._grasp_latched = True
        do_close = bool(closing_cmd and getattr(self, "_grasp_latched", False))
        g_target = -1.0 if do_close else 4.2       # clamp HARD while grasped; else open FULLY (pads clear the fabric)
        g_gain = 4.0 if do_close else 5.0          # open fast so the fabric is let go cleanly
        dq[g0] = (g_target - q[g0]) * g_gain; dq[g1] = (g_target - q[g1]) * g_gain
        # DWELL on BOTH ends: hold the arm still while the pads finish SHUTTING (grasp) AND while they
        # finish OPENING on release -> the fabric/eraser DROPS before the arm moves, so it is never
        # dragged/carried after a drop (the sticking you saw).
        pads_open = min(q[g0], q[g1]) > 3.2
        if do_close and max(q[g0], q[g1]) > 0.3:
            pass                                              # closing: freeze until shut
        elif (not closing_cmd) and (not pads_open):
            pass                                              # RELEASE: freeze until the pads open + object drops
        else:
            # SMOOTH tracking: feed-forward the cuRobo trajectory VELOCITY (so the arm moves AT the
            # planned speed instead of a laggy catch-up) + a small position correction. Pure (qt-q)
            # drive read as jerky. Segments are position-continuous, so a forward difference is clean.
            i2 = min(i + 1, len(self.exec_q) - 1)
            dt_c = max(1e-3, float(self.exec_t[i2] - self.exec_t[i]))
            vff = (self.exec_q[i2] - self.exec_q[i]) / dt_c
            dq[:na] = vff + (qt - q[:na]) * 3.0
        self.target_joint_qd.assign(dq.astype(np.float32))
        self._cur_interval = int(np.clip(np.searchsorted(self.key_times, self.sim_time), 0, len(self.targets) - 1))
        self.target = self.targets[self._cur_interval]
        # ERASER (rigid) grasp: attach ONLY once the fingers have ARRIVED ON the eraser (pinch within
        # ~9cm) and closed -> a small, correct offset so it's carried BETWEEN the pads, not floating.
        # Release (drop at the placement) when the trajectory commands OPEN.
        spec = self.kf_spec.get(self._cur_interval)
        if self.rigid_grasp is None and do_close and spec is not None and spec.get("kind") == "rigid":
            bid = spec["body"]; pinch = self._ee_pinch(state_in)
            epos = state_in.body_q.numpy()[bid, :3]
            if float(np.linalg.norm(pinch - epos)) < 9.0:
                self.rigid_grasp = {"body": bid, "off": (epos - pinch).astype(np.float64)}
                self._active_rigid_spec = spec
                print(f"[plan] physical GRASP rigid '{spec['name']}' (pinch-eraser "
                      f"{float(np.linalg.norm(pinch - epos)):.1f}cm) t={self.sim_time:.1f}s", flush=True)
        elif self.rigid_grasp is not None and not closing_cmd:
            rs = getattr(self, "_active_rigid_spec", None)
            if rs is not None:
                self._release_rigid(rs); self._active_rigid_spec = None

    def _control(self, state_in):
        if getattr(self, "exec_mode", False):
            self._exec_control(state_in)
            return
        interval = int(np.searchsorted(self.key_times, self.sim_time))
        interval = min(interval, len(self.targets) - 1)
        self.target = self.targets[interval]
        self._cur_interval = interval
        # SMOOTH the arm: interpolate the EE position along the segment from the previous keyframe
        # to this one (cosine ease) instead of snapping to each keyframe -> no jerky yanks, and the
        # cloth is carried gently (violent target jumps were also destabilizing the fold).
        t_to = float(self.key_times[interval])
        t_from = float(self.key_times[interval - 1]) if interval > 0 else 0.0
        prev = self.targets[interval - 1] if interval > 0 else self.targets[interval]
        a = float(np.clip((self.sim_time - t_from) / max(1e-3, t_to - t_from), 0.0, 1.0))
        a = 0.5 - 0.5 * np.cos(np.pi * a)                 # ease in/out
        pos = np.array(prev[:3], float) + a * (np.array(self.target[:3], float) - np.array(prev[:3], float))
        tgt = np.array([pos[0], pos[1], pos[2], *self.target[3:7]], dtype=float)
        rb = np.array([self.robot_base[0], self.robot_base[1]])
        dxy = tgt[:2] - rb; r = float(np.linalg.norm(dxy))
        mr = float(self.pose_reach[interval])
        if r > mr:
            tgt[:2] = rb + dxy / r * mr
        # adaptive grasp height: land the fingertip on the ACTUAL cloth-pile top near the grasp XY
        if interval in self.fold_grasp_iv:
            pq = state_in.particle_q.numpy()
            near = (pq[:, 0] - tgt[0]) ** 2 + (pq[:, 1] - tgt[1]) ** 2 < 36.0
            if near.any():
                ctop = float(pq[near, 2].max())
                if self.physical_grip:
                    # CLAMP needs the finger PADS (not just the [0,0,22] tips) on the fabric. The tips
                    # sit below the pads, so drive the tips ~4cm BELOW the cloth top -> the pads close
                    # ON the cloth instead of clamping air above it (the "dry run" you saw).
                    tgt[2] = float(np.clip(ctop - 4.0, TABLE_TOP - 3.0, 42.0))
                else:
                    tgt[2] = float(np.clip(ctop + 0.3, CLOTH_TOP, 42.0))
        wp.launch(compute_ee_delta, dim=1,
                  inputs=[state_in.body_q, self.endeffector_offset, self.endeffector_id,
                          self.bodies_per_world, wp.transform(*tgt)], outputs=[self.ee_delta])
        self._jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta = self.ee_delta.numpy()[0]
        na = self.n_arm
        J3 = J[:3, :na]
        Jinv = _dls_pinv(J3)
        dq_pos = Jinv @ delta[:3] * 3.0
        # null-space bias toward the tidy nominal config -> the wrist/gripper stays pointed down
        # instead of twisting (this only moves the arm within the redundancy; EE position unaffected)
        N = np.eye(na) - Jinv @ J3
        q_arm = state_in.joint_q.numpy()[:na]
        dq_null = N @ (self.arm_nominal - q_arm) * 0.7
        delta_q = np.zeros(self.model.joint_dof_count)
        delta_q[:na] = dq_pos + dq_null
        if self.physical_grip:
            # REFERENCE-STYLE physical CLAMP (example_cloth_franka): the franka fingers physically clamp
            # a whole REGION of the fabric (no pin, no thin-edge drag), so it folds as a coherent flap.
            # Close only once the EE has ARRIVED at the target (no mid-air clamp); hold through the drag;
            # open FULLY on release with a dwell so the fabric/eraser DROPS before the arm moves.
            g0, g1 = self.grip_dofs
            qj = state_in.joint_q.numpy()
            grip_cmd_closed = float(self.target[-1]) < 0.4
            # ABSOLUTE arrival gate: close ONLY once the real finger tip has reached the actual grasp
            # point -- correct XY *and* down at the fabric. The old test used the EASED interpolation
            # delta, which is ~0 for the whole segment, so the pads shut the instant the CLOSE keyframe
            # began (mid-air, before the descent finished -> "grasps in the air then moves down closed",
            # and a shut gripper can never take the cloth). Measuring EE_tip -> FINAL target instead
            # keeps the fingers OPEN through the entire descent and fires the clamp only on contact.
            ee_tip = self._ee_pinch(state_in)
            f_xy = np.array(self.target[:2], float)
            d_xy = f_xy - rb; rf = float(np.linalg.norm(d_xy))
            if rf > mr:
                f_xy = rb + d_xy / rf * mr
            ftgt = np.array([f_xy[0], f_xy[1], float(tgt[2])])   # tgt[2] already dug-in at grasp keyframes
            arm_arrived = float(np.linalg.norm(ee_tip - ftgt)) < 4.0
            self._last_arrive_d = float(np.linalg.norm(ee_tip - ftgt))
            if not grip_cmd_closed:
                self._clamp_latched = False
            elif arm_arrived:
                if not getattr(self, "_clamp_latched", False):
                    print(f"[plan] CLAMP arrived t={self.sim_time:.1f}s tip=({ee_tip[0]:.0f},{ee_tip[1]:.0f},{ee_tip[2]:.0f}) "
                          f"grasp=({ftgt[0]:.0f},{ftgt[1]:.0f},{ftgt[2]:.0f}) d={self._last_arrive_d:.1f}cm", flush=True)
                self._clamp_latched = True
            do_clamp = bool(grip_cmd_closed and getattr(self, "_clamp_latched", False))
            g_target = self.g_close if do_clamp else self.g_open   # clamp HARD; else open FULLY
            ggain = 4.0 if do_clamp else 5.0
            delta_q[g0] = (g_target - qj[g0]) * ggain; delta_q[g1] = (g_target - qj[g1]) * ggain
            # gripper-agnostic closure fraction: 0 = fully open, 1 = fully shut (works for the franka
            # fingers AND the UR5e jaws, whose slide sign is opposite).
            qg = 0.5 * (qj[g0] + qj[g1])
            frac = (qg - self.g_open_ref) / (self.g_close_ref - self.g_open_ref + 1e-6)
            jaws_shut = frac > 0.85
            jaws_open = frac < 0.15
            if (do_clamp and not jaws_shut) or ((not grip_cmd_closed) and (not jaws_open)):
                delta_q[:na] = 0.0                       # DWELL: hold arm still while pads shut / open
            # ERASER: attach once the fingers have ARRIVED on it (small offset -> held, not floating)
            spec = self.kf_spec.get(self._cur_interval)
            if self.rigid_grasp is None and do_clamp and spec is not None and spec.get("kind") == "rigid":
                bid = spec["body"]; pinch = self._ee_pinch(state_in)
                epos = state_in.body_q.numpy()[bid, :3]
                if float(np.linalg.norm(pinch - epos)) < 9.0:
                    self.rigid_grasp = {"body": bid, "off": (epos - pinch).astype(np.float64)}
                    self._active_rigid_spec = spec
                    print(f"[plan] physical GRASP rigid '{spec['name']}' t={self.sim_time:.1f}s", flush=True)
            elif self.rigid_grasp is not None and not grip_cmd_closed:
                rs = getattr(self, "_active_rigid_spec", None)
                if rs is not None:
                    self._release_rigid(rs); self._active_rigid_spec = None
        self.target_joint_qd.assign(delta_q.astype(np.float32))

    # ----------------------------- grasp / pin (range-aware) -----------------------------
    def _grasp(self):
        if self.physical_grip:
            return                       # physical grasp: fingers hold the cloth by CONTACT (no pin)
        spec = self.kf_spec.get(self._cur_interval)
        if spec is None:
            return
        p0, n = spec["range"]
        bq = self.state_0.body_q.numpy()
        hand = bq[self.endeffector_id, :3]; q6 = bq[self.endeffector_id, 3:7]

        def qrot(qq, v):
            x, y, z, w = qq; t = 2.0 * np.cross([x, y, z], v)
            return np.asarray(v) + w * t + np.cross([x, y, z], t)
        pinch = hand + qrot(q6, np.array(list(self.ee_off_vec)))

        pq = self.state_0.particle_q.numpy()
        sub = pq[p0:p0 + n]
        kind = spec["kind"]
        if kind == "bundle":
            local = np.arange(n)
        elif kind == "edge":
            # pin the WHOLE leading edge as a coherent flap. Both folds grasp the MIN side along
            # their axis, so grab the bottom ~20% of the CURRENT fabric along that axis (robust to
            # the fold-1 result — a fixed line from the original bbox missed the moved fabric).
            ax = 0 if spec["axis"] == "x" else 1
            vals = sub[:, ax]
            # WIDE band (was a thin 14% edge). The July-21 fold kept its shape because the gripper held
            # a whole REGION of fabric; a thin-edge pin drags a few particles and lets the rest crumple
            # (loses shape after a few folds). Pin ~45% of the flap being folded so a coherent sheet
            # moves together and the garment stays garment-shaped.
            thr = float(np.percentile(vals, 45.0))
            local = np.where(vals <= thr)[0]
            if len(local) < 8:
                local = np.argsort(vals)[:max(30, int(0.42 * len(sub)))]
            elif len(local) > 260:                                # cap so it stays a flap, not the whole cloth
                local = local[np.argsort(vals[local])[:260]]
        else:                                                     # legacy point pinch
            local = np.argsort(np.linalg.norm(sub[:, :2] - pinch[:2], axis=1))[:40]
        idx = local + p0
        if kind in ("bundle", "edge"):
            # keep the grabbed set's SHAPE and hang it under the fingertip -> the edge stays straight
            anchor = np.array([pq[idx, 0].mean(), pq[idx, 1].mean(), pinch[2]])
        else:
            anchor = pinch
        self.pin_idx = wp.array(idx.astype(np.int32), dtype=wp.int32)
        self.pin_off = wp.array((pq[idx] - anchor).astype(np.float32), dtype=wp.vec3)
        im = self.orig_inv_mass.copy(); im[idx] = 0.0
        self.model.particle_inv_mass.assign(im)
        self.grasp_active = True
        gap = float(pinch[2] - pq[idx, 2].mean())
        print(f"[plan] GRASP {spec['kind']}/{spec.get('axis','')} ({len(idx)} particles) "
              f"t={self.sim_time:.1f}s fingertip z={pinch[2]:.1f} cloth z={pq[idx,2].mean():.1f} (gap {gap:+.1f})", flush=True)

    def _release(self):
        # GENTLE release: zero the freed particles' velocity so the stretch the pin held doesn't snap
        # them into a spike (the fold-2 release NaN). Restoring inv_mass alone let them fling.
        if self.pin_idx is not None:
            idx = self.pin_idx.numpy()
            for st in (self.state_0, self.state_1):
                qd = st.particle_qd.numpy()
                qd[idx] = 0.0
                st.particle_qd.assign(qd)
        self.model.particle_inv_mass.assign(self.orig_inv_mass)
        self.grasp_active = False
        print(f"[plan] RELEASE t={self.sim_time:.1f}s", flush=True)

    def _ee_pinch(self, state):
        bq = state.body_q.numpy()
        hand = bq[self.endeffector_id, :3]; q6 = bq[self.endeffector_id, 3:7]
        return hand + _qrot(q6, np.array(list(self.ee_off_vec)))

    def _grasp_rigid(self, spec):
        bid = spec["body"]
        pinch = self._ee_pinch(self.state_0)
        # Hold the object CENTERED directly under the fingertip (hanging just below it), instead of
        # freezing whatever offset it happened to have at grasp time -> it looked like the gripper was
        # holding the cup by one wall / off to the side. A clean centered offset reads as a real pick.
        hz = float(getattr(self, "rigid_hz", 1.5))
        off = np.array([0.0, 0.0, -(hz + 1.0)], dtype=np.float64)
        self.rigid_grasp = {"body": bid, "off": off}
        newp = (pinch + off)
        for st in (self.state_0, self.state_1):
            bq = st.body_q.numpy(); bq[bid, :3] = newp; st.body_q.assign(bq)
        self.rigid_pose[bid][:3] = newp
        print(f"[plan] GRASP rigid '{spec['name']}' t={self.sim_time:.1f}s (centered under tip)", flush=True)

    def _release_rigid(self, spec):
        px, py, pz = spec["place"]
        for st in (self.state_0, self.state_1):
            bq = st.body_q.numpy()
            bq[spec["body"], :3] = [px, py, pz]
            st.body_q.assign(bq)
        self.rigid_pose[spec["body"]][:3] = [px, py, pz]     # so render re-stamps the PLACED pose
        self.rigid_grasp = None
        print(f"[plan] PLACE rigid '{spec['name']}' -> ({px:.0f},{py:.0f})cm t={self.sim_time:.1f}s", flush=True)

    def _update_grasp(self):
        if getattr(self, "exec_mode", False):
            return                       # franka: cloth held by contact, eraser grasp handled in _exec_control (arrival-gated)
        grip = float(self.target[-1])
        if self.prev_grip > 0.4 and grip <= 0.4:            # close
            spec = self.kf_spec.get(self._cur_interval)
            self._active_spec = spec
            if spec and spec.get("kind") == "rigid":
                self._grasp_rigid(spec)
            else:
                self._grasp()
        elif self.prev_grip <= 0.4 and grip > 0.4:          # open
            spec = getattr(self, "_active_spec", None)
            if spec and spec.get("kind") == "rigid":
                self._release_rigid(spec)
            else:
                self._release()
            self._active_spec = None
        self.prev_grip = grip

    # ----------------------------- coupled simulate + render -----------------------------
    def _prof_add(self, key, dt):
        self.prof[key] = self.prof.get(key, 0.0) + dt

    def _simulate(self):
        prof = self.profile
        if prof:
            wp.synchronize(); _t0 = time.perf_counter()
        self.cloth_solver.rebuild_bvh(self.state_0)
        if prof:
            wp.synchronize(); self._prof_add("bvh", time.perf_counter() - _t0)
        for _ in range(self.substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            if prof:
                wp.synchronize(); _t0 = time.perf_counter()
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if prof:
                wp.synchronize(); self._prof_add("robot", time.perf_counter() - _t0)
            self.state_0.particle_f.zero_()
            self.model.particle_count = pc
            self.model.gravity.assign(self.gravity_earth)
            if self.rigid_grasp is not None:                 # carry the grasped eraser under the EE
                pinch = self._ee_pinch(self.state_1)
                bq = self.state_1.body_q.numpy()
                newp = pinch + self.rigid_grasp["off"]
                bq[self.rigid_grasp["body"], :3] = newp
                self.state_1.body_q.assign(bq)
                self.rigid_pose[self.rigid_grasp["body"]][:3] = newp   # render follows the carry
            if self.grasp_active and self.pin_idx is not None and not self.physical_grip:
                wp.launch(pin_to_ee, dim=len(self.pin_idx),
                          inputs=[self.state_1.body_q, self.endeffector_id, self.ee_off_vec,
                                  self.pin_idx, self.pin_off],
                          outputs=[self.state_0.particle_q, self.state_0.particle_qd])
            if prof:
                wp.synchronize(); _t0 = time.perf_counter()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            if prof:
                wp.synchronize(); self._prof_add("collide", time.perf_counter() - _t0); _t0 = time.perf_counter()
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # anti-NaN velocity guard on the freshly-stepped cloth (state_1): cap spikes before they grow
            if self.model.particle_count > 0:
                wp.launch(_stabilize_particle_vel, dim=self.model.particle_count,
                          inputs=[self.state_1.particle_qd, 800.0])
            if prof:
                wp.synchronize(); self._prof_add("cloth", time.perf_counter() - _t0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _pose_gripper(self):
        if self.grip_closed is None or getattr(self, "exec_mode", False):
            return                                            # franka: gripper is driven in _exec_control
        grip = float(self.target[-1]) if self.sim_time < self.total_time else OPEN
        tgt = self.grip_closed if grip <= 0.4 else self.grip_open
        q = self.state_0.joint_q.numpy()
        for j, d in enumerate(self.grip_dofs):
            q[d] = tgt[j]
        for d in self.free_loop_dofs:
            q[d] = 0.0
        self.state_0.joint_q.assign(q)
        eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        if self.profile:
            wp.synchronize(); _t0 = time.perf_counter()
            self._control(self.state_0)
            wp.synchronize(); self._prof_add("control", time.perf_counter() - _t0)
            self._update_grasp()
            self._simulate()
            wp.synchronize(); _t0 = time.perf_counter()
            self._pose_gripper()
            wp.synchronize(); self._prof_add("gripper", time.perf_counter() - _t0)
            self.prof_n += 1
        else:
            self._control(self.state_0)
            self._update_grasp()
            self._simulate()
            self._pose_gripper()
        self.sim_time += self.frame_dt

    def _setup_render(self):
        v = self.viewer
        v.set_model(self.model)
        v.show_triangles = False; v.show_particles = False
        # kill the strong BLUE sky tint (default sky_upper is blue -> washes the whole scene blue,
        # worse when zoomed): use a neutral, bright studio ambient + white light.
        r = getattr(v, "renderer", None)
        if r is not None:
            try:
                r.draw_sky = False                    # solid background (the sky dome blended as a
                #                                       grey haze over the scene -> washed out)
                r.sky_upper = (0.55, 0.57, 0.60)      # still used for ambient light (neutral grey)
                r.sky_lower = (0.32, 0.33, 0.35)
                r._light_color = (1.0, 1.0, 0.97)
            except Exception:
                pass
        self.tex_img = load_texture(os.path.join(self.capture_dir, "rgb.png"))
        self.g_uv = {k: wp.array(g["uvs"].astype(np.float32), dtype=wp.vec2) for k, g in self.garments.items()}
        self.g_fc = {k: wp.array(g["faces"].astype(np.int32), dtype=wp.int32) for k, g in self.garments.items()}

    def export_plan(self, path):
        """Stage 1 (franka): write the fold+pack keyframe fingertip poses + obstacles in the FRANKA-BASE
        frame (metres) for cuRobo. Transform = R_base^T @ (world - base) using the ACTUAL base pose
        (translation + yaw) -- matches how Newton places the arm, so cuRobo plans in the right frame."""
        import json
        base = np.array(self.robot_base, dtype=float)
        theta = np.radians(self.base_yaw_deg)             # must match the franka placement rotation
        c_, s_ = np.cos(theta), np.sin(theta)

        def to_franka(world_xyz):
            d = np.asarray(world_xyz, float) - base
            return np.array([d[0] * c_ + d[1] * s_, -d[0] * s_ + d[1] * c_, d[2]]) / 100.0

        kfs = []
        for row in self.key_poses:
            p = to_franka([row[1], row[2], row[3]])
            if p[2] < 0.02:
                p[2] -= 0.06                              # dig-in: fingers reach INTO the fabric/object
            kfs.append({"dur": float(row[0]), "fingertip_m": p.tolist(),
                        "gripper": "open" if float(row[8]) > 0.4 else "closed"})
        table_top_franka = (TABLE_TOP - base[2]) / 100.0
        obstacles = [{"name": "table", "center_m": [0.5, 0.0, round(table_top_franka - 0.1, 4)],
                      "dims_m": [1.6, 1.6, 0.2]}]
        json.dump({"keyframes": kfs, "obstacles": obstacles,
                   "down_quat_wxyz": [0.0, 1.0, 0.0, 0.0], "ee_offset_z": 0.105,
                   "home_js": [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]},
                  open(path, "w"), indent=2)
        print(f"[plan] exported {len(kfs)} keyframes + {len(obstacles)} obstacles -> {path}", flush=True)

    def load_exec_plan(self, path):
        """Stage 3 (franka): load a cuRobo joint trajectory + gripper events for exec_mode."""
        import json
        traj = json.load(open(path))
        ts, qs, gs = [], [], []
        t = 0.0
        for s in traj["segments"]:
            q = s["q"]; dur = float(s["dur"]); grip = 0.02 if s["gripper"] == "closed" else 0.8
            n = len(q)
            for j, qj in enumerate(q):
                ts.append(t + dur * (j / (n - 1) if n > 1 else 0.0)); qs.append(qj); gs.append(grip)
            t += dur
        self.exec_t = np.array(ts); self.exec_q = np.array(qs, np.float64); self.exec_g = np.array(gs)
        self.exec_total = t; self.exec_mode = True; self.cur_grip = 0.8
        print(f"[plan] EXEC cuRobo trajectory: {len(ts)} waypoints over {t:.1f}s", flush=True)

    def export_ee_traj(self, path):
        """Write the fold+pack EE (TCP) waypoints in the ur5e_base_link frame (metres) + gripper
        state for the real UR5e transfer. The sim already runs in that frame at cm scale (robot at
        the base origin), so base-frame metres = keyframe (x,y,z)/100. Plan-driven ordering preserved."""
        wps = []
        for row in self.key_poses:
            dur, x, y, z = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            wps.append({"dur": dur, "tcp_m": [x / M2CM, y / M2CM, z / M2CM],
                        "gripper": "open" if float(row[8]) > 0.4 else "closed"})
        out = {"frame": "ur5e_base_link", "units": "m", "plan": os.path.basename(self.args.plan),
               "actions": [{"op": op, "target": tgt} for op, tgt in self.actions],
               "placements_m": {norm_label(k): [float(c) for c in v["pos"]]
                                for k, v in self.placements.items()},
               "waypoints": wps}
        json.dump(out, open(path, "w"), indent=2)
        print(f"[plan] exported {len(wps)} EE waypoints (ur5e_base_link, m) -> {path}", flush=True)

    def render(self):
        v = self.viewer
        v.begin_frame(self.sim_time)
        if self.rigid_pose:                          # re-stamp eraser poses (cloth solver drops them)
            bq = self.state_0.body_q.numpy()
            for bid, pose in self.rigid_pose.items():
                bq[bid] = pose
            self.state_0.body_q.assign(bq)
        pq = self.state_0.particle_q.numpy()
        for k, g in self.garments.items():
            verts = pq[g["p0"]:g["p0"] + g["n"]].astype(np.float32)
            v.log_mesh(f"g_{k}", wp.array(verts, dtype=wp.vec3), self.g_fc[k], uvs=self.g_uv[k],
                       texture=self.tex_img, color=(1, 1, 1), backface_culling=False)
            ob = getattr(v, "objects", {}).get(f"g_{k}")
            if ob is not None:
                r, m, c, _ = ob.material; ob.material = (r, m, c, 1.0)
        v.log_state(self.state_0)
        v.end_frame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--vbd_iters", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0, help="arm-motion speed-up; compresses the keyframe schedule (normal=1.0; brisker=2.2)")
    ap.add_argument("--profile", action="store_true", help="time each phase of the hot loop (control/bvh/robot/collide/cloth/gripper/render/nan) and print a per-frame breakdown")
    ap.add_argument("--self_contact_interval", type=int, default=1, help="detect cloth self-contact every N substeps; >1 cuts the dominant per-frame cost (frees budget for fold fidelity)")
    ap.add_argument("--self_contact_buf", type=int, default=64, help="cloth self-contact buffer size (smaller = cheaper)")
    ap.add_argument("--garment_max_cm", type=float, default=0.0, help="rescale each garment so its largest span == this many cm (0 = keep depth-reconstructed size); fixes oversized/off-table garments from noisy fabric depth")
    ap.add_argument("--reach", type=float, default=86.0, help="EE reach clamp (cm) from the base")
    ap.add_argument("--mask_step", type=int, default=9, help="cloth_from_mask sampling stride (px); smaller = finer/more realistic, more particles")
    ap.add_argument("--cell", type=float, default=2.0, help="[unused: legacy grid] cloth grid cell (cm)")
    # NORMAL (original) cloth defaults — this is the baseline the pipeline shipped with. The
    # shape-holding "adjusted" values are: --stretch_ke 3e5 --bend_ke 12 --tri_kd 1e-3 --edge_kd 5e-2
    ap.add_argument("--stretch_ke", type=float, default=1.0e4, help="cloth STRETCH (membrane) stiffness (normal=1e4; adjusted=3e5 for inextensible shape-holding fabric)")
    ap.add_argument("--bend_ke", type=float, default=3.0, help="cloth BEND stiffness (normal=3; adjusted=12)")
    ap.add_argument("--tri_kd", type=float, default=1.5e-6, help="cloth triangle (stretch) damping (normal=1.5e-6; adjusted=1e-3)")
    ap.add_argument("--edge_kd", type=float, default=1.0e-2, help="cloth edge (bend) damping (normal=1e-2; adjusted=5e-2)")
    ap.add_argument("--only_fold", default=None, help="debug: schedule ONLY this garment's fold (fast fidelity iteration)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=600.0)
    ap.add_argument("--render_every", type=int, default=2)
    ap.add_argument("--screenshot", default=None, help="PNG base; milestone frames saved at each release")
    ap.add_argument("--frames_dir", default=None, help="dump every rendered frame here as frame_%06d.png (for video)")
    ap.add_argument("--export_ee_traj", default=None, help="write the fold+pack EE trajectory (base frame, m) then exit")
    # --- Franka + PHYSICAL grasp (cuRobo-planned) path in the FAITHFUL scene ---
    ap.add_argument("--arm", default="ur5e", choices=["ur5e", "franka"], help="ur5e (2F-85 + pin) or franka (physical contact grasp via cuRobo)")
    ap.add_argument("--physical_grasp", action="store_true", help="franka: pinch+hold the cloth by CONTACT+friction (no pin)")
    ap.add_argument("--base_yaw", type=float, default=0.0, help="extra base-facing yaw (deg) for the franka")
    ap.add_argument("--export_plan", default=None, help="stage 1: write the fold+pack keyframe poses + obstacles (franka-base frame) for cuRobo, then exit")
    ap.add_argument("--exec_plan", default=None, help="stage 3: execute a cuRobo joint trajectory (from curobo_fold_plan.py)")
    ap.add_argument("--cam", default=None)
    args = ap.parse_args()

    wp.init()
    # Newton's GL shape shader bakes distance fog in WORLD units: fog_start=20, fog_end=200.
    # Franka demos are metre-scale so tabletop objects never reach 20 m and fog never shows.
    # Our scene is CM-scale with the camera ~55 cm out, so the whole table sits inside 20-200 cm
    # of fog -> a grey haze/veil washes everything toward sky_lower. Push fog past the scene so
    # it renders crisp (the way the metre-scale demos look).
    if args.viewer == "gl":
        try:
            from newton._src.viewer.gl import shaders as _sh
            _sh.shape_fragment_shader = (_sh.shape_fragment_shader
                .replace("float fog_start = 20.0;", "float fog_start = 100000.0;")
                .replace("float fog_end   = 200.0;", "float fog_end   = 200000.0;"))
        except Exception:
            pass
    viewer = ViewerGL() if args.viewer == "gl" else ViewerNull(num_frames=10**9)
    ex = PlanExec(viewer, args)
    if args.export_ee_traj:
        ex.export_ee_traj(args.export_ee_traj)
        return
    if args.export_plan:                                 # stage 1 (franka): export poses for cuRobo
        ex.export_plan(args.export_plan)
        return
    if args.exec_plan:                                   # stage 3 (franka): run cuRobo joint trajectory
        ex.load_exec_plan(args.exec_plan)
    if args.viewer == "gl":
        if args.cam:
            cx, cy, cz, cp, cyaw = (float(v) for v in args.cam.split(","))
        else:
            cx, cy, cz, cp, cyaw = ex.tcx - 55.0, ex.tcy - 70.0, 60.0, -32.0, 55.0
        viewer.set_camera(wp.vec3(cx, cy, cz), cp, cyaw)

    sched_time = ex.exec_total if getattr(ex, "exec_mode", False) else ex.total_time
    n_frames = int(sched_time * ex.fps) + 60
    shots, base = {}, (args.screenshot.replace(".png", "") if args.screenshot else None)
    if base:
        grips = ex.key_poses[:, -1]
        rels = [i for i in range(1, len(grips)) if grips[i] > 0.4 >= grips[i - 1]]
        for k, i in enumerate(rels):
            shots[int(ex.key_times[i] * ex.fps)] = f"{base}_step{k+1}.png"
        shots[3] = f"{base}_start.png"
    vid_i = 0
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    print(f"[plan] running {n_frames} frames ({ex.total_time:.0f}s schedule)...", flush=True)
    prof_wall = 0.0
    for f in range(n_frames):
        if args.profile:
            wp.synchronize(); _fw0 = time.perf_counter()
        ex.step()
        if f % args.render_every == 0 or f in shots:
            if args.profile:
                wp.synchronize(); _t0 = time.perf_counter()
            ex.render()
            if args.profile:
                wp.synchronize(); ex._prof_add("render", time.perf_counter() - _t0)
            if args.frames_dir and args.viewer == "gl":
                from PIL import Image
                Image.fromarray(viewer.get_frame().numpy()).save(
                    os.path.join(args.frames_dir, f"frame_{vid_i:06d}.png"))
                vid_i += 1
        if args.profile:
            wp.synchronize(); _t0 = time.perf_counter()
        q = ex.state_0.particle_q.numpy()
        naninf = not np.isfinite(q).all()
        if args.profile:
            wp.synchronize(); ex._prof_add("nan_check", time.perf_counter() - _t0)
            prof_wall += time.perf_counter() - _fw0
            if ex.prof_n > 0 and f % 60 == 59:
                n = ex.prof_n
                parts = " ".join(f"{k}={1000*v/n:.1f}ms" for k, v in sorted(ex.prof.items(), key=lambda kv: -kv[1]))
                fps = n / prof_wall if prof_wall > 0 else 0
                rt = fps / ex.fps
                print(f"[prof] frame {f}: {1000*prof_wall/n:.1f}ms/frame ({fps:.1f} fps, {rt:.2f}x real-time) | {parts}", flush=True)
        if naninf:
            print(f"[plan] NaN at frame {f} (t={ex.sim_time:.1f}s)", flush=True)
            break
        if base and f in shots and args.viewer == "gl":
            from PIL import Image
            Image.fromarray(viewer.get_frame().numpy()).save(shots[f])
            print(f"[plan] shot -> {shots[f]}", flush=True)
    # final footprint report per garment
    q = ex.state_0.particle_q.numpy()
    for k, g in ex.garments.items():
        sub = q[g["p0"]:g["p0"] + g["n"]]
        fw, fd = np.ptp(sub[:, 0]), np.ptp(sub[:, 1])
        iw = g["bbox"][1] - g["bbox"][0]; ih = g["bbox"][3] - g["bbox"][2]
        print(f"[plan] {g['label']} folded footprint {fw:.0f}x{fd:.0f}cm "
              f"({100*fw*fd/max(1,iw*ih):.0f}% of flat {iw:.0f}x{ih:.0f})", flush=True)
    if base and args.viewer == "gl":
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(f"{base}_done.png")
        print(f"[plan] shot -> {base}_done.png | finite={np.isfinite(q).all()}", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and (args.viewer != "gl" or viewer.is_running()):
        ex.render()


if __name__ == "__main__":
    main()
