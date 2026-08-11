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
CELL = 2.0              # cloth grid cell (cm)
ROBOT_Z = 25.0


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
        self.robot_base = (0.0, 0.0, ROBOT_Z)
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
        thx = float(np.ptp(allxy[:, 0])) / 2 + 22.0
        thy = float(np.ptp(allxy[:, 1])) / 2 + 22.0
        self.table_hz = 10.0
        self.table_top = TABLE_TOP
        tbl_idx = scene.shape_count
        # VISIBLE light table (like the RGBD) so the dark erasers read against it
        scene.add_shape_box(-1, wp.transform(wp.vec3(self.tcx, self.tcy, TABLE_TOP - 10.0), wp.quat_identity()),
                            hx=thx, hy=thy, hz=10.0, cfg=cfg, color=wp.vec3(0.80, 0.80, 0.82))
        ped_idx = scene.shape_count
        scene.add_shape_box(-1, wp.transform(wp.vec3(0.0, 0.0, ROBOT_Z / 2), wp.quat_identity()),
                            hx=14.0, hy=14.0, hz=ROBOT_Z / 2, cfg=cfg)

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
            gx, gy = plan_xy[norm_label(nm)]             # place the real garment shape AT its plan pose
            v[:, 0] += gx - v[:, 0].mean()
            v[:, 1] += gy - v[:, 1].mean()
            v[:, 2] += CLOTH_TOP - v[:, 2].min()         # lie flat on the table
            faces = faces.reshape(-1, 3).astype(np.int32)
            n = len(v)
            scene.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
                                 vertices=[wp.vec3(*x) for x in v], indices=faces.reshape(-1).tolist(),
                                 density=0.02, tri_ke=self.stretch_ke, tri_ka=self.stretch_ke, tri_kd=1.5e-6,
                                 edge_ke=self.bend_ke, edge_kd=1.0e-2, particle_radius=0.8)
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
                    placed = True
                except Exception as e:
                    print(f"[plan] eraser mesh '{nm}' failed ({e}); box fallback", flush=True)
            if not placed:
                scene.add_shape_box(b, hx=4.5, hy=2.5, hz=hz, cfg=cfg, color=wp.vec3(*r["color"]))
            self.rigids[nm] = b
            self.rigid_xy[nm] = (rx, ry)
            print(f"[plan] rigid '{nm}' (real mesh={'yes' if placed else 'no'}) @({rx:.0f},{ry:.0f})cm", flush=True)
        self.rigid_grasp = None

        scene.color(include_bending=True)
        scene.add_ground_plane(cfg=cfg)
        self.model = scene.finalize()

        flags = self.model.shape_flags.numpy()
        for si in (ped_idx, box_liner_idx):          # keep the TABLE visible (erasers read on it)
            flags[si] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # HIGH cloth-table friction so the un-grabbed fabric STAYS on the table while the grabbed
        # flap folds over (the crease forms and the cloth deforms — instead of the whole sheet
        # sliding rigidly). Firm normal contact so it rests flat.
        self.model.soft_contact_ke = 2.0e4; self.model.soft_contact_kd = 8.0e-2
        self.model.soft_contact_mu = 1.2; self.model.soft_contact_margin = 1.0

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=1.2)
        self.contacts = self.collision_pipeline.contacts()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, GRAVITY), dtype=wp.vec3)

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.substeps)
        self.cloth_solver = SolverVBD(
            self.model, iterations=args.vbd_iters, integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5, particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=64,
            particle_conservative_bound_relaxation=0.5, particle_collision_detection_interval=1)
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
        folds = [
            # fold the -x half over the +x half: grasp the x=x0 edge, hinge over cx, lay at x1
            {"axis": "x", "edge": x0, "to": x1, "hinge": cx, "along": (y0, y1), "mid": cy},
            # fold the -y half over the +y half (of the now half-width garment): grasp y=y0 edge
            {"axis": "y", "edge": y0, "to": y1, "hinge": cy, "along": (cx, x1), "mid": (cx + x1) / 2.0},
        ]
        poses, specs, adaptive = [], {}, set()
        for i, f in enumerate(folds):
            base = len(poses)
            z_edge = TABLE_TOP + 0.6 + i * 1.6       # grasp just onto the (possibly doubled) fabric
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
        return poses, specs, adaptive

    def _in_box(self, x, y, margin=5.0):
        """Clamp an XY into the box interior (keep `margin` cm off the walls) so bundles/erasers
        settle flat on the floor instead of perching on a rim."""
        cx, cy = self.box_open[0], self.box_open[1]
        hx = max(1.0, self.box_hw[0] - margin); hy = max(1.0, self.box_hw[1] - margin)
        return cx + float(np.clip(x - cx, -hx, hx)), cy + float(np.clip(y - cy, -hy, hy))

    def _place_xy(self, key):
        """Map the cuTAMP in-box arrangement (offset from the plan box centre) onto the REAL box
        opening, clamped inside the interior. Preserves which object goes where inside the box."""
        p = self.placements.get(key)
        if p is not None:
            dxp = (float(p["pos"][0]) - float(self.plan_box_center[0])) * M2CM
            dyp = (float(p["pos"][1]) - float(self.plan_box_center[1])) * M2CM
            return self._in_box(self.box_open[0] + dxp, self.box_open[1] + dyp)
        return self.box_open[0], self.box_open[1]

    def _rigid_block(self, nm):
        """Pick a rigid (eraser) from its real spot and Place it at the cuTAMP placement in the box."""
        bid = self.rigids[nm]
        rx, ry = self.rigid_xy[nm]
        px, py = self._place_xy(nm)
        q = DOWN_Q
        ZHI = 42.0
        zgrab = TABLE_TOP + 2.0 * self.rigid_hz + 0.5      # fingertip on top of the eraser
        zdrop = TABLE_TOP + 2.0 * self.rigid_hz + 1.0
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
        px, py = self._place_xy(gkey)
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
            self.key_times = np.cumsum(self.key_poses[:, 0])
            self.targets = self.key_poses[:, 1:]
            self.target = self.targets[0]
            self.pose_reach = np.full(len(self.targets), self.max_reach, dtype=np.float32)
            self.total_time = float(self.key_times[-1])
            print(f"[plan] ONLY_FOLD({self.only_fold}): {len(self.targets)} kf, {self.total_time:.0f}s", flush=True)
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

    def _control(self, state_in):
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
        self.target_joint_qd.assign(delta_q.astype(np.float32))

    # ----------------------------- grasp / pin (range-aware) -----------------------------
    def _grasp(self):
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
            thr = float(np.percentile(vals, 14.0))                # thin edge -> the REST is free to deform
            local = np.where(vals <= thr)[0]
            if len(local) < 6:
                local = np.argsort(vals)[:max(16, int(0.12 * len(sub)))]
            elif len(local) > 70:                                 # pin only the edge, not a big rigid chunk
                local = local[np.argsort(vals[local])[:70]]
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
        off = self.state_0.body_q.numpy()[bid, :3] - pinch
        self.rigid_grasp = {"body": bid, "off": off.astype(np.float64)}
        print(f"[plan] GRASP rigid '{spec['name']}' t={self.sim_time:.1f}s", flush=True)

    def _release_rigid(self, spec):
        px, py, pz = spec["place"]
        for st in (self.state_0, self.state_1):
            bq = st.body_q.numpy()
            bq[spec["body"], :3] = [px, py, pz]
            st.body_q.assign(bq)
        self.rigid_grasp = None
        print(f"[plan] PLACE rigid '{spec['name']}' -> ({px:.0f},{py:.0f})cm t={self.sim_time:.1f}s", flush=True)

    def _update_grasp(self):
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
    def _simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.substeps):
            self.state_0.clear_forces(); self.state_1.clear_forces()
            pc = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = pc
            self.model.gravity.assign(self.gravity_earth)
            if self.rigid_grasp is not None:                 # carry the grasped eraser under the EE
                pinch = self._ee_pinch(self.state_1)
                bq = self.state_1.body_q.numpy()
                bq[self.rigid_grasp["body"], :3] = pinch + self.rigid_grasp["off"]
                self.state_1.body_q.assign(bq)
            if self.grasp_active and self.pin_idx is not None:
                wp.launch(pin_to_ee, dim=len(self.pin_idx),
                          inputs=[self.state_1.body_q, self.endeffector_id, self.ee_off_vec,
                                  self.pin_idx, self.pin_off],
                          outputs=[self.state_0.particle_q, self.state_0.particle_qd])
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _pose_gripper(self):
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
        self._control(self.state_0)
        self._update_grasp()
        self._simulate()
        self._pose_gripper()
        self.sim_time += self.frame_dt

    def _setup_render(self):
        v = self.viewer
        v.set_model(self.model)
        v.show_triangles = False; v.show_particles = False
        self.tex_img = load_texture(os.path.join(self.capture_dir, "rgb.png"))
        self.g_uv = {k: wp.array(g["uvs"].astype(np.float32), dtype=wp.vec2) for k, g in self.garments.items()}
        self.g_fc = {k: wp.array(g["faces"].astype(np.int32), dtype=wp.int32) for k, g in self.garments.items()}

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
    ap.add_argument("--reach", type=float, default=82.0)
    ap.add_argument("--mask_step", type=int, default=9, help="cloth_from_mask sampling stride (px); smaller = finer/more realistic, more particles")
    ap.add_argument("--cell", type=float, default=2.0, help="[unused: legacy grid] cloth grid cell (cm)")
    ap.add_argument("--stretch_ke", type=float, default=1.0e4, help="cloth STRETCH stiffness (deformable)")
    ap.add_argument("--bend_ke", type=float, default=3.0, help="cloth BEND stiffness (low = drapes/folds)")
    ap.add_argument("--only_fold", default=None, help="debug: schedule ONLY this garment's fold (fast fidelity iteration)")
    ap.add_argument("--viewer", default="gl", choices=["gl", "null"])
    ap.add_argument("--hold", type=float, default=600.0)
    ap.add_argument("--render_every", type=int, default=2)
    ap.add_argument("--screenshot", default=None, help="PNG base; milestone frames saved at each release")
    ap.add_argument("--export_ee_traj", default=None, help="write the fold+pack EE trajectory (base frame, m) then exit")
    ap.add_argument("--cam", default=None)
    args = ap.parse_args()

    wp.init()
    viewer = ViewerGL() if args.viewer == "gl" else ViewerNull(num_frames=10**9)
    ex = PlanExec(viewer, args)
    if args.export_ee_traj:
        ex.export_ee_traj(args.export_ee_traj)
        return
    if args.viewer == "gl":
        if args.cam:
            cx, cy, cz, cp, cyaw = (float(v) for v in args.cam.split(","))
        else:
            cx, cy, cz, cp, cyaw = ex.tcx - 55.0, ex.tcy - 70.0, 60.0, -32.0, 55.0
        viewer.set_camera(wp.vec3(cx, cy, cz), cp, cyaw)

    n_frames = int(ex.total_time * ex.fps) + 60
    shots, base = {}, (args.screenshot.replace(".png", "") if args.screenshot else None)
    if base:
        grips = ex.key_poses[:, -1]
        rels = [i for i in range(1, len(grips)) if grips[i] > 0.4 >= grips[i - 1]]
        for k, i in enumerate(rels):
            shots[int(ex.key_times[i] * ex.fps)] = f"{base}_step{k+1}.png"
        shots[3] = f"{base}_start.png"
    print(f"[plan] running {n_frames} frames ({ex.total_time:.0f}s schedule)...", flush=True)
    for f in range(n_frames):
        ex.step()
        if f % args.render_every == 0 or f in shots:
            ex.render()
        q = ex.state_0.particle_q.numpy()
        if not np.isfinite(q).all():
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
