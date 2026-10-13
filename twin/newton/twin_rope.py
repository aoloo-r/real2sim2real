"""Dual-arm rope coiling twin: execute a tamp_rope.py plan in Newton and record it to USD.

Separate from the cloth twin (twin_stockfold.py) on purpose - nothing here touches the cloth path.

  rope   Newton cable: capsule bodies + cable joints (stretch/bend/twist), SolverVBD, real contact
         with the table, the box and itself. Units: cm, kg, s (as in the cloth twin).
  arms   two UR5e + Robotiq 2F-85, same bases as the cloth scene. They are KINEMATIC here: joint
         angles come from damped-least-squares IK on the tip pose and the links do not collide
         with anything. What the arms do to the rope is the grasp only.
  grasp  a kinematic pin: when the jaws close within ROPE_GRASP_TOL_CM of a rope segment, that
         segment follows the tip until release. A miss is reported, not papered over.

Usage: twin_rope.py --plan plan.json --out DIR      (writes DIR/rope.usd, DIR/rope_audit.json)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp

wp.config.quiet = True
import newton
from newton import ModelBuilder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ur5e_gripper import add_ur5e_gripper

E = os.environ.get
TABLE_TOP = 20.0
TABLE = ((0.0, -42.0, 10.0), 45.0, 52.0, 10.0)            # centre, half extents
BOX_XY = (0.0, -14.0)                                     # same place as the cloth scene's box
ARM_BASES = (("right", 30.0), ("left", -30.0))            # x; y = -2, z = table + 4
UR_HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
SEG = float(E("ROPE_SEG_CM", "2.5"))
GRASP_TOL = float(E("ROPE_GRASP_TOL_CM", "1.5"))
LAY_SPEED = float(E("ROPE_LAY_SPEED", "7"))               # cm of rope laid per second
MOVE_SPEED = float(E("ROPE_MOVE_SPEED", "28"))
FPS, SUBSTEPS = 60, int(E("ROPE_SUBSTEPS", "10"))
DT = 1.0 / FPS


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def rotvec(R):
    """Rotation vector of R (log map)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * v if a < 1e-6 else v * (a / (2.0 * math.sin(a)))


class Arms:
    """Two kinematic UR5e arms on a CPU model: FK by newton.eval_fk, IK by damped least squares."""

    def __init__(self):
        b = ModelBuilder()
        self.hands = {}
        for name, bx in ARM_BASES:
            xf = wp.transform((bx, -2.0, TABLE_TOP + 4.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0))
            j0, b0 = b.joint_count, b.body_count
            w3, _d0, _na, pinch = add_ur5e_gripper(b, xf, scale=100.0, home=False, loop_joints=False,
                                                   finger_cloth_contact=False)
            coords = [int(b.joint_q_start[j0 + i]) for i in range(6)]
            for i, c in enumerate(coords):
                b.joint_q[c] = UR_HOME[i]
            keys = getattr(b, "joint_key", None) or getattr(b, "joint_label", None)
            fd = []
            for j in range(j0, b.joint_count):
                k = str(keys[j])
                if "robotiq" not in k:
                    continue
                if k.endswith("driver_joint") or k.endswith("spring_link_joint"):
                    fd.append((int(b.joint_q_start[j]), 1.0))
                elif k.endswith("follower_joint"):
                    fd.append((int(b.joint_q_start[j]), -1.0))
            self.hands[name] = dict(name=name, ee=int(w3), pinch=np.array([float(v) for v in pinch]), coords=coords, fd=fd,
                                    bodies=list(range(b0, b.body_count)), base=np.array([bx, -2.0, TABLE_TOP + 4.0]),
                                    grip=0.0, grip_cur=0.0, yaw=0.0, busy=False)
        blabels = getattr(b, "body_key", None) or getattr(b, "body_label", None)
        self.model = b.finalize(device="cpu")
        self.state = self.model.state()
        self.q = self.model.joint_q.numpy().astype(np.float32).copy()
        self._qd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device="cpu")
        bq = self.fk(self.q)
        for h in self.hands.values():
            h["R_home"] = q2R(bq[h["ee"], 3:7])
            h["tip"], h["R"] = self._tip(h, bq)
            h["tgt"] = h["tip"].copy()
            # jaw axis at yaw 0, from the two finger pads
            pads = [i for i in h["bodies"] if "pad" in str(blabels[i])]
            ax = bq[pads[0], :3] - bq[pads[-1], :3] if len(pads) >= 2 else np.array([1.0, 0.0, 0.0])
            ax[2] = 0.0
            h["jaw0"] = math.atan2(ax[1], ax[0]) if np.linalg.norm(ax) > 1e-3 else 0.0
        self.body_q = bq

    def fk(self, q):
        newton.eval_fk(self.model, wp.array(q, dtype=wp.float32, device="cpu"), self._qd, self.state)
        return self.state.body_q.numpy().astype(float)

    @staticmethod
    def _tip(h, bq):
        R = q2R(bq[h["ee"], 3:7])
        return bq[h["ee"], :3] + R @ h["pinch"], R

    def step(self):
        """One frame: move every hand's joints toward its tip target (position + tool-down yaw)."""
        hs = list(self.hands.values())
        for _it in range(2):
            bq = self.fk(self.q)
            cur = [self._tip(h, bq) for h in hs]
            J = [np.zeros((6, 6)) for _ in hs]
            eps = 1e-3
            for i in range(6):
                q2 = self.q.copy()
                for h in hs:
                    q2[h["coords"][i]] += eps
                b2 = self.fk(q2)
                for k, h in enumerate(hs):
                    p2, R2 = self._tip(h, b2)
                    J[k][:3, i] = (p2 - cur[k][0]) / eps
                    J[k][3:, i] = rotvec(R2 @ cur[k][1].T) / eps
            for k, h in enumerate(hs):
                Rt = rz(h["yaw"]) @ h["R_home"]
                e = np.concatenate([h["tgt"] - cur[k][0], 20.0 * rotvec(Rt @ cur[k][1].T)])
                Jk = J[k].copy(); Jk[3:] *= 20.0                    # 1 rad weighs like 20 cm
                dq = Jk.T @ np.linalg.solve(Jk @ Jk.T + 0.25 * np.eye(6), e)
                self.q[h["coords"]] += np.clip(dq, -0.035, 0.035).astype(np.float32)
        for h in hs:
            h["grip_cur"] += float(np.clip(h["grip"] - h["grip_cur"], -0.08, 0.08))
            for c, sgn in h["fd"]:
                self.q[c] = sgn * 0.8 * h["grip_cur"]
        self.body_q = self.fk(self.q)
        for h in hs:
            h["tip"], h["R"] = self._tip(h, self.body_q)


def serpentine(L, y0, z, half_x=28.0, turn_r=6.0, centre_x=0.0):
    """Centreline of a rope dumped on the table in rows: points every SEG cm of arc length."""
    pts, x, y, d = [], centre_x - half_x, y0, 1.0
    dense = [np.array([x, y])]
    total = 0.0
    while total < L + 2 * SEG:
        for t in np.linspace(0.0, 1.0, 200)[1:]:
            dense.append(np.array([x + d * 2 * half_x * t, y]))
        x += d * 2 * half_x; total += 2 * half_x
        for a in np.linspace(0.0, math.pi, 80)[1:]:
            dense.append(np.array([x + d * turn_r * math.sin(a), y - turn_r * (1.0 - math.cos(a))]))
        y -= 2 * turn_r; total += math.pi * turn_r; d = -d
    dense = np.array(dense)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    for k in range(int(round(L / SEG)) + 1):
        pts.append(wp.vec3(float(np.interp(k * SEG, s, dense[:, 0])), float(np.interp(k * SEG, s, dense[:, 1])), z))
    return pts


class RopeTwin:
    def __init__(self, plan, out):
        self.plan, self.out = plan, out
        sc = plan["scene"]
        coil = plan.get("coil") or dict(radius_cm=8.0, loop_cm=50.0, hang_frac=0.85, centre_dxy=[0.0, 0.0], loops=0.0)
        self.task = plan.get("task", "coil_rope")
        self.table = (TABLE[0], float(sc.get("table_half_x_cm", TABLE[1])), TABLE[2], TABLE[3])
        self.L, self.rad = float(sc["rope"]["length_cm"]), 0.5 * float(sc["rope"]["diameter_cm"])
        self.L = SEG * 2 * round(self.L / SEG / 2)                 # even segment count: the middle is a joint
        self.N = int(round(self.L / SEG))
        self.hin = 0.5 * float(sc["box"]["inner_cm"][0])
        self.bh, self.wall = float(sc["box"]["h_cm"]), float(sc["box"].get("wall_cm", 1.0))
        self.floor = TABLE_TOP + 0.5
        self.rim = self.floor + self.bh
        self.lift_max = float(sc.get("lift_max_cm", 50.0))
        self.Rc, self.loop, self.hang = float(coil["radius_cm"]), float(coil["loop_cm"]), float(coil["hang_frac"])
        self.C = np.array([BOX_XY[0] + coil["centre_dxy"][0], BOX_XY[1] + coil["centre_dxy"][1]])
        self.F = self.C + np.array([0.0, -self.Rc])               # front point of the coil: every loop starts and ends here

        # ---- rope world -----------------------------------------------------------------
        b = ModelBuilder(gravity=-981.0)
        b.rigid_gap = 0.0
        stat = ModelBuilder.ShapeConfig(mu=float(E("ROPE_MU_TABLE", "0.6")), ke=1.0e5, kd=0.0)
        (tx, ty, tz), thx, thy, thz = self.table
        b.add_shape_box(-1, xform=wp.transform((tx, ty, tz), wp.quat_identity()), hx=thx, hy=thy, hz=thz, cfg=stat)
        bx, by, hin, w = BOX_XY[0], BOX_XY[1], self.hin, self.wall
        self.box_shapes = [dict(p=(bx, by, self.floor - 0.25), h=(hin + w, hin + w, 0.25), name="floor")]
        for nm, dx, dy, hx, hy in (("right", hin + w / 2, 0.0, w / 2, hin + w), ("left", -hin - w / 2, 0.0, w / 2, hin + w),
                                   ("back", 0.0, hin + w / 2, hin + w, w / 2), ("front", 0.0, -hin - w / 2, hin + w, w / 2)):
            self.box_shapes.append(dict(p=(bx + dx, by + dy, self.floor + self.bh / 2), h=(hx, hy, self.bh / 2), name=nm))
        for s in self.box_shapes:
            b.add_shape_box(-1, xform=wp.transform(s["p"], wp.quat_identity()), hx=s["h"][0], hy=s["h"][1], hz=s["h"][2], cfg=stat)
        cfg = ModelBuilder.ShapeConfig(density=float(E("ROPE_DENSITY", "4e-4")), ke=1.0e5, kd=0.0, mu=float(E("ROPE_MU", "0.6")))
        bend = float(E("ROPE_BEND", str(sc["rope"].get("bend", 1000))))                        # kg cm^2/s^2 per rad = 1e-4 N m: a 1 cm cotton rope (60 kinks like a chain)
        rows = sc["rope"].get("rows", {})
        if self.task == "sling":
            # straight along x through the work spot: the block will be set on its middle
            self.W = np.array([float(v) for v in sc.get("work_xy", [0.0, -50.0])])
            lay = plan.get("layout")
            if lay and lay["type"] == "keyhole":
                # doubled, with a round eye: end -> +y strand -> eye on the left -> -y strand -> end
                g, y0, tx, ex, r, nx = (float(lay[k]) for k in ("half_gap", "y", "tip_x", "ends_x", "eye_r", "neck_x"))
                cxe = tx + r
                dense = [np.array([x, y0 + g]) for x in np.linspace(ex, nx, 400)]
                dense += [np.array([cxe + r * math.cos(a), y0 + r * math.sin(a)]) for a in np.linspace(math.radians(60), math.radians(300), 120)]
                dense += [np.array([x, y0 - g]) for x in np.linspace(nx, ex, 400)]
                dense = np.array(dense)
                sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
                self.N = 2 * int(sl[-1] / SEG / 2)                 # the rope is as long as the layout needs
                self.L = self.N * SEG
                off = 0.5 * (sl[-1] - self.N * SEG)
                pts = [wp.vec3(float(np.interp(off + i * SEG, sl, dense[:, 0])), float(np.interp(off + i * SEG, sl, dense[:, 1])),
                               TABLE_TOP + self.rad + 0.1) for i in range(self.N + 1)]
            elif lay and lay["type"] == "straight":
                self.N = 2 * int((float(lay["x1"]) - float(lay["x0"])) / SEG / 2)
                self.L = self.N * SEG
                pts = [wp.vec3(float(lay["x0"]) + i * SEG, float(lay["y"]), TABLE_TOP + self.rad + 0.1) for i in range(self.N + 1)]
            elif lay and lay["type"] == "U":
                # doubled: end -> along +y strand to the bight on the left -> back along the -y strand
                g, y0, tx, ex = float(lay["half_gap"]), float(lay["y"]), float(lay["tip_x"]), float(lay["ends_x"])
                dense = [np.array([x, y0 + g]) for x in np.linspace(ex, tx + g, 400)]
                dense += [np.array([tx + g - g * math.sin(a), y0 + g * math.cos(a)]) for a in np.linspace(0.0, math.pi, 80)[1:]]
                dense += [np.array([x, y0 - g]) for x in np.linspace(tx + g, ex, 400)[1:]]
                dense = np.array(dense)
                sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
                off = 0.5 * (sl[-1] - self.N * SEG)                # centre the rope on the bight
                pts = [wp.vec3(float(np.interp(off + i * SEG, sl, dense[:, 0])), float(np.interp(off + i * SEG, sl, dense[:, 1])),
                               TABLE_TOP + self.rad + 0.1) for i in range(self.N + 1)]
            else:
                pts = [wp.vec3(float(self.W[0] - self.N * SEG / 2 + i * SEG), float(self.W[1]), TABLE_TOP + self.rad + 0.1) for i in range(self.N + 1)]
        else:
            pts = serpentine(self.N * SEG, y0=float(E("ROPE_Y0", "-46")), z=TABLE_TOP + self.rad + 0.1,
                             half_x=float(rows.get("half_x", 28.0)), centre_x=float(rows.get("centre_x", 0.0)))
        # built STRAIGHT (that is the rest shape the cable joints remember), then posed in rows
        straight = newton.utils.create_straight_cable_points(start=wp.vec3(0.0, 0.0, 200.0), direction=wp.vec3(1.0, 0.0, 0.0),
                                                             length=self.N * SEG, num_segments=self.N)
        self.rb, _ = b.add_rod(positions=straight, quaternions=newton.utils.create_parallel_transport_cable_quaternions(straight, twist_total=0.0),
                               radius=self.rad, cfg=cfg, stretch_stiffness=5.0e5, bend_stiffness=bend,
                               bend_damping=float(E("ROPE_BEND_DAMP", str(0.5 * bend))), label="rope", body_frame_origin="com")
        qs = newton.utils.create_parallel_transport_cable_quaternions(pts, twist_total=0.0)
        for i, body in enumerate(self.rb):
            mid = 0.5 * (np.array(pts[i]) + np.array(pts[i + 1]))
            b.body_q[body] = wp.transform(wp.vec3(*map(float, mid)), qs[i])
        # rigid blocks: dynamic bodies in the SAME world, so rope and blocks push on each other
        self.blocks = {}
        for g in sc.get("rigids", []) + ([sc["object"]] if "object" in sc else []):
            w, dd, hh = [float(v) for v in g["dims_cm"]]
            body = b.add_body(xform=wp.transform((float(g["start_xy"][0]), float(g["start_xy"][1]), TABLE_TOP + hh / 2 + 0.05), wp.quat_identity()),
                              label=g["name"])
            b.add_shape_box(body, hx=w / 2, hy=dd / 2, hz=hh / 2,
                            cfg=ModelBuilder.ShapeConfig(density=float(g.get("density_kg_cm3", E("ROPE_BLOCK_DENSITY", "5e-4"))), ke=1.0e5, kd=0.0,
                                                         mu=float(g.get("mu", 0.6))))
            self.blocks[g["name"]] = dict(body=int(body), dims=np.array([w, dd, hh]), plan=plan.get("rigids", {}).get(g["name"]))
        if self.task == "sling":
            self.sling_blk = self.blocks[sc["object"]["name"]]
            self.sling_blk["plan"] = dict(dxy=[float(self.W[0] - BOX_XY[0]), float(self.W[1] - BOX_XY[1])], z_on=0.0, role="slung",
                                          base_z=TABLE_TOP + 2.0 * self.rad)
        b.color()
        self.model = b.finalize()
        self.pipe = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.pipe.contacts()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=int(E("ROPE_ITERS", "5")),
                                               rigid_body_contact_buffer_size=256, rigid_contact_history=True)
        self.s0, self.s1, self.ctl = self.model.state(), self.model.state(), self.model.control()
        self.rb = np.asarray(self.rb, int)
        self.pins = {}                                            # hand name -> (body, offset in tip frame, R_rel)
        self.arms = Arms()
        for h in self.arms.hands.values():
            h["park"] = np.array([math.copysign(34.0, h["base"][0]), -46.0, 58.0])
        self.t, self.frame = 0.0, 0
        self.bg = []
        self.tip_min, self.both_in_box, self.miss, self.log = 1e9, 0, [], []
        self.tip_in_block, self.tool_through = 0, 0
        self.ik_err_max, self.ik_err_at = 0.0, None
        self._rec_init()
        print(f"[twin-rope] rope {self.L:.0f} cm x {2 * self.rad:.1f} cm, {self.N} segments ({float(self.model.body_mass.numpy()[self.rb[0]]) * self.N * 1000:.0f} g), "
              f"box inner {2 * self.hin:.0f} cm h {self.bh:.0f} at ({bx:.0f},{by:.0f}); coil R {self.Rc} at ({self.C[0]:.1f},{self.C[1]:.1f}), "
              f"loop {self.loop:.1f} cm; device {self.model.device}", flush=True)

    # ---- recording ----------------------------------------------------------------------
    def _rec_init(self):
        from pxr import Gf, UsdGeom
        from usd_recorder import UsdRecorder
        self.rec = UsdRecorder(self.arms.model, os.path.join(self.out, "rope.usd"), every=4, fps=FPS, table=self.table)
        st = self.rec.stage
        hide = set(E("ROPE_HIDE_WALLS", "").split(","))
        for s in self.box_shapes:
            c = UsdGeom.Cube.Define(st, f"/World/box/{s['name']}")
            c.CreateSizeAttr(1.0)
            c.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.62, 0.38)])
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, s["p"])))
            xf.AddScaleOp().Set(Gf.Vec3f(*[2.0 * float(v) for v in s["h"]]))
            if s["name"] in hide:                                 # the wall is still there in the physics
                UsdGeom.Imageable(c).MakeInvisible()
        self._blk_ops = {}
        for i, (nm, blk) in enumerate(self.blocks.items()):
            c = UsdGeom.Cube.Define(st, f"/World/blocks/{nm}")
            c.CreateSizeAttr(1.0)
            c.CreateDisplayColorAttr([[Gf.Vec3f(0.85, 0.30, 0.25), Gf.Vec3f(0.25, 0.50, 0.80), Gf.Vec3f(0.30, 0.65, 0.35)][i % 3]])
            xf = UsdGeom.Xformable(c)
            self._blk_ops[nm] = xf.AddTransformOp()
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float, blk["dims"])))
        K, M = 10, 3                                                # ring sides; render samples per segment
        m = UsdGeom.Mesh.Define(st, "/World/rope")
        idx = []
        for i in range(self.N * M):
            for k in range(K):
                a, bb = i * K + k, i * K + (k + 1) % K
                idx += [a, bb, bb + K, a + K]
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateFaceVertexCountsAttr([4] * (self.N * M * K))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(True)
        cols = [Gf.Vec3f(0.93, 0.80, 0.52) if (i // 2) % 2 == 0 else Gf.Vec3f(0.78, 0.33, 0.20) for i in range(self.N) for _ in range(M * K)]
        m.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(cols)
        self._rope_pts, self._K, self._M = m.GetPointsAttr(), K, M

    def _rope_nodes(self, bq):
        p = bq[self.rb, :3]
        z = np.array([q2R(q)[:, 2] for q in bq[self.rb, 3:7]])
        return np.vstack([p - 0.5 * SEG * z, (p[-1] + 0.5 * SEG * z[-1])[None]])

    def _rec_sample(self, bq):
        from pxr import Gf, Usd
        if (self.rec.frame + 1) % self.rec.every == 0 and np.isfinite(bq).all():
            c = self._rope_nodes(bq)
            # the drawn centreline is a Catmull-Rom curve through the capsule end points (the
            # simulated rope is the polyline; this only rounds its corners in the picture)
            e = np.vstack([2 * c[0] - c[1], c, 2 * c[-1] - c[-2]])
            out = []
            for j in range(self._M):
                u = j / self._M
                out.append(0.5 * ((2 * e[1:-2]) + (-e[:-3] + e[2:-1]) * u + (2 * e[:-3] - 5 * e[1:-2] + 4 * e[2:-1] - e[3:]) * u * u
                                  + (-e[:-3] + 3 * e[1:-2] - 3 * e[2:-1] + e[3:]) * u ** 3))
            c = np.vstack([np.stack(out, 1).reshape(-1, 3), c[-1][None]])
            t = np.gradient(c, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            n = np.cross(t[0], [0.0, 0.0, 1.0]); n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else np.array([1.0, 0.0, 0.0])
            ring = []
            ang = np.linspace(0.0, 2 * math.pi, self._K, endpoint=False)
            for i in range(len(c)):
                n = n - t[i] * float(n @ t[i]); n /= max(np.linalg.norm(n), 1e-9)       # parallel transport
                bn = np.cross(t[i], n)
                ring.append(c[i][None] + self.rad * (np.cos(ang)[:, None] * n[None] + np.sin(ang)[:, None] * bn[None]))
            self._rope_pts.Set([Gf.Vec3f(*map(float, v)) for v in np.vstack(ring)], Usd.TimeCode(self.rec.t))
            for nm, blk in self.blocks.items():
                self._blk_ops[nm].Set(self.rec._mat(bq[blk["body"]]), Usd.TimeCode(self.rec.t))
        self.rec.sample(self.arms.state, 0)

    # ---- simulation ---------------------------------------------------------------------
    def seg_pos(self):
        return self.s0.body_q.numpy()[self.rb, :3].astype(float)

    def _set_kin(self, body, on):
        fl = self.model.body_flags.numpy()
        fl[body] = int(newton.BodyFlags.KINEMATIC if on else newton.BodyFlags.DYNAMIC)
        self.model.body_flags.assign(fl)
        self.solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    def step_frame(self):
        hs = self.arms.hands
        old = {n: (h["tip"].copy(), h["R"].copy()) for n, h in hs.items()}
        self.bg = [g for g in self.bg if next(g, "done") != "done"]
        self.arms.step()
        for h in hs.values():
            _e = float(np.linalg.norm(h["tip"] - h["tgt"]))
            if _e > self.ik_err_max:
                self.ik_err_max, self.ik_err_at = _e, (round(self.t, 1), h["name"], [round(float(v), 1) for v in h["tgt"]])
        for k in range(SUBSTEPS):
            if self.pins:
                a = (k + 1) / SUBSTEPS
                bq, bv = self.s0.body_q.numpy(), self.s0.body_qd.numpy()
                for key, (body, off, Rrel) in self.pins.items():
                    n = key.split("#")[0]
                    h = hs[n]
                    tip = old[n][0] + (h["tip"] - old[n][0]) * a
                    bq[body, :3] = tip + h["R"] @ off
                    bq[body, 3:7] = R2q(h["R"] @ Rrel)
                    bv[body] = 0.0
                    bv[body, :3] = (h["tip"] - old[n][0]) / DT
                self.s0.body_q.assign(bq); self.s0.body_qd.assign(bv)
            self.s0.clear_forces()
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.ctl, self.contacts, DT / SUBSTEPS)
            self.s0, self.s1 = self.s1, self.s0
        bq = self.s0.body_q.numpy()
        if not np.isfinite(bq).all():
            raise RuntimeError(f"rope diverged (NaN) at t={self.t:.2f}s")
        self._rec_sample(bq)
        tips = [h["tip"] for h in hs.values()]
        self.tip_min = min(self.tip_min, float(np.linalg.norm(tips[0] - tips[1])))
        if all(abs(p[0] - BOX_XY[0]) < self.hin + self.wall and abs(p[1] - BOX_XY[1]) < self.hin + self.wall and p[2] < self.rim + 3.0 for p in tips):
            self.both_in_box += 1
        # the arms do not collide with the rope, so count the frames in which a gripper's tool axis
        # (tip -> wrist) cuts through rope it is not holding or reaching for
        rp = bq[self.rb, :3]
        for n, h in hs.items():
            a0, a1 = h["tip"], self.arms.body_q[h["ee"], :3]
            ax = a1 - a0
            tpar = np.clip(((rp - a0) @ ax) / max(float(ax @ ax), 1e-9), 0.0, 1.0)
            dist = np.linalg.norm(rp - (a0 + tpar[:, None] * ax), axis=1)
            for key in (n, n + "#2"):
                if key in self.pins and self.pins[key][0] in self.rb:
                    k0 = int(np.where(self.rb == self.pins[key][0])[0][0])
                    dist[max(0, k0 - 5):k0 + 6] = 99.0
            if h.get("aim_seg") is not None:
                dist[max(0, h["aim_seg"] - 5):h["aim_seg"] + 6] = 99.0
            if dist.min() < float(E("ROPE_TOOL_CLEAR_CM", "1.5")):
                self.tool_through += 1
        for blk in self.blocks.values():
            Rb, pb = q2R(bq[blk["body"], 3:7]), bq[blk["body"], :3]
            held = [n.split("#")[0] for n, pn in self.pins.items() if pn[0] == blk["body"]]
            for n, h in hs.items():
                if n not in held and h.get("picking") != blk["body"] and np.all(np.abs(Rb.T @ (h["tip"] - pb)) < blk["dims"] / 2 - 0.3):
                    self.tip_in_block += 1
        if E("ROPE_DEBUG") and getattr(self, "sling_blk", None) and self.frame % 12 == 0 and float(E("ROPE_DEBUG_T0", "0")) <= self.t <= float(E("ROPE_DEBUG_T1", "1e9")):
            pb, tl = self._blk_state(self.sling_blk)
            gaps = np.linalg.norm(np.diff(rp, axis=0), axis=1)
            hh_ = [h for h in hs.values() if h["name"] in self.pins]
            tp = hh_[0]["tip"] if hh_ else np.zeros(3)
            print(f"[dbg] t={self.t:5.2f} tip ({tp[0]:6.1f},{tp[1]:6.1f},{tp[2]:5.1f}) block ({pb[0]:5.1f},{pb[1]:6.1f},{pb[2]:5.1f}) tilt {tl:4.0f} "
                  f"max seg gap {gaps.max():.2f} at {int(gaps.argmax())}  rope z max {rp[:, 2].max():.1f}", flush=True)
            if E("ROPE_DEBUG_DUMP"):
                print("[dump] " + " ".join(f"{i}:({q_[0]:.0f},{q_[1]:.0f},{q_[2]:.0f})" for i, q_ in enumerate(rp[:int(E("ROPE_DEBUG_DUMP"))])), flush=True)
        self.t += DT; self.frame += 1

    # ---- motion primitives (generators: one yield per frame) -----------------------------
    def g_move(self, h, goal, speed=MOVE_SPEED, yaw=None, tol=0.5, anchor=None, radius=None):
        start, goal = h["tgt"].copy(), np.asarray(goal, float)
        n = max(1, int(math.ceil(np.linalg.norm(goal - start) / speed * FPS)))
        y0 = h["yaw"]
        for i in range(1, n + 1):
            a = i / n; a = a * a * (3 - 2 * a)
            p = start + (goal - start) * a
            if anchor is not None and np.linalg.norm(p - anchor) > radius:
                p = anchor + (p - anchor) * radius / np.linalg.norm(p - anchor)
            h["tgt"] = p
            if yaw is not None:
                h["yaw"] = y0 + (yaw - y0) * a
            yield
        for _ in range(120):
            if np.linalg.norm(h["tip"] - h["tgt"]) < tol:
                break
            yield

    def g_wait(self, sec):
        for _ in range(int(sec * FPS)):
            yield

    def g_grasp(self, h, seg, tag):
        """Close the jaws on rope segment `seg`; pins it only if the tip really is on it."""
        h["aim_seg"] = int(seg)
        for attempt in range(3):
            p = self.seg_pos()
            lo, hi = max(0, seg - 1), min(self.N - 1, seg + 1)
            tan = p[hi] - p[lo]
            want = math.atan2(tan[1], tan[0]) + math.pi / 2.0 - h["jaw0"]      # jaws across the rope
            yaw = h["yaw"] + (want - h["yaw"] + math.pi / 2) % math.pi - math.pi / 2
            yield from self.g_move(h, p[seg] + np.array([0.0, 0.0, 12.0]), yaw=yaw)
            p = self.seg_pos()
            yield from self.g_move(h, p[seg] + np.array([0.0, 0.0, 0.2]), speed=12.0, tol=0.3)
            h["grip"] = 1.0
            yield from self.g_wait(0.35)
            p = self.seg_pos()
            d = float(np.linalg.norm(h["tip"] - p[seg]))
            where = self._where(p[seg])
            if d <= GRASP_TOL:
                body = int(self.rb[seg])
                self.pins[h["name"]] = (body, h["R"].T @ (p[seg] - h["tip"]), h["R"].T @ q2R(self.s0.body_q.numpy()[body, 3:7]))
                self._set_kin(body, True)
                print(f"[twin-rope] t={self.t:5.1f}s {tag}: {h['name']} grasped s={seg * SEG + SEG / 2:.0f} cm ({where}) "
                      f"at ({p[seg][0]:.1f},{p[seg][1]:.1f},{p[seg][2]:.1f}), tip-to-rope {d:.2f} cm", flush=True)
                self.log.append(dict(t=round(self.t, 2), tag=tag, hand=h["name"], seg=int(seg), where=where, grasp_err_cm=round(d, 3)))
                return
            print(f"[twin-rope] t={self.t:5.1f}s {tag}: {h['name']} MISSED s={seg * SEG:.0f} cm by {d:.2f} cm (attempt {attempt + 1})", flush=True)
            self.miss.append(dict(t=round(self.t, 2), tag=tag, err_cm=round(d, 3)))
            h["grip"] = 0.0
            yield from self.g_wait(0.3)
        raise RuntimeError(f"{tag}: could not grasp segment {seg}")

    def g_release(self, h, rise=16.0):
        if h["name"] + "#2" in self.pins:
            self._set_kin(self.pins.pop(h["name"] + "#2")[0], False)
        h["aim_seg"] = int(np.where(self.rb == self.pins[h["name"]][0])[0][0]) if h["name"] in self.pins and self.pins[h["name"]][0] in self.rb else h.get("aim_seg")
        if h["name"] in self.pins:
            self._set_kin(self.pins.pop(h["name"])[0], False)
        h["grip"] = 0.0
        yield from self.g_wait(0.35)
        yield from self.g_move(h, h["tgt"] + np.array([0.0, 0.0, rise]), speed=20.0)

    def g_park(self, h):
        h["busy"] = True
        yield from self.g_move(h, h["park"], yaw=0.0)
        h["aim_seg"] = None
        h["busy"] = False

    def _where(self, p):
        inx, iny = abs(p[0] - BOX_XY[0]), abs(p[1] - BOX_XY[1])
        if inx < self.hin and iny < self.hin:
            return "in box"
        if inx < self.hin + self.wall + 1.0 and iny < self.hin + self.wall + 1.0:
            return "on the wall"
        return "on the table" if p[2] < TABLE_TOP + 3.0 else "hanging outside"

    def g_pick_place(self, h, name):
        """Top grasp of a block (jaws across its narrow side), carry over the rim, set down, release."""
        while h["busy"]:
            yield
        blk = self.blocks[name]
        body, (w, dd, hh), pl = blk["body"], blk["dims"], blk["plan"]
        h["picking"] = body                                          # the pinch point is inside this block by design
        bq = self.s0.body_q.numpy()[body].astype(float)
        want = (0.0 if w <= dd else math.pi / 2.0) - h["jaw0"]
        yaw = h["yaw"] + (want - h["yaw"] + math.pi / 2) % math.pi - math.pi / 2
        grip_dz = max(0.0, hh / 2 - 2.5)                             # tip 2.5 cm below the top face
        yield from self.g_move(h, bq[:3] + np.array([0.0, 0.0, hh / 2 + 12.0]), yaw=yaw)
        yield from self.g_move(h, bq[:3] + np.array([0.0, 0.0, grip_dz]), speed=12.0, tol=0.3)
        h["grip"] = 0.6
        yield from self.g_wait(0.35)
        bq = self.s0.body_q.numpy()[body].astype(float)
        err = float(np.linalg.norm((h["tip"] - bq[:3])[:2]))
        if err > GRASP_TOL:
            raise RuntimeError(f"pick {name}: tip {err:.1f} cm off the block centre")
        self.pins[h["name"]] = (body, h["R"].T @ (bq[:3] - h["tip"]), h["R"].T @ q2R(bq[3:7]))
        self._set_kin(body, True)
        goal = np.array([BOX_XY[0] + pl["dxy"][0], BOX_XY[1] + pl["dxy"][1], pl.get("base_z", self.floor) + pl["z_on"] + hh / 2 + grip_dz + 0.3])
        top = (self.rim if "base_z" not in pl else pl["base_z"]) + hh + 8.0
        yield from self.g_move(h, np.array([h["tgt"][0], h["tgt"][1], top]), speed=20.0)
        yield from self.g_move(h, np.array([goal[0], goal[1], top]), speed=22.0)
        # set down on whatever is there now (rope may have crept into the spot): stop at first contact height
        yield from self.g_move(h, goal, speed=10.0)
        yield from self.g_release(h)
        h["picking"] = None
        p = self.s0.body_q.numpy()[body, :3]
        print(f"[twin-rope] t={self.t:5.1f}s pick_place {name} ({pl['role']}): {h['name']} set it at ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}), "
              f"plan ({goal[0]:.1f},{goal[1]:.1f}), grasp {err:.2f} cm off centre", flush=True)
        self.log.append(dict(t=round(self.t, 2), tag=f"pick_place {name}", hand=h["name"], grasp_err_cm=round(err, 3)))
        self.bg.append(self.g_park(h))

    # ---- sling: wrap the rope around a block and lift the block by the rope ---------------
    def g_par(self, *gens):
        gens = list(gens)
        while gens:
            gens = [g for g in gens if next(g, "done") != "done"]
            yield

    def _blk_state(self, blk):
        bq = self.s0.body_q.numpy()[blk["body"]].astype(float)
        return bq[:3], math.degrees(math.acos(float(np.clip(q2R(bq[3:7])[2, 2], -1.0, 1.0))))

    def g_drape(self, h, st):
        """Carry one rope end over the block and lay it on the table beyond the far face."""
        while h["busy"]:
            yield
        seg, anc, side = self._seg(st["s_cm"]), self._seg(st["anchor_s_cm"]), float(st["to_side"])
        w, _, hh = self.sling_blk["dims"]
        yield from self.g_grasp(h, seg, "drape")
        P = self.seg_pos()[anc]                                     # where this strand comes out from under the block
        arc = abs(seg - anc) * SEG
        bp, _ = self._blk_state(self.sling_blk)
        top = bp[2] + hh / 2
        cap = TABLE_TOP + 52.0
        dy = float(st.get("land_dy", 0.0))
        for goal in ([P[0] - side * 2.0, P[1], min(TABLE_TOP + 0.85 * arc, cap)],
                     [bp[0] + side * (w / 2 + 3.0), P[1] + 0.5 * dy, min(top + max(4.0, 0.8 * (arc - hh - w - 3.0)), cap)],
                     [bp[0] + side * (w / 2 + float(st["land_cm"])), P[1] + dy, TABLE_TOP + self.rad + 0.6]):
            yield from self.g_move(h, np.array(goal), speed=14.0, anchor=P, radius=0.95 * arc)
        yield from self.g_release(h)
        bp2, tilt = self._blk_state(self.sling_blk)
        print(f"[twin-rope] t={self.t:5.1f}s drape: {h['name']} laid its end at x={self.seg_pos()[seg][0]:.1f}; block moved "
              f"{np.linalg.norm(bp2[:2] - bp[:2]):.1f} cm, tilt {tilt:.0f} deg", flush=True)
        self.bg.append(self.g_park(h))

    def g_bight_over(self, h, st):
        """Carry the bight (the rope's middle) over the block and lay it across the two ends."""
        while h["busy"]:
            yield
        blk = self.sling_blk
        w, _, hh = blk["dims"]
        mid = self.N // 2
        yield from self.g_grasp(h, mid, "bight_over")
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        j = min(range(2, mid), key=lambda i: abs(p[i][0] - (bp[0] - w / 2)))   # where a strand leaves the block's underside
        arc = (mid - j) * SEG
        P = np.array([bp[0] - w / 2, bp[1], TABLE_TOP])
        top, cap = bp[2] + hh / 2, TABLE_TOP + 52.0
        for goal in ([P[0] - 2.0, P[1], min(TABLE_TOP + 0.85 * arc, cap)],
                     [bp[0] + w / 2 + 3.0, P[1], min(top + max(4.0, 0.8 * (arc - hh - w - 3.0)), cap)],
                     [bp[0] + w / 2 + float(st["land_cm"]), P[1], TABLE_TOP + 3.0 * self.rad + 0.6]):
            yield from self.g_move(h, np.array(goal), speed=12.0, anchor=P, radius=0.95 * arc)
        yield from self.g_release(h)
        bp2, tilt = self._blk_state(blk)
        tip = self.seg_pos()[mid]
        print(f"[twin-rope] t={self.t:5.1f}s bight_over: bight tip laid at ({tip[0]:.1f},{tip[1]:.1f},{tip[2]:.1f}), {tip[0] - (bp2[0] + w / 2):.1f} cm beyond the face; "
              f"block moved {np.linalg.norm(bp2[:2] - bp[:2]):.1f} cm, tilt {tilt:.0f} deg", flush=True)
        self.bg.append(self.g_park(h))

    def _tip(self):
        """Segment at the eye's tip: where the robot's fold actually put it, else the rope's middle."""
        return int(getattr(self, "fold", {}).get("tip_seg", self.N // 2))

    def _strand(self, name):
        return range(1, self._tip() - 3) if name == "P" else range(self._tip() + 3, self.N - 1)

    def g_end_over(self, h, st):
        """Lay one rope end over the block: up the right face, across the top, down the left face."""
        while h["busy"]:
            yield
        blk = self.sling_blk
        w, _, hh = blk["dims"]
        strand = self._strand(st["strand"])
        seg = strand[0] if st["strand"] == "P" else strand[-1]      # 1 segment in from the end
        yield from self.g_grasp(h, seg, f"end_over {st['strand']}")
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        anc = min(strand, key=lambda i: abs(p[i][0] - (bp[0] + w / 2)))  # where the strand leaves the block's underside
        P, arc = p[anc].copy(), abs(seg - anc) * SEG
        top, cap = bp[2] + hh / 2, TABLE_TOP + 52.0
        y_land = bp[1] + float(st["land_y"])
        # Setting the end down 4 cm from the face wedged the rope between the descending jaws and the
        # block's top corner and rolled the block over (tilt 90 deg, with and without turning the
        # hand). The plan now lands it 8 cm out; ROPE_END_TURN=1 also turns the hand half a revolution.
        yaw0 = h["yaw"]
        turn = math.pi if E("ROPE_END_TURN", "0") == "1" else 0.0
        for goal, yaw in (([P[0] + 2.0, P[1], min(TABLE_TOP + 0.85 * arc, cap)], yaw0),
                          ([bp[0] - w / 2 - 3.0, y_land, min(top + max(4.0, 0.8 * (arc - hh - w - 3.0)), cap)], yaw0 + turn),
                          ([bp[0] - w / 2 - float(st["land_cm"]), y_land, TABLE_TOP + 3.0 * self.rad + 0.6], yaw0 + turn)):
            yield from self.g_move(h, np.array(goal), speed=12.0, yaw=yaw, anchor=P, radius=0.95 * arc)
        yield from self.g_release(h)
        bp2, tilt = self._blk_state(blk)
        e = self.seg_pos()[seg]
        print(f"[twin-rope] t={self.t:5.1f}s end_over {st['strand']}: end laid at ({e[0]:.1f},{e[1]:.1f},{e[2]:.1f}); block moved "
              f"{np.linalg.norm(bp2[:2] - bp[:2]):.1f} cm, tilt {tilt:.0f} deg", flush=True)
        self.bg.append(self.g_park(h))

    def _pinch_two(self, h, segs, tag, squeeze=True):
        """Close the jaws on two strands lying side by side: both are pinned, and (as closing jaws
        would) pressed together to one rope diameter apart."""
        p = self.seg_pos()
        aim = 0.5 * (p[segs[0]] + p[segs[1]])
        h["aim_seg"] = int(segs[0])
        tan = p[min(self.N - 1, segs[0] + 1)] - p[max(0, segs[0] - 1)]
        want = math.atan2(tan[1], tan[0]) + math.pi / 2.0 - h["jaw0"]
        yaw = h["yaw"] + (want - h["yaw"] + math.pi / 2) % math.pi - math.pi / 2
        yield from self.g_move(h, aim + np.array([0.0, 0.0, 12.0]), yaw=yaw)
        p = self.seg_pos()
        aim = 0.5 * (p[segs[0]] + p[segs[1]])
        yield from self.g_move(h, aim + np.array([0.0, 0.0, 0.2]), speed=10.0, tol=0.3)
        h["grip"] = 1.0
        yield from self.g_wait(0.35)
        p = self.seg_pos()
        offs = [float(np.linalg.norm(p[sg] - h["tip"])) for sg in segs]
        if max(offs) > float(E("ROPE_TWO_STRAND_TOL_CM", "3.0")):
            raise RuntimeError(f"{tag}: the two strands are {offs[0]:.1f} and {offs[1]:.1f} cm from the jaws - cannot hold both")
        bqn = self.s0.body_q.numpy()
        keys = (h["name"], h["name"] + "#2")
        for key, sg in zip(keys, segs):
            body = int(self.rb[sg])
            self.pins[key] = (body, h["R"].T @ (p[sg] - h["tip"]), h["R"].T @ q2R(bqn[body, 3:7]))
            self._set_kin(body, True)
        if squeeze:
            start = [self.pins[k][1].copy() for k in keys]
            goal = [o * min(1.0, (self.rad + 0.1) / max(float(np.linalg.norm(o)), 1e-6)) for o in start]
            for i in range(1, 19):
                for k, o0, o1 in zip(keys, start, goal):
                    self.pins[k] = (self.pins[k][0], o0 + (o1 - o0) * i / 18.0, self.pins[k][2])
                yield
        print(f"[twin-rope] t={self.t:5.1f}s {tag}: {h['name']} pinched two strands ({offs[0]:.1f} and {offs[1]:.1f} cm from the jaw centre) at "
              f"({h['tip'][0]:.1f},{h['tip'][1]:.1f},{h['tip'][2]:.1f})", flush=True)
        self.log.append(dict(t=round(self.t, 2), tag=tag, hand=h["name"], grasp_err_cm=round(max(offs), 3)))

    def g_ends_over(self, h, st):
        """Pinch both rope ends together and lay them over the block as a pair."""
        while h["busy"]:
            yield
        blk = self.sling_blk
        w, _, hh = blk["dims"]
        sP, sM = self._strand("P"), self._strand("M")
        # the fold rarely lands exactly on the rope's middle, so one end reaches further than the
        # other: take both strands side by side at the SHORTER end; the longer one keeps a tail
        p = self.seg_pos()
        e = sP[1] if p[sP[1]][0] <= p[sM[-2]][0] else sM[-2]
        o = sM if e == sP[1] else sP
        segs = sorted([e, min(o, key=lambda i: abs(p[i][0] - p[e][0]))])
        yield from self._pinch_two(h, segs, "ends_over")
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        anc = min(sP, key=lambda i: abs(p[i][0] - (bp[0] + w / 2)))
        P, arc = np.array([bp[0] + w / 2, bp[1], TABLE_TOP]), abs(segs[0] - anc) * SEG
        top, cap = bp[2] + hh / 2, TABLE_TOP + 52.0
        for goal in ([P[0] + 2.0, P[1], min(TABLE_TOP + 0.85 * arc, cap)],
                     [bp[0] - w / 2 - 3.0, P[1], min(top + max(4.0, 0.8 * (arc - hh - w - 3.0)), cap)],
                     [bp[0] - w / 2 - float(st["land_cm"]), P[1], TABLE_TOP + 3.0 * self.rad + 0.6]):
            yield from self.g_move(h, np.array(goal), speed=12.0, anchor=P, radius=0.95 * arc)
        yield from self.g_release(h)
        yield from self.g_wait(0.5)
        p, (bp2, tilt) = self.seg_pos(), self._blk_state(blk)
        on_top = lambda rng: [p[i][1] for i in rng if abs(p[i][2] - top) < 2.5 and abs(p[i][0] - bp2[0]) < 2.5]
        yp, ym = on_top(sP), on_top(sM)
        sep = abs(np.mean(yp) - np.mean(ym)) if yp and ym else float("nan")
        print(f"[twin-rope] t={self.t:5.1f}s ends_over: both ends laid over the block, {sep:.1f} cm apart where they cross the top's mid-line; "
              f"block moved {np.linalg.norm(bp2[:2] - bp[:2]):.1f} cm, tilt {tilt:.0f} deg", flush=True)
        self.bg.append(self.g_park(h))

    def g_fold_rope(self, h, st):
        """Fold the straight rope in half on the table: carry the left end over to the right and lay
        the left half beside the right half. Two hands: the left arm cannot reach the end's place."""
        h2 = self.arms.hands[st["then"]]
        y_lay, mid = float(st["y_lay"]), self.N // 2
        x_mid = float(self.seg_pos()[mid][0])
        up = TABLE_TOP + float(E("ROPE_FOLD_UP_CM", "9"))
        yield from self.g_grasp(h, 1, "fold_rope")
        x_h = x_mid + float(st["handover_dx"])
        for goal in ([h["tgt"][0], h["tgt"][1], up], [x_h, y_lay, up], [x_h, y_lay, TABLE_TOP + self.rad + 0.5]):
            yield from self.g_move(h, np.array(goal), speed=14.0)
        yield from self.g_release(h)
        self.bg.append(self.g_park(h))
        while h2["busy"]:
            yield
        yield from self.g_grasp(h2, 1, "fold_rope")
        p = self.seg_pos()
        # carry the end until it lies beside the OTHER end: the rope then doubles back at its middle
        # (aiming from where the fold stood at hand-over left the eye 10 cm off the middle)
        x_end = float(p[-2][0]) - float(E("ROPE_FOLD_SHORT_CM", "1.0"))
        for goal in ([h2["tgt"][0], y_lay, up], [x_end, y_lay, up], [x_end, y_lay, TABLE_TOP + self.rad + 0.5]):
            yield from self.g_move(h2, np.array(goal), speed=14.0)
        yield from self.g_release(h2)
        yield from self.g_wait(0.5)
        p = self.seg_pos()
        tip = int(np.argmin(p[:, 0]))
        self.fold = dict(tip_x=float(p[tip][0]), tip_y=float(p[tip][1]), tip_seg=tip)
        xs = self.fold["tip_x"] + 20.0
        yP = float(p[min(self._strand("P"), key=lambda i: abs(p[i][0] - xs))][1])
        yM = float(p[min(self._strand("M"), key=lambda i: abs(p[i][0] - xs))][1])
        eye_w = float(np.ptp(p[max(0, tip - 4):tip + 5, 1]))
        print(f"[twin-rope] t={self.t:5.1f}s fold_rope: rope doubled; eye tip at ({p[tip][0]:.1f},{p[tip][1]:.1f}) (segment {tip}, middle is {mid}), eye {eye_w:.1f} cm wide, "
              f"strands {abs(yP - yM):.1f} cm apart 20 cm from the tip, ends at x {p[1][0]:.1f} and {p[-2][0]:.1f}", flush=True)
        self.bg.append(self.g_park(h2))

    def g_gather_ends(self, h, st):
        """Dress the rope: slide one end, where it lies on the block's top, next to the other, so
        the eye's tip (only ~4 cm wide once it hangs from the jaws) can lie across both."""
        while h["busy"]:
            yield
        blk = self.sling_blk
        hh = blk["dims"][2]
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        top = bp[2] + hh / 2
        on_top = lambda rng: [i for i in rng if abs(p[i][2] - top) < 2.5 and abs(p[i][0] - bp[0]) < 2.5]
        mv, ref = on_top(self._strand(st["move"])), on_top(self._strand(st["beside"]))
        if not mv or not ref:
            raise RuntimeError("gather_ends: an end is not lying across the block's top")
        seg = min(mv, key=lambda i: abs(p[i][0] - bp[0]))
        y_ref = float(np.mean([p[i][1] for i in ref]))
        y_goal = y_ref + math.copysign(float(st["gap_cm"]), p[seg][1] - y_ref)
        before = abs(p[seg][1] - y_ref)
        yield from self.g_grasp(h, seg, "gather_ends")
        for goal in ([h["tgt"][0], h["tgt"][1], top + 3.0], [h["tgt"][0], y_goal, top + 3.0], [h["tgt"][0], y_goal, top + self.rad + 0.3]):
            yield from self.g_move(h, np.array(goal), speed=8.0)
        yield from self.g_release(h)
        yield from self.g_wait(0.5)
        p = self.seg_pos()
        print(f"[twin-rope] t={self.t:5.1f}s gather_ends: the two ends were {before:.1f} cm apart on the top, now {abs(p[seg][1] - y_ref):.1f} cm", flush=True)
        self.bg.append(self.g_park(h))

    def g_eye_on_top(self, h, st):
        """Lift the eye by its tip and lay it on the block's top, over the two ends, tip at the centre."""
        while h["busy"]:
            yield
        blk = self.sling_blk
        w, _, hh = blk["dims"]
        mid = self._tip()
        yield from self.g_grasp(h, mid, "eye_on_top")
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        top = bp[2] + hh / 2

        def y_on_top(rng, x0):
            c = [i for i in rng if abs(p[i][2] - top) < 2.5 and abs(p[i][0] - x0) < 2.0]
            return float(np.mean([p[i][1] for i in c])) if c else None

        # FEEDBACK: the ends do not lie where the plan drew them (each one shifts sideways where it
        # leaves the underside). Measure where the two ends cross the top's mid-line and put the
        # eye's tip across both of them.
        ends = [y_on_top(self._strand("P"), bp[0]), y_on_top(self._strand("M"), bp[0])]
        if None in ends:
            raise RuntimeError(f"eye_on_top: an end is not lying across the block's top (y at the mid-line: {ends})")
        y_tip = 0.5 * (ends[0] + ends[1])
        x_tip = bp[0] + float(E("ROPE_EYE_DX", "0.5"))
        eye = range(mid - 4, mid + 5)
        tries = 1 + int(E("ROPE_EYE_RETRY", "0"))               # re-laying chased the ends in tests (they move with the eye); off by default
        for attempt in range(tries):
            for goal in ([h["tgt"][0], y_tip, top + 5.0], [x_tip, y_tip, top + 5.0], [x_tip, y_tip, top + 3.0 * self.rad + 0.3]):
                yield from self.g_move(h, np.array(goal), speed=10.0)
            yield from self.g_release(h)
            yield from self.g_wait(0.5)
            p, (bp2, tilt) = self.seg_pos(), self._blk_state(blk)
            ends = [y_on_top(self._strand("P"), p[mid][0] - 1.0), y_on_top(self._strand("M"), p[mid][0] - 1.0)]
            lo, hi = min(p[i][1] for i in eye), max(p[i][1] for i in eye)   # the tip's span in y
            self.threaded = bool(None not in ends and all(lo + 0.3 < y < hi - 0.3 for y in ends))
            if self.threaded or None in ends or attempt == tries - 1:
                break
            # CHECK AND CORRECT: the tip landed to one side of the two ends - pick it up and re-lay it
            # shifted by the measured error
            err = 0.5 * (ends[0] + ends[1]) - 0.5 * (lo + hi)
            print(f"[twin-rope] t={self.t:5.1f}s eye_on_top: tip spans y {lo:.1f}..{hi:.1f} but the ends are at {ends[0]:.1f} and {ends[1]:.1f} - re-laying it {err:+.1f} cm", flush=True)
            yield from self.g_grasp(h, mid, "eye_on_top (correct)")
            y_tip = float(h["tip"][1]) + err
            yield from self.g_move(h, np.array([h["tgt"][0], h["tgt"][1], top + 5.0]), speed=10.0)
        print(f"[twin-rope] t={self.t:5.1f}s eye_on_top: tip at ({p[mid][0]:.1f},{p[mid][1]:.1f},{p[mid][2]:.1f}), {p[mid][0] - bp2[0]:+.1f} cm from the block centre, "
              f"{p[mid][2] - top:.1f} cm above the top; tip spans y {lo:.1f}..{hi:.1f}, ends there at y {ends[0]} and {ends[1]} -> "
              f"both ends under the tip: {self.threaded}; block moved {np.linalg.norm(bp2[:2] - bp[:2]):.1f} cm, tilt {tilt:.0f} deg", flush=True)
        self.bg.append(self.g_park(h))

    def g_hitch_lift(self, h, st):
        """The other hand holds the eye (a finger on the knot) while this hand takes both ends,
        pulls the slack out of the wrap, then lifts above the eye, holds, carries the block over
        the box and lowers it in - one hand on the rope from the moment the knot is snug."""
        other = [o for o in self.arms.hands.values() if o is not h][0]
        while h["busy"] or other["busy"]:
            yield
        blk = self.sling_blk
        w, _, hh = blk["dims"]
        mid = self._tip()
        # first the hold: taking the ends disturbs the eye, whose tip rides on them
        cinch = E("ROPE_CINCH", "0") == "1"                      # hold-the-eye-and-cinch variant; not yet better than lifting directly
        if cinch:
            yield from self.g_grasp(other, mid, "hold_eye")
        p, (bp, _) = self.seg_pos(), self._blk_state(blk)
        cP = [i for i in self._strand("P") if p[i][0] < bp[0] - w / 2 - 2.0]
        cM = [j for j in self._strand("M") if p[j][0] < bp[0] - w / 2 - 2.0]
        pairs = [(i, j) for i in cP for j in cM if np.linalg.norm(p[i] - p[j]) <= float(E("ROPE_PAIR_MAX_CM", "3.5"))]
        if not pairs:
            raise RuntimeError("hitch_lift: the two ends do not run close enough together beside the block to hold both")
        segs = list(max(pairs, key=lambda ij: p[ij[0]][2] + p[ij[1]][2]))
        yield from self._pinch_two(h, segs, "hitch_lift", squeeze=False)
        z0 = self._blk_state(blk)[0][2]
        res = dict(lift_plan_cm=float(st["lift_cm"]), one_hand=True, threaded=bool(getattr(self, "threaded", False)))
        cap = TABLE_TOP + float(E("ROPE_TIP_MAX_CM", "55"))
        top = bp[2] + hh / 2
        # CINCH before lifting. With slack left in the wrap the eye lifted the block's left side
        # ~9 cm before the right side took any load, and the block rolled over (it still came up,
        # upside down). Pull the ends sideways, level with the top, until the block itself starts
        # to move: then the loop is snug all the way round.
        # ... Pulling alone dragged the eye off the top with the ends (the tip rides on them), so the
        # OTHER hand first holds the eye's tip where it lies - a finger on the knot - and the pull
        # is exactly the slack measured in the wrap on the far side.
        edge = np.array([bp[0] - w / 2, bp[1], top])
        if cinch:
            yield from self.g_move(h, np.array([edge[0] - 10.0, h["tgt"][1], top + 1.5]), speed=8.0)
            p = self.seg_pos()
            slack = []
            for rng, sg in ((self._strand("P"), segs[0]), (self._strand("M"), segs[1])):
                i_a = min(rng, key=lambda i: abs(p[i][0] - (bp[0] + w / 2)) + abs(p[i][2] - TABLE_TOP))
                geo = hh + 2.0 * self.rad + w + float(np.linalg.norm(p[sg] - edge))     # up the far face, across the top, out to the hand
                slack.append(abs(sg - i_a) * SEG - geo - float(E("ROPE_SNUG_EXTRA_CM", "1.5")))
            pull, pulled = float(np.clip(min(slack), 0.0, 30.0)), 0.0
            while pulled < pull:
                h["tgt"] = h["tgt"] + np.array([-4.0 * DT, 0.0, 0.0]); pulled += 4.0 * DT
                yield
            yield from self.g_wait(0.3)
            res["cinch_cm"] = round(pulled, 1)
            tipn = self.seg_pos()[mid]
            pb, tl = self._blk_state(blk)
            print(f"[twin-rope] t={self.t:5.1f}s hitch: {other['name']} held the eye, {h['name']} cinched {pulled:.1f} cm of slack out (measured {slack[0]:.1f} and {slack[1]:.1f}); "
                  f"eye tip {tipn[0] - pb[0]:+.1f} cm from the block centre, block moved {np.linalg.norm(pb[:2] - bp[:2]):.1f} cm, tilt {tl:.0f} deg", flush=True)
            yield from self.g_release(other)
            park = self.g_park(other)
            while np.linalg.norm(other["tip"] - tipn) < 22.0:           # clear of the space above the knot
                if next(park, "done") == "done":
                    break
                yield
            self.bg.append(park)
        reach = float(np.linalg.norm(h["tip"] - edge))
        tipn = self.seg_pos()[self._tip()]                          # lift from above the eye's tip
        reach += float(abs(tipn[0] - edge[0]))
        yield from self.g_move(h, np.array([tipn[0], tipn[1], top + max(float(E("ROPE_HITCH_START_CM", "9")), reach - 1.0)]), speed=8.0)   # above the centre, slack still in
        while self._blk_state(blk)[0][2] - z0 < st["lift_cm"] and h["tgt"][2] < cap:
            h["tgt"] = h["tgt"] + np.array([0.0, 0.0, 6.0 * DT])
            yield
        track = dict(lift=[], tilt=[])

        def watch(gen, key):
            for _ in gen:
                pb, tilt = self._blk_state(blk)
                track["lift"].append(pb[2] - z0); track["tilt"].append(tilt)
                yield
            res[key] = dict(min_lift_cm=round(float(min(track["lift"])), 2), max_tilt_deg=round(float(max(track["tilt"])), 1))
            track["lift"].clear(); track["tilt"].clear()

        yield from watch(self.g_wait(float(st["hold_s"])), "hold")
        print(f"[twin-rope] t={self.t:5.1f}s hitch: block hangs {res['hold']['min_lift_cm']:.1f} cm above where it stood for {st['hold_s']:.0f}s from ONE hand, "
              f"tilt up to {res['hold']['max_tilt_deg']:.0f} deg; hand {h['tip'][2] - TABLE_TOP:.0f} cm above the table", flush=True)
        self.sling_res = res
        if res["hold"]["min_lift_cm"] < 3.0:
            raise RuntimeError("the block did not come up with the rope")
        d = np.array([BOX_XY[0], BOX_XY[1]]) - self._blk_state(blk)[0][:2]
        yield from watch(self.g_move(h, h["tgt"] + np.array([d[0], d[1], 0.0]), speed=10.0), "carry")
        while self._blk_state(blk)[0][2] - hh / 2 > self.floor + 0.3 and h["tgt"][2] > self.floor + 4.0:
            h["tgt"] = h["tgt"] - np.array([0.0, 0.0, 6.0 * DT])
            yield
        yield from self.g_wait(0.5)
        yield from self.g_release(h)
        self.bg.append(self.g_park(h))

    def g_swing_end(self, h, st):
        """Move a rope end that lies beside the block to the other y side, passing OUTSIDE the
        other strand's standing part on that face - the second crossing of the half knot."""
        while h["busy"]:
            yield
        seg = self._seg(st["s_cm"])
        yield from self.g_grasp(h, seg, "swing_end")
        bp, _ = self._blk_state(self.sling_blk)
        w = self.sling_blk["dims"][0]
        x = bp[0] + float(st["x_side"]) * (w / 2 + float(st["land_cm"]))
        up = float(E("ROPE_SWING_UP_CM", "7"))
        for goal in ([h["tgt"][0], h["tgt"][1], TABLE_TOP + up], [x, bp[1] + float(st["to_dy"]), TABLE_TOP + up],
                     [x, bp[1] + float(st["to_dy"]), TABLE_TOP + self.rad + 0.6]):
            yield from self.g_move(h, np.array(goal), speed=12.0)
        yield from self.g_release(h)
        print(f"[twin-rope] t={self.t:5.1f}s swing_end: {h['name']} moved the end to y{st['to_dy']:+.0f}; twist of the two strands "
              f"(Gauss linking) {self.linking():+.2f}", flush=True)
        self.bg.append(self.g_park(h))

    def linking(self):
        """Gauss linking integral of the two free strands (each from where it leaves the block's
        underside to its end). About 0 when they merely lie across each other, about +-1 when
        they are twisted around each other once."""
        p = self.seg_pos()
        w = self.sling_blk["dims"][0]
        ia, ib = self._seg(self.L / 2 - w / 2), self._seg(self.L / 2 + w / 2)
        A, B = p[:ia + 1][::-1], p[ib:]
        dA, dB = np.diff(A, axis=0), np.diff(B, axis=0)
        mA, mB = 0.5 * (A[1:] + A[:-1]), 0.5 * (B[1:] + B[:-1])
        r = mA[:, None, :] - mB[None, :, :]
        n = np.maximum(np.linalg.norm(r, axis=2), 1.0) ** 3
        return float((np.einsum("ijk,ijk->ij", r, np.cross(dA[:, None, :], dB[None, :, :])) / n).sum() / (4 * math.pi))

    def g_lift_carry_lower(self, st):
        hl, hr = self.arms.hands["left"], self.arms.hands["right"]
        while hl["busy"] or hr["busy"]:
            yield
        blk = self.sling_blk
        hh = blk["dims"][2]
        # Re-ground the plan's re-grasp points on the measured rope. ROPE_REGRASP=table takes each
        # strand where it lies on the table beyond the face (hands ~25 cm apart): lifting from there
        # pulls the wrap open and the rope slides out from under the block (run kept as open_wrap).
        def ground(s_cm, step, side):
            p, bp = self.seg_pos(), self._blk_state(blk)[0]
            i = self._seg(s_cm)
            mode = E("ROPE_REGRASP", st.get("regrasp", "plan"))
            if mode == "plan":                                     # the planner's arc lengths, as given
                return i
            if mode == "top":
                # grip each strand where it leaves the block's top, on the hand's own side: the two
                # hands then pull from nearly one point above the block and the wrap stays closed
                strand = range(2, self.N // 2) if step < 0 else range(self.N // 2, self.N - 2)
                c = [j for j in strand if 1.0 <= side * (p[j][0] - bp[0]) - blk["dims"][0] / 2 <= 4.0]
                return max(c, key=lambda j: p[j][2]) if c else i
            while 2 < i < self.N - 3 and not (side * (p[i][0] - bp[0]) >= blk["dims"][0] / 2 + float(E("ROPE_REGRASP_CM", "6"))
                                              and p[i][2] < TABLE_TOP + 2.5):
                i += step
            return i
        if "grasps" in st:
            # choker hitch: each hand takes one END strand on the table beyond the laid bight
            p, bp = self.seg_pos(), self._blk_state(blk)[0]
            for gsp in st["grasps"]:
                strand = range(2, self.N // 2 - 4) if gsp["strand"] == "P" else range(self.N // 2 + 4, self.N - 2)
                xt = bp[0] + blk["dims"][0] / 2 + float(gsp["beyond_cm"])
                seg = min(strand, key=lambda i: abs(p[i][0] - xt) + 10.0 * max(0.0, p[i][2] - TABLE_TOP - 2.5))
                yield from self.g_grasp(self.arms.hands[gsp["hand"]], seg, "hitch")
        else:
            seg_r, seg_l = ground(st["right_s_cm"], -1, +1.0), ground(st["left_s_cm"], +1, -1.0)
            yield from self.g_grasp(hr, seg_r, "sling")
            yield from self.g_grasp(hl, seg_l, "sling")
        z0 = self._blk_state(blk)[0][2]
        res = dict(lift_plan_cm=float(st["lift_cm"]))
        cap = TABLE_TOP + 52.0
        y_c = self._blk_state(blk)[0][1]
        while self._blk_state(blk)[0][2] - z0 < st["lift_cm"] and hl["tgt"][2] < cap:
            for h in (hl, hr):
                dy = float(np.clip(y_c - h["tgt"][1], -4.0 * DT, 4.0 * DT)) if st.get("converge") else 0.0   # bring the two ends together
                h["tgt"] = h["tgt"] + np.array([0.0, dy, 8.0 * DT])
            yield
        res["hand_rise_cm"] = float(hl["tgt"][2] - TABLE_TOP)
        track = dict(lift=[], tilt=[])

        def watch(gen, key):
            for _ in gen:
                p, tilt = self._blk_state(blk)
                track["lift"].append(p[2] - z0); track["tilt"].append(tilt)
                yield
            res[key] = dict(min_lift_cm=round(float(min(track["lift"])), 2), max_tilt_deg=round(float(max(track["tilt"])), 1))
            track["lift"].clear(); track["tilt"].clear()

        yield from watch(self.g_wait(float(st["hold_s"])), "hold")
        print(f"[twin-rope] t={self.t:5.1f}s sling: block hangs {res['hold']['min_lift_cm']:.1f} cm above where it stood for {st['hold_s']:.0f}s, "
              f"tilt up to {res['hold']['max_tilt_deg']:.0f} deg; hands {np.linalg.norm(hl['tip'] - hr['tip']):.0f} cm apart", flush=True)
        self.sling_res = res
        if res["hold"]["min_lift_cm"] < 3.0:
            raise RuntimeError("the block did not come up with the rope")
        d = np.array([BOX_XY[0], BOX_XY[1]]) - self._blk_state(blk)[0][:2]
        d3 = np.array([d[0], d[1], 0.0])
        yield from watch(self.g_par(self.g_move(hl, hl["tgt"] + d3, speed=10.0), self.g_move(hr, hr["tgt"] + d3, speed=10.0)), "carry")
        while self._blk_state(blk)[0][2] - hh / 2 > self.floor + 0.3 and hl["tgt"][2] > self.floor + 4.0:
            for h in (hl, hr):
                h["tgt"] = h["tgt"] - np.array([0.0, 0.0, 6.0 * DT])
            yield
        yield from self.g_wait(0.5)
        yield from self.g_par(self.g_release(hl), self.g_release(hr))
        self.bg += [self.g_park(hl), self.g_park(hr)]

    def _seg(self, s_cm):
        return int(np.clip(round(s_cm / SEG - 0.5), 1, self.N - 2))

    def g_place_end(self, h, seg):
        yield from self.g_grasp(h, seg, "place_end")
        top = max(self.rim + 10.0, TABLE_TOP + 26.0)
        yield from self.g_move(h, np.array([h["tgt"][0], h["tgt"][1], top]), speed=20.0)
        yield from self.g_move(h, np.array([self.F[0], self.F[1], top]), speed=18.0)
        yield from self.g_move(h, np.array([self.F[0], self.F[1], self.floor + self.rad + 0.3]), speed=14.0)
        yield from self.g_release(h)
        self.bg.append(self.g_park(h))

    def g_lay_loop(self, h, seg, prev_seg, k):
        """Grasp `seg`, hang the rope between it and the last laid point, lower it in one circle."""
        while h["busy"]:
            yield
        tag = f"lay_loop {k}"
        arc = (seg - prev_seg) * SEG
        yield from self.g_grasp(h, seg, tag)
        A = self.seg_pos()[prev_seg]                               # where the laid part ends, measured
        hang = min(self.hang * arc, self.lift_max)
        z_end = self.floor + self.rad + 0.3 + 2.0 * self.rad * 0.9 * k   # on top of the loops already there
        th0, sweep = -math.pi / 2.0, 2.0 * math.pi * arc / self.loop
        off = A[:2] - self.F
        lead = math.radians(float(E("ROPE_LEAD_DEG", "20")))       # hand ahead of the touchdown point, like laying cable

        former = next((bk for bk in self.blocks.values() if (bk["plan"] or {}).get("role") == "former"), None)
        # the trailing rope runs from the hand to the front rim and sags in between; at block height
        # + 3 cm it still dragged across the former and tipped it, so the hand stays above the RIM
        z_clear = (max(self.floor + former["dims"][2] + 3.0, self.rim + float(E("ROPE_FORMER_CLEAR_CM", "5")))
                   if former is not None else None)
        u_drop = 1.0 - float(E("ROPE_DROP_DEG", "45")) / max(1.0, math.degrees(sweep))

        def pose(u):
            th = th0 + sweep * u
            rem = (1.0 - u) * hang
            xy = (self.C + self.Rc * np.array([math.cos(th), math.sin(th)]) + off * (1.0 - u)
                  + rem * math.sin(lead) * np.array([-math.sin(th), math.cos(th)]))
            z = z_end + rem * math.cos(lead) * (self.floor + hang - z_end) / hang
            if z_clear is not None:
                # a former stands in the ring: stay above it until the last stretch before the front
                # point, so the trailing rope (hand -> front rim) passes OVER the block, not through it
                z = max(z, z_clear) if u < u_drop else z_end + (max(z_clear, z_end) - z_end) * (1.0 - u) / (1.0 - u_drop)
            return np.array([xy[0], xy[1], z])

        yield from self.g_move(h, pose(0.0), speed=20.0, anchor=A, radius=0.95 * arc)
        yield from self.g_wait(0.5)
        n = max(1, int(arc / LAY_SPEED * FPS))
        for i in range(1, n + 1):
            h["tgt"] = pose(i / n)
            yield
        yield from self.g_wait(0.4)
        yield from self.g_release(h)
        self.bg.append(self.g_park(h))

    def run(self):
        def main():
            yield from self.g_wait(1.0)                            # let the rope settle on the table
            hs = self.arms.hands
            prev = None
            k = 0
            for st in self.plan["plan"]:
                if st["op"] == "lift_carry_lower":
                    yield from self.g_lift_carry_lower(st)
                    continue
                h = hs[st["hand"]]
                if st["op"] == "drape":
                    yield from self.g_drape(h, st)
                    continue
                if st["op"] in ("end_over", "ends_over", "gather_ends", "eye_on_top", "hitch_lift"):
                    yield from getattr(self, "g_" + st["op"])(h, st)
                    continue
                if st["op"] == "bight_over":
                    yield from self.g_bight_over(h, st)
                    continue
                if st["op"] == "swing_end":
                    yield from self.g_swing_end(h, st)
                    continue
                if st["op"] == "fold_rope":
                    yield from self.g_fold_rope(h, st)
                    continue
                if st["op"] == "place_on_rope":
                    if "tip_to_centre_cm" in st and getattr(self, "fold", None):
                        # put the block where the fold actually ended up, centred on the two strands
                        p = self.seg_pos()
                        xb = self.fold["tip_x"] + float(st["tip_to_centre_cm"])
                        yb = 0.5 * (p[min(self._strand("P"), key=lambda i: abs(p[i][0] - xb))][1]
                                    + p[min(self._strand("M"), key=lambda i: abs(p[i][0] - xb))][1])
                        self.sling_blk["plan"]["dxy"] = [float(xb - BOX_XY[0]), float(yb - BOX_XY[1])]
                        while h["busy"]:
                            yield
                    yield from self.g_pick_place(h, st["name"])
                    yield from self.g_wait(1.0)
                    continue
                if st["op"] == "pick_place":
                    yield from self.g_pick_place(h, st["name"])
                    self.audit(f"after pick_place {st['name']}")
                    continue
                seg = int(np.clip(round(st["s_cm"] / SEG - 0.5), 1, self.N - 2))
                if st["op"] == "place_end":
                    yield from self.g_place_end(h, seg)
                else:
                    yield from self.g_lay_loop(h, seg, prev, k)
                    k += 1
                prev = seg
                self.audit(f"after {st['op']} s={st['s_cm']}")
            while self.bg:
                yield
            yield from self.g_wait(3.0)
        t0 = time.time()
        ok = True
        try:
            for _ in main():
                self.step_frame()
                if self.frame % (FPS * 10) == 0:
                    print(f"[twin-rope] t={self.t:5.1f}s  wall {time.time() - t0:.0f}s", flush=True)
        except RuntimeError as e:
            print(f"[twin-rope] ABORTED: {e}", flush=True)
            ok = False
        if self.task == "sling":
            return self._finish_sling(ok, t0)
        final = self.audit("FINAL")
        final["completed"] = ok
        coil = self.plan["coil"]
        final["success"] = bool(ok and final["in_box_frac"] >= 0.98 and final["winding_turns"] >= coil["loops"] - 0.75
                                and final["top_above_floor_cm"] <= self.bh and final["on_ring_frac"] >= float(E("ROPE_RING_MIN", "0.8"))
                                and all(bk["in_box"] and bk["tilt_deg"] < 15.0 and bk["off_plan_cm"] < 3.0 for bk in final["blocks"].values())
                                and final.get("turns_around_former", 99.0) >= coil["loops"] - 0.75)
        res = dict(final=final, grasps=self.log, misses=self.miss, sim_s=round(self.t, 1), wall_s=round(time.time() - t0, 1),
                   tip_min_cm=round(self.tip_min, 1), both_tips_in_box_frames=self.both_in_box, frames=self.frame,
                   ik_err_max_cm=round(self.ik_err_max, 2), tip_in_block_frames=self.tip_in_block, stages=self._stages)
        json.dump(res, open(os.path.join(self.out, "rope_audit.json"), "w"), indent=1)
        self.rec.save()
        print(f"[twin-rope] ROPE result: {'SUCCESS' if final['success'] else 'FAIL'} - in box {final['in_box_frac']:.0%}, "
              f"{final['winding_turns']:.2f} turns (plan {coil['loops']}), top {final['top_above_floor_cm']:.1f} cm above the floor (rim {self.bh:.0f}), "
              f"coil radius {final['radius_mean_cm']:.1f}+-{final['radius_std_cm']:.1f} cm (plan {self.Rc}), {final['on_ring_frac']:.0%} on the planned ring, kink p95 {final['turn_deg_p95']:.0f} deg; {len(self.log)} grasps, {len(self.miss)} misses, "
              f"{self.t:.0f}s sim / {time.time() - t0:.0f}s wall; tips closest {self.tip_min:.1f} cm, both tips in the box {self.both_in_box}/{self.frame} frames, "
              f"worst IK lag {self.ik_err_max:.1f} cm {self.ik_err_at or ''}", flush=True)
        if self.blocks:
            print("[twin-rope] BLOCKS: " + "; ".join(f"{n} ({bk['role']}) in box {bk['in_box']}, {bk['off_plan_cm']:.1f} cm off plan, tilt {bk['tilt_deg']:.0f} deg, "
                  f"bottom {bk['bottom_above_floor_cm']:.1f} cm above the floor" for n, bk in final["blocks"].items())
                  + (f"; rope makes {final['turns_around_former']:.2f} turns around the centre block" if "turns_around_former" in final else "")
                  + f"; tip inside a block {self.tip_in_block} frames", flush=True)
        return final["success"]

    def _finish_sling(self, ok, t0):
        blk = self.sling_blk
        p, tilt = self._blk_state(blk)
        res = dict(getattr(self, "sling_res", {}))
        in_box = bool(abs(p[0] - BOX_XY[0]) < self.hin and abs(p[1] - BOX_XY[1]) < self.hin and p[2] < self.rim)
        rp = self.seg_pos()
        res.update(completed=ok, block_xyz=[round(float(v), 2) for v in p], block_tilt_deg=round(tilt, 1), block_in_box=in_box,
                   rope_in_box_frac=float(((np.abs(rp[:, 0] - BOX_XY[0]) < self.hin) & (np.abs(rp[:, 1] - BOX_XY[1]) < self.hin)).mean()),
                   grasps=self.log, misses=self.miss, sim_s=round(self.t, 1), wall_s=round(time.time() - t0, 1), tip_min_cm=round(self.tip_min, 1),
                   ik_err_max_cm=round(self.ik_err_max, 2), ik_err_at=self.ik_err_at, tip_in_block_frames=self.tip_in_block, tool_through_rope_frames=self.tool_through,
                   linking=round(self.linking(), 3))
        hold, carry = res.get("hold", {}), res.get("carry", {})
        res["knot"] = self.plan.get("knot", "none")
        res["success"] = bool(ok and hold.get("min_lift_cm", 0) >= 10.0 and hold.get("max_tilt_deg", 99) < 25.0
                              and carry.get("min_lift_cm", 0) >= self.rim - TABLE_TOP and in_box and tilt < 25.0)
        json.dump(res, open(os.path.join(self.out, "rope_audit.json"), "w"), indent=1)
        self.rec.save()
        print(f"[twin-rope] SLING result: {'SUCCESS' if res['success'] else 'FAIL'} - hold: lowest {hold.get('min_lift_cm', 0):.1f} cm up, tilt up to {hold.get('max_tilt_deg', 0):.0f} deg; "
              f"carry: lowest {carry.get('min_lift_cm', 0):.1f} cm up (rim needs {self.rim - TABLE_TOP:.1f}), tilt up to {carry.get('max_tilt_deg', 0):.0f} deg; "
              f"block ends at ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) in box {in_box}, tilt {tilt:.0f} deg; rope {res['rope_in_box_frac']:.0%} in the box; "
              f"{len(self.log)} grasps, {len(self.miss)} misses, {self.t:.0f}s sim / {time.time() - t0:.0f}s wall; tips closest {self.tip_min:.1f} cm, "
              f"worst IK lag {self.ik_err_max:.1f} cm {self.ik_err_at or ''}, tip inside the block {self.tip_in_block} frames, gripper through the rope {self.tool_through} frames", flush=True)
        return res["success"]

    _stages = []

    def audit(self, tag):
        p = self.seg_pos()
        inb = (np.abs(p[:, 0] - BOX_XY[0]) < self.hin) & (np.abs(p[:, 1] - BOX_XY[1]) < self.hin) & (p[:, 2] < self.rim + self.rad)
        q = p[inb] if inb.any() else p
        cen = q[:, :2].mean(0)
        ang = np.unwrap(np.arctan2(q[:, 1] - cen[1], q[:, 0] - cen[0]))
        r = np.linalg.norm(q[:, :2] - cen, axis=1)
        gaps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        # neatness: how much of the rope lies on the planned ring, and how sharply it kinks
        on_ring = np.abs(np.linalg.norm(p[:, :2] - self.C, axis=1) - self.Rc) < float(E("ROPE_RING_TOL_CM", "3.0"))
        d = np.diff(p, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        turn = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1.0, 1.0)))
        a = dict(tag=tag, t=round(self.t, 2), in_box_frac=float(inb.mean()),
                 winding_turns=float(abs(ang[-1] - ang[0]) / (2 * math.pi)) if inb.sum() > 3 else 0.0,
                 top_above_floor_cm=float(q[:, 2].max() + self.rad - self.floor), radius_mean_cm=float(r.mean()), radius_std_cm=float(r.std()),
                 centre_xy=[float(cen[0]), float(cen[1])], min_z_cm=float(p[:, 2].min()),
                 on_ring_frac=float((on_ring & inb).mean()), turn_deg_p95=float(np.percentile(turn, 95)), turn_deg_max=float(turn.max()),
                 seg_gap_cm=[float(gaps.min()), float(gaps.max())])
        bqa = self.s0.body_q.numpy()
        a["blocks"] = {}
        for nm, blk in self.blocks.items():
            pb, Rb = bqa[blk["body"], :3].astype(float), q2R(bqa[blk["body"], 3:7])
            pl = blk["plan"] or {}
            tgt = np.array([BOX_XY[0] + pl["dxy"][0], BOX_XY[1] + pl["dxy"][1]]) if pl else pb[:2]
            a["blocks"][nm] = dict(role=pl.get("role"), xyz=[round(float(v), 2) for v in pb],
                                   in_box=bool(abs(pb[0] - BOX_XY[0]) < self.hin and abs(pb[1] - BOX_XY[1]) < self.hin and pb[2] < self.rim),
                                   off_plan_cm=round(float(np.linalg.norm(pb[:2] - tgt)), 2), tilt_deg=round(math.degrees(math.acos(np.clip(Rb[2, 2], -1, 1))), 1),
                                   bottom_above_floor_cm=round(float(pb[2] - blk["dims"][2] / 2 - self.floor), 2))
            if pl.get("role") in ("former", "keeper") and inb.sum() > 3:
                an = np.unwrap(np.arctan2(q[:, 1] - pb[1], q[:, 0] - pb[0]))
                a["turns_around_former"] = float(abs(an[-1] - an[0]) / (2 * math.pi))
        self._stages = self._stages + [a]
        print(f"[twin-rope] audit {tag}: in box {a['in_box_frac']:.0%}, {a['winding_turns']:.2f} turns, top {a['top_above_floor_cm']:.1f} cm, "
              f"radius {a['radius_mean_cm']:.1f}+-{a['radius_std_cm']:.1f}, on the planned ring {a['on_ring_frac']:.0%}, "
              f"kink p95 {a['turn_deg_p95']:.0f} deg (ideal {math.degrees(SEG / self.Rc):.0f}), lowest point z {a['min_z_cm']:.1f}", flush=True)
        return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.exit(0 if RopeTwin(json.load(open(a.plan)), a.out).run() else 1)
