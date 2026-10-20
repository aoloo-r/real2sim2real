"""Sled twin: a towel used as a tool to bring out-of-reach objects to two arms, then folded and
packed with them. One coupled Newton VBD world (cloth particles + dynamic rigid boxes/cylinders +
table + box), so objects ride, slip or topple by friction and inertia, and the towel really folds.

  towel    cloth grid; grasped by pinning the particles under the closed jaws (all layers)
  objects  dynamic boxes, a mug and a bottle (props.py) standing on the towel
  arms     two kinematic UR5e + 2F-85, IK on the tip pose; they collide with nothing.
           `with_arm=False` replaces them by points on the same tip paths - identical physics,
           used by the planner's rollouts.

Usage: twin_sled.py --plan plan.json --out DIR     (writes DIR/sled.usd, DIR/sled_audit.json)
Units: cm, kg, s.
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
from twin_rope import Arms, q2R, R2q, TABLE_TOP
import props

E = os.environ.get
FPS, SUBSTEPS = 60, int(E("SLED_SUBSTEPS", "10"))
DT = 1.0 / FPS
TABLE = ((0.0, -50.0, 10.0), 50.0, 80.0, 10.0)
ARM_BASES = (("right", 30.0), ("left", -30.0))            # x; y = -2
BASES_XY = {"right": np.array([30.0, -2.0]), "left": np.array([-30.0, -2.0])}
GRIP_R = float(E("SLED_GRIP_R_CM", "2.5"))
TILT_OK = 15.0
# Farthest a straight-down grasp can be from an arm's base axis. MEASURED on this arm model: the tip
# tracks exactly to 92 cm and is 9.7 cm short (tool 8 deg off vertical) at 96 cm; 88 keeps a margin.
R_PICK = float(E("SLED_REACH_CM", "88"))
BOX_XY, BOX_WALL = (0.0, float(E("SLED_BOX_Y", "6"))), 1.0
# arm mount height above the table. 4 cm: the upper arms graze the box walls and the towel pack puts each
# hand ~16 cm from its own base axis (both infeasible for a real UR5e; found by the motion layer, motion.py)
BASE_LIFT = float(E("SLED_BASE_LIFT_CM", "4"))


def reach_dist(xy):
    return float(min(np.linalg.norm(np.asarray(xy, float)[:2] - b) for b in BASES_XY.values()))


def reachable(xy, hand=None):
    if hand:
        return bool(np.linalg.norm(np.asarray(xy, float)[:2] - BASES_XY[hand]) <= R_PICK)
    return reach_dist(xy) <= R_PICK


class SledSim:
    def __init__(self, plan, out=None, with_arm=True, record=True, verbose=True):
        self.plan, self.out, self.verbose = plan, out, verbose
        sc = plan["scene"]
        self.sc = sc
        tw = sc["towel"]
        self.tw_c = np.array([float(v) for v in tw["centre_xy"]])
        self.tw_w, self.tw_l, sp = float(tw["size_cm"][0]), float(tw["size_cm"][1]), float(tw.get("spacing_cm", 1.0))   # width along x, length along y
        self.hin, self.bh = 0.5 * float(sc["box"]["inner_cm"][0]), float(sc["box"]["h_cm"])
        self.floor = TABLE_TOP + 0.5
        b = ModelBuilder(gravity=-981.0)
        b.rigid_gap = 0.0
        (tx, ty, tz), thx, thy, thz = TABLE
        b.add_shape_box(-1, xform=wp.transform((tx, ty, tz), wp.quat_identity()), hx=thx, hy=thy, hz=thz,
                        cfg=ModelBuilder.ShapeConfig(mu=float(sc.get("mu_table", 0.3)), ke=1.0e5, kd=0.0))
        stat = ModelBuilder.ShapeConfig(mu=0.5, ke=1.0e5, kd=0.0)
        bx, by, hin, w_ = BOX_XY[0], BOX_XY[1], self.hin, BOX_WALL
        self.box_shapes = [dict(p=(bx, by, self.floor - 0.25), h=(hin + w_, hin + w_, 0.25), name="floor")]
        for nm, dx, dy, hx, hy in (("right", hin + w_ / 2, 0.0, w_ / 2, hin + w_), ("left", -hin - w_ / 2, 0.0, w_ / 2, hin + w_),
                                   ("back", 0.0, hin + w_ / 2, hin + w_, w_ / 2), ("front", 0.0, -hin - w_ / 2, hin + w_, w_ / 2)):
            self.box_shapes.append(dict(p=(bx + dx, by + dy, self.floor + self.bh / 2), h=(hx, hy, self.bh / 2), name=nm))
        for s_ in self.box_shapes:
            b.add_shape_box(-1, xform=wp.transform(s_["p"], wp.quat_identity()), hx=s_["h"][0], hy=s_["h"][1], hz=s_["h"][2], cfg=stat)
        nx, ny = int(round(self.tw_w / sp)), int(round(self.tw_l / sp))
        b.add_cloth_grid(pos=wp.vec3(float(self.tw_c[0] - self.tw_w / 2), float(self.tw_c[1] - self.tw_l / 2), TABLE_TOP + 0.4),
                         rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=nx, dim_y=ny, cell_x=sp, cell_y=sp,
                         mass=float(tw.get("mass_kg", 0.25)) / ((nx + 1) * (ny + 1)),
                         tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-2, edge_ke=float(tw.get("bend", 5.0)), edge_kd=1.0e-2, particle_radius=0.4)
        self.objs = {}
        pre = plan.get("preplaced", {})                            # planner rollouts: objects packed before the pull start in the box
        for o in sc["objects"]:
            w, d, h = [float(v) for v in o["dims_cm"]]
            if o["name"] in pre:
                xy, z = np.array(pre[o["name"]], float), self.floor + h / 2 + 0.05
            else:
                xy, z = self.tw_c + np.array([float(v) for v in o["on_towel_xy"]]), TABLE_TOP + 0.9 + h / 2
            body = b.add_body(xform=wp.transform((float(xy[0]), float(xy[1]), z), wp.quat_identity()), label=o["name"])
            mu = float(o.get("mu", 0.5))
            props.add_shapes(b, body, o, mu)                          # box / cylinder / mug / bottle (see props.py)
            self.objs[o["name"]] = dict(body=int(body), dims=np.array([w, d, h]), spec=o, start=np.array([xy[0], xy[1], z]))
        b.color()
        self.model = b.finalize()
        self.model.soft_contact_ke, self.model.soft_contact_kd = 1.0e4, 1.0e1
        self.model.soft_contact_mu = float(tw.get("mu", 0.5))
        self.pipe = newton.CollisionPipeline(self.model, soft_contact_margin=0.5, contact_matching="sticky")
        self.contacts = self.pipe.contacts()
        # the towel folds onto itself only if the plan folds it; rollouts of the pull skip that cost
        self.self_contact = any(s_["op"] in ("fold", "pack_towel") for s_ in plan["plan"]) and E("SLED_SELF_CONTACT", "1") == "1"
        self.model.edge_rest_angle.zero_()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=int(E("SLED_ITERS", "8")), rigid_contact_history=True,
                                               particle_enable_self_contact=self.self_contact,
                                               particle_self_contact_radius=0.2, particle_self_contact_margin=0.35,
                                               particle_topological_contact_filter_threshold=1, particle_rest_shape_contact_exclusion_radius=0.5)
        self.s0, self.s1, self.ctl = self.model.state(), self.model.state(), self.model.control()
        self.flags0 = self.model.particle_flags.numpy().copy()
        q0 = self.model.particle_q.numpy()
        self.nx, self.ny = nx, ny                                  # particle at grid node (i, j)
        ii = np.rint((q0[:, 0] - (self.tw_c[0] - self.tw_w / 2)) / sp).astype(int)
        jj = np.rint((q0[:, 1] - (self.tw_c[1] - self.tw_l / 2)) / sp).astype(int)
        self.grid = -np.ones((nx + 1, ny + 1), int)
        self.grid[ii, jj] = np.arange(len(q0))
        self.arms = Arms(bases=ARM_BASES) if with_arm else None
        if with_arm:
            self.hands = self.arms.hands
        else:
            self.hands = {n: dict(name=n, tip=np.array([math.copysign(36.0, bx_), -34.0, 55.0]), R=np.eye(3), yaw=0.0, grip=0.0, jaw0=0.0)
                          for n, bx_ in ARM_BASES}
        for n, h in self.hands.items():
            h["park"] = np.array([math.copysign(36.0, BASES_XY[n][0]), -34.0, 55.0])
            h["tgt"] = h["tip"].copy()
        self.pin_p, self.pin_b = {}, {}                            # hand -> (indices, offsets) / (body, offset, R_rel)
        self.t, self.frame, self.log, self.ik_err_max, self.tip_min = 0.0, 0, [], 0.0, 1e9
        self.rec = None
        if record and with_arm and out:
            self._rec_init()

    # ---- recording ----------------------------------------------------------------------
    def _rec_init(self):
        from pxr import Gf, UsdGeom
        from usd_recorder import UsdRecorder
        self.rec = UsdRecorder(self.arms.model, os.path.join(self.out, "sled.usd"), every=4, fps=FPS, table=TABLE)
        st = self.rec.stage
        for s_ in self.box_shapes:
            c = UsdGeom.Cube.Define(st, f"/World/box/{s_['name']}")
            c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.62, 0.38)])
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, s_["p"])))
            xf.AddScaleOp().Set(Gf.Vec3f(*[2.0 * float(v) for v in s_["h"]]))
            if s_["name"] in E("SLED_HIDE_WALLS", "front").split(","):   # drawn open so the inside shows; the wall is still there in the physics
                UsdGeom.Imageable(c).MakeInvisible()
        m = UsdGeom.Mesh.Define(st, "/World/towel")
        tris = self.model.tri_indices.numpy().reshape(-1, 3)
        m.CreateFaceVertexIndicesAttr(tris.flatten().tolist()); m.CreateFaceVertexCountsAttr([3] * len(tris))
        m.CreateDoubleSidedAttr(True); m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.42, 0.68)])
        self._tw_pts, self._obj_ops = m.GetPointsAttr(), {}
        cols = [Gf.Vec3f(0.85, 0.30, 0.25), Gf.Vec3f(0.95, 0.75, 0.25), Gf.Vec3f(0.92, 0.92, 0.95), Gf.Vec3f(0.30, 0.65, 0.35), Gf.Vec3f(0.45, 0.35, 0.65)]
        for i, (nm, o) in enumerate(self.objs.items()):
            op = props.define(st, f"/World/objects/{nm}", o["spec"], tuple(cols[i % len(cols)]))
            if op is not None:                                       # mug, bottle: their own meshes
                self._obj_ops[nm] = op
            elif o["spec"].get("shape", "box") == "cylinder":
                c = UsdGeom.Cylinder.Define(st, f"/World/objects/{nm}")
                c.CreateAxisAttr("Z"); c.CreateHeightAttr(float(o["dims"][2])); c.CreateRadiusAttr(float(o["dims"][0]) / 2)
                c.CreateDisplayColorAttr([cols[i % len(cols)]])
                self._obj_ops[nm] = UsdGeom.Xformable(c).AddTransformOp()
            else:
                c = UsdGeom.Cube.Define(st, f"/World/objects/{nm}")
                c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([cols[i % len(cols)]])
                xf = UsdGeom.Xformable(c)
                self._obj_ops[nm] = xf.AddTransformOp()
                xf.AddScaleOp().Set(Gf.Vec3f(*map(float, o["dims"])))

    def _rec_sample(self):
        from pxr import Gf, Usd
        if (self.rec.frame + 1) % self.rec.every == 0:
            tc = Usd.TimeCode(self.rec.t)
            self._tw_pts.Set([Gf.Vec3f(*map(float, v)) for v in self.s0.particle_q.numpy()], tc)
            bq = self.s0.body_q.numpy()
            for nm, o in self.objs.items():
                self._obj_ops[nm].Set(self.rec._mat(bq[o["body"]]), tc)
        self.rec.sample(self.arms.state, 0)

    # ---- simulation ---------------------------------------------------------------------
    def step_frame(self):
        hs = self.hands
        old = {n: h["tip"].copy() for n, h in hs.items()}
        if self.arms:
            self.arms.step()
            self.ik_err_max = max([self.ik_err_max] + [float(np.linalg.norm(h["tip"] - h["tgt"])) for h in hs.values()])
        else:
            for h in hs.values():
                h["tip"] = h["tgt"].copy()
        vel = {n: (h["tip"] - old[n]) / DT for n, h in hs.items()}
        if self.self_contact:
            self.solver.rebuild_bvh(self.s0)
        for k in range(SUBSTEPS):
            a = (k + 1) / SUBSTEPS
            if self.pin_p:
                pq, pv = self.s0.particle_q.numpy(), self.s0.particle_qd.numpy()
                for n, (idx, off) in self.pin_p.items():
                    pq[idx] = old[n] + (hs[n]["tip"] - old[n]) * a + off; pv[idx] = vel[n]
                self.s0.particle_q.assign(pq); self.s0.particle_qd.assign(pv)
            if self.pin_b:
                bq, bv = self.s0.body_q.numpy(), self.s0.body_qd.numpy()
                for n, (body, off, Rrel) in self.pin_b.items():
                    bq[body, :3] = old[n] + (hs[n]["tip"] - old[n]) * a + hs[n]["R"] @ off; bq[body, 3:7] = R2q(hs[n]["R"] @ Rrel)
                    bv[body] = 0.0; bv[body, :3] = vel[n]
                self.s0.body_q.assign(bq); self.s0.body_qd.assign(bv)
            self.s0.clear_forces()
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.ctl, self.contacts, DT / SUBSTEPS)
            self.s0, self.s1 = self.s1, self.s0
        if self.frame % 20 == 0 and not np.isfinite(self.s0.particle_q.numpy()).all():
            raise RuntimeError(f"towel diverged (NaN) at t={self.t:.2f}s")
        if self.rec:
            self._rec_sample()
        tips = [h["tip"] for h in hs.values()]
        self.tip_min = min(self.tip_min, float(np.linalg.norm(tips[0] - tips[1])))
        self.t += DT; self.frame += 1

    def obj_state(self, o):
        bq = self.s0.body_q.numpy()[o["body"]].astype(float)
        return bq[:3], math.degrees(math.acos(float(np.clip(q2R(bq[3:7])[2, 2], -1.0, 1.0))))

    def node(self, u, v):
        """Towel particle at material coordinates: u across the width (x), v along the length (0 = far end)."""
        return int(self.grid[int(round(u * self.nx)), int(round(v * self.ny))])

    def towel_box(self):
        q = self.s0.particle_q.numpy()
        return q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), q[:, 2].max()

    # ---- primitives (generators: one yield per frame) -----------------------------------
    def g_move(self, h, goal, speed=25.0, yaw=None, tol=0.5):
        start, goal = h["tgt"].copy(), np.asarray(goal, float)
        n = max(1, int(math.ceil(np.linalg.norm(goal - start) / speed * FPS)))
        y0 = h["yaw"]
        for i in range(1, n + 1):
            a = i / n; a = a * a * (3 - 2 * a)
            h["tgt"] = start + (goal - start) * a
            if yaw is not None:
                h["yaw"] = y0 + (yaw - y0) * a
            yield
        for _ in range(120):
            if np.linalg.norm(h["tip"] - h["tgt"]) < tol:
                break
            yield

    def g_home(self, h):
        """Back to the park pose. From inside the box the straight line to it passes ~10 cm from the
        arm's own base axis, where the IK flipped the arm over its back and never recovered (lag
        38-53 cm): leave through a point in front of the box first, the way the hand came in."""
        if h["tgt"][1] > BOX_XY[1] - self.hin - 4.0:
            yield from self.g_move(h, np.array([math.copysign(9.0, BASES_XY[h["name"]][0]), BOX_XY[1] - self.hin - 26.0, h["tgt"][2]]), speed=24.0)
        yield from self.g_move(h, h["park"], yaw=0.0)

    def g_wait(self, sec):
        for _ in range(int(sec * FPS)):
            yield

    def g_par(self, *gens):
        gens = list(gens)
        while gens:
            gens = [g for g in gens if next(g, "done") != "done"]
            yield

    def _yaw(self, h, direction):
        want = math.atan2(direction[1], direction[0]) - h.get("jaw0", 0.0)
        return h["yaw"] + (want - h["yaw"] + math.pi / 2) % math.pi - math.pi / 2

    def _pinch(self, h, half_along=None, along=None):
        """Pin every towel particle under the closed jaws (all layers). Returns how many."""
        pq = self.s0.particle_q.numpy()
        d = pq - h["tip"]
        if half_along is None:
            idx = np.where((np.linalg.norm(d[:, :2], axis=1) < GRIP_R) & (np.abs(d[:, 2]) < 3.0))[0]
        else:                                                       # a band the width of the jaws, along the edge
            a = np.array([along[0], along[1]], float); c = np.array([-a[1], a[0]])
            idx = np.where((np.abs(d[:, :2] @ a) < half_along) & (np.abs(d[:, :2] @ c) < 1.8) & (np.abs(d[:, 2]) < 3.5))[0]
        self.pin_p[h["name"]] = (idx, pq[idx] - h["tip"])
        self._flags()
        return len(idx)

    def _unpin(self, h):
        self.pin_p.pop(h["name"], None)
        self._flags()
        h["grip"] = 0.0

    def _flags(self):
        fl = self.flags0.copy()
        for idx, _ in self.pin_p.values():
            fl[idx] = fl[idx] & ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(fl)

    def _grasp_towel(self, h, uv, jaw_dir, band=None):
        g = self.s0.particle_q.numpy()[self.node(*uv)].astype(float)
        if not reachable(g, h["name"]):
            raise RuntimeError(f"{h['name']} hand: towel point is {np.linalg.norm(g[:2] - BASES_XY[h['name']]):.0f} cm from its base, reach is {R_PICK:.0f}")
        yield from self.g_move(h, g + np.array([0.0, 0.0, 12.0]), yaw=self._yaw(h, jaw_dir))
        g = self.s0.particle_q.numpy()[self.node(*uv)].astype(float)
        yield from self.g_move(h, np.array([g[0], g[1], max(g[2] - 0.3, TABLE_TOP + 0.5)]), speed=12.0, tol=0.3)
        h["grip"] = 1.0
        yield from self.g_wait(0.3)
        n = self._pinch(h, half_along=band, along=jaw_dir)
        if n < 3:
            raise RuntimeError(f"{h['name']} hand: only {n} towel particles under the jaws - no grasp")
        h["n_pinched"] = n

    def g_drag(self, st):
        """Both hands pinch the towel's near edge and pull it with a trapezoid speed profile
        (accelerate, cruise, decelerate - an abrupt stop topples tall objects)."""
        hl, hr = self.hands["left"], self.hands["right"]
        d = np.array([float(st["dir"][0]), float(st["dir"][1]), 0.0]); d /= np.linalg.norm(d)
        D, acc, vmax = float(st["dist_cm"]), float(st["accel"]), float(st["vmax"])
        yield from self.g_par(self._grasp_towel(hl, st["grasp_uv"]["left"], d), self._grasp_towel(hr, st["grasp_uv"]["right"], d))
        c0 = self.s0.particle_q.numpy().mean(0)
        lift = np.array([0.0, 0.0, float(st.get("lift_cm", 1.0))])
        yield from self.g_par(self.g_move(hl, hl["tgt"] + lift, speed=6.0, tol=0.3), self.g_move(hr, hr["tgt"] + lift, speed=6.0, tol=0.3))
        t_acc = vmax / acc
        if acc * t_acc * t_acc > D:
            t_acc = math.sqrt(D / acc); vmax = acc * t_acc
        t_cruise = (D - acc * t_acc * t_acc) / vmax
        T = 2 * t_acc + t_cruise
        s0 = {n: self.hands[n]["tgt"].copy() for n in ("left", "right")}
        for i in range(1, int(math.ceil(T * FPS)) + 1):
            t = min(i * DT, T)
            x = (0.5 * acc * t * t if t < t_acc else
                 0.5 * acc * t_acc * t_acc + vmax * (t - t_acc) if t < t_acc + t_cruise else
                 D - 0.5 * acc * (T - t) ** 2)
            for n in s0:
                self.hands[n]["tgt"] = s0[n] + d * x
            yield
        yield from self.g_wait(0.6)
        yield from self.g_par(self.g_move(hl, hl["tgt"] - lift, speed=6.0, tol=0.3), self.g_move(hr, hr["tgt"] - lift, speed=6.0, tol=0.3))
        self._unpin(hl); self._unpin(hr)
        yield from self.g_wait(0.3)
        moved = self.s0.particle_q.numpy().mean(0) - c0
        if self.verbose:
            print(f"[twin-sled] t={self.t:5.1f}s pull: {D:.0f} cm in {T:.1f}s (accel {acc:.0f} cm/s2, top speed {vmax:.0f} cm/s), two hands "
                  f"({hl['n_pinched']} + {hr['n_pinched']} towel particles); towel centre moved ({moved[0]:.1f},{moved[1]:.1f})", flush=True)
        self.log.append(dict(op="drag", t=round(self.t, 2), pulled_cm=D, duration_s=round(T, 2), towel_moved=[round(float(v), 2) for v in moved[:2]]))
        up = np.array([0.0, 0.0, 14.0])
        yield from self.g_par(self.g_move(hl, hl["tgt"] + up, speed=20.0), self.g_move(hr, hr["tgt"] + up, speed=20.0))
        yield from self.g_par(self.g_home(hl), self.g_home(hr))

    def g_pick_place(self, st):
        """Top grasp of an object; set it down in the box at its planned slot. The nearer arm does it."""
        o = self.objs[st["name"]]
        w, dd, hh = o["dims"]
        p, tilt0 = self.obj_state(o)
        name = st.get("hand") or ("left" if p[0] < 0 else "right")
        if not reachable(p, name):
            name = "left" if name == "right" else "right"
        h = self.hands[name]
        if not reachable(p, name):
            raise RuntimeError(f"pick {st['name']}: {reach_dist(p):.0f} cm from the nearest arm's base, reach is {R_PICK:.0f}")
        if tilt0 > TILT_OK:
            raise RuntimeError(f"pick {st['name']}: it has fallen over (tilt {tilt0:.0f} deg) - no top grasp")
        # where and how this object is grasped (props.grasp): a block or the bottle's neck at the centre,
        # the mug by its rim wall opposite the handle, jaws across the wall
        off, jaw, depth = props.grasp(o["spec"])
        jaw = st.get("jaw") or jaw                                  # the motion layer may have chosen the other way
        Ro = q2R(self.s0.body_q.numpy()[o["body"]].astype(float)[3:7])
        psi = math.atan2(Ro[1, 0], Ro[0, 0])                        # how far the object has turned about the vertical
        gz = max(0.0, hh / 2 - depth)
        g_off = Ro @ np.array([off[0], off[1], 0.0]); g_off[2] = 0.0
        yield from self.g_move(h, p + g_off + np.array([0.0, 0.0, hh / 2 + 12.0]), yaw=self._yaw(h, (Ro @ np.array([jaw[0], jaw[1], 0.0]))[:2]))
        yield from self.g_move(h, p + g_off + np.array([0.0, 0.0, gz]), speed=12.0, tol=0.3)
        h["grip"] = 0.6
        yield from self.g_wait(0.3)
        bq = self.s0.body_q.numpy()[o["body"]].astype(float)
        g_off = q2R(bq[3:7]) @ np.array([off[0], off[1], 0.0])
        err = float(np.linalg.norm((h["tip"] - bq[:3] - g_off)[:2]))
        if err > 1.5:
            raise RuntimeError(f"pick {st['name']}: tip {err:.1f} cm off the grasp point")
        self.pin_b[name] = (o["body"], h["R"].T @ (bq[:3] - h["tip"]), h["R"].T @ q2R(bq[3:7]))
        self._body_kin(o["body"], True)
        top = self.floor + self.bh + hh + 8.0
        # set it down square to the box (handle along +x, as the planner's footprint assumes): turn the
        # hand back by the object's yaw on the way; the grasp point then sits at `off` from the slot
        goal = np.array([float(st["to_xy"][0]) + off[0], float(st["to_xy"][1]) + off[1], self.floor + hh / 2 + gz + 0.3])
        yield from self.g_move(h, np.array([h["tgt"][0], h["tgt"][1], top]), speed=20.0)
        yield from self.g_move(h, np.array([goal[0], goal[1], top]), speed=24.0, yaw=h["yaw"] - psi)
        yield from self.g_move(h, goal, speed=10.0)
        self.pin_b.pop(name)
        self._body_kin(o["body"], False)
        h["grip"] = 0.0
        yield from self.g_wait(0.3)
        if self.verbose:
            p2, _ = self.obj_state(o)
            R2 = q2R(self.s0.body_q.numpy()[o["body"]].astype(float)[3:7])
            print(f"[twin-sled] t={self.t:5.1f}s pack {st['name']} ({name} hand): set at ({p2[0]:.1f},{p2[1]:.1f}), plan ({st['to_xy'][0]:.1f},{st['to_xy'][1]:.1f}); "
                  f"it had turned {math.degrees(psi):.0f} deg, set down at {math.degrees(math.atan2(R2[1, 0], R2[0, 0])):.0f} deg", flush=True)
        self.log.append(dict(op="pick_place", name=st["name"], hand=name, t=round(self.t, 2)))
        yield from self.g_move(h, np.array([h["tgt"][0], h["tgt"][1], top]), speed=20.0)
        yield from self.g_home(h)

    def _body_kin(self, body, on):
        fl = self.model.body_flags.numpy(); fl[body] = int(newton.BodyFlags.KINEMATIC if on else newton.BodyFlags.DYNAMIC)
        self.model.body_flags.assign(fl)
        self.solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    def g_fold(self, st):
        """One fold with both hands: each carries a corner on ITS side of the towel onto its partner,
        at the same time. The hands never cross."""
        def one(h, frm, to):
            yield from self._grasp_towel(h, frm, (1.0, 0.0))
            pq = self.s0.particle_q.numpy()
            a, b_ = pq[self.node(*frm)].astype(float), pq[self.node(*to)].astype(float)
            if not reachable(b_, h["name"]):
                raise RuntimeError(f"fold: target {np.linalg.norm(b_[:2] - BASES_XY[h['name']]):.0f} cm from the {h['name']} base")
            h["fold_goal"] = (a, b_)
        hl, hr = self.hands["left"], self.hands["right"]
        mv = {m["hand"]: m for m in st["moves"]}
        yield from self.g_par(one(hl, mv["left"]["from"], mv["left"]["to"]), one(hr, mv["right"]["from"], mv["right"]["to"]))
        # Carry each corner on a SEMICIRCLE about the fold line (the mid-point between where it is and
        # where it goes): the flap swings over like a page and nothing pulls on the part that stays.
        # A low straight carry dragged the lower layers along (second fold came out 27 cm, plan 15).
        n = max(1, int(math.pi * 0.5 * float(np.linalg.norm(hl["fold_goal"][1][:2] - hl["fold_goal"][0][:2])) / float(E("SLED_FOLD_SPEED", "12")) * FPS))
        for i in range(1, n + 1):
            th = math.pi * i / n
            for h in (hl, hr):
                a, b_ = h["fold_goal"]
                mid, r = 0.5 * (a[:2] + b_[:2]), 0.5 * (b_[:2] - a[:2])
                xy = mid - r * math.cos(th)
                z0 = a[2] + (b_[2] + 0.8 - a[2]) * i / n
                h["tgt"] = np.array([xy[0], xy[1], z0 + float(np.linalg.norm(r)) * math.sin(th) * float(E("SLED_FOLD_ARC", "0.9"))])
            yield
        yield from self.g_wait(0.3)
        self._unpin(hl); self._unpin(hr)
        yield from self.g_wait(0.4)
        pq = self.s0.particle_q.numpy()
        res = [float(np.linalg.norm(pq[self.node(*mv[n]["from"])][:2] - pq[self.node(*mv[n]["to"])][:2])) for n in ("left", "right")]
        up_ = np.array([0.0, 0.0, 10.0])
        yield from self.g_par(self.g_move(hl, hl["tgt"] + up_, speed=20.0), self.g_move(hr, hr["tgt"] + up_, speed=20.0))
        yield from self.g_wait(0.6)
        x0, x1, y0, y1, zt = self.towel_box()
        if self.verbose:
            print(f"[twin-sled] t={self.t:5.1f}s fold {st['name']}: towel now {x1 - x0:.0f} x {y1 - y0:.0f} cm (plan {st['to_cm'][0]:.0f} x {st['to_cm'][1]:.0f}), "
                  f"{zt - TABLE_TOP:.1f} cm high; corners landed {res[0]:.1f} and {res[1]:.1f} cm from their partners", flush=True)
        self.log.append(dict(op="fold", name=st["name"], t=round(self.t, 2), size_cm=[round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
                             residual_cm=[round(r, 2) for r in res]))

    def g_pack_towel(self, st):
        """Both hands take the folded towel by its two short edges (all layers), carry it level
        over the rim and set it down in its planned place."""
        hl, hr = self.hands["left"], self.hands["right"]
        gu = st["grasp_uv"]
        yield from self.g_par(self._grasp_towel(hl, gu["left"], (0.0, 1.0), band=float(st.get("band_half_cm", 4.0))),
                              self._grasp_towel(hr, gu["right"], (0.0, 1.0), band=float(st.get("band_half_cm", 4.0))))
        c = 0.5 * (hl["tip"] + hr["tip"])
        top = self.floor + self.bh + float(st.get("clear_cm", 10.0))
        goal = np.array([float(st["to_xy"][0]), float(st["to_xy"][1])])
        d = np.array([goal[0] - c[0], goal[1] - c[1], 0.0])
        g0 = {h["name"]: h["tgt"].copy() for h in (hl, hr)}

        def both(z, mv, sp_):
            return self.g_par(*[self.g_move(h, np.array([g0[h["name"]][0] + mv[0], g0[h["name"]][1] + mv[1], z]), speed=sp_) for h in (hl, hr)])
        # a little tension first, so the folded towel does not belly between the hands
        tn = float(st.get("tension_cm", 1.5))
        yield from self.g_par(self.g_move(hl, hl["tgt"] + np.array([-tn, 0.0, 0.5]), speed=4.0), self.g_move(hr, hr["tgt"] + np.array([tn, 0.0, 0.5]), speed=4.0))
        g0 = {h["name"]: h["tgt"].copy() for h in (hl, hr)}
        yield from both(top, np.zeros(3), 10.0)
        yield from self.g_wait(0.5)
        sag = float(min(hl["tip"][2], hr["tip"][2]) - self.s0.particle_q.numpy()[:, 2].min())
        # the towel bellies between the hands: carry it high enough that its LOWEST point clears the rim
        # (at a fixed height the belly, 11-12 cm deep, caught the front wall and the towel landed askew)
        need = self.floor + self.bh + sag + float(st.get("rim_clear_cm", 3.0))
        if need > top:
            top = need
            yield from both(top, np.zeros(3), 10.0)
        # steer by the TOWEL, not by the hands: hanging from its two short edges, the folded towel's
        # centre sat ~4 cm from the line between the hands (the layers hang unevenly), and that offset
        # was carried into the box (runs at 84 % / 82 % in place). Measured in the air, the robot sees it.
        if st.get("steer", E("SLED_PACK_STEER", "centroid")) == "centroid":
            cq = self.s0.particle_q.numpy()[:, :2].mean(axis=0)
            d = np.array([goal[0] - cq[0], goal[1] - cq[1], 0.0])
            if self.verbose:
                hc = 0.5 * (hl["tip"] + hr["tip"])
                print(f"[twin-sled] t={self.t:5.1f}s pack towel: in the air its centre is ({cq[0] - hc[0]:+.1f}, {cq[1] - hc[1]:+.1f}) cm from the hands' midpoint; steering by the centre", flush=True)
        yield from both(top, d, 14.0)
        yield from both(self.floor + float(st.get("release_cm", 1.5)), d, 8.0)
        self._unpin(hl); self._unpin(hr)
        yield from self.g_wait(0.4)
        if self.verbose:
            print(f"[twin-sled] t={self.t:5.1f}s pack towel: two hands {np.linalg.norm(hl['tip'] - hr['tip']):.0f} cm apart ({hl['n_pinched']} + {hr['n_pinched']} particles), "
                  f"towel sagged {sag:.1f} cm between them in the air", flush=True)
        self.log.append(dict(op="pack_towel", t=round(self.t, 2), sag_cm=round(sag, 1)))
        up = np.array([0.0, 0.0, 18.0])
        yield from self.g_par(self.g_move(hl, hl["tgt"] + up, speed=18.0), self.g_move(hr, hr["tgt"] + up, speed=18.0))
        yield from self.g_par(self.g_home(hl), self.g_home(hr))

    # ---- run + audit --------------------------------------------------------------------
    def run(self):
        drag = next((s for s in self.plan["plan"] if s["op"] == "drag"), None)
        dvec = (np.array([float(drag["dir"][0]), float(drag["dir"][1])]) / np.linalg.norm(drag["dir"]) * float(drag["dist_cm"])) if drag else np.zeros(2)
        ops = {"drag": self.g_drag, "pick_place": self.g_pick_place, "fold": self.g_fold, "pack_towel": self.g_pack_towel}

        def main():
            yield from self.g_wait(1.0)
            for st in self.plan["plan"]:
                if st["op"] == "pick_place" and st["name"] in self.plan.get("preplaced", {}):
                    continue
                yield from ops[st["op"]](st)
            yield from self.g_wait(1.5)

        t0, ok, err = time.time(), True, None
        try:
            for _ in main():
                self.step_frame()
        except RuntimeError as e:
            ok, err = False, str(e)
        res = dict(completed=ok, error=err, objects={}, sim_s=round(self.t, 1), wall_s=round(time.time() - t0, 1), log=self.log,
                   ik_err_max_cm=round(self.ik_err_max, 2), tip_min_cm=round(self.tip_min, 1))
        fast = self.arms is None                                    # a planner rollout: only the pull is simulated
        for nm, o in self.objs.items():
            p, tilt = self.obj_state(o)
            in_box = bool(abs(p[0] - BOX_XY[0]) < self.hin and abs(p[1] - BOX_XY[1]) < self.hin and p[2] < self.floor + o["dims"][2])
            rode = nm not in self.plan.get("preplaced", {}) and nm in self.plan.get("riders", [])
            slip = float(np.linalg.norm(p[:2] - (o["start"][:2] + dvec))) if rode else 0.0
            if fast and rode:                                       # after the pull: can it be picked from where it ended?
                status = "toppled" if tilt > TILT_OK else "unreachable" if not reachable(p) else "ok"
            elif fast:
                status = "ok"
            else:
                status = "ok" if (in_box and tilt < TILT_OK) else "toppled" if tilt >= TILT_OK else "not in box"
            res["objects"][nm] = dict(mode="rode" if rode else "packed first", xy=[round(float(p[0]), 1), round(float(p[1]), 1)], tilt_deg=round(tilt, 1),
                                      slip_cm=round(slip, 1), from_base_cm=round(reach_dist(p), 1), in_box=in_box, status=status)
        res["success"] = bool(ok and all(v["status"] == "ok" for v in res["objects"].values()))
        tp = self.plan.get("towel_place")
        if tp and not fast:
            q = self.s0.particle_q.numpy()
            inb = (np.abs(q[:, 0] - BOX_XY[0]) < self.hin) & (np.abs(q[:, 1] - BOX_XY[1]) < self.hin) & (q[:, 2] < self.floor + self.bh)
            r = tp["rect"]                                           # planned place on the box floor: x0, x1, y0, y1
            inr = (q[:, 0] > r[0] - 1.5) & (q[:, 0] < r[1] + 1.5) & (q[:, 1] > r[2] - 1.5) & (q[:, 1] < r[3] + 1.5) & inb
            qi = q[inb] if inb.any() else q
            res["towel"] = dict(in_box_frac=float(inb.mean()), in_planned_place_frac=float(inr.mean()),
                                size_cm=[round(float(np.ptp(qi[:, 0])), 1), round(float(np.ptp(qi[:, 1])), 1)], plan_cm=tp["size_cm"],
                                top_above_floor_cm=round(float(qi[:, 2].max() - self.floor), 1), folds=tp["folds"])
            res["success"] = bool(res["success"] and res["towel"]["in_box_frac"] >= 0.98 and res["towel"]["in_planned_place_frac"] >= 0.9)
        gs = self.plan.get("goal")
        if gs and not fast:
            res["goal_check"] = self.goal_check(gs, res)
            packed = {s_["name"] for s_ in self.plan["plan"] if s_["op"] == "pick_place"}
            # the goal's predicates must hold, and every object the plan moved must have landed well
            res["success"] = bool(ok and all(res["goal_check"].values()) and
                                  all(v["status"] == "ok" for n, v in res["objects"].items() if n in packed))
        if self.rec:
            self.rec.save()
        if self.out:
            json.dump(res, open(os.path.join(self.out, "sled_audit.json"), "w"), indent=1)
        return res

    def goal_check(self, gs, res):
        """Each goal predicate, measured on the end state (the same vocabulary goal_parse.py writes)."""
        out = {}
        for o in gs["in_box"]:
            v = res["objects"][o]
            out[f"in_box({o})"] = bool(v["in_box"] and v["tilt_deg"] < TILT_OK)
        if gs["folds_min"]:
            folds = [e for e in self.log if e["op"] == "fold"]
            L = float(self.plan["scene"]["towel"]["size_cm"][1])
            k = gs["folds_min"]
            # k folds done, and the towel's length shows it: at most 1.25x the planned folded length
            size = res["towel"]["size_cm"] if "towel" in res else self.towel_size()
            out[f"folded(towel, {k})"] = bool(len(folds) >= k and size[1] <= 1.25 * L / 2 ** len(folds))
        if gs["towel_in_box"]:
            t = res.get("towel", {})
            out["towel_in_box"] = bool(t.get("in_box_frac", 0) >= 0.98 and t.get("in_planned_place_frac", 0) >= 0.9)
        t_pack = {e["name"]: e["t"] for e in self.log if e["op"] == "pick_place"}
        for a, b in gs["before"]:
            out[f"before({a}, {b})"] = bool(a in t_pack and b in t_pack and t_pack[a] < t_pack[b])
        return out

    def towel_size(self):
        x0, x1, y0, y1, _ = self.towel_box()
        return [round(float(x1 - x0), 1), round(float(y1 - y0), 1)]


def summary(res):
    return "; ".join(f"{n} ({v['mode']}) -> {v['status']}" for n, v in res["objects"].items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = SledSim(json.load(open(a.plan)), a.out).run()
    if "towel" in r:
        t = r["towel"]
        print(f"[twin-sled] TOWEL: {t['in_box_frac']:.0%} in the box, {t['in_planned_place_frac']:.0%} in its planned place, lies {t['size_cm'][0]:.0f} x {t['size_cm'][1]:.0f} cm "
              f"(plan {t['plan_cm'][0]:.0f} x {t['plan_cm'][1]:.0f} after {t['folds']} folds), top {t['top_above_floor_cm']:.1f} cm above the floor", flush=True)
    for p, v in r.get("goal_check", {}).items():
        print(f"[twin-sled] GOAL {p}: {'holds' if v else 'FAILS'}", flush=True)
    print(f"[twin-sled] SLED result: {'SUCCESS' if r['success'] else 'FAIL'} - {summary(r)}; {r['sim_s']:.0f}s sim / {r['wall_s']:.0f}s wall, "
          f"worst IK lag {r['ik_err_max_cm']:.1f} cm, hands closest {r['tip_min_cm']:.1f} cm" + (f"; ABORTED: {r['error']}" if r["error"] else ""), flush=True)
    sys.exit(0 if r["success"] else 1)
