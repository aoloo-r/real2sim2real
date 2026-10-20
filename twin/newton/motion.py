"""Motion layer of the sled TAMP: is an action's arm motion collision-free, and what is it?

Each check follows the executor's motion script for that action (twin_sled.py) segment by segment:
free-space segments (home -> above the grasp, carrying over the rim, back home) are PLANNED by cuRobo
around the table, the box, the objects, the towel, the other arm and the carried object; contact
segments (descend, lift, lower, pull, the fold's arc) are Cartesian paths whose every waypoint must
have a collision-free IK continuous with the last (the gripper may touch what it grasps).

The world is built from a planner state (tamp_sym.S): what is in the box sits at its slot, riders
stand where the last rollout left them, the towel is a slab (thicker per fold). Objects are their
bounding boxes (the mug's includes its handle).

cuRobo runs in another env: motion_server.py, started once, spoken to over a pipe.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props  # noqa: E402

CUROBO_PY = os.environ.get("CUROBO_PY", os.path.expanduser("~/miniforge3/envs/isaaclab/bin/python"))
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_TOP = 20.0
UR_HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
BASE_Z = TABLE_TOP + float(os.environ.get("SLED_BASE_LIFT_CM", "4"))   # arm mount height above the table (twin_sled.BASE_LIFT)
HANDS = ("right", "left")


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def q2R(q):                                       # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R):                                       # wxyz
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1); return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s; q[i + 1] = 0.25 * s; q[j + 1] = (R[j, i] + R[i, j]) / s; q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


class Motion:
    def __init__(self, bases_xy, verbose=True, world=None):
        """bases_xy: {hand: (x, y)} in cm, as in twin_sled.BASES_XY. world: fixed obstacles (cuboids)
        the park pose must clear."""
        self.verbose = verbose
        # cuRobo's UR5e base frame is the twin's base frame turned by -90 deg (FK checked to 0.1 cm)
        # MIRRORED (default): the left arm is turned 180 deg so both upper arms sit outboard; mounted
        # like the right one, the left upper arm hangs 0.2 cm above the box's left wall at home
        self.mirrored = os.environ.get("MOTION_MIRRORED", "1") == "1"
        yaw = {"right": -math.pi / 2, "left": math.pi / 2 if self.mirrored else -math.pi / 2}
        bases = {h: [float(bases_xy[h][0]) / 100, float(bases_xy[h][1]) / 100, BASE_Z / 100, yaw[h]] for h in HANDS}
        arg = json.dumps(dict(bases=bases, home=UR_HOME * 2, out_dir=os.path.join(HERE, ".motion_cache")))
        t0 = time.time()
        self.p = subprocess.Popen([CUROBO_PY, os.path.join(HERE, "motion_server.py"), arg], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.path.join(HERE, ".motion_cache.log"), "a"), text=True, bufsize=1)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"motion server did not start: {ready}")
        self.jn, self.limits, self.bases = ready.get("joints", []), ready.get("limits", []), bases
        self.calls, self.wall = 0, 0.0
        self.call(cmd="world", cuboids=world or [])
        ur = self.call(cmd="fk", q=UR_HOME * 2)["tcp"]
        self.R_home = {h: q2R(ur[h][3:]) for h in HANDS}
        # start and end every action at the executor's park pose: tip (+-36, -34, 55) cm, tool down
        park = {h: [self.pose(h, [math.copysign(36.0, bases_xy[h][0]), -34.0, 55.0])] for h in HANDS}
        # park solves are IK, not paths: any branch may be reached from the UR home (max_jump off)
        r = self.call(cmd="path", q=UR_HOME * 2, waypoints={"right": park["right"]}, gripper_contact=False, max_jump=10.0)
        if not r["ok"]:
            raise RuntimeError(f"no IK for the right park pose: {r['why']}")
        qr = r["traj"][-1][:6]
        # the left arm in the SAME branch as the right (elbow up, wrist down): seeded from the right
        # arm's solution turned about the pan axis; an unseeded IK put it folded back over its base
        best, cands = None, []
        for pan in (0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi):
            for lift_sign in (1.0, -1.0):
                l0 = [qr[0] + pan, qr[1], lift_sign * qr[2], qr[3], qr[4], qr[5]] if lift_sign > 0 else \
                     [qr[0] + pan, -math.pi - qr[1], -qr[2], -math.pi - qr[3], qr[4], qr[5]]
                rl = self.call(cmd="path", q=qr + l0, waypoints={"left": park["left"]}, gripper_contact=False, max_jump=10.0)
                if rl["ok"]:
                    ql = rl["traj"][-1][6:]
                    cands.append([round(v, 2) for v in ql])
                    # elbow up and not folded back over the base, like the right arm
                    d = abs(ql[1] - qr[1]) + abs(abs(ql[2]) - abs(qr[2]))
                    if best is None or d < best[0]:
                        best = (d, ql)
        if verbose:
            print(f"[motion] left park candidates (collision-free): {cands}", flush=True)
        if best is None:
            raise RuntimeError("no IK for the left park pose")
        self.home_q = qr + best[1]
        self.home_tcp = self.call(cmd="fk", q=self.home_q)["tcp"]
        if verbose:
            print(f"[motion] cuRobo dual-UR5e server ready ({time.time() - t0:.0f}s)", flush=True)

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("motion server exited (see twin/newton/.motion_cache.log)")
            if line.startswith("{"):
                return json.loads(line)

    def call(self, **req):
        t0 = time.time()
        if os.environ.get("MOTION_DUMP"):
            with open(os.environ["MOTION_DUMP"], "a") as f:
                f.write(json.dumps(req) + "\n")
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        r = self._read()
        self.calls += 1; self.wall += time.time() - t0
        return r

    def close(self):
        try:
            self.call(cmd="quit")
        except Exception:
            pass
        self.p.kill()

    # ---- poses ------------------------------------------------------------------------
    def pose(self, hand, tip_cm, yaw=0.0):
        """Tool down, turned by yaw about the vertical (the executor's convention) -> tcp pose (m, wxyz)."""
        R = rz(yaw) @ self.R_home[hand]
        return [float(v) / 100 for v in tip_cm] + [float(v) for v in R2q(R)]

    @staticmethod
    def jaw_yaw(direction, cur=0.0):
        want = math.atan2(direction[1], direction[0])
        return cur + (want - cur + math.pi / 2) % math.pi - math.pi / 2

    @staticmethod
    def line(a, b, n=None, step=3.0):
        """Waypoints every `step` cm (at least n) from a (excluded) to b: near an arm's own base the joints
        turn fast, and coarse steps read as a flip."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(n or 3, int(math.ceil(np.linalg.norm(b - a) / step)))
        return [a + (b - a) * i / n for i in range(1, n + 1)]


class SledMotion:
    """Motion checks for the sled operators, on the planner's symbolic states."""

    def __init__(self, D, verbose=True):
        from twin_sled import BASES_XY, BOX_XY, BOX_WALL, TABLE
        self.D, self.verbose = D, verbose
        self.BOX, self.WALL, self.TABLE = BOX_XY, BOX_WALL, TABLE
        self.hin, self.bh = 0.5 * float(D.scene["box"]["inner_cm"][0]), float(D.scene["box"]["h_cm"])
        self.floor = TABLE_TOP + 0.5
        fixed = [c for c in self.world_fixed()]
        self.m = Motion(BASES_XY, verbose, world=fixed)
        sc = D.scene
        self.hin, self.bh = 0.5 * float(sc["box"]["inner_cm"][0]), float(sc["box"]["h_cm"])
        self.floor = TABLE_TOP + 0.5
        self.cache, self.log = {}, []

    # ---- the world ----------------------------------------------------------------------
    def cub(self, name, c, half):
        return dict(name=name, pose=[c[0] / 100, c[1] / 100, c[2] / 100, 1.0, 0.0, 0.0, 0.0], dims=[2 * half[0] / 100, 2 * half[1] / 100, 2 * half[2] / 100])

    def obj_box(self, name, xy, z_base):
        o = self.D.objs[name]
        fw, fd, dx = props.footprint(o)
        h = float(o["dims_cm"][2])
        return self.cub(name, (xy[0] - dx, xy[1], z_base + h / 2), (fw / 2, fd / 2, h / 2))

    def world_fixed(self):
        (tx, ty, tz), thx, thy, thz = self.TABLE
        cs = [self.cub("table", (tx, ty, tz), (thx, thy, thz))]
        bx, by, hin, w = self.BOX[0], self.BOX[1], self.hin, self.WALL
        cs.append(self.cub("box_floor", (bx, by, self.floor - 0.25), (hin + w, hin + w, 0.25)))
        for nm, dx, dy, hx, hy in (("box_right", hin + w / 2, 0.0, w / 2, hin + w), ("box_left", -hin - w / 2, 0.0, w / 2, hin + w),
                                   ("box_back", 0.0, hin + w / 2, hin + w, w / 2), ("box_front", 0.0, -hin - w / 2, hin + w, w / 2)):
            cs.append(self.cub(nm, (bx + dx, by + dy, self.floor + self.bh / 2), (hx, hy, self.bh / 2)))
        return cs

    def world(self, st, folds=None, towel_at=None):
        """Obstacles for a planner state (cm in, metres out)."""
        cs = self.world_fixed()
        for o in st.in_box:
            cs.append(self.obj_box(o, self.D.slot[o], self.floor))
        for o in st.on_towel:
            if o not in st.toppled:
                cs.append(self.obj_box(o, st.pos[o], TABLE_TOP + 0.8))
        k = st.folds if folds is None else folds
        if not st.towel_in_box:
            W, L = self.D.W, self.D.L / 2 ** k
            c = towel_at or (self.D.cx, self.D.cy + st.towel_dy + self.D.L / 2 - L / 2)
            cs.append(self.cub("towel", (c[0], c[1], TABLE_TOP + 0.4 * 2 ** k), (W / 2, L / 2, 0.4 * 2 ** k)))
        else:
            W, L = self.D.W, self.D.L / 2 ** k
            cs.append(self.cub("towel", (self.BOX[0], self.D.strip[2] + 0.5 + L / 2, self.floor + 0.4 * 2 ** k), (W / 2, L / 2, 0.4 * 2 ** k)))
        return cs

    # ---- segment runner -----------------------------------------------------------------
    def run(self, label, segs, world):
        """segs: list of ('plan'|'path'|'world', {hand: pose or [poses]}, attach, off[, contact]). Stops at the
        first failure. contact: the gripper may touch the world (pinching cloth on the table); the arm
        links and anything carried may not. MOTION_EXPORT=file: append every problem posed and its
        outcome as JSON (astra_eval/p2_motion poses the same problems to other planners)."""
        r = self._run(label, segs, world)
        if os.environ.get("MOTION_EXPORT"):
            with open(os.environ["MOTION_EXPORT"], "a") as f:
                f.write(json.dumps(dict(label=label, world=world, segs=segs, result=dict(ok=r["ok"], why=r["why"]),
                                        trajs=[s["traj"] for s in r["segments"]]),
                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else list(o)) + "\n")
        return r

    def _run(self, label, segs, world):
        cur = [dict(c) for c in world]
        self.m.call(cmd="world", cuboids=cur)
        q, t_motion, out, held = list(self.m.home_q), 0.0, [], {}
        for i, sg in enumerate(segs):
            kind, goals, attach, off = sg[:4]
            contact = sg[4] if len(sg) > 4 else False
            if kind == "world":
                cur = [dict(c) for c in goals]
                self.m.call(cmd="world", cuboids=cur)
                continue
            if attach:
                # a carried object rides with the hand: its offset from the tcp is fixed when it is first
                # carried (at the grasp); before each later segment it is moved to where the hand now is
                tcp = self.m.call(cmd="fk", q=q)["tcp"]
                moved = False
                for hand, names in attach.items():
                    for c in cur:
                        if c["name"] in names:
                            if c["name"] not in held:
                                held[c["name"]] = np.array(c["pose"][:3]) - np.array(tcp[hand][:3])
                            else:
                                c["pose"] = [float(v) for v in np.array(tcp[hand][:3]) + held[c["name"]]] + c["pose"][3:]
                                moved = True
                if moved:
                    self.m.call(cmd="world", cuboids=cur)
                if os.environ.get("MOTION_DEBUG"):
                    print(f"[motion]   seg {i} {kind}: tcp {[round(v * 100, 1) for v in tcp[hand][:3]]}, held "
                          f"{[(c['name'], [round(v * 100, 1) for v in c['pose'][:3]]) for c in cur if c['name'] in held]}", flush=True)
            if kind == "plan":
                extra = dict(pair_tol=sg[5]) if len(sg) > 5 else {}
                r = self.m.call(cmd="plan", q=q, goals=goals, attach=attach, off=off, **extra)
            else:
                r = self.m.call(cmd="path", q=q, waypoints=goals, attach=attach, off=off, gripper_contact=contact)
            if not r["ok"]:
                return dict(ok=False, why=f"{label}: segment {i} ({kind}): {r['why']}", segments=out)
            out.append(dict(kind=kind, traj=r["traj"], dt=r.get("dt")))
            q = r["traj"][-1]
            t_motion += r.get("motion_s", 0.0)
        return dict(ok=True, why="", segments=out, motion_s=round(t_motion, 2))

    def P(self, hand, tip, yaw=0.0):
        return self.m.pose(hand, tip, yaw)

    def home(self, hand):
        return self.m.home_tcp[hand]

    # ---- operators ----------------------------------------------------------------------
    def pick_place(self, st, o, hand, jaw=None, flip=None):
        """jaw / flip None: try the allowed jaw directions (props.grasp_jaws), each with the wrist as is
        and turned 180 deg (the same grip; the wrist on the other side), keep the first that works."""
        if jaw is None or flip is None:
            r = None
            for j in ([jaw] if jaw else props.grasp_jaws(self.D.objs[o])):
                for f in ((flip,) if flip is not None else (0, 1)):
                    r = self.pick_place(st, o, hand, j, f)
                    if r["ok"]:
                        return r
            return r
        key = ("pick", o, hand, tuple(jaw), flip, st.in_box, tuple(sorted((n, round(x), round(y)) for n, (x, y) in st.pos.items())), round(st.towel_dy), st.folds)
        if key in self.cache:
            return self.cache[key]
        spec = self.D.objs[o]
        hh = float(spec["dims_cm"][2])
        off, _, depth = props.grasp(spec)
        x, y = st.pos[o]
        zb = TABLE_TOP + 0.8
        gz = max(0.0, hh / 2 - depth)
        g = np.array([x + off[0], y + off[1], zb + hh / 2 + gz])
        pre = g + np.array([0.0, 0.0, hh / 2 - gz + 12.0])
        yaw = Motion.jaw_yaw(jaw) + math.pi * flip
        top = self.floor + self.bh + hh + 8.0
        sx, sy = self.D.slot[o]
        goal = np.array([sx + off[0], sy + off[1], self.floor + hh / 2 + gz + 0.3])
        P = lambda p: self.P(hand, p, yaw)   # noqa: E731
        att = {hand: [o]}
        after = st.copy(on_towel=st.on_towel - {o}, in_box=st.in_box | {o}, pos={n: p for n, p in st.pos.items() if n != o})
        segs = [("plan", {hand: P(pre)}, {}, []),
                ("path", {hand: [P(p) for p in Motion.line(pre, g)]}, {}, [o, "towel"]),
                # lifting: the object may touch what it stood on (the towel, the table) as it leaves it
                ("path", {hand: [P(p) for p in Motion.line(g, [g[0], g[1], top])]}, att, ["towel", "table"]),
                ("plan", {hand: P([goal[0], goal[1], top])}, att, []),
                ("path", {hand: [P(p) for p in Motion.line([goal[0], goal[1], top], goal + np.array([0.0, 0.0, 1.0]))]}, att, []),
                ("world", self.world(after), None, None),
                ("path", {hand: [P(p) for p in Motion.line(goal, [goal[0], goal[1], top])]}, {}, [o]),
                ("plan", {hand: self.home(hand)}, {}, [])]
        r = self.run(f"pick_place({o}, {hand} arm, jaws {jaw[0]:+.0f},{jaw[1]:+.0f}{', wrist turned' if flip else ''})", segs, self.world(st))
        r.update(hand=hand, jaw=list(jaw), flip=flip, yaw=yaw)
        self.cache[key] = r
        return r

    @staticmethod
    def flips():
        return ({"right": 0, "left": 0}, {"right": 0, "left": 1}, {"right": 1, "left": 0}, {"right": 1, "left": 1})

    def two_arm(self, fn, *a):
        """A two-arm operator with each wrist as is or turned 180 deg; the first combination that works."""
        r = None
        for f in self.flips():
            r = fn(*a, flip=f)
            if r["ok"]:
                break
        return r

    def fold(self, st, k, flip=None):
        if flip is None:
            return self.two_arm(self.fold, st, k)
        key = ("fold", k, round(st.towel_dy), st.in_box, tuple(flip.values()))
        if key in self.cache:
            return self.cache[key]
        D = self.D
        y0 = D.cy + st.towel_dy - D.L / 2                       # far edge (v = 0)
        corners = {}
        for mv in __import__("tamp_sled").fold_moves(k):
            u, v = mv["from"]; u2, v2 = mv["to"]
            a = np.array([D.cx - D.W / 2 + u * D.W, y0 + v * D.L, TABLE_TOP + 0.5])
            b = np.array([D.cx - D.W / 2 + u2 * D.W, y0 + v2 * D.L, TABLE_TOP + 0.5 + 0.8 * 2 ** (k - 1)])
            corners[mv["hand"]] = (a, b)
        yaw = 0.0                                                # jaws along x (the executor grasps fold corners with jaw_dir (1, 0))
        P = lambda h, p: self.P(h, p, yaw + math.pi * flip[h])   # noqa: E731
        pre = {h: P(h, a + np.array([0.0, 0.0, 12.0])) for h, (a, b) in corners.items()}
        down = {h: [P(h, p) for p in Motion.line(a + np.array([0.0, 0.0, 12.0]), a)] for h, (a, b) in corners.items()}
        arc = {}
        for h, (a, b) in corners.items():                        # the executor's semicircle about the fold line
            mid, r = 0.5 * (a[:2] + b[:2]), 0.5 * (b[:2] - a[:2])
            pts = []
            for i in range(1, 13):
                th = math.pi * i / 12
                xy = mid - r * math.cos(th)
                z = a[2] + (b[2] + 0.8 - a[2]) * i / 12 + float(np.linalg.norm(r)) * math.sin(th) * 0.9
                pts.append(P(h, [xy[0], xy[1], z]))
            arc[h] = pts
        up = {h: [P(h, p) for p in Motion.line(b + np.array([0.0, 0.0, 0.8]), b + np.array([0.0, 0.0, 10.8]), 3)] for h, (a, b) in corners.items()}
        segs = [("plan", pre, {}, []), ("path", down, {}, ["towel"], True), ("path", arc, {}, ["towel"], True),
                ("path", up, {}, ["towel"], True), ("plan", {h: self.home(h) for h in HANDS}, {}, [])]
        r = self.run(f"fold {k}", segs, self.world(st, folds=k - 1))
        r.update(flip=dict(flip), yaw={h: yaw + math.pi * flip[h] for h in HANDS})
        self.cache[key] = r
        return r

    def pack_towel(self, st, k, flip=None):
        if flip is None:
            return self.two_arm(self.pack_towel, st, k)
        key = ("pack_towel", k, round(st.towel_dy), st.in_box, tuple(flip.values()))
        if key in self.cache:
            return self.cache[key]
        D = self.D
        L = D.L / 2 ** k
        cy = D.cy + st.towel_dy + D.L / 2 - L / 2               # folded far-to-near: it lies against the near edge
        z = TABLE_TOP + 0.5
        grasp = {"left": np.array([D.cx - 0.45 * D.W, cy, z]), "right": np.array([D.cx + 0.45 * D.W, cy, z])}
        yaw0 = Motion.jaw_yaw((0.0, 1.0))
        P = lambda h, p: self.P(h, p, yaw0 + math.pi * flip[h])   # noqa: E731
        top = self.floor + self.bh + 0.5 * L + 3.0 + 2.0         # its belly hangs ~half its free length (measured sag 11 cm for 15 cm)
        top = max(top, self.floor + self.bh + float(os.environ.get("MOTION_PACK_CARRY_CM", "10")))
        gy = D.strip[2] + 0.5 + L / 2
        dxy = np.array([self.BOX[0] - D.cx, gy - cy, 0.0])
        segs = [("plan", {h: P(h, g + np.array([0.0, 0.0, 12.0])) for h, g in grasp.items()}, {}, []),
                ("path", {h: [P(h, p) for p in Motion.line(g + np.array([0.0, 0.0, 12.0]), g)] for h, g in grasp.items()}, {}, ["towel"], True),
                ("path", {h: [P(h, p) for p in Motion.line(g, [g[0], g[1], top])] for h, g in grasp.items()}, {}, ["towel"], True),
                # the carry: a planned two-arm move (a straight one passes each arm's own base, where it
                # would have to flip); the towel hangs between the hands, so their distance may change
                # by at most 3 cm on the way
                (("plan", {h: P(h, np.array([g[0], g[1], top]) + dxy) for h, g in grasp.items()}, {}, ["towel"], False, 0.03)
                 if os.environ.get("MOTION_PACK_CARRY", "path") == "plan" else
                 ("path", {h: [P(h, p) for p in Motion.line([g[0], g[1], top], np.array([g[0], g[1], top]) + dxy)] for h, g in grasp.items()}, {}, ["towel"], True)),
                ("path", {h: [P(h, p) for p in Motion.line(np.array([g[0], g[1], top]) + dxy, np.array([g[0], g[1], self.floor + 1.5]) + dxy)] for h, g in grasp.items()}, {}, ["towel"], True),
                ("path", {h: [P(h, p) for p in Motion.line(np.array([g[0], g[1], self.floor + 1.5]) + dxy, np.array([g[0], g[1], self.floor + 19.5]) + dxy, 4)] for h, g in grasp.items()}, {}, ["towel"], True),
                ("plan", {h: self.home(h) for h in HANDS}, {}, [])]
        r = self.run(f"pack_towel({k} folds)", segs, self.world(st))
        r.update(flip=dict(flip), yaw={h: yaw0 + math.pi * flip[h] for h in HANDS})
        self.cache[key] = r
        return r

    def pull(self, st, d, flip=None):
        if flip is None:
            return self.two_arm(self.pull, st, d)
        key = ("pull", round(d), round(st.towel_dy), st.in_box, st.on_towel, tuple(flip.values()))
        if key in self.cache:
            return self.cache[key]
        D = self.D
        y_near = D.cy + st.towel_dy + D.L / 2 - 0.02 * D.L
        grasp = {"left": np.array([D.cx - 0.25 * D.W, y_near, TABLE_TOP + 0.5]), "right": np.array([D.cx + 0.25 * D.W, y_near, TABLE_TOP + 0.5])}
        yaw = Motion.jaw_yaw((0.0, 1.0))
        P = lambda h, p: self.P(h, p, yaw + math.pi * flip[h])   # noqa: E731
        lift = np.array([0.0, 0.0, 1.0])
        moved = st.copy(towel_dy=st.towel_dy + d, pos={o: (p[0], p[1] + d) for o, p in st.pos.items()})
        segs = [("plan", {h: P(h, g + np.array([0.0, 0.0, 12.0])) for h, g in grasp.items()}, {}, []),
                ("path", {h: [P(h, p) for p in Motion.line(g + np.array([0.0, 0.0, 12.0]), g)] for h, g in grasp.items()}, {}, ["towel"], True),
                ("path", {h: [P(h, p) for p in Motion.line(g + lift, g + lift + np.array([0.0, d, 0.0]), max(4, int(d / 4)))] for h, g in grasp.items()}, {}, ["towel"], True),
                ("world", self.world(moved), None, None),
                ("path", {h: [P(h, p) for p in Motion.line(g + np.array([0.0, d, 0.0]), g + np.array([0.0, d, 14.0]), 3)] for h, g in grasp.items()}, {}, ["towel"], True),
                ("plan", {h: self.home(h) for h in HANDS}, {}, [])]
        r = self.run(f"pull {d:.0f} cm", segs, self.world(st))
        r.update(flip=dict(flip), yaw={h: yaw + math.pi * flip[h] for h in HANDS})
        self.cache[key] = r
        return r
