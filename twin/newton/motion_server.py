"""Collision-aware motion for the two UR5e arms of the twins: a cuRobo process the planner talks to.

Runs in the isaaclab env (cuRobo + torch); the twin and the task planner run in the newton env and
talk to it through motion.py (one JSON request per line on stdin, one JSON reply per line on stdout).
Units here are METRES and the frame is the twin's world frame (twin cm / 100).

The robot: two cuRobo UR5e arms (ur5e.urdf) on a common world link, bases where the twin puts them,
each with the Robotiq 2F-85 of the twin as collision spheres on tool0 and a `tcp` link at the
fingertip pinch point. Checked against the twin's own UR5e model (newton MJCF): same joint
convention; tool0 FK agrees to 0.1 cm once the base is turned by -90 deg; the pinch point is 12.35 cm
along tool0 z. Joint order: right arm (no suffix) then left arm (suffix _1).

Requests:
  {"cmd": "world", "cuboids": [{"name", "dims": [x,y,z], "pose": [x,y,z,qw,qx,qy,qz]}]}
  {"cmd": "fk", "q": [12]}                                        -> tcp poses
  {"cmd": "plan", "q": [12], "goals": {"right": pose7|null, "left": pose7|null},
                  "attach": {"right": [names]}, "off": [names]}   -> collision-free joint trajectory
        free-space motion; an arm with goal null holds its tcp where it is.
  {"cmd": "path", "q": [12], "waypoints": {"right": [pose7...], "left": [...]},
                  "attach": {...}, "off": [names], "gripper_contact": true}
        a Cartesian path (descend, lift, carry, fold arc), solved waypoint by waypoint by IK seeded
        from the previous one and checked for collisions; with gripper_contact the gripper (tool0)
        may touch the world, as it must to pinch cloth on the table; the arm links and any held
        object may not. `off` disables named obstacles (the object being grasped).
Reply: {"ok": bool, "why": str, "traj": [[12]...], "dt": s, ...}
"""
import sys
OUT, sys.stdout = sys.stdout, sys.stderr          # replies only on the real stdout; library chatter goes to stderr
import copy
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

TCP_Z = 0.1235                     # tool0 -> fingertip pinch point (measured on the twin's 2F-85)
ARMS = (("right", ""), ("left", "_1"))
# the 2F-85, open, in tool0 coordinates (jaw axis = tool0 x; measured from the twin's gripper bodies)
GRIPPER_SPHERES = [dict(center=[0.0, 0.0, 0.03], radius=0.045), dict(center=[0.0, 0.0, 0.07], radius=0.035),
                   dict(center=[0.045, 0.0, 0.065], radius=0.022), dict(center=[-0.045, 0.0, 0.065], radius=0.022),
                   dict(center=[0.052, 0.0, 0.10], radius=0.016), dict(center=[-0.052, 0.0, 0.10], radius=0.016),
                   dict(center=[0.055, 0.0, 0.125], radius=0.013), dict(center=[-0.055, 0.0, 0.125], radius=0.013)]


def log(*a):
    print("[motion]", *a, file=sys.stderr, flush=True)


def build_robot(bases, home, out_dir):
    """bases: {arm: [x, y, z, yaw]} in metres / rad (cuRobo base frame in the world)."""
    src_dir = join_path(get_assets_path(), "robot/ur_description")
    src = ET.parse(join_path(src_dir, "ur5e.urdf")).getroot()
    root = ET.Element("robot", name="dual_ur5e")
    ET.SubElement(root, "link", name="world")
    for arm, sf in ARMS:
        for el in src:
            if el.tag not in ("link", "joint"):
                continue
            e = copy.deepcopy(el)
            e.set("name", e.get("name") + sf)
            if e.tag == "joint":
                for k in ("parent", "child"):
                    e.find(k).set("link", e.find(k).get("link") + sf)
            root.append(e)
        x, y, z, yaw = bases[arm]
        j = ET.SubElement(root, "joint", name=f"world_to_base{sf}", type="fixed")
        ET.SubElement(j, "parent", link="world"); ET.SubElement(j, "child", link=f"base_link{sf}")
        ET.SubElement(j, "origin", xyz=f"{x} {y} {z}", rpy=f"0 0 {yaw}")
        ET.SubElement(root, "link", name=f"tcp{sf}")
        j = ET.SubElement(root, "joint", name=f"tool0_to_tcp{sf}", type="fixed")
        ET.SubElement(j, "parent", link=f"tool0{sf}"); ET.SubElement(j, "child", link=f"tcp{sf}")
        ET.SubElement(j, "origin", xyz=f"0 0 {TCP_Z}", rpy="0 0 0")
    os.makedirs(out_dir, exist_ok=True)
    urdf = os.path.join(out_dir, "dual_ur5e.urdf")
    ET.ElementTree(root).write(urdf)

    one = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))["robot_cfg"]["kinematics"]
    links = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'tool0']
    spheres, coll, ignore, extra, extra_n, buf = {}, [], {}, {}, {}, {}
    for arm, sf in ARMS:
        for l in links:
            spheres[l + sf] = GRIPPER_SPHERES if l == "tool0" else one["collision_spheres"][l]
            coll.append(l + sf)
        att = "attached_object" + sf
        extra[att] = dict(parent_link_name="tcp" + sf, link_name=att, fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_type="FIXED", joint_name="attach_joint" + sf)
        extra_n[att] = 40
        coll.append(att)
        for k, v in one["self_collision_ignore"].items():
            ignore[k + sf] = [n + sf for n in v]
        ignore["wrist_2_link" + sf] = ignore.get("wrist_2_link" + sf, []) + [att]
        ignore["wrist_3_link" + sf] = ignore.get("wrist_3_link" + sf, []) + [att]
        ignore["tool0" + sf] = ignore.get("tool0" + sf, []) + [att]
        buf.update({l + sf: 0.0 for l in links})
        buf[att] = 0.0
    jn = [n + sf for _, sf in ARMS for n in ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')]
    cfg = dict(kinematics=dict(
        urdf_path=urdf, asset_root_path=src_dir, base_link="world", ee_link="tcp", link_names=["tcp", "tcp_1"],
        lock_joints=None, extra_links=extra, collision_link_names=coll, collision_spheres=spheres,
        collision_sphere_buffer=0.005, extra_collision_spheres=extra_n, self_collision_ignore=ignore,
        self_collision_buffer=buf, use_global_cumul=True, mesh_link_names=[],
        cspace=dict(joint_names=jn, retract_config=list(home), null_space_weight=[1.0] * 12, cspace_distance_weight=[1.0] * 12,
                    max_jerk=500.0, max_acceleration=12.0, position_limit_clip=0.1)))
    return cfg, jn


class Server:
    def __init__(self, bases, home, out_dir):
        self.t = TensorDeviceType()
        cfg, self.jn = build_robot(bases, home, out_dir)
        self.ignore = cfg["kinematics"]["self_collision_ignore"]
        self.rc = RobotConfig.from_dict(cfg, self.t)
        world = WorldConfig(cuboid=[Cuboid(name="_dummy", pose=[0, 0, -5, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])])
        mc = MotionGenConfig.load_from_robot_config(
            self.rc, world, self.t, interpolation_dt=0.02, collision_cache={"obb": 64}, use_cuda_graph=False,
            num_trajopt_seeds=8, num_graph_seeds=4, num_ik_seeds=64, collision_activation_distance=0.01)
        self.mg = MotionGen(mc)
        self.mg.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.names = []
        self.home = list(home)
        self.lim = self.mg.kinematics.get_joint_limits().position[1].cpu().numpy() - 1e-3

    # ---- helpers ----------------------------------------------------------------------
    def js(self, q):
        return JointState.from_position(self.t.to_device(torch.tensor([q], dtype=torch.float32)), joint_names=self.jn)

    def fk(self, q):
        st = self.mg.kinematics.get_state(self.t.to_device(torch.tensor([q], dtype=torch.float32)))
        out = {}
        for arm, sf in ARMS:
            p = st.link_poses["tcp" + sf]
            out[arm] = p.position[0].cpu().tolist() + p.quaternion[0].cpu().tolist()
        return out

    def set_world(self, cuboids):
        self.cub = {c["name"]: c for c in cuboids}
        cs = [Cuboid(name=c["name"], pose=list(c["pose"]), dims=list(c["dims"])) for c in cuboids]
        self.mg.update_world(WorldConfig(cuboid=cs))
        self.names = [c["name"] for c in cuboids]

    def collisions(self, q, off=(), skip_links=()):
        """What touches what at q: the deepest robot-sphere / obstacle contacts and the nearest pairs of
        spheres of different links (self-contact), in cm. For explaining failures, not for planning."""
        kc = self.mg.kinematics.kinematics_config
        idx2link = {v: k for k, v in kc.link_name_to_idx_map.items()}
        st = self.mg.kinematics.get_state(self.t.to_device(torch.tensor([q], dtype=torch.float32)))
        sp = st.link_spheres_tensor.view(-1, 4).cpu().numpy()
        lk = [idx2link.get(int(i), "?") for i in kc.link_sphere_idx_map.cpu().numpy()]
        use = [i for i in range(len(sp)) if sp[i, 3] > 0 and lk[i] not in skip_links]
        out = []
        for c in getattr(self, "cub", {}).values():
            if c["name"] in off:
                continue
            ctr, half = np.array(c["pose"][:3]), np.array(c["dims"]) / 2   # axis-aligned (all our cuboids are)
            for i in use:
                d = np.linalg.norm(np.maximum(np.abs(sp[i, :3] - ctr) - half, 0.0)) - sp[i, 3]
                if d < 0.005:
                    out.append((round(float(d) * 100, 1), lk[i], c["name"]))
        best = {}
        for d, l, o in sorted(out):
            best.setdefault((l, o), d)
        world = [f"{l}~{o} {d:+.1f}cm" for (l, o), d in sorted(best.items(), key=lambda kv: kv[1])[:4]]
        selfc = []
        ign = self.rc.kinematics.kinematics_config.self_collision_config if False else None
        for i in use:
            for j in use:
                if j <= i or lk[i] == lk[j]:
                    continue
                if lk[i].endswith("_1") == lk[j].endswith("_1"):   # same arm: skip the pairs cuRobo ignores
                    if lk[j] in self.ignore.get(lk[i], []) or lk[i] in self.ignore.get(lk[j], []):
                        continue
                d = float(np.linalg.norm(sp[i, :3] - sp[j, :3]) - sp[i, 3] - sp[j, 3])
                if d < 0.01:
                    selfc.append((round(d * 100, 1), lk[i], lk[j]))
        selfc = [f"{a}~{b} {d:+.1f}cm" for d, a, b in sorted(selfc)[:3]]
        return world, selfc

    def _enable(self, names, flag):
        for n in names:
            if n in self.names:
                self.mg.world_coll_checker.enable_obstacle(n, flag)

    def _attach(self, q, attach):
        for arm, sf in ARMS:
            objs = [n for n in attach.get(arm, []) if n in self.names]
            if objs:
                # from the pose the client sent last (attach_objects_to_robot read a stale copy of the world)
                ext = [Cuboid(name=n, pose=list(self.cub[n]["pose"]), dims=list(self.cub[n]["dims"])) for n in objs]
                # cuRobo expresses an attached object in the frame of ITS end-effector (the right tcp) whatever
                # link it goes on; for the left arm, shift by right_tcp * left_tcp^-1 so the frame is the left tcp
                off = None
                if sf:
                    fk = self.fk(q)
                    P = lambda v: Pose.from_list(list(v), tensor_args=self.t)  # noqa: E731
                    off = P(fk["right"]).multiply(P(fk[arm]).inverse())
                # cuRobo fits spheres by sampling the cuboid surface with numpy's global RNG; unseeded, the carried
                # object's spheres move a few mm between runs and a check at the clearance limit flips pass/fail
                np.random.seed(0)
                self.mg.attach_external_objects_to_robot(self.js(q), ext, link_name="attached_object" + sf, surface_sphere_radius=0.005,
                                                         world_objects_pose_offset=off)
                self._enable(objs, False)                        # carried, not an obstacle

    def _detach(self, attach):
        for arm, sf in ARMS:
            if attach.get(arm):
                self.mg.detach_object_from_robot("attached_object" + sf)
                self._enable(attach[arm], True)

    def _poses(self, q, goals):
        cur = self.fk(q)
        g = {arm: (goals.get(arm) or cur[arm]) for arm, _ in ARMS}
        P = lambda v: Pose.from_list(list(v), tensor_args=self.t)  # noqa: E731
        return P(g["right"]), {"tcp": P(g["right"]), "tcp_1": P(g["left"])}

    # ---- requests ---------------------------------------------------------------------
    def pair_dev(self, traj, q0):
        """Largest change of the distance between the two tcps along a trajectory (m)."""
        st = self.mg.kinematics.get_state(self.t.to_device(torch.tensor(traj, dtype=torch.float32)))
        a, b = st.link_poses["tcp"].position, st.link_poses["tcp_1"].position
        d = torch.linalg.norm(a - b, dim=-1).cpu().numpy()
        f0 = self.fk(q0)
        d0 = float(np.linalg.norm(np.array(f0["right"][:3]) - np.array(f0["left"][:3])))
        return float(np.abs(d - d0).max())

    def plan(self, q, goals, attach=None, off=(), max_attempts=8, pair_tol=None, tries=4):
        """Both arms at once; if that fails and both move, one after the other (right, then left;
        then left, then right). pair_tol: the two hands carry one thing (a towel between them) - the
        distance between them may change by at most this much (m) along the motion; the plan is
        redrawn up to `tries` times, and never done one arm after the other."""
        if pair_tol is not None:
            worst = None
            for _ in range(tries):
                r = self.plan1(q, goals, attach, off, max_attempts)
                if not r["ok"]:
                    return r
                dev = self.pair_dev(r["traj"], q)
                worst = dev if worst is None else min(worst, dev)
                if dev <= pair_tol:
                    r["pair_dev_cm"] = round(dev * 100, 1)
                    return r
            return dict(ok=False, why=f"every plan changed the distance between the hands by >= {worst * 100:.1f} cm (allowed {pair_tol * 100:.1f})")
        r = self.plan1(q, goals, attach, off, max_attempts)
        moving = [a for a, _ in ARMS if goals.get(a)]
        if r["ok"] or len(moving) < 2:
            return r
        for order in (("right", "left"), ("left", "right")):
            r1 = self.plan1(q, {order[0]: goals[order[0]]}, attach, off, max_attempts)
            if not r1["ok"]:
                continue
            r2 = self.plan1(r1["traj"][-1], {order[1]: goals[order[1]]}, attach, off, max_attempts)
            if r2["ok"]:
                return dict(ok=True, why="", traj=r1["traj"] + r2["traj"], dt=r1["dt"], motion_s=r1["motion_s"] + r2["motion_s"],
                            plan_s=r1["plan_s"] + r2["plan_s"], sequential=list(order))
        r["why"] += "; one arm after the other failed too"
        return r

    def plan1(self, q, goals, attach=None, off=(), max_attempts=8):
        attach = attach or {}
        self._enable(off, False)
        self._attach(q, attach)
        try:
            goal, lp = self._poses(q, goals)
            t0 = time.time()
            r = self.mg.plan_single(self.js(q), goal, MotionGenPlanConfig(max_attempts=max_attempts, enable_graph=True,
                                                                           enable_finetune_trajopt=True), link_poses=lp)
            ok = bool(r.success.item())
            out = dict(ok=ok, why=str(r.status) if not ok else "", plan_s=round(time.time() - t0, 2))
            if ok:
                tr = r.get_interpolated_plan()
                out.update(traj=tr.position.cpu().numpy().round(5).tolist(), dt=float(r.interpolation_dt), motion_s=float(r.motion_time))
            else:
                out["why"] += self._diagnose(q, goals, attach)
            return out
        finally:
            self._detach(attach)
            self._enable(off, True)

    def _diagnose(self, q, goals, attach):
        """Why a plan failed: is the goal itself unreachable, or in collision?"""
        goal, lp = self._poses(q, goals)
        w0, s0 = self.collisions(q, off=sum(attach.values(), []))
        if w0 or s0:
            return f"; at the START: {w0 + s0}"
        ik = self.mg.ik_solver.solve_single(goal, link_poses=lp)
        if not bool(ik.success.item()):
            pe = float(ik.position_error.max()) * 100
            w, sc = self.collisions(ik.solution[0, 0].cpu().numpy().tolist(), off=sum(attach.values(), []))
            return f"; no collision-free IK at the goal (best pose error {pe:.1f} cm; touches {w + sc})"
        return "; the goal has a collision-free IK, no collision-free path was found"

    def path(self, q, waypoints, attach=None, off=(), gripper_contact=True, max_jump=0.6):
        attach = attach or {}
        n = max(len(v) for v in waypoints.values())
        cur = self.fk(q)
        wps = {arm: (waypoints.get(arm) or [cur[arm]] * n) for arm, _ in ARMS}
        self._enable(off, False)
        self._attach(q, attach)
        if gripper_contact:
            self.mg.toggle_link_collision(["tool0", "tool0_1"], False)
        try:
            seed, traj = list(q), []
            for i in range(n):
                g = {a: wps[a][min(i, len(wps[a]) - 1)] for a, _ in ARMS}
                goal, lp = self._poses(seed, g)
                s = self.t.to_device(torch.tensor([seed], dtype=torch.float32))
                # many seeds (the last solution among them), keep the feasible solution nearest the last one
                ns = 32
                seeds = s.view(1, 1, -1).repeat(1, ns, 1) + 0.15 * torch.randn(1, ns, s.shape[-1], device=s.device) * \
                    (torch.arange(ns, device=s.device) > 0).float().view(1, ns, 1)
                ik = self.mg.ik_solver.solve_single(goal, retract_config=s, seed_config=seeds, link_poses=lp, num_seeds=ns, return_seeds=ns)
                okm = ik.success.view(-1).cpu().numpy()
                # ik.success measures the ee link (the right tcp) only; the left tcp is a link-pose constraint
                # whose error is not in the flag, so check both hands' poses by FK (0.5 cm, 3 deg)
                ks = self.mg.kinematics.get_state(ik.solution.view(ns, -1))
                for arm, sf in ARMS:
                    gp = np.array(g[arm], dtype=float)
                    pos = ks.link_poses["tcp" + sf].position.cpu().numpy()
                    qt = ks.link_poses["tcp" + sf].quaternion.cpu().numpy()
                    perr = np.linalg.norm(pos - gp[:3], axis=1)
                    aerr = 2 * np.arccos(np.clip(np.abs(qt @ gp[3:]), 0, 1))
                    okm = okm & (perr < 0.005) & (aerr < math.radians(3))
                if not okm.any():
                    k = int(torch.argmin(ik.position_error.view(-1)))
                    qb = ik.solution.view(ns, -1)[k].cpu().numpy().tolist()
                    w, sc = self.collisions(qb, off=list(off) + sum(attach.values(), []), skip_links=("tool0", "tool0_1") if gripper_contact else ())
                    return dict(ok=False, why=f"waypoint {i}/{n}: no collision-free IK (pose error {float(ik.position_error.min()) * 100:.1f} cm); "
                                              f"best try touches {w + sc or 'nothing (limits?)'}", at=i, traj=traj)
                sols = ik.solution.view(ns, -1).cpu().numpy().astype(float)[okm]
                # the same configuration a turn of 2 pi away on a wrist (limits are +-2 pi) is not a flip
                d = (sols - np.array(seed) + math.pi) % (2 * math.pi) - math.pi
                alt = np.array(seed) + d
                sols = np.where(np.abs(alt) <= self.lim, alt, sols)
                jumps = np.abs(sols - np.array(seed)).max(axis=1)
                qn = sols[int(np.argmin(jumps))]
                jump = float(jumps.min())
                if jump > max_jump:                                  # also for the first waypoint: the path starts from q
                    return dict(ok=False, why=f"waypoint {i}/{n}: the arm would flip (joint jump {math.degrees(jump):.0f} deg)", at=i, traj=traj)
                traj.append(qn.round(5).tolist())
                seed = qn.tolist()
            return dict(ok=True, why="", traj=traj)
        finally:
            if gripper_contact:
                self.mg.toggle_link_collision(["tool0", "tool0_1"], True)
            self._detach(attach)
            self._enable(off, True)

    def check(self, q0, traj, kind, goals, attach=None, off=(), gripper_contact=False, pair_tol=None, max_jump=0.6,
              step=0.05, pos_tol=0.01, ang_tol_deg=5.0):
        """Verify a joint trajectory somebody else planned, under the rules plan/path impose: starts at q0,
        within limits, steps <= max_jump, no contact (checked every `step` rad along each straight joint
        move), goals met (path: every pose in order; plan: the last), idle hands end where they began."""
        attach = attach or {}
        tr = [[float(v) for v in p] for p in traj]
        if not tr:
            return dict(ok=False, why="empty trajectory")
        if max(abs(a - b) for a, b in zip(tr[0], q0)) > 1e-3:       # the start is implicit (path() omits it); the step to it is checked
            tr = [[float(v) for v in q0]] + tr
        for i, p in enumerate(tr):
            if len(p) != len(self.lim) or any(abs(v) > l for v, l in zip(p, self.lim)):
                return dict(ok=False, why=f"point {i}: outside the joint limits")
        jumps = [max(abs(a - b) for a, b in zip(tr[i], tr[i - 1])) for i in range(1, len(tr))]
        if jumps and max(jumps) > max_jump:
            return dict(ok=False, why=f"point {int(np.argmax(jumps)) + 1}: joint step {math.degrees(max(jumps)):.0f} deg > {math.degrees(max_jump):.0f}")
        dense = [tr[0]]
        for i in range(1, len(tr)):
            n = max(1, int(math.ceil(jumps[i - 1] / step)))
            dense += [[a + (b - a) * k / n for a, b in zip(tr[i - 1], tr[i])] for k in range(1, n + 1)]
        self._enable(off, False)
        self._attach(q0, attach)
        try:
            skip = ("tool0", "tool0_1") if gripper_contact else ()
            offs = list(off) + sum(attach.values(), [])
            # the sphere model sits outside the meshes; cuRobo's own trajectories overlap spheres by a few mm
            # (soft costs), so a penetration deeper than `tol` cm fails and the worst clearance is reported
            tol, tol_self = -0.5, -1.0                               # our own plans overlap the two arms' spheres by up to 0.9 cm
            worst_w, worst_s = (0.5, ""), (1.0, "")
            for i, p in enumerate(dense):
                w, sc = self.collisions(p, off=offs, skip_links=skip)
                if w:                                                  # listed = closer than 0.5 cm
                    d = float(w[0].split()[-1][:-2])
                    if d < worst_w[0]:
                        worst_w = (d, f"point {i}: {w[0]}")
                for s_ in sc:                                          # listed = closer than 1 cm
                    d = float(s_.split()[-1][:-2])
                    if d < worst_s[0]:
                        worst_s = (d, f"point {i}: {s_}")
            clear = dict(world_min_cm=worst_w[0], self_min_cm=worst_s[0])
            if worst_w[0] < tol:
                return dict(ok=False, why=f"contact {worst_w[1]}", **clear)
            if worst_s[0] < tol_self:
                return dict(ok=False, why=f"arm-arm contact {worst_s[1]}", **clear)
            fks = [self.fk(p) for p in tr]
            ang = lambda qa, qb: math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(qa, qb))))))  # noqa: E731
            near = lambda f, g: np.linalg.norm(np.array(f[:3]) - np.array(g[:3])) <= pos_tol and ang(f[3:], g[3:]) <= ang_tol_deg  # noqa: E731
            for arm, _ in ARMS:
                g = goals.get(arm)
                if not g:
                    if np.linalg.norm(np.array(fks[-1][arm][:3]) - np.array(fks[0][arm][:3])) > pos_tol:
                        return dict(ok=False, why=f"{arm} hand had no goal but moved away ({np.linalg.norm(np.array(fks[-1][arm][:3]) - np.array(fks[0][arm][:3])) * 100:.1f} cm)")
                    continue
                poses = g if kind == "path" else [g]
                if kind == "path" and poses and not isinstance(poses[0], list):
                    poses = [poses]
                j = 0
                for k, gp in enumerate(poses):
                    while j < len(fks) and not near(fks[j][arm], gp):
                        j += 1
                    if j >= len(fks):
                        e = min(np.linalg.norm(np.array(f[arm][:3]) - np.array(gp[:3])) for f in fks) * 100
                        return dict(ok=False, why=f"{arm} hand misses goal pose {k + 1}/{len(poses)} (nearest {e:.1f} cm)")
                if kind == "plan" and not near(fks[-1][arm], poses[-1]):
                    return dict(ok=False, why=f"{arm} hand does not end at the goal pose")
            if pair_tol is not None:
                dev = self.pair_dev(tr, q0)
                if dev > pair_tol:
                    return dict(ok=False, why=f"hand-to-hand distance changes by {dev * 100:.1f} cm > {pair_tol * 100:.1f}")
            return dict(ok=True, why="", points=len(tr), checked=len(dense), max_step_deg=round(math.degrees(max(jumps)) if jumps else 0.0, 1), **clear)
        finally:
            self._detach(attach)
            self._enable(off, True)


def main():
    a = json.loads(sys.argv[1])
    srv = Server(a["bases"], a["home"], a["out_dir"])
    lim = srv.mg.kinematics.get_joint_limits().position.cpu().numpy()
    print(json.dumps(dict(ok=True, ready=True, joints=srv.jn, limits=[[float(lo), float(hi)] for lo, hi in zip(lim[0], lim[1])])), file=OUT, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            c = req["cmd"]
            if c == "world":
                srv.set_world(req["cuboids"]); rep = dict(ok=True)
            elif c == "fk":
                rep = dict(ok=True, tcp=srv.fk(req["q"]))
            elif c == "plan":
                rep = srv.plan(req["q"], req["goals"], req.get("attach"), req.get("off", []), pair_tol=req.get("pair_tol"))
            elif c == "path":
                rep = srv.path(req["q"], req["waypoints"], req.get("attach"), req.get("off", []), req.get("gripper_contact", True),
                               max_jump=req.get("max_jump", 0.6))
            elif c == "check":                                       # somebody else's trajectory, our rules (astra_eval/p2_motion)
                rep = srv.check(req["q0"], req["traj"], req["kind"], req["goals"], req.get("attach"), req.get("off", []),
                                req.get("gripper_contact", False), req.get("pair_tol"))
            elif c == "quit":
                break
            else:
                rep = dict(ok=False, why=f"unknown cmd {c}")
        except Exception as e:                                    # report, keep serving
            import traceback
            log(traceback.format_exc())
            rep = dict(ok=False, why=f"server error: {e}")
        print(json.dumps(rep), file=OUT, flush=True)


if __name__ == "__main__":
    main()
