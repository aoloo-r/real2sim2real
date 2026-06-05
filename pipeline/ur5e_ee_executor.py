#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute a sim-exported EE (TCP) trajectory on the REAL UR5e via MoveIt (ROS1).

Runs ON the robot (segbot, ROS Melodic / Python 2). Reads an ee_trajectory.json
produced by real2sim_franka.py (--export_ee_traj): EE/TCP pose keyframes in the
`ur5e_base_link` frame + gripper events. For each pose it sets a MoveIt pose
target (MoveIt does the UR5e IK + collision-aware planning) and either previews
or executes it.

SAFETY (a real arm moves):
  * DRY-RUN by default — only plans + reports; NO motion. Watch it in RViz.
  * --execute is required to move; a countdown lets you Ctrl-C first.
  * move-only by default (gripper ignored); --with-gripper enables Robotiq.
  * low velocity scaling (default 0.1).

Usage (on the robot):
  source /opt/ros/melodic/setup.bash; source ~/catkin_ws/devel/setup.bash
  python ur5e_ee_executor.py --traj ee_trajectory.json                 # DRY-RUN
  python ur5e_ee_executor.py --traj ee_trajectory.json --execute       # move only
  python ur5e_ee_executor.py --traj ee_trajectory.json --execute --with-gripper
"""
from __future__ import print_function
import argparse
import json
import sys
import time

import math

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped


def quat_mul(a, b):
    """Hamilton product, both [w,x,y,z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quat_from_axis_angle(axis, ang):
    """Rotation quaternion [w,x,y,z] of `ang` rad about `axis`."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), x / n * s, y / n * s, z / n * s]


def quat_to_R(q):
    """[w,x,y,z] -> 3x3 rotation matrix (row-major lists)."""
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]]


def candidate_quats(base_q, gx, gy, is_grasp):
    """Orientation candidates (each [w,x,y,z]) to try for one waypoint, in order
    of increasing wrist tilt. The commanded (straight-down/yawed) orientation is
    tried FIRST so easy poses stay clean; if that wrist config is unreachable we
    LEAN the gripper toward the robot base (the natural way to reach far/low/front
    poses) and then add side-tilts. The smallest tilt that MoveIt can plan wins,
    so the robot grasps from positions a fixed straight-down pose can't reach.
    Grasp waypoints are capped at a modest tilt (still grasps a cup/bowl wall);
    free-space moves may tilt more."""
    cands = [("down", list(base_q))]
    n = math.hypot(gx, gy) or 1.0
    lean_axis = [-gy / n, gx / n, 0.0]   # rotate the down-axis toward base origin
    # Leaning the wrist is what unlocks LOW reach (the front/far low zone is
    # unreachable straight-down but reachable tilted) — allow steep leans for
    # grasp legs too, capped only enough to still pinch a cup/bowl wall.
    max_t = 60
    for deg in (12, 20, 28, 35, 45, 55, 60):
        if deg > max_t:
            break
        cands.append(("lean%d" % deg,
                      quat_mul(quat_from_axis_angle(lean_axis, math.radians(deg)), base_q)))
    for ax, nm in (([1, 0, 0], "tx"), ([0, 1, 0], "ty")):
        for deg in (20, -20, 35, -35, 50, -50):
            if abs(deg) > max_t:
                continue
            cands.append(("%s%+d" % (nm, deg),
                          quat_mul(quat_from_axis_angle(ax, math.radians(deg)), base_q)))
    return cands


def connect_gripper():
    """Return a publisher + msg class for the Robotiq 2F driver, or (None, None)."""
    try:
        from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as out_mod
        Msg = out_mod.Robotiq2FGripper_robot_output
    except Exception:
        try:
            from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output as Msg
        except Exception as e:
            rospy.logwarn("Robotiq msg not importable (%r); gripper disabled" % e)
            return None, None
    pub = rospy.Publisher("/Robotiq2FGripperRobotOutput", Msg, queue_size=1)
    rospy.sleep(0.5)
    return pub, Msg


def gripper_cmd(pub, Msg, action):
    """action: 'open' | 'close'. Standard Robotiq 2F activate+go-to command."""
    if pub is None:
        rospy.loginfo("[gripper] (disabled) would %s", action); return
    m = Msg()
    m.rACT = 1; m.rGTO = 1; m.rSP = 120; m.rFR = 120     # activate, go, mid speed/force
    m.rPR = 255 if action == "close" else 0              # 255 closed, 0 open
    pub.publish(m); rospy.loginfo("[gripper] %s", action); rospy.sleep(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="ee_trajectory.json path")
    ap.add_argument("--group", default="manipulator", help="MoveIt planning group")
    ap.add_argument("--ee-link", default=None, help="override end-effector link")
    ap.add_argument("--vel", type=float, default=None, help="velocity scaling 0..1")
    ap.add_argument("--tcp-offset", type=float, default=0.0,
                    help="tool0->fingertip distance (m); raises each target along "
                         "+Z so the GRIPPER TIP (not tool0) reaches the pose. "
                         "UR5e+HandE ~0.187. Assumes gripper-down orientation.")
    ap.add_argument("--skip-labels", default="",
                    help="comma-separated waypoint labels to skip (e.g. hover_object)")
    ap.add_argument("--no-collision", action="store_true",
                    help="do NOT load the scene's collision_objects (table + other "
                         "objects) into MoveIt — disables collision-free planning")
    ap.add_argument("--execute", action="store_true",
                    help="ACTUALLY MOVE the arm (default: dry-run plan only)")
    ap.add_argument("--with-gripper", action="store_true",
                    help="also actuate the Robotiq gripper (default: move-only)")
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to abort (Ctrl-C) before each real move")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    # json gives unicode in py2; MoveIt's C++ bindings require str (bytes)
    frame = str(traj.get("frame", "ur5e_base_link"))
    vel = args.vel if args.vel is not None else traj.get("default_vel_scale", 0.1)
    wps = traj.get("waypoints", [])
    skip = set(s.strip() for s in args.skip_labels.split(",") if s.strip())

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5e_ee_executor", anonymous=True, disable_signals=True)
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.group)
    if args.ee_link:
        group.set_end_effector_link(args.ee_link)
    group.set_max_velocity_scaling_factor(max(0.01, min(1.0, vel)))
    group.set_max_acceleration_scaling_factor(max(0.01, min(1.0, vel)))
    # Kept SHORT: we try many orientation candidates per waypoint, so each plan
    # must be fast or the search takes minutes. 1 attempt @ 1.5s; the first
    # reachable candidate wins. (Was 5x5s — far too slow for the candidate sweep.)
    group.set_planning_time(1.5)
    group.set_num_planning_attempts(1)
    group.set_pose_reference_frame(frame)   # interpret targets in the traj frame (Cartesian too)
    # We enumerate discrete orientation candidates ourselves (candidate_quats),
    # so each individual plan is held TIGHT to its commanded orientation — the
    # reachability search comes from trying multiple leans/tilts, not from a loose
    # tolerance (a big tolerance let MoveIt pick weird wrist tilts on easy poses).
    group.set_goal_orientation_tolerance(0.087)   # ~5 deg slack per candidate
    group.set_goal_position_tolerance(0.005)      # 5 mm

    # Load the scene's collision geometry (table + non-target objects) into MoveIt
    # so it plans a COLLISION-FREE path to each waypoint (instead of an empty scene).
    coll = traj.get("collision_objects", []) or []
    n_coll = 0
    if coll and not args.no_collision:
        from geometry_msgs.msg import PoseStamped as _PS
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        rospy.sleep(1.0)
        for c in coll:
            try:
                ps = _PS()
                ps.header.frame_id = frame
                cx, cy, cz = c["center"]
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = cx, cy, cz
                ps.pose.orientation.w = 1.0
                scene.add_box(str(c["name"]), ps, tuple(c["size"]))
                n_coll += 1
            except Exception as e:
                rospy.logwarn("collision obj %r failed: %r" % (c.get("name"), e))
        rospy.sleep(1.0)
        print("  collision scene : %d obstacles loaded (table + objects)" % n_coll)

    mode = "EXECUTE" if args.execute else "DRY-RUN (no motion)"
    print("=" * 64)
    print("UR5e EE executor  | mode=%s  group=%s  ee_link=%s" % (
        mode, args.group, group.get_end_effector_link()))
    print("  planning frame : %s   traj frame: %s" % (group.get_planning_frame(), frame))
    print("  vel scaling    : %.2f   gripper: %s" % (
        vel, "ON" if args.with_gripper else "off (move-only)"))
    print("  waypoints      : %d   from %s" % (len(wps), args.traj))
    print("=" * 64)

    gpub, GMsg = (connect_gripper() if (args.execute and args.with_gripper) else (None, None))
    ok, fail = 0, 0
    held_q = None   # wrist orientation chosen at grasp; reused for lift/place so
                    # the held object isn't twisted, cleared when the gripper opens
    if args.tcp_offset:
        print("  tcp-offset     : +%.3f m along Z (tool0 -> fingertip)" % args.tcp_offset)
    if skip:
        print("  skip labels    : %s" % ", ".join(sorted(skip)))
    for i, wp in enumerate(wps):
        label = wp.get("label", "wp%d" % i)
        if label in skip:
            print("  [%d] %-14s SKIPPED (--skip-labels)" % (i, label))
            continue
        if wp.get("position") is None:                  # gripper-only event
            g = wp.get("gripper", "none")
            if g == "open":
                held_q = None                           # released: wrist free again
            if g in ("open", "close") and args.with_gripper:
                if args.execute:
                    gripper_cmd(gpub, GMsg, g)
                else:
                    print("  [%d] %-14s gripper %s (dry-run)" % (i, label, g))
            else:
                print("  [%d] %-14s gripper %s (skipped: move-only)" % (i, label, g))
            continue

        tip = list(wp["position"])                      # desired FINGERTIP pose
        base_q = wp["quaternion"]                        # commanded orientation [w,x,y,z]
        is_grasp = label in ("hover_object", "descend_to_grasp", "lower_to_place")

        # Reachability search: try the commanded orientation first, then leans/
        # tilts, until MoveIt can plan one. While holding the object, try the
        # already-grasped wrist orientation FIRST so we don't twist it.
        cands = candidate_quats(base_q, tip[0], tip[1], is_grasp)
        if held_q is not None:
            cands = [("hold", list(held_q))] + cands

        plan = None; jt = []; how = ""; used = ""; chosen_q = None
        for nm, q in cands:
            R = quat_to_R(q)
            # The fingertip sits tcp_offset along the gripper's local +Z axis from
            # tool0. Place tool0 = fingertip - R@[0,0,1]*offset so the FINGERTIP
            # (not the flange) reaches `tip` for ANY orientation, not just down.
            off = [R[0][2] * args.tcp_offset, R[1][2] * args.tcp_offset, R[2][2] * args.tcp_offset]
            p = [tip[0] - off[0], tip[1] - off[1], tip[2] - off[2]]
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p
            ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = q
            group.set_start_state_to_current_state()
            group.clear_pose_targets()
            # CARTESIAN only for grasp/place legs (a clean straight-line descend/
            # lower that free-space OMPL often can't connect); skip it elsewhere —
            # it's slow and rarely valid across a big orientation change.
            if is_grasp:
                try:
                    cart_plan, frac = group.compute_cartesian_path([ps.pose], 0.01, 0.0)
                    cjt = getattr(getattr(cart_plan, "joint_trajectory", None), "points", [])
                    if frac >= 0.9 and cjt:
                        # un-timed path -> controller rejects it; retime to vel scaling.
                        cart_plan = group.retime_trajectory(
                            robot.get_current_state(), cart_plan,
                            velocity_scaling_factor=max(0.01, min(1.0, vel)),
                            acceleration_scaling_factor=max(0.01, min(1.0, vel)))
                        plan, jt, how, used, chosen_q = (
                            cart_plan, cart_plan.joint_trajectory.points,
                            "cart %.0f%%" % (frac * 100), nm, q)
                        break
                except Exception as e:
                    rospy.logwarn("cartesian (%s) failed: %r" % (nm, e))
            group.set_pose_target(ps)
            oplan = group.plan()
            ojt = getattr(getattr(oplan, "joint_trajectory", None), "points", [])
            if ojt:
                plan, jt, how, used, chosen_q = oplan, ojt, "ompl", nm, q
                break

        if not jt:
            print("  [%d] %-14s (%.3f,%.3f,%.3f)  PLAN FAILED (tried %d orientations)" % (
                i, label, tip[0], tip[1], tip[2], len(cands)))
            fail += 1
            continue
        # Lock the grasp wrist so lift/place keep it (no twisting the held object).
        if label == "descend_to_grasp":
            held_q = chosen_q
        print("  [%d] %-14s (%.3f,%.3f,%.3f)  planned (%d pts, %s, ori=%s)" % (
            i, label, tip[0], tip[1], tip[2], len(jt), how, used))
        ok += 1
        if args.execute:
            if args.countdown > 0:
                for s in range(args.countdown, 0, -1):
                    sys.stdout.write("\r    moving in %d ... (Ctrl-C to abort) " % s)
                    sys.stdout.flush(); time.sleep(1)
                print("")
            group.execute(plan, wait=True)
            group.stop()

    group.clear_pose_targets()
    print("-" * 64)
    print("done: %d planned ok, %d failed%s" % (
        ok, fail, "" if args.execute else "  (DRY-RUN — nothing moved)"))
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
