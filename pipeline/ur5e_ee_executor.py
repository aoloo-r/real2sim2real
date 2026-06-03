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

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped


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
    ap.add_argument("--execute", action="store_true",
                    help="ACTUALLY MOVE the arm (default: dry-run plan only)")
    ap.add_argument("--with-gripper", action="store_true",
                    help="also actuate the Robotiq gripper (default: move-only)")
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to abort (Ctrl-C) before each real move")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    frame = traj.get("frame", "ur5e_base_link")
    vel = args.vel if args.vel is not None else traj.get("default_vel_scale", 0.1)
    wps = traj.get("waypoints", [])

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("ur5e_ee_executor", anonymous=True, disable_signals=True)
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander(args.group)
    if args.ee_link:
        group.set_end_effector_link(args.ee_link)
    group.set_max_velocity_scaling_factor(max(0.01, min(1.0, vel)))
    group.set_max_acceleration_scaling_factor(max(0.01, min(1.0, vel)))
    group.set_planning_time(5.0)
    group.set_num_planning_attempts(5)

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
    for i, wp in enumerate(wps):
        label = wp.get("label", "wp%d" % i)
        if wp.get("position") is None:                  # gripper-only event
            g = wp.get("gripper", "none")
            if g in ("open", "close") and args.with_gripper:
                if args.execute:
                    gripper_cmd(gpub, GMsg, g)
                else:
                    print("  [%d] %-14s gripper %s (dry-run)" % (i, label, g))
            else:
                print("  [%d] %-14s gripper %s (skipped: move-only)" % (i, label, g))
            continue

        p = wp["position"]; q = wp["quaternion"]
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p
        # JSON quaternion is [w,x,y,z]; ROS Quaternion is x,y,z,w
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = q
        group.set_start_state_to_current_state()
        group.clear_pose_targets()
        group.set_pose_target(ps)
        plan = group.plan()
        # Melodic returns a RobotTrajectory; success = non-empty joint_trajectory
        jt = getattr(getattr(plan, "joint_trajectory", None), "points", [])
        if not jt:
            print("  [%d] %-14s (%.3f,%.3f,%.3f)  PLAN FAILED" % (i, label, p[0], p[1], p[2]))
            fail += 1
            continue
        print("  [%d] %-14s (%.3f,%.3f,%.3f)  planned (%d pts)" % (
            i, label, p[0], p[1], p[2], len(jt)))
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
