#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot RGB-D capture from a running ROS1 realsense2_camera node.

Designed for ROS Melodic / Python2 on the robot PC ("segbot"), where neither
pyrealsense2 nor cv_bridge are reliably importable. It decodes the
sensor_msgs/Image byte buffers directly with numpy (no cv_bridge, no cv2), so
it has no dependency beyond rospy + numpy.

Subscribes (one message each) to:
  /camera/color/image_raw                       (RGB, rgb8)
  /camera/aligned_depth_to_color/image_raw      (depth aligned to color, 16UC1 mm)
  /camera/color/camera_info                     (intrinsics)

Saves into <output_dir>:
  color_raw.npy   — (H, W, 3) uint8, channel order recorded in capture_meta.json
  depth.npy       — (H, W) float32 depth in METERS
  intrinsics.json — fx, fy, cx, cy, width, height, distortion_model, D
  capture_meta.json — encodings + topic info for the workstation-side converter

The workstation then converts color_raw.npy -> rgb.png (it has cv2) and runs the
reconstruction pipeline. Schema matches the original ROS2 capture_rgbd.py.

Usage (on segbot):
  source /opt/ros/melodic/setup.bash
  python capture_rgbd_ros1.py /home/test/real2sim_captures/robot_TIMESTAMP
"""

import json
import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo


def capture_extrinsics(out, optical_frame="camera_color_optical_frame"):
    """Look up and save the calibrated camera->robot transforms via TF.

    Saves extrinsics.json mapping camera-optical-frame points into each robot
    base frame: p_base = T_base_cam * p_cam. Non-fatal if TF is unavailable.
    """
    try:
        import tf as tf_ros
        import tf.transformations as tft
    except Exception as e:
        print("[extrinsics] tf not importable (%r); skipping" % e)
        return
    listener = tf_ros.TransformListener()
    bases = ["base_link", "base", "ur5e_base_link"]
    result = {"optical_frame": optical_frame, "transforms": {}}
    deadline = rospy.Time.now() + rospy.Duration(8.0)
    for base in bases:
        got = False
        while rospy.Time.now() < deadline and not got:
            try:
                listener.waitForTransform(base, optical_frame, rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = listener.lookupTransform(base, optical_frame, rospy.Time(0))
                T = tft.quaternion_matrix(rot)  # 4x4, rot = [x,y,z,w]
                T[0:3, 3] = trans
                result["transforms"][base] = {
                    "translation": list(trans),
                    "quaternion_xyzw": list(rot),
                    "T_base_cam": [list(map(float, row)) for row in T],
                }
                print("[extrinsics] %s <- %s: t=%s" % (base, optical_frame,
                      [round(v, 4) for v in trans]))
                got = True
            except Exception:
                rospy.sleep(0.2)
        if not got:
            print("[extrinsics] could not get %s <- %s" % (base, optical_frame))
    with open(os.path.join(out, "extrinsics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[extrinsics] saved %d transform(s) to extrinsics.json" % len(result["transforms"]))


def decode_image(msg):
    """Decode a sensor_msgs/Image into a numpy array, honoring row step/padding."""
    enc = msg.encoding
    h, w, step = msg.height, msg.width, msg.step

    if enc in ("rgb8", "bgr8"):
        channels = 3
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
        img = buf[:, : w * channels].reshape(h, w, channels)
        return img, enc
    if enc in ("rgba8", "bgra8"):
        channels = 4
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
        img = buf[:, : w * channels].reshape(h, w, channels)
        return img, enc
    if enc in ("16UC1", "mono16"):
        buf = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, step // 2)
        img = buf[:, :w].copy()
        return img, enc
    raise ValueError("Unsupported image encoding: %s" % enc)


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: python capture_rgbd_ros1.py <output_dir> [timeout_s]\n")
        sys.exit(2)
    out = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    if not os.path.isdir(out):
        os.makedirs(out)

    rospy.init_node("rgbd_oneshot_capture", anonymous=True, disable_signals=True)

    print("[capture] waiting for color/depth/info (timeout=%.1fs each)..." % timeout)
    color_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=timeout)
    depth_msg = rospy.wait_for_message(
        "/camera/aligned_depth_to_color/image_raw", Image, timeout=timeout
    )
    info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=timeout)

    color, color_enc = decode_image(color_msg)
    depth_raw, depth_enc = decode_image(depth_msg)
    depth_m = depth_raw.astype(np.float32) / 1000.0  # mm -> m

    np.save(os.path.join(out, "color_raw.npy"), color)
    np.save(os.path.join(out, "depth.npy"), depth_m)

    intrinsics = {
        "fx": info_msg.K[0],
        "fy": info_msg.K[4],
        "cx": info_msg.K[2],
        "cy": info_msg.K[5],
        "width": int(info_msg.width),
        "height": int(info_msg.height),
        "distortion_model": info_msg.distortion_model,
        "D": list(info_msg.D),
    }
    with open(os.path.join(out, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    meta = {
        "color_encoding": color_enc,
        "depth_encoding": depth_enc,
        "color_shape": list(color.shape),
        "depth_shape": list(depth_m.shape),
        "color_topic": "/camera/color/image_raw",
        "depth_topic": "/camera/aligned_depth_to_color/image_raw",
        "info_topic": "/camera/color/camera_info",
        "depth_units": "meters(float32) from 16UC1 mm",
    }
    with open(os.path.join(out, "capture_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Calibrated camera->robot extrinsics from TF (non-fatal if unavailable)
    capture_extrinsics(out)

    valid = depth_m[depth_m > 0]
    dmin = float(valid.min()) if valid.size else 0.0
    dmax = float(depth_m.max())
    print("[capture] color: %s enc=%s" % (color.shape, color_enc))
    print("[capture] depth: %s  range=%.3f..%.3f m  (%.1f%% valid)" % (
        depth_m.shape, dmin, dmax, 100.0 * valid.size / depth_m.size))
    print("[capture] intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f %dx%d" % (
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
        intrinsics["width"], intrinsics["height"]))
    print("[capture] saved to: %s" % out)


if __name__ == "__main__":
    main()
