# real2sim2real

A real-to-sim pipeline for a **UR5e + Robotiq** mobile manipulator ("segbot"):
capture a tabletop scene with a wrist/overhead RealSense, reconstruct the
objects in 3D, and load them into **NVIDIA Isaac Lab** as physics-accurate,
correctly-placed assets — in one command.

> Built incrementally on top of `facebookresearch/sam-3d-objects`,
> `isaac-sim/IsaacLab`, and Hunyuan3D-2. This repo holds the **pipeline glue +
> fixes** we wrote; the heavy models live in their own conda envs (see Setup).

## What it does (one command)

```bash
bash pipeline/run_robot_pipeline.sh        # FULL: capture -> reconstruct -> load Isaac Sim
```

Stages:
0. **Preflight** – bring the robot Wi-Fi link up, verify reachable, ensure the
   hand-eye TF (`base -> camera`) is published.
1. **Capture** – RGBD + camera→robot **extrinsics** on the robot (ROS1 Melodic),
   pulled back to the workstation (`capture_rgbd_ros1.py`).
2. **Detect** – a VLM (Gemini) finds the manipulable objects — **general, no
   hardcoded categories** (`gemini_manip.py`); optional `MANIP_TARGETS` for
   task-conditioned grasping.
3. **Reconstruct** – **SAM 3D** builds the real mesh per object; an adapter
   (`sam3d_to_scene.py`) gives it metric size (from depth, `depth_scale.py`),
   calibrated placement, orientation (incl. flip-open-side-up for dishes), and a
   **render-and-compare QA gate** with **self-repair** fallbacks
   (depth-carve → close primitive) only when SAM 3D genuinely fails.
4. **Load Isaac Sim** – open the scene in the GUI (`load_in_isaaclab.sh`).

Backend is swappable: `BACKEND=sam3d` (default) or `BACKEND=hunyuan`.

## Design principles

- **No predefining.** Objects are reconstructed as their *real* shapes (SAM 3D).
  Primitives (ellipsoid/cylinder/dish/box) appear **only** as a flagged fallback
  when SAM 3D fails the QA gate (e.g. an object too small for the sensor).
- **Calibrated, not guessed.** Object placement uses the published hand-eye
  extrinsic (`ur5e_base_link <- camera`), not heuristic camera-yaw flags.
- **Stable physics.** Objects spawn bottom-on-table (no freefall), with high
  friction + gentle depenetration so they settle in place and don't scatter.
- **Quality is gated.** Every object is render-verified vs its mask; bad ones get
  repaired or flagged `needs_recapture` instead of shipping junk.

## Setup / dependencies (external)

The scripts orchestrate three environments and a robot; they are NOT vendored:
- **conda env `sam3d-objects`** — SAM 3D (`facebookresearch/sam-3d-objects` + checkpoints).
- **conda env `hunyuan3d`** — cv2/trimesh/open3d/SAM-2/Hunyuan3D-2; runs the adapter + QA.
- **conda env `isaaclab`** — `isaac-sim/IsaacLab`; runs `real2sim_franka.py`.
- **Robot (segbot)** — Ubuntu 18.04 / ROS1 Melodic with `realsense2_camera`,
  the UR driver, and an `easy_handeye` calibration published via
  `realsense_handeye_publish.launch`.

Runtime paths currently assume `~/sam-3d-objects` (pipeline scripts) and
`~/IsaacLab` (`scripts/real2sim_franka.py`); this repo is the versioned copy.
