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
4. **Load Isaac Sim** – open the scene in the GUI, VRAM-guarded so it can't
   OOM-crash other GPU jobs (`load_in_isaaclab.sh`, `vram_guard.sh`).

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

## File map

| File | Role |
|------|------|
| `pipeline/run_robot_pipeline.sh` | one-command end-to-end orchestrator |
| `pipeline/capture_rgbd_ros1.py` | ROS1 RGBD + extrinsics capture (runs on the robot) |
| `pipeline/gemini_manip.py` | VLM manipulable-object detection (general / targeted) |
| `pipeline/object_focus.py` | keep central/reachable objects |
| `pipeline/depth_scale.py` | robust metric sizing from depth |
| `pipeline/sam3d_to_scene.py` | SAM 3D → metric/calibrated scene + QA self-repair |
| `pipeline/render_compare.py` | Open3D render-and-compare QA (silhouette IoU) |
| `pipeline/primitive_fit.py` | close-shape fallback primitives (fallback only) |
| `pipeline/depth_mesh.py` | depth-carve fallback (real measured surface) |
| `pipeline/hunyuan_demo.py` | Hunyuan3D-2 backend + shared mesh/scene utilities |
| `pipeline/demo.py` | SAM 3D backend (Gemini+SAM2 → SAM 3D) |
| `pipeline/vram_guard.sh` / `load_in_isaaclab.sh` | VRAM guard + guarded Isaac Sim loader |
| `pipeline/inspect_meshes.py` / `measure_objects.py` / `tune_qa.py` | diagnostics |
| `isaaclab/real2sim_franka.py` | Isaac Lab scene: spawn assets, calibrated placement, physics, demos |

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

## Status

Working end-to-end on the segbot: capture → SAM 3D → QA → Isaac Sim, objects
reconstructed as their real shapes, correctly placed and stable. Known limit:
objects below ~5 cm (or transparent/occluded) can fail QA and fall back / flag
for a closer recapture — a sensor-resolution limit, surfaced honestly by the gate.

Open next step: **end-effector trajectory transfer** (plan in sim → execute on
the real UR5e via its ROS1 MoveIt / joint-trajectory controllers).
