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

## Repository layout

Organized by pipeline stage. Each directory is one stage of real → sim → real.

```
real2sim2real/
├── perception/   real → scene:  capture (RGBD+extrinsics) → detect (Gemini) →
│                 reconstruct (SAM 3D) → metric/calibrated scene + render-compare QA
├── tamp/         task+motion planning:  language → task plan → N candidate plans →
│                 EE-trajectory compiler;  shared grasp/place geometry (ee_geometry.py)
├── twin/         Isaac Lab digital twin:  scene loader, multi-step replay + per-step
│                 verdict, and the PARALLEL task-plan search engine (twin_plan_eval.py)
├── transfer/     sim → real UR5e:  MoveIt EE executor + transfer orchestration
├── scripts/      orchestration + utils (run_robot_pipeline.sh, load_in_isaaclab.sh)
├── docs/         TWIN_TAMP.md (the TAMP + parallel-twin layer)
├── engines/      local symlinks to the live heavy installs (gitignored, this machine):
│                   IsaacLab/ -> ~/IsaacLab,  sam-3d-objects/ -> ~/sam-3d-objects,
│                   Hunyuan3D-2/ -> ~/Hunyuan3D-2
└── third_party/  upstream submodules (sam-3d-objects, IsaacLab) — pinned commits for portable clone
```

**Why symlinks, not copies, for the engines:** IsaacLab is editable-installed into
the `isaaclab` conda env and sam-3d-objects is ~19 G (with data + the SAM3D package);
moving/copying them would break the conda installs and duplicate tens of GB. `engines/`
gives one-folder access to the live installs without either cost. For a *portable*
checkout, `git clone --recursive` fetches the pinned upstream commits into `third_party/`.
Recreate the symlinks on a new machine: `mkdir -p engines && ln -sfn <path> engines/<name>`.

### Key files
| File | Role |
|------|------|
| `scripts/run_robot_pipeline.sh` | one-command end-to-end orchestrator |
| `perception/capture_rgbd_ros1.py` | ROS1 RGBD + extrinsics capture (on the robot) |
| `perception/gemini_manip.py` | VLM manipulable-object detection (general / targeted) |
| `perception/sam3d_to_scene.py` | SAM 3D → metric/calibrated scene + QA self-repair |
| `perception/render_compare.py` | Open3D render-and-compare QA (silhouette IoU) |
| `perception/demo.py` / `hunyuan_demo.py` | SAM 3D / Hunyuan3D-2 reconstruction backends |
| `tamp/tamp_plan.py` | language → grounded goal + ordered pick/place actions (Gemini) |
| `tamp/twin_plan_search.py` | goal → N candidate plans (vary order + grasp + placement) |
| `tamp/ee_geometry.py` | **shared** grasp/place geometry (single source of truth) |
| `tamp/tamp_to_ee.py` / `cgn_to_ee_traj.py` | plan → EE-trajectory compilers |
| `twin/real2sim_franka.py` | Isaac Lab scene + `--replay_plan` multi-step exec + verdict |
| `twin/twin_plan_eval.py` | **parallel task-plan search** (N envs, physics-scored, pick winner) |
| `transfer/ur5e_ee_executor.py` | MoveIt EE executor on the real UR5e |

See **`docs/TWIN_TAMP.md`** for the TAMP + parallel-twin layer.

## Upstream repos (git submodules)

The two big upstream codebases are pinned as submodules under `third_party/`
(NOT vendored — they're 17 GB / 1.2 GB):

| Submodule | Pinned commit |
|-----------|---------------|
| `third_party/sam-3d-objects` (facebookresearch) | `81a8237` |
| `third_party/IsaacLab` (isaac-sim) | `f4aa17f` |

```bash
git clone --recursive https://github.com/aoloo-r/real2sim2real.git
# or, in an existing clone:
git submodule update --init   # fetches the pinned upstream versions
```

Our edited copies of a few upstream files live in `pipeline/` (`demo.py`,
`depth_mesh.py`, `vertex_colors.py`) and `isaaclab/real2sim_franka.py`; apply
them over the submodules (copy/symlink into `third_party/sam-3d-objects/` and
`third_party/IsaacLab/scripts/`). SAM 3D checkpoints + the conda envs are still
downloaded/built separately (below).

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
