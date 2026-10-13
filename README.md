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

## Dual-arm fold-and-pack in simulation (TAMP, rigid + deformable)

Two UR5e arms fold a shirt and two towels, pack them into an open box, and place
three rigid blocks on top. A planner picks the fold level and orientation of each
garment and the position of every item so that all of it fits; Newton 1.5 runs
the physics; Isaac Sim plays the recording back.

Code is in `twin/newton/`:

| File | Role |
|---|---|
| `tamp_pack.py` | Planner: fold level per garment, orientation, height-map packing, action order |
| `twin_stockfold.py` | Executor: cloth (VBD) + arms (Featherstone) + rigid blocks (XPBD), `TWIN_SCENE=task1` |
| `towel_labels.py` | Grasp labels (corners, quarter points) for each towel |
| `fold_predicates.py` | Symbolic audit: fold predicates, fraction of each garment in the box |
| `usd_recorder.py`, `play_fold_isaac.py` | Record the run to USD; play it in Isaac Sim |

Needs two conda envs: `newton-1.5` (Newton 1.5, simulation) and `isaaclab`
(Isaac Sim 5.1, playback). The run scripts look for them under `~/miniforge3/envs`;
override with `NEWTON_PY=` and `ISAAC_PY=`.

### Saved run: `runs_2026-10/task2_pack_s34/`

Shirt + bath towel (40×30 cm) + hand towel (34×24 cm) + three blocks into a
38 cm box with 12 cm walls. 11 actions, 330 s of simulated time.

```bash
cd runs_2026-10/task2_pack_s34

# 1. Plan (about a second). Regenerates plan_t38b.json from the scene description.
python ../../twin/newton/tamp_pack.py --scene scene_t38.json --out plan.json

# 2. Simulate (about 35 minutes on a 24 GB GPU). Runs detached.
./rerun.sh                      # writes rerun_<date>/execute.log and fold_robot.usd
tail -f rerun_*/execute.log     # ends with a "TASK 2 result" line and the audit

# 3. Watch a recording in Isaac Sim (no physics; loads in seconds once Isaac is up).
./show.sh                       # plays ./fold_robot.usd; optional argument = speed
```

The recording itself (`fold_robot.usd`, about 0.9 GB) is too large for GitHub and
is not in the repository. On a fresh clone, run step 2 first and point `show.sh`
at the new recording, or copy the file in.

To pack a different set of objects, edit a copy of `scene_t38.json` (box size,
garments with their flat size, rigid blocks with their dimensions), re-plan, and
set the matching `TWIN_TOWELS` / `TWIN_SHIRTS` / `TWIN_BLOCKS` /
`TWIN_BOX_INNER_CM` in a copy of `rerun.sh`.

Things to know before relying on a re-run:

- A re-run does not reproduce a recording exactly. The cloth solver occasionally
  bunches differently or diverges; a diverged run does not exit and has to be
  killed by PID.
- `execute.log` and `fold_robot_task2_audit.json` hold the result of the saved
  run: 7 of 8 audit checks, every item inside the box, no arm-against-arm overlap.
- `probes/` has the measurement scripts used on the recording (arm clearance,
  arm links inside the box, cloth shape while carried). They were written against
  scratch paths and need their paths edited.
