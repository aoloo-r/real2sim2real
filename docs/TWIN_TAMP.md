# TAMP + Parallel Digital-Twin layer

Built on top of the real2sim pipeline. Turns a **natural-language instruction**
into a **physically-validated multi-step plan** by searching candidate plans in
parallel inside the Isaac Lab twin, then executes the winner. This is the
"twin-enhanced TAMP" direction: instead of trusting one open-loop plan (TiPToP /
VLA), we search the plan space in the twin and return the plan that physically
works best.

## Flow

```
LANGUAGE ──► TASK PLAN ──► N CANDIDATE PLANS ──► PARALLEL SEARCH ──► EXECUTE WINNER
 (Gemini)    (ordered      (vary order +         (N cloned envs,     (replay the
             pick/place)    grasp + placement)    physics-scored)     chosen plan)
```

## Files

### `tamp/` (runs in the sam-3d-objects conda env, no Isaac)
| File | Role |
|------|------|
| `tamp_plan.py` | Language instruction + scene → grounded symbolic goal + **ordered pick/place actions** (Gemini, object-agnostic). |
| `twin_plan_search.py` | One goal → **N candidate plans**, varying task (action order) + motion (grasp strategy, placement). |
| `ee_geometry.py` | **Shared grasp/place geometry** (single source of truth) — extracted from `real2sim_franka.export_ee_trajectory`; pure numpy, imported by both the twin and the compiler. |
| `tamp_to_ee.py` | TAMP action plan → EE-trajectory JSON(s) (the motion compiler), via `ee_geometry`. |

### `perception/` — capture → detect (Gemini) → reconstruct (SAM 3D) → QA → `scene_layout.json`
### `transfer/` — sim → real UR5e (`ur5e_ee_executor.py` MoveIt, `transfer_to_robot.sh`)
### `scripts/` — orchestration (`run_robot_pipeline.sh`, `load_in_isaaclab.sh`)

### `twin/` (runs via `./isaaclab.sh -p`, Isaac env)
| File | Role |
|------|------|
| `real2sim_franka.py` | Loads the reconstructed scene; `--replay_ee_traj` / `--replay_plan` execute a (multi-step) plan with a per-step grasp-success verdict. |
| `twin_plan_eval.py` | **Parallel task-plan search engine**: clones the real twin into N envs, rolls out N candidate plans simultaneously, scores each (goals + stability + collisions + efficiency), picks the winner. |
| `twin_grasp_eval.py` | Parallel grasp-candidate evaluator (nested sub-case: ranks grasps for one object). |
| `twin_parallel_spike.py` | De-risking spike: confirms N-env cloning + cuRobo `plan_batch`. |
| `_ur_assets.py` | UR5e/Robotiq asset configs. |

## Run

```bash
# 1. language -> task plan
python tamp/tamp_plan.py --scene_dir <out> --capture_dir <cap> \
    --instruction "put the kiwi in the bowl and the cup on the plate" --out /tmp/task_plan.json
# 2. task plan -> N candidate plans
python tamp/twin_plan_search.py --task_plan /tmp/task_plan.json --n 8 --out /tmp/candidates.json
# 3. PARALLEL SEARCH in the twin -> ranked, winner (headless = fast; GUI = watch the N envs)
./isaaclab.sh -p twin/twin_plan_eval.py --scene_dir <out> --capture_dir <cap> \
    --candidates /tmp/candidates.json [--headless]
# 3b. RESOLVE the grasp orientation against the REAL UR5e (cuRobo ur5e.yml IK, scene-
#     aware) and bake it into the trajectory -> the twin and the real robot then run
#     the SAME orientation verbatim (sim == real grasp).
python twin/resolve_orientation.py --traj <ee.json> --out <ee_resolved.json> --tcp_offset 0.187
# 4. execute the RESOLVED trajectory — sim twin (replay) AND real robot run it verbatim
./isaaclab.sh -p twin/real2sim_franka.py --scene_dir <out> --capture_dir <cap> \
    --robot franka --sim_attach --replay_plan <plan_manifest.json>   # resolved files
```

## Sim ≡ Real invariant (grasp orientation)
The EE trajectory is the **single source of truth**: every TCP pose + orientation is
fixed up front and **both** the sim twin and the real `ur5e_ee_executor.py` execute it
**verbatim** — no runtime improvisation. The grasp orientation is resolved **once** by
`twin/resolve_orientation.py` against the real UR5e's kinematics (cuRobo `ur5e.yml` IK,
table+object collision from the trajectory), so the orientation validated in sim is the
one the robot performs. (This fixes the bug Xiaohan caught: the old executor searched
orientations at runtime — straight-down → lean → tilt — so the sim grasped one face and
the real robot another. That search now happens once, offline, baked in.) Use the **same
`--tcp_offset`** (UR5e+HandE ≈ 0.187) in the resolver and the executor.

## Scoring (twin_plan_eval)
- **goals** — each object settled within its target's footprint
- **stability** — objects at rest (low velocity) after release
- **collisions** — bystander objects not displaced / knocked over
- **efficiency** — fewer planning failures

## Status / known limits
- Validated end-to-end on the real reconstructed meshes (`mesh_obb.usd`).
- Object **stacking** respected (e.g. a kiwi resting *on* a plate spawns on top, not embedded).
- A flat-disc bowl can't truly *contain* a round object — full "in a bowl" fidelity
  needs the concave raw mesh (`mesh.obj` + convex-decomposition collision).
- GUI rollout of many envs is slow (rendering); use `--headless` for the scores,
  GUI to watch the parallel search.
- Next: Stage-3 evolutionary loop (mutate top-K plans, iterate); RDT2 VLA baseline; MolmoSpaces eval.
