#!/usr/bin/env bash
# THE FULL PIPELINE, one command: real RGBD -> reconstructed garment -> twin -> plan -> robot.
#
#   ./run_full_pipeline.sh [workdir]
#
# Stages, in order, each visible in the Newton viewer where it has something to show:
#   1 capture      RGBD off the robot's RealSense                        (~20s)
#   2 reconstruct  SAM3D: mask + mesh for whatever is on the table       (~3min, GPU alone)
#   3 scene        depth-grounded layout in the ur5e_base_link frame     (~30s)
#   4 observe      twin builds THAT garment (two-layer, mask-derived),
#                  settles it, dumps the state + the real robot frame    (~40s)
#                  ...or, when the capture shows the shirt already FOLDED (footprint
#                  < 85% of flat, or TWIN_STATE_FIT=1): fold_state_fit simulates fold
#                  primitives from the flat tuned shirt and keeps the one whose
#                  settled state matches the capture through its camera   (~2-10min)
#                  TWIN_FLAT_TS=<capture of the same shirt lying flat> sets the rest-shape
#                  scale for that path (else the stock asset size).
#   5 plan         symbolic skeletons, all validated in ONE batched pass  (~3-5min)
#   6 execute      the robot folds the winning plan in sim, and exports
#                  the EE waypoints in the real robot's frame            (~4min)
#
# Then, separately and never automatically:
#   ../../transfer/transfer_to_robot.sh $WORK/fold_ee.json --search-orientations   # DRY RUN
#   ...and only with a human watching, add --execute.
#
# The garment is whatever the camera saw. Nothing here assumes a t-shirt: the cloth is built
# from the capture's own mask + depth, and the fold goal is scored against that garment's own
# flat footprint.
set -euo pipefail
PY=/home/aoloo/miniforge3/envs/newton-1.5/bin/python
ISAAC_PY=/home/aoloo/miniforge3/envs/isaaclab/bin/python
# Everything visual happens in ISAAC SIM - Newton runs headless throughout. The recordings save
# incrementally, so the player opens them while they are still being written (accept the
# "fetch changes" toast to pull newer frames, or just let it loop).
ISAAC_PID=""
show_in_isaac() {  # $1 = usd file, $2.. = camera numbers
  local f="$1"; shift || true
  for _ in $(seq 1 60); do [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 1000000 ] && break; sleep 5; done
  [ -f "$f" ] || { echo "   (no recording appeared at $f)"; return; }
  pkill -f play_fold_isaac.py 2>/dev/null && sleep 3
  DISPLAY="${DISPLAY:-:1}" setsid nohup "$ISAAC_PY" "$HERE/play_fold_isaac.py" "$f" "$@" \
      > "$WORK/isaac_player.log" 2>&1 < /dev/null &
  ISAAC_PID=$!
  echo "   -> showing $f in Isaac Sim"
}
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-/tmp/foldrun}"
# Optional 2nd arg: an EXISTING capture id, to re-run everything downstream without taking a new
# RGBD (useful when the robot link is down but the table has not changed).
TS_ARG="${2:-}"
N="${N:-9}"
mkdir -p "$WORK"
cd "$HERE"

if [ -n "$TS_ARG" ]; then
  TS="$TS_ARG"
  echo "== 1/6  capture SKIPPED - reusing $TS (passed explicitly)"
else
echo "== 1/6  capture"
# /tmp/last_capture_dest.txt persists from the PREVIOUS run, so a capture that fails on the
# flaky link leaves a perfectly valid-looking id behind and the pipeline quietly reconstructs
# a stale capture. Remember what was there and refuse to continue if it did not change.
PREV_TS=$(cat /tmp/last_capture_dest.txt 2>/dev/null || echo none)
bash run_twin_pipeline.sh capture 2>&1 | tee "$WORK/capture.log" | grep -E "capture\]|CAPTURE OK|net\]|ERROR" || true
TS_PATH=$(cat /tmp/last_capture_dest.txt 2>/dev/null || echo "")
if [ -z "$TS_PATH" ] || [ "$TS_PATH" = "$PREV_TS" ]; then
  echo "   CAPTURE FAILED - no new capture (still '$PREV_TS'). Robot link down?"
  echo "   Refusing to run the pipeline on a stale capture."
  exit 1
fi
TS=$(basename "$TS_PATH")
echo "   TS=$TS"
fi

echo "== 2/6  reconstruct $TS  (needs the GPU alone)"
# Idempotent: a capture that is already reconstructed + laid out is not done again - rerunning
# the pipeline on the same TS goes straight to the twin.
# ...but only if the layout actually CONTAINS the garment. A run killed mid-reconstruct once
# left a valid-looking scene_layout.json with zero objects, and the skip walked straight into it.
if "$PY" -c "
import json,sys
d=json.load(open('/home/aoloo/sam-3d-objects/outputs/$TS/scene_layout.json'))
sys.exit(0 if d.get('objects') else 1)" 2>/dev/null; then
  echo "   already reconstructed - skipping stages 2 and 3"
  SKIP_RECON=1
else
  SKIP_RECON=0
fi
if [ "$SKIP_RECON" = 0 ]; then
# SAM3D needs ~16GB; a leftover Isaac player holds ~3GB each and OOMed a rerun (which then
# clobbered a good reconstruction with a 0-object layout). Clear the screen before the GPU work.
pkill -f play_fold_isaac.py 2>/dev/null || true
sleep 2
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free_mb" -lt 12000 ]; then
  echo "   WARNING: only ${free_mb}MiB free; SAM3D wants ~12000 and will OOM."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/     holding: /'
fi
bash run_twin_pipeline.sh reconstruct "$TS" > "$WORK/reconstruct.log" 2>&1 \
  || { echo "   reconstruct FAILED - see $WORK/reconstruct.log"; tail -3 "$WORK/reconstruct.log"; exit 1; }
grep -E "objects reconstructed" "$WORK/reconstruct.log" | sed 's/^/   /' || true

echo "== 3/6  scene layout"
bash run_twin_pipeline.sh scene "$TS" > "$WORK/scene.log" 2>&1 || true
fi
"$PY" -c "
import json; d=json.load(open('/home/aoloo/sam-3d-objects/outputs/$TS/scene_layout.json'))
for o in d['objects']:
    di=o.get('depth_info',{})
    print('   %s  size %.2fm' % (di.get('label') or o.get('label'), di.get('physical_size_m') or 0))"

# WHICH CLOTH. unisex_shirt.usd is a properly modelled garment - real sleeve and seam topology,
# 6436 particles, tuned over weeks. The mask-built sheet is a two-layer extrusion of the SAM2
# silhouette: faithful in outline and pose, but no seams and a noisy boundary. So use the ASSET
# when the capture actually shows a shirt, and only reconstruct when there is no asset for what
# is on the table (trousers, towels, anything new). Hardcoding 'mask' here quietly replaced a
# tuned shirt with a cutout the moment we went back to folding a shirt.
# WHICH GARMENT. A capture may show several (2026-09-11: purple shirt + red tee). TWIN_GARMENT
# = object id or label substring selects the one THIS run folds; every stage below reads the
# same selector (cloth_mesh.select_garment), so masks, pose, planner cloth and twin agree.
GARMENT="${TWIN_GARMENT:-}"
export TWIN_GARMENT="$GARMENT"
LABEL=$("$PY" -c "
import sys; sys.path.insert(0, '$HERE')
from cloth_mesh import select_garment, _obj_label
o = select_garment('$TS', '$GARMENT'); print(_obj_label(o))") \
  || { echo "   cannot pick a garment (TWIN_GARMENT='$GARMENT') - see the labels above"; exit 1; }
echo "   garment = '$LABEL'${GARMENT:+  (TWIN_GARMENT=$GARMENT)}"
if [ -n "${TWIN_CLOTH:-}" ]; then
  CLOTH="$TWIN_CLOTH"; WHY="forced by TWIN_CLOTH"
elif echo "$LABEL" | grep -qE "shirt|tee|t-shirt"; then
  # 'fit' = the EXACT tuned USD, warped to the capture: same topology and seams, but at the
  # measured pose/size. Plain 'usd' puts the shirt at the asset's stock spot, which makes the
  # reach verdicts answer a question about the wrong table.
  CLOTH="fit";  WHY="'$LABEL' matches the tuned unisex_shirt asset, fitted to this capture"
else
  CLOTH="mask"; WHY="no asset for '$LABEL' - reconstructing it from the capture"
fi
echo "   cloth = $CLOTH  ($WHY)"

# FOLDED START? Only the tuned shirt can be state-fitted (the fold primitives are the shirt's).
STATE_FIT="${TWIN_STATE_FIT:-auto}"
FLAT_ARGS=()
[ -n "${TWIN_FLAT_TS:-}" ] && FLAT_ARGS=(--flat-ts "$TWIN_FLAT_TS")
if [ "$CLOTH" = "fit" ] && [ "$STATE_FIT" = "auto" ]; then
  AREA=$("$PY" fold_state_fit.py --ts "$TS" --probe "${FLAT_ARGS[@]}" 2>/dev/null | sed -n 's/^AREA_FRAC=//p')
  echo "   garment footprint = ${AREA:-?} of flat"
  if [ -n "$AREA" ] && "$PY" -c "import sys; sys.exit(0 if float('$AREA') < 0.85 else 1)"; then
    STATE_FIT=1
  else
    STATE_FIT=0
  fi
fi
FIT_SCALE=""
# A stale observed.npy from an earlier run in the same WORK dir passed the "-s" checks below and
# let a CRASHED observe stage go unnoticed (2026-09-11): remove it first, so no state = no run.
rm -f "$WORK/observed.npy" "$WORK/observed.npy.robot.json"
if [ "$STATE_FIT" = "1" ]; then
  echo "== 4/6  observe: shirt is NOT flat - fitting a fold state (Newton, headless; recorded)"
  # The fit's search grid (9 flat shirts each trying one fold) is recorded for QA but NOT put
  # on screen: the first Isaac window should be the planner starting from the half-folded
  # state the camera saw. TWIN_SHOW_FIT=1 shows the search instead.
  [ "${TWIN_SHOW_FIT:-0}" = "1" ] && show_in_isaac "$WORK/fit_grid.usd" 300 80 260 120 80 20 &
  stdbuf -oL "$PY" fold_state_fit.py --ts "$TS" --out "$WORK" --depth "${TWIN_STATE_FIT_DEPTH:-1}" \
      --worlds "$N" "${FLAT_ARGS[@]}" --rec-usd "$WORK/fit_grid.usd" 2>&1 | tee "$WORK/observe.log" \
      | grep --line-buffered -E "^\[statefit\]|^\[fold_env\]" || true
  wait || true
  [ -s "$WORK/observed.npy" ] || { echo "   state fit produced no state"; exit 1; }
  FIT_SCALE=$("$PY" -c "import json; print(','.join('%.4f' % v for v in json.load(open('$WORK/observed.npy.robot.json'))['fit_scale']))")
  echo "   fitted state: $("$PY" -c "import json; print(json.load(open('$WORK/fit_report.json'))['symbolic']['label'])")  rest-shape scale $FIT_SCALE  (QA: $WORK/fit_overlay.png)"
else
echo "== 4/6  observe: build the garment in the twin and settle it"
# Rest-shape scale from the FLAT reference capture when one is given, not from this capture's
# silhouette: a hem cut off by the image border (capture 141425) otherwise shrinks the asset
# to 56x44 for the whole run.
REF_SCALE=""
if [ -n "${TWIN_FLAT_TS:-}" ]; then
  REF_SCALE=$("$PY" -c "
import sys; sys.path.insert(0, '$HERE')
from fold_state_fit import flat_scale_from_capture as f
print(','.join('%.4f' % v for v in f('$TWIN_FLAT_TS')))" 2>/dev/null || true)
  [ -n "$REF_SCALE" ] && echo "   rest-shape scale from flat capture $TWIN_FLAT_TS: $REF_SCALE"
fi
TWIN_TS="$TS" TWIN_CLOTH="$CLOTH" TWIN_FOLD=grasp TWIN_DUMP_STATE="$WORK/observed.npy" \
  TWIN_FIT_SCALE="${REF_SCALE}" \
  stdbuf -oL "$PY" twin_stockfold.py --viewer null > "$WORK/observe.log" 2>&1 || true
grep -E "cloth from mask|observed state|robot frame" "$WORK/observe.log" | sed 's/^/   /' || true
[ -s "$WORK/observed.npy" ] || { echo "   observe produced no state - see $WORK/observe.log:"
                                 grep -E "Error|Traceback" -A2 "$WORK/observe.log" | tail -4 | sed 's/^/   /'; exit 1; }
# the twin's fitted rest-shape scale, so the planner does not load the state into a stock shape
FIT_SCALE=$("$PY" -c "
import json
try: print(','.join('%.4f' % v for v in json.load(open('$WORK/observed.npy.robot.json')).get('fit_scale', [1, 1])))
except Exception: print('1,1')")
fi

echo "== 5/6  plan: $N skeletons validated in physics - WATCH THEM IN ISAAC SIM"
CLOTH_ARG=$([ "$CLOTH" = "mask" ] && echo "mask:$TS${GARMENT:+:$GARMENT}" || echo "shirt")
show_in_isaac "$WORK/plan_grid.usd" 300 80 260 120 80 20 &
stdbuf -oL "$PY" fold_tamp.py --start "$WORK/observed.npy" --cloth "$CLOTH_ARG" \
    --scale "${FIT_SCALE:-1,1}" ${TWIN_GOAL_AREA_FRAC:+--goal-area $TWIN_GOAL_AREA_FRAC} \
    --depth 4 --validate "$N" --worlds "$N" --spacing 130 --rec-usd "$WORK/plan_grid.usd" \
    --out "$WORK/plan.txt" \
    2>&1 | tee "$WORK/plan.log" | grep --line-buffered -E "^\[tamp\]|^\[fold_env\]" || true
wait || true

if [ ! -s "$WORK/plan.txt" ]; then
  echo "== no plan survives physics from this state. That is a result, not a crash:"
  grep -E "^\[tamp\] (WORKS|FAILS)" "$WORK/plan.log" | sed 's/^/   /' || true
  exit 1
fi

echo "== 6/6  execute: the ROBOT folds the chosen plan ($(cat "$WORK/plan.txt")) - IN ISAAC SIM"
show_in_isaac "$WORK/fold_robot.usd" 110 -150 85 -5 -45 22 &
TWIN_TS="$TS" TWIN_CLOTH="$CLOTH" TWIN_PLAN_FILE="$WORK/plan.txt" TWIN_LOAD_STATE="$WORK/observed.npy" \
  TWIN_FIT_SCALE="${FIT_SCALE:-}" \
  TWIN_FOLD=grasp TWIN_EXPORT_EE="$WORK/fold_ee.json" TWIN_REC_USD="$WORK/fold_robot.usd" \
  stdbuf -oL "$PY" twin_stockfold.py --viewer null 2>&1 | tee "$WORK/execute.log" \
  | grep --line-buffered -E "^\[twin-stock\]" || true
wait || true

echo
echo "== done.  capture=$TS  plan=$(cat "$WORK/plan.txt")"
if [ -s "$WORK/fold_ee.json" ]; then
  echo "   EE trajectory: $WORK/fold_ee.json"
  echo "   DRY RUN on the real arm (plans + RViz, no motion):"
  echo "     ../../transfer/transfer_to_robot.sh $WORK/fold_ee.json --search-orientations"
fi
