#!/usr/bin/env bash
# FOLD EVERY GARMENT ON THE TABLE AND STACK THEM, in ONE twin run.
#
#   ./run_two_garments.sh <workdir> <capture-id> [anchor-garment-id] [garment-ids in fold order...]
#   e.g. ./run_two_garments.sh /tmp/foldrun8 robot_20260911_120313 1 1 0
#
# The capture must already be reconstructed (run_full_pipeline.sh stages 2-3, or
# run_twin_pipeline.sh reconstruct+scene). Stages:
#   observe   one headless twin per garment, ALL garments in the scene, world pinned to the
#             ANCHOR garment -> <work>/g<id>/observed.npy (that garment's particles only)
#   plan      one fold_tamp per garment from its observed state and its own rest-shape scale
#   execute   ONE twin: TWIN_PLANS folds the garments in order, then TWIN_STACK_ON lays the
#             last one on the anchor; recorded to <work>/both/fold_robot.usd and shown in Isaac
# Only shirts are handled today (TWIN_CLOTH=fit warps the tuned unisex_shirt.usd to each mask).
set -euo pipefail
PY=/home/aoloo/miniforge3/envs/newton-1.5/bin/python
ISAAC_PY=/home/aoloo/miniforge3/envs/isaaclab/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?workdir}"; TS="${2:?capture id}"; ANCHOR="${3:-}"
shift 3 || true
ORDER=("$@")
cd "$HERE"
LAY=/home/aoloo/sam-3d-objects/outputs/$TS/scene_layout.json
[ -s "$LAY" ] || { echo "no scene layout for $TS - reconstruct first"; exit 1; }
if [ ${#ORDER[@]} -eq 0 ]; then
  mapfile -t ORDER < <("$PY" -c "
import json, sys; sys.path.insert(0, '$HERE')
from cloth_mesh import GARMENT_WORDS, _obj_label
for o in json.load(open('$LAY'))['objects']:
    if any(w in _obj_label(o) for w in GARMENT_WORDS): print(o['id'])")
fi
[ -n "$ANCHOR" ] || ANCHOR="${ORDER[0]}"
LAST="${ORDER[-1]}"
echo "== garments in fold order: ${ORDER[*]}   anchor (stack target): $ANCHOR"
export TWIN_TS="$TS" TWIN_ANCHOR="$ANCHOR" TWIN_CLOTH=fit TWIN_FOLD=grasp
export DISPLAY="${DISPLAY:-:1}"

PLANS=""
for G in "${ORDER[@]}"; do
  W="$WORK/g$G"; mkdir -p "$W"
  echo "== observe garment $G (all garments in the scene, world pinned to $ANCHOR)"
  rm -f "$W/observed.npy" "$W/observed.npy.robot.json"
  TWIN_GARMENT="$G" TWIN_DUMP_STATE="$W/observed.npy" "$PY" twin_stockfold.py --viewer null \
      > "$W/observe.log" 2>&1 || true
  [ -s "$W/observed.npy" ] || { echo "   observe failed - see $W/observe.log"; exit 1; }
  grep -E "garments in the scene|observed state" "$W/observe.log" | sed 's/^/   /'
  FS=$("$PY" -c "import json; print(','.join('%.4f' % v for v in json.load(open('$W/observed.npy.robot.json'))['fit_scale']))")
  echo "== plan garment $G (rest-shape scale $FS)"
  "$PY" fold_tamp.py --start "$W/observed.npy" --cloth shirt --scale "$FS" --depth 4 \
      --validate "${N:-9}" --worlds "${N:-9}" --spacing 130 ${TWIN_GOAL_AREA_FRAC:+--goal-area $TWIN_GOAL_AREA_FRAC} \
      --out "$W/plan.txt" > "$W/plan.log" 2>&1 || true
  [ -s "$W/plan.txt" ] || { echo "   no plan survives physics for garment $G:"; grep -E "^\[tamp\] (WORKS|FAILS)" "$W/plan.log" | sed 's/^/   /'; exit 1; }
  echo "   plan: $(cat "$W/plan.txt")"
  PLANS="${PLANS:+$PLANS;}$G:$W/plan.txt"
done

W="$WORK/both"; mkdir -p "$W"
echo "== execute: ONE run - $PLANS ; then stack $LAST on $ANCHOR - IN ISAAC SIM"
TWIN_GARMENT="$ANCHOR" TWIN_PLANS="$PLANS" TWIN_STACK_ON="$ANCHOR" \
  TWIN_EXPORT_EE="$W/fold_ee.json" TWIN_REC_USD="$W/fold_robot.usd" TWIN_FINAL_STATE="$W/final_all.npy" \
  setsid nohup "$PY" twin_stockfold.py --viewer null > "$W/execute.log" 2>&1 < /dev/null &
TWIN_PID=$!
for _ in $(seq 1 60); do [ -f "$W/fold_robot.usd" ] && [ "$(stat -c%s "$W/fold_robot.usd")" -gt 1000000 ] && break; sleep 5; done
pkill -f play_fold_isaac.py 2>/dev/null && sleep 3
setsid nohup "$ISAAC_PY" play_fold_isaac.py "$W/fold_robot.usd" 110 -150 85 -5 -45 22 > "$W/isaac_player.log" 2>&1 < /dev/null &
wait "$TWIN_PID" || true
grep -E "=== active|: fold|stack:|AUDIT|DIVERGED|plan complete|exported" "$W/execute.log" | grep -v narrow_phase | sed 's/^/   /'
# the player loops only the frames it had at open: reopen on the finished recording
pkill -f play_fold_isaac.py 2>/dev/null && sleep 3
setsid nohup "$ISAAC_PY" play_fold_isaac.py "$W/fold_robot.usd" 110 -150 85 -5 -45 22 > "$W/isaac_player_final.log" 2>&1 < /dev/null &
echo "== done. recording $W/fold_robot.usd  EE export $W/fold_ee.json (check per-garment z before any real run)"
