#!/usr/bin/env bash
# One command, end to end. The system observes the garment, plans symbolically, validates every
# skeleton in one batched GPU pass, chooses, and hands the robot a plan. Nobody reads a log and
# ranks candidates by hand.
#
#   ./fold_pipeline.sh [workdir] [n_worlds]
#
# Each stage writes a RAW log and streams line-buffered. Piping through a plain grep buffers 4KB
# at a time, which made a working stage look like it had hung for ten minutes.
set -euo pipefail
PY=/home/aoloo/miniforge3/envs/newton-1.7/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-/tmp/foldrun}"
N="${2:-16}"
mkdir -p "$WORK"
cd "$HERE"

echo "== 1/3  observe: settle the shirt and record the state the robot sees"
TWIN_DUMP_STATE="$WORK/observed.npy" TWIN_FOLD=grasp \
  stdbuf -oL "$PY" twin_stockfold.py > "$WORK/observe.log" 2>&1 || true
grep -E "observed state" "$WORK/observe.log" || true

echo "== 2/3  plan: symbolic skeletons, all validated in ONE batched pass ($N worlds)"
# --view so the validation is watchable: N shirts side by side, each trying a different
# skeleton. Set FOLD_PIPELINE_HEADLESS=1 to drop it and run the planning stage faster.
VIEW="--view"; [ "${FOLD_PIPELINE_HEADLESS:-0}" = "1" ] && VIEW=""
stdbuf -oL "$PY" fold_tamp.py --start "$WORK/observed.npy" --depth 4 --validate "$N" \
    --worlds "$N" --spacing 80 $VIEW --out "$WORK/plan.txt" 2>&1 | tee "$WORK/plan.log" \
  | grep --line-buffered -E "^\[tamp\]|camera framed"

if [ ! -s "$WORK/plan.txt" ]; then
  echo "== no plan works from this state - stopping rather than running one that does not"
  exit 1
fi

echo "== 3/3  execute: the robot folds the plan the system chose ($(cat "$WORK/plan.txt"))"
# TWIN_EXPORT_EE also writes the arm's commanded waypoints in the REAL ur5e_base_link frame, so
# the same run that folds it in sim produces the trajectory the UR5e will execute.
TWIN_PLAN_FILE="$WORK/plan.txt" TWIN_LOAD_STATE="$WORK/observed.npy" TWIN_FOLD=grasp \
  TWIN_EXPORT_EE="$WORK/fold_ee.json" \
  stdbuf -oL "$PY" twin_stockfold.py 2>&1 | tee "$WORK/execute.log" \
  | grep --line-buffered -E "^\[twin-stock\]"

echo
if [ -s "$WORK/fold_ee.json" ]; then
  echo "== 4/4  transfer: DRY-RUN on the real UR5e (MoveIt plans + RViz preview, NO motion)"
  echo "   ../../transfer/transfer_to_robot.sh $WORK/fold_ee.json"
  echo "   add --execute (and --with-gripper) only when the preview looks right"
fi
