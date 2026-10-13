#!/usr/bin/env bash
# Closed loop: the robot observes the cloth, the planner rolls out from THAT state, the robot
# executes what the planner returned. No hand-copied plan strings anywhere in the chain.
set -euo pipefail
PY=/home/aoloo/miniforge3/envs/newton-1.5/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-/tmp/foldloop}"
DEPTH="${DEPTH:-3}"
mkdir -p "$WORK"

echo "== 1/3  observe: the twin settles the shirt and dumps what it sees"
cd "$HERE" && TWIN_DUMP_STATE="$WORK/observed.npy" TWIN_FOLD=grasp \
    $PY twin_stockfold.py 2>&1 | grep -E "^\[twin-stock\]" || true

echo "== 2/3  plan: roll out from the observed state, keep what WORKS"
cd "$HERE" && $PY fold_search.py --start "$WORK/observed.npy" \
    --mode success --depth "$DEPTH" --out "$WORK/plan.txt" 2>&1 | grep -E "^\[success\]" || true

if [ ! -s "$WORK/plan.txt" ]; then
    echo "== no plan works from this state - stopping rather than executing something that does not"
    exit 1
fi

echo "== 3/3  execute: the robot runs the planner's answer ($(cat "$WORK/plan.txt"))"
cd "$HERE" && TWIN_PLAN_FILE="$WORK/plan.txt" TWIN_FOLD=grasp \
    $PY twin_stockfold.py 2>&1 | grep -E "^\[twin-stock\]"
