#!/usr/bin/env bash
# Plan and simulate wrap + lift + carry into the box (about 2 minutes). Usage: rerun.sh [plan|top|table|hitch] [out_dir]
#   hitch choker hitch (eye on top, both ends under it), ONE-hand lift, from a STRAIGHT rope (scene_fold.json): the robot folds it first
#   plan  (default) grip each strand at the planner's arc length - lifts and carries the block
#   table grip the ends where they lie on the table (hands 25 cm apart) - the wrap opens, no lift
#   top   grip each strand at the highest point beside the top edge - drags the block, no lift
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-plan}; W=${2:-$HERE/rerun_$(date +%m%d_%H%M)}; mkdir -p $W
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.5/bin/python}
cd "$HERE/../../twin/newton"
if [ "$MODE" = hitch ]; then
  $PY tamp_sling.py --scene $HERE/scene_fold.json --out $W/plan.json --knot girth && $PY twin_rope.py --plan $W/plan.json --out $W 2>&1 | tee $W/execute.log | grep "twin-rope"; exit
fi
$PY tamp_sling.py --scene $HERE/scene.json --out $W/plan.json && ROPE_REGRASP=$MODE $PY twin_rope.py --plan $W/plan.json --out $W 2>&1 | tee $W/execute.log | grep "twin-rope"
