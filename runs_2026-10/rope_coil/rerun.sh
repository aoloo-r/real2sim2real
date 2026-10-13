#!/usr/bin/env bash
# Plan and simulate the rope coil (about 2 minutes). Usage: rerun.sh [scene.json] [out_dir]
HERE=$(cd "$(dirname "$0")" && pwd)
SCENE=${1:-$HERE/scene.json}; W=${2:-$HERE/rerun_$(date +%m%d_%H%M)}; mkdir -p $W
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.5/bin/python}
cd "$HERE/../../twin/newton"
$PY tamp_rope.py --scene $SCENE --out $W/plan.json && $PY twin_rope.py --plan $W/plan.json --out $W 2>&1 | tee $W/execute.log | grep "twin-rope"
