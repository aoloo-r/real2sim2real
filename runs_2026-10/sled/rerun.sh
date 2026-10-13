#!/usr/bin/env bash
# Plan (with rollouts in the twin) and execute the towel-sled task. About 8 minutes. Usage: rerun.sh [scene.json] [out_dir]
HERE=$(cd "$(dirname "$0")" && pwd)
SCENE=${1:-$HERE/scene.json}; W=${2:-$HERE/rerun_$(date +%m%d_%H%M)}; mkdir -p $W
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.5/bin/python}
cd "$HERE/../../twin/newton"
$PY tamp_sled.py --scene $SCENE --out $W/plan.json 2>&1 | grep "sled-plan" && $PY twin_sled.py --plan $W/plan.json --out $W 2>&1 | tee $W/execute.log | grep "twin-sled"
