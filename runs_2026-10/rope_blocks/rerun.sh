#!/usr/bin/env bash
# Plan and simulate rope + blocks (about 3 minutes). Usage: rerun.sh [first|last] [out_dir]
#   last  (default): coil the rope, then set the centre block into the coil  - works
#   first: centre block first, coil around it                                 - the trailing rope tips the block or drapes over it
HERE=$(cd "$(dirname "$0")" && pwd)
CORE=${1:-last}; W=${2:-$HERE/rerun_$(date +%m%d_%H%M)}; mkdir -p $W
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.5/bin/python}
cd "$HERE/../../twin/newton"
$PY tamp_rope.py --scene $HERE/scene.json --out $W/plan.json --core $CORE && $PY twin_rope.py --plan $W/plan.json --out $W 2>&1 | tee $W/execute.log | grep "twin-rope"
