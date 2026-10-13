#!/usr/bin/env bash
# Play the recorded s34 run in Isaac Sim (no physics re-run). Usage: show.sh [speed]
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../twin/newton"
DISPLAY=${DISPLAY:-:1} ${ISAAC_PY:-$HOME/miniforge3/envs/isaaclab/bin/python} play_fold_isaac.py \
  $HERE/fold_robot.usd 0 -95 150 0 -28 22 ${1:-1.0}
