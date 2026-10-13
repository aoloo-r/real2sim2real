#!/usr/bin/env bash
# Play a recorded sling run in Isaac Sim. Usage: show.sh [fold_best|best|open_wrap|top_edge] [speed]
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../twin/newton"
DISPLAY=${DISPLAY:-:1} ${ISAAC_PY:-$HOME/miniforge3/envs/isaaclab/bin/python} play_fold_isaac.py \
  $HERE/${1:-fold_best}/rope.usd 35 -115 75 0 -38 28 ${2:-1.0}
