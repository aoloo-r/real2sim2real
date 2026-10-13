#!/usr/bin/env bash
# Play a recorded sled run in Isaac Sim. Usage: show.sh [best|pick_only] [speed]
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../twin/newton"
DISPLAY=${DISPLAY:-:1} ${ISAAC_PY:-$HOME/miniforge3/envs/isaaclab/bin/python} play_fold_isaac.py \
  $HERE/${1:-best}/sled.usd 45 -125 100 0 -30 24 ${2:-1.0}
