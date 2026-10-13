#!/usr/bin/env bash
# Play a recorded rope run in Isaac Sim (no physics re-run). Usage: show.sh [run_dir] [speed]
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../twin/newton"
DISPLAY=${DISPLAY:-:1} ${ISAAC_PY:-$HOME/miniforge3/envs/isaaclab/bin/python} play_fold_isaac.py \
  $HERE/${1:-best}/rope.usd 15 -72 125 0 -22 22 ${2:-1.0}
