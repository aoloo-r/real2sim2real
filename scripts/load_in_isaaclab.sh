#!/usr/bin/env bash
# Guarded loader: open a reconstructed scene in Isaac Lab (GUI) only if there is
# enough free VRAM, so it can't OOM-crash an in-progress training run.
#
# Usage: load_in_isaaclab.sh <scene_dir> <capture_dir> [extra real2sim_franka args...]
#   e.g. load_in_isaaclab.sh outputs/sam3d_scene captures/robot_XXXX --demo none
set -euo pipefail

SCENE="${1:?usage: load_in_isaaclab.sh <scene_dir> <capture_dir> [args...]}"
CAP="${2:?capture_dir required}"
shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"

# Resolve to ABSOLUTE paths NOW (before any cd) so relative args still work.
SCENE="$(cd "$SCENE" 2>/dev/null && pwd || echo "$SCENE")"
CAP="$(cd "$CAP" 2>/dev/null && pwd || echo "$CAP")"
if [ ! -f "$SCENE/scene_layout.json" ]; then
  echo "ERROR: no scene_layout.json in $SCENE"; exit 2
fi

bash "$HERE/vram_guard.sh" 6500 "Isaac Lab real2sim sim" || {
  echo "Not launching Isaac Lab (insufficient VRAM)."; exit 3; }

export DISPLAY="${DISPLAY:-:1}"   # always show the GUI (user preference)
cd "${ISAACLAB_DIR:-$HOME/IsaacLab}"
exec ./isaaclab.sh -p scripts/real2sim_franka.py \
  --scene_dir "$SCENE" --capture_dir "$CAP" "$@"
