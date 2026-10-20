#!/usr/bin/env bash
# Our own real2sim stage on the same capture: Gemini (objects) -> SAM 2 (masks) -> SAM 3D (meshes, metric pose)
# -> sled scene JSON. Usage: baseline.sh inputs/capture OUT_DIR
# The stages are the fold pipeline's (perception/); layout_to_sled.py turns their layout into the sled format.
# NOTE: layout_to_sled.py has to be validated on the first real capture; the layout schema is read generically.
set -e
CAP=$(cd "$1" && pwd); OUT=$2; mkdir -p "$OUT"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SAM=${SAM3D_PY:-$HOME/miniforge3/envs/sam3d-objects/bin/python}
cd "$ROOT/perception"
$SAM gemini_manip.py --capture_dir "$CAP" --out "$OUT/detect" 2>&1 | tee "$OUT/detect.log"
$SAM sam3d_to_scene.py --capture_dir "$CAP" --detections "$OUT/detect" --out "$OUT/recon" 2>&1 | tee "$OUT/recon.log"
python3 "$ROOT/astra_eval/p1_real2sim/layout_to_sled.py" --recon "$OUT/recon" --frame "$CAP/../frame.json" --out "$OUT/scene.json"
echo "[baseline] wrote $OUT/scene.json; score it with score.py"
