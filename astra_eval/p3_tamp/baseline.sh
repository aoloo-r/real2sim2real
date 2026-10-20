#!/usr/bin/env bash
# Our stage on the same goals: tamp_sym.py (A* + Newton rollouts) then twin_sled.py. About 10 min per goal.
# Usage: baseline.sh [OUT_DIR]   (default astra_eval/runs/p3_tamp/baseline)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/astra_eval/runs/p3_tamp/baseline}; mkdir -p "$OUT"
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.7/bin/python}
IN=$ROOT/astra_eval/p3_tamp/inputs
cd "$ROOT/twin/newton"
for g in $(python3 -c "import json;print(' '.join(json.load(open('$IN/goals.json')).keys()))"); do
  SC=$(python3 -c "import json;print(json.load(open('$IN/goals.json'))['$g']['scene'])")
  mkdir -p "$OUT/$g"
  echo "[baseline] $g (scene $SC)"
  $PY tamp_sym.py --scene "$IN/$SC" --goal "$IN/goal_$g.json" --out "$OUT/$g/plan.json" 2>&1 | tee "$OUT/$g/plan.log" | grep "\[sym\]"
  $PY twin_sled.py --plan "$OUT/$g/plan.json" --out "$OUT/$g" 2>&1 | tee "$OUT/$g/execute.log" | grep "twin-sled"
done
echo "[baseline] done -> $OUT; score with score.py --run $OUT"
