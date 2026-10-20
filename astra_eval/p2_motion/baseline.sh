#!/usr/bin/env bash
# Our stage: the cuRobo layer on the sled actions (also regenerates the packet inputs). About 3 min.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.7/bin/python}
export SLED_BOX_Y=${SLED_BOX_Y:-36} SLED_BASE_LIFT_CM=${SLED_BASE_LIFT_CM:-14}
$PY "$ROOT/astra_eval/p2_motion/export_inputs.py" "$@"
python3 - <<'EOF'
import json, os
p = json.load(open(os.path.join(os.environ.get("ROOT", "."), "astra_eval/p2_motion/inputs/actions.json")))
acts = {}
for q in p:
    acts.setdefault(q["id"], []).append(bool((q.get("our_result") or {}).get("ok")))
print("our layer per action:", {k: any(v) for k, v in acts.items()})
EOF
