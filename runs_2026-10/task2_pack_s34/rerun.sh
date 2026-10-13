#!/usr/bin/env bash
# Re-simulate Task 2 (shirt + 2 towels + 3 blocks, 38 cm box) from the saved plan. ~35 min wall on the 24 GB GPU.
# Usage: rerun.sh [out_dir]   (detached; follow <out_dir>/execute.log)
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HERE/rerun_$(date +%m%d_%H%M)}; mkdir -p $W
export TWIN_TOWELS="towelA:30,40:0,0:1;towelB:34,24:0,0:2"
export TWIN_SHIRTS="shirt:0,0:0:0.55"
export TWIN_BLOCKS="block1:10,6,4:30,10;block2:8,8,6:30,-4;block3:12,5,3:30,-18"
cd "$HERE/../../twin/newton"
PY=${NEWTON_PY:-$HOME/miniforge3/envs/newton-1.5/bin/python}
TWIN_CARRY_JAW_CM=13 TWIN_DEBUG_NO_GRAPH=1 TWIN_PIN_KINEMATIC=1 \
  TWIN_SCENE=task1 TWIN_BOX_INNER_CM=38 TWIN_BOX_H_CM=12 TWIN_BOX_DY_CM=36 TWIN_TOWEL_PILE_GAP_CM=1.5 \
  TWIN_ASIDE_CM="28,20;-28,20" TWIN_REACH_ONE_HAND_CM=75 TWIN_RIGID_DYNAMIC=1 TWIN_VBD_ITERS=10 TWIN_PACK_SLOW=2 \
  TWIN_ARM=ur5e2 TWIN_ARM_LAYOUT=side TWIN_CLOTH=towel TWIN_FOLD=grasp TWIN_COORD=independent TWIN_GOAL=quarter \
  TWIN_FOLD_ARC_FRAC=0.35 TWIN_FOLD_OVERSHOOT_CM=3 TWIN_FOLD_BAND_CM=5 TWIN_FOLD_HOLD_MAX=1400 TWIN_GRAB_INSET_CM=2 \
  TWIN_IK_GAIN=3 TWIN_FOLD_PIN_CM=2.5 TWIN_GRIP_HARD=0.06 TWIN_STACK_CARRY=2.5 TWIN_CARRY_PROBE_CM=1.5 \
  TWIN_REC_THICK_CM=0.6 TWIN_LABEL_PLAN=$HERE/plan_t38b.json TWIN_LABELS=$HERE/labels_t38.json \
  TWIN_REC_USD=$W/fold_robot.usd \
  setsid nohup $PY twin_stockfold.py --viewer null > $W/execute.log 2>&1 < /dev/null &
echo "started $! -> $W/execute.log"
