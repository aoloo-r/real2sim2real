#!/usr/bin/env bash
# Task 2 (orientation-aware plan): shirt + 2 towels + 3 dynamic blocks. Usage: exec_g3.sh <plan> <out>
PLAN=$1; W=$2; mkdir -p $W
source /tmp/foldrun13/task2/env_t38.sh
  TWIN_BLOCKS="block1:10,6,4:30,10;block2:8,8,6:30,-4;block3:12,5,3:30,-18"
  export TWIN_BLOCKS
cd /home/aoloo/real2sim2real/twin/newton
PY=/home/aoloo/miniforge3/envs/newton-1.5/bin/python
TWIN_SCENE=task1 TWIN_BOX_INNER_CM=38 TWIN_BOX_H_CM=12 TWIN_BOX_DY_CM=36 TWIN_TOWEL_PILE_GAP_CM=1.5 \
  TWIN_ASIDE_CM="28,20;-28,20" TWIN_REACH_ONE_HAND_CM=${REACH:-75} TWIN_RIGID_DYNAMIC=1 TWIN_PIN_KINEMATIC=${PINK:-0} TWIN_VBD_ITERS=10 TWIN_PACK_SLOW=2 \
  TWIN_ARM=ur5e2 TWIN_ARM_LAYOUT=side TWIN_CLOTH=towel TWIN_FOLD=grasp TWIN_COORD=independent TWIN_GOAL=quarter \
  TWIN_FOLD_ARC_FRAC=0.35 TWIN_FOLD_OVERSHOOT_CM=3 TWIN_FOLD_BAND_CM=5 TWIN_FOLD_HOLD_MAX=1400 TWIN_GRAB_INSET_CM=2 \
  TWIN_IK_GAIN=3 TWIN_FOLD_PIN_CM=2.5 TWIN_GRIP_HARD=0.06 TWIN_STACK_CARRY=2.5 TWIN_CARRY_PROBE_CM=1.5 \
  TWIN_REC_THICK_CM=0.6 TWIN_LABEL_PLAN=$PLAN TWIN_LABELS=/tmp/foldrun13/task2/labels_t38.json \
  TWIN_REC_USD=$W/fold_robot.usd \
  setsid nohup $PY twin_stockfold.py --viewer null > $W/execute.log 2>&1 < /dev/null &
echo "started $! $(date +%H:%M:%S)"
