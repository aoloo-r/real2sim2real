#!/bin/bash
# Full RGBD -> digital-twin pipeline. Run stages in order, or "all" for everything.
#
#   bash run_twin_pipeline.sh capture              # 1) take a new RGBD on the robot
#   bash run_twin_pipeline.sh reconstruct <TS>     # 2) SAM + Gemini + SAM3D meshes
#   bash run_twin_pipeline.sh scene <TS>           # 3) depth-grounded scene layout
#   bash run_twin_pipeline.sh sim <TS>             # 4) robot twin sim (viewer on $DISPLAY)
#   bash run_twin_pipeline.sh all                  # capture -> reconstruct -> scene -> sim
#
# <TS> is the capture id, e.g. robot_20260819_123718 (printed by the capture stage).
set -e
SAM=/home/aoloo/sam-3d-objects
PY_SAM=/home/aoloo/miniforge3/envs/sam3d-objects/bin/python
PY_HY="${PY_HY:-/home/aoloo/miniforge3/envs/hunyuan3d/bin/python}"
[ -x "$PY_HY" ] || PY_HY=/home/aoloo/miniforge3/envs/sam3d-objects/bin/python  # hunyuan3d env removed
PY_NEWTON=/home/aoloo/miniforge3/envs/newton-1.7/bin/python
TWIN=/home/aoloo/real2sim2real/twin/newton/twin_robot.py

stage="${1:?usage: run_twin_pipeline.sh capture|reconstruct|scene|sim|all [TS]}"
TS="${2:-}"

do_capture() {
    echo "== [1/4] capture (robot must be on RAO NGAJI wifi) =="
    # The link runs at 60-80% packet loss, so a single ping is a coin flip and the stage aborted
    # on a robot that was actually there. Accept ANY reply out of a burst, and only give up if a
    # whole burst is lost.
    ping -c 8 -i 0.3 -W 1 172.20.10.4 >/dev/null 2>&1 || true     # warm the flaky link
    _up=0
    for _try in 1 2 3; do
        if ping -c 5 -i 0.3 -W 6 172.20.10.4 2>/dev/null | grep -qE "[1-5] received"; then _up=1; break; fi
        echo "  [net] burst $_try: no reply, retrying"
    done
    [ "$_up" = 1 ] || { echo "robot unreachable - check wifi"; exit 1; }
    ( cd "$SAM" && bash ./run_robot_pipeline.sh capture )
    TS=$(basename "$(cat /tmp/last_capture_dest.txt)")
    echo "CAPTURE OK: TS=$TS"
}

do_reconstruct() {
    echo "== [2/4] reconstruct $TS (SAM + Gemini + SAM3D; needs GPU alone, ~5 min) =="
    ( cd "$SAM" && $PY_SAM demo.py \
        --image_path "captures/$TS/rgb.png" \
        --output_dir "outputs/${TS}_full" \
        --use_gemini )
}

do_scene() {
    echo "== [3/4] depth-grounded scene layout for $TS =="
    ( cd "$SAM" && $PY_HY sam3d_to_scene.py \
        --sam3d_dir "outputs/${TS}_full" \
        --capture_dir "captures/$TS" \
        --output_dir "outputs/$TS" \
        --no_focus )
}

do_sim() {
    echo "== [4/4] robot twin sim for $TS (viewer on ${DISPLAY:-:1}) =="
    DISPLAY="${DISPLAY:-:1}" $PY_NEWTON "$TWIN" "$TS"
}

case "$stage" in
  capture)      do_capture ;;
  reconstruct)  : "${TS:?need TS}"; do_reconstruct ;;
  scene)        : "${TS:?need TS}"; do_scene ;;
  sim)          : "${TS:?need TS}"; do_sim ;;
  all)          do_capture; do_reconstruct; do_scene; do_sim ;;
  *) echo "unknown stage: $stage"; exit 1 ;;
esac
