#!/usr/bin/env bash
# End-to-end real-to-sim from the segbot robot, in ONE command:
#   capture (ROS1 on segbot) -> pull -> convert -> reconstruct (SAM 3D + QA
#   self-repair) -> LOAD INTO ISAAC SIM (GUI).
#
# Usage:
#   run_robot_pipeline.sh            # FULL: capture + reconstruct + load Isaac Sim
#   run_robot_pipeline.sh full       # same as default
#   run_robot_pipeline.sh all        # capture + reconstruct (no sim)
#   run_robot_pipeline.sh capture    # capture + pull + convert only
# Env overrides:
#   BACKEND=sam3d|hunyuan   MANIP_TARGETS="cup,bowl"   MAX_OBJECTS=4
#   DEMO=none|curobo_pick|pick   ROBOT=test@172.20.10.4
set -euo pipefail

ROBOT="${ROBOT:-test@172.20.10.4}"
# All workstation paths default under $HOME and are env-overridable. SAM_DIR is
# the sam-3d-objects checkout (holds the pipeline scripts + the SAM 3D package);
# captures/outputs are written there.
SAM_DIR="${SAM_DIR:-$HOME/sam-3d-objects}"
HUNYUAN_PY="${HUNYUAN_PY:-$HOME/miniforge3/envs/hunyuan3d/bin/python}"
SAM3D_PY="${SAM3D_PY:-$HOME/miniforge3/envs/sam3d-objects/bin/python}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"; export ISAACLAB_DIR
ROBOT_HOME="${ROBOT_HOME:-/home/test}"            # robot-side home (segbot user)
ROBOT_CATKIN="${ROBOT_CATKIN:-$ROBOT_HOME/catkin_ws}"
STAGE="${1:-full}"
TS="$(date +%Y%m%d_%H%M%S)"
ROBOT_DIR="$ROBOT_HOME/real2sim_captures/robot_${TS}"
DEST="${SAM_DIR}/captures/robot_${TS}"
OUT="${SAM_DIR}/outputs/robot_${TS}"

HOSTIP="${ROBOT##*@}"
echo "==> [0/5] Preflight: robot link + camera hand-eye TF"
# 0a. ensure the RAO NGAJI interface has an IP (re-DHCP if the link dropped)
if ! ip -4 addr show wlp5s0 2>/dev/null | grep -q "inet "; then
  echo "  [net] wlp5s0 has no IP — bringing up RAO NGAJI on it"
  nmcli con up "RAO NGAJI" ifname wlp5s0 >/dev/null 2>&1 || true
  sleep 3
fi
# 0b. robot reachable?
if ! ping -c1 -W3 "$HOSTIP" >/dev/null 2>&1; then
  echo "  [net] ERROR: robot $HOSTIP unreachable. Connect this machine to RAO NGAJI (wlp5s0)."
  exit 2
fi
# 0c. ensure hand-eye TF (camera<->arm) is published; force-launch if missing
HE=$(ssh -o BatchMode=yes "$ROBOT" "bash -lc '
  source /opt/ros/melodic/setup.bash 2>/dev/null; source ${ROBOT_CATKIN}/devel/setup.bash 2>/dev/null
  export ROS_MASTER_URI=http://localhost:11311
  rosrun tf tf_echo ur5e_base_link camera_color_optical_frame 2>&1 & P=\$!; sleep 3; kill \$P 2>/dev/null
'" 2>/dev/null | grep -c Translation || true)
if [ "${HE:-0}" -eq 0 ]; then
  echo "  [tf] hand-eye not published — launching realsense_handeye_publish"
  ssh -o BatchMode=yes "$ROBOT" "bash -lc '
    source /opt/ros/melodic/setup.bash 2>/dev/null; source ${ROBOT_CATKIN}/devel/setup.bash 2>/dev/null
    export ROS_MASTER_URI=http://localhost:11311
    setsid nohup roslaunch segbot_bringup realsense_handeye_publish.launch with_aruco:=false >${ROBOT_HOME}/handeye_publish.log 2>&1 </dev/null & sleep 10
  '" 2>/dev/null || true
else
  echo "  [tf] hand-eye TF OK"
fi

echo "==> [1/5] Capture on segbot ($ROBOT) into $ROBOT_DIR"
scp -o BatchMode=yes "${SAM_DIR}/capture_rgbd_ros1.py" "${ROBOT}:${ROBOT_HOME}/capture_rgbd_ros1.py"
ssh -o BatchMode=yes "$ROBOT" "bash -lc '
  source /opt/ros/melodic/setup.bash 2>/dev/null
  export ROS_MASTER_URI=http://localhost:11311
  python ${ROBOT_HOME}/capture_rgbd_ros1.py ${ROBOT_DIR} 12
'"

echo "==> [2/5] Pull capture to $DEST"
mkdir -p "$DEST"
scp -o BatchMode=yes -r "${ROBOT}:${ROBOT_DIR}/*" "$DEST/"

echo "==> [3/5] Convert color_raw.npy -> rgb.png (+ depth viz)"
"$HUNYUAN_PY" - "$DEST" <<'PYEOF'
import sys, os, json, numpy as np, cv2
d = sys.argv[1]
meta = json.load(open(os.path.join(d,"capture_meta.json")))
color = np.load(os.path.join(d,"color_raw.npy"))
bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) if meta.get("color_encoding")=="rgb8" else color
cv2.imwrite(os.path.join(d,"rgb.png"), bgr)
depth = np.load(os.path.join(d,"depth.npy")); viz=depth.copy(); viz[viz==0]=np.nan
norm = cv2.normalize(np.nan_to_num(viz,nan=0),None,0,255,cv2.NORM_MINMAX)
col = cv2.applyColorMap(norm.astype(np.uint8),cv2.COLORMAP_JET); col[depth==0]=0
cv2.imwrite(os.path.join(d,"depth_viz.png"), col)
print("converted:", bgr.shape)
PYEOF

echo "CAPTURE_DEST=$DEST"
echo "$DEST" > /tmp/last_capture_dest.txt
echo "$OUT" > /tmp/last_recon_out.txt

if [ "$STAGE" = "capture" ]; then
  echo "==> done (capture stage). RGB: $DEST/rgb.png  Depth: $DEST/depth_viz.png"
  exit 0
fi

# Object selection is GENERAL by default (generalist robot): the VLM decides
# which manipulable objects are present in ANY scene — no hardcoded categories.
# Optionally set MANIP_TARGETS="cup,bowl,..." for task-conditioned grasping of
# specific objects; empty/unset = general manipulable detection.
export MANIP_TARGETS="${MANIP_TARGETS:-}"
[ -n "$MANIP_TARGETS" ] && echo "==> VLM targeted to: $MANIP_TARGETS" || echo "==> VLM: general manipulable-object detection"

# Reconstruction backend: SAM 3D (default, best geometry) or Hunyuan3D-2.
BACKEND="${BACKEND:-sam3d}"
cd "$SAM_DIR"
if [ "$BACKEND" = "sam3d" ]; then
  echo "==> [4/5] Reconstruct (Gemini -> SAM2 -> SAM 3D, focus) -> $OUT"
  # SAM 3D peaks ~18-20GB (spconv/SLAT sparse decoding) — effectively needs the
  # card to itself; pause other GPU jobs first.
  RAW="${OUT}_sam3d_raw"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$SAM3D_PY" demo.py \
    --image_path "$DEST/rgb.png" --output_dir "$RAW" \
    --use_gemini --tag hf --max_objects "${MAX_OBJECTS:-6}"
  # adapt SAM 3D output -> metric, calibrated, real2sim-ready scene
  "$HUNYUAN_PY" sam3d_to_scene.py \
    --sam3d_dir "$RAW" --capture_dir "$DEST" --output_dir "$OUT" \
    --robot_base_frame "${ROBOT_BASE_FRAME:-ur5e_base_link}" \
    --max_objects "${MAX_OBJECTS:-4}"
else
  echo "==> [4/5] Reconstruct (Gemini -> SAM2 -> Hunyuan3D-2, focus + QA) -> $OUT"
  "$HUNYUAN_PY" hunyuan_demo.py \
    --image_path "$DEST/rgb.png" \
    --depth_path "$DEST/depth.npy" \
    --intrinsics_path "$DEST/intrinsics.json" \
    --extrinsics_path "$DEST/extrinsics.json" \
    --robot_base_frame "${ROBOT_BASE_FRAME:-ur5e_base_link}" \
    --max_objects "${MAX_OBJECTS:-4}" \
    --output_dir "$OUT" \
    --skip_texture
fi
echo "==> reconstruction done. scene_layout.json at $OUT/scene_layout.json"
echo "$OUT" > /tmp/last_recon_out.txt

if [ "$STAGE" != "full" ]; then
  echo "==> done ($STAGE). To load in Isaac Sim:"
  echo "  bash $SAM_DIR/load_in_isaaclab.sh $OUT $DEST --demo ${DEMO:-none}"
  exit 0
fi

echo "==> [5/5] Load into Isaac Sim (GUI) — scene=$OUT"
# Guarded GUI launch (refuses if VRAM too low instead of OOM-ing other jobs).
exec bash "$SAM_DIR/load_in_isaaclab.sh" "$OUT" "$DEST" --demo "${DEMO:-none}"
