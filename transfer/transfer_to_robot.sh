#!/usr/bin/env bash
# Transfer a sim-exported EE trajectory to the real UR5e and run it via MoveIt.
#
#   scp ee_trajectory.json + ur5e_ee_executor.py -> segbot, then run the executor.
#   DRY-RUN by default (MoveIt plans + RViz preview, NO motion).
#
# Usage:
#   transfer_to_robot.sh <ee_trajectory.json>                 # DRY-RUN (safe)
#   transfer_to_robot.sh <ee_trajectory.json> --execute       # MOVE (move-only)
#   transfer_to_robot.sh <ee_trajectory.json> --execute --with-gripper
# Env: ROBOT=test@172.20.10.4  ROBOT_HOME=/home/test  GROUP=manipulator  VEL=0.1
set -euo pipefail

TRAJ="${1:?usage: transfer_to_robot.sh <ee_trajectory.json> [--execute] [--with-gripper]}"
shift || true
EXTRA_ARGS="$*"                              # passed through to the executor (--execute etc.)
ROBOT="${ROBOT:-test@172.20.10.4}"
ROBOT_HOME="${ROBOT_HOME:-/home/test}"
ROBOT_CATKIN="${ROBOT_CATKIN:-$ROBOT_HOME/catkin_ws}"
GROUP="${GROUP:-manipulator}"
VEL="${VEL:-0.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTIP="${ROBOT##*@}"

[ -f "$TRAJ" ] || { echo "ERROR: trajectory not found: $TRAJ"; exit 2; }

echo "==> [preflight] robot link"
if ! ip -4 addr show wlp5s0 2>/dev/null | grep -q "inet "; then
  echo "  [net] wlp5s0 has no IP — bringing up RAO NGAJI"
  nmcli con up "RAO NGAJI" ifname wlp5s0 >/dev/null 2>&1 || true; sleep 3
fi
ping -c1 -W3 "$HOSTIP" >/dev/null 2>&1 || { echo "  [net] ERROR: $HOSTIP unreachable"; exit 2; }

echo "==> [copy] trajectory + executor -> $ROBOT"
scp -o BatchMode=yes "$TRAJ" "${ROBOT}:${ROBOT_HOME}/ee_trajectory.json"
scp -o BatchMode=yes "$HERE/ur5e_ee_executor.py" "${ROBOT}:${ROBOT_HOME}/ur5e_ee_executor.py"

case " $EXTRA_ARGS " in
  *" --execute "*) echo "==> [run] EXECUTE on real arm (vel=$VEL) — watch the robot!";;
  *)              echo "==> [run] DRY-RUN (plan + RViz preview, NO motion)";;
esac

# moveit_commander needs an unbuffered, login-sourced ROS env on the robot.
ssh -t -o BatchMode=yes "$ROBOT" "bash -lc '
  source /opt/ros/melodic/setup.bash 2>/dev/null
  source ${ROBOT_CATKIN}/devel/setup.bash 2>/dev/null
  export ROS_MASTER_URI=http://localhost:11311
  python -u ${ROBOT_HOME}/ur5e_ee_executor.py --traj ${ROBOT_HOME}/ee_trajectory.json \
    --group ${GROUP} --vel ${VEL} ${EXTRA_ARGS}
'"
