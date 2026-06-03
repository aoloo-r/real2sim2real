#!/usr/bin/env bash
# VRAM guard: refuse to launch a GPU job unless enough VRAM is free, so a new
# job can't OOM-crash other GPU users (e.g. an in-progress training run).
#
# Usage:   vram_guard.sh <required_mib> "<job description>"
# Exit:    0 = enough free (or GPU unreadable -> warn + allow)
#          3 = NOT enough free (caller should abort)
#
# Override: set VRAM_GUARD_FORCE=1 to bypass the check (logs a warning).
set -u
req="${1:?usage: vram_guard.sh <required_mib> <description>}"
desc="${2:-GPU job}"

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
if [ -z "${free:-}" ]; then
  echo "[VRAM-GUARD] WARNING: could not read GPU memory; allowing '$desc' to proceed."
  exit 0
fi

if [ "${VRAM_GUARD_FORCE:-0}" = "1" ]; then
  echo "[VRAM-GUARD] FORCE set — launching '$desc' with ${free} MiB free (needs ${req} MiB)."
  exit 0
fi

if [ "$free" -lt "$req" ]; then
  echo "[VRAM-GUARD] ❌ REFUSING to launch '$desc': needs ${req} MiB but only ${free} MiB free."
  echo "[VRAM-GUARD] Current GPU processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/    /' || echo "    (none reported)"
  echo "[VRAM-GUARD] Free VRAM (close a sim / pause a job) and retry, or set VRAM_GUARD_FORCE=1 to override."
  exit 3
fi

echo "[VRAM-GUARD] ✓ ${free} MiB free >= ${req} MiB needed for '$desc'."
exit 0
