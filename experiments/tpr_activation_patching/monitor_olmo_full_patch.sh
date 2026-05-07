#!/bin/bash
set -euo pipefail

JOB_ID="${1:-1919753}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${2:-${DEFAULT_ROOT}}"
RESULT_ROOT="${ROOT}/results/olmo-2-1124-13b-all-layers-all-test"
MONITOR_LOG="${RESULT_ROOT}/logs/monitor-${JOB_ID}.log"

mkdir -p "${RESULT_ROOT}/logs"

while true; do
  {
    echo "===== $(date) ====="
    squeue -j "${JOB_ID}" -o '%i|%T|%M|%l|%D|%R|%b|%S' || true
    for path in \
      "${ROOT}/logs/olmo-full-patch-${JOB_ID}.out" \
      "${ROOT}/logs/olmo-full-patch-${JOB_ID}.err" \
      "${RESULT_ROOT}/logs/activation_patching.log" \
      "${RESULT_ROOT}/logs/tpe_activation_patching.log" \
      "${RESULT_ROOT}/activation_patching/summary.json" \
      "${RESULT_ROOT}/tpe_activation_patching/summary.json"; do
      if [[ -e "${path}" ]]; then
        stat -c '%y %s %n' "${path}"
      else
        echo "missing ${path}"
      fi
    done
  } >> "${MONITOR_LOG}" 2>&1

  if ! squeue -j "${JOB_ID}" -h >/dev/null 2>&1; then
    break
  fi
  sleep 300
done
