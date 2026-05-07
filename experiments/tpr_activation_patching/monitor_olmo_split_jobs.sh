#!/bin/bash
set -euo pipefail

ACT_JOB="${1:?activation job id required}"
TPE_JOB="${2:?tpe job id required}"
PLOT_JOB="${3:?plot job id required}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${4:-${DEFAULT_ROOT}}"
RESULT_ROOT="${ROOT}/results/olmo-2-1124-13b-all-layers-all-test"
MONITOR_LOG="${RESULT_ROOT}/logs/monitor-${ACT_JOB}-${TPE_JOB}-${PLOT_JOB}.log"

mkdir -p "${RESULT_ROOT}/logs"

while true; do
  {
    echo "===== $(date) ====="
    squeue -j "${ACT_JOB},${TPE_JOB},${PLOT_JOB}" -o '%i|%j|%T|%M|%l|%D|%R|%b|%S' || true
    for path in \
      "${ROOT}/logs/olmo-act-full-${ACT_JOB}.out" \
      "${ROOT}/logs/olmo-act-full-${ACT_JOB}.err" \
      "${ROOT}/logs/olmo-tpe-full-${TPE_JOB}.out" \
      "${ROOT}/logs/olmo-tpe-full-${TPE_JOB}.err" \
      "${ROOT}/logs/olmo-patch-plot-${PLOT_JOB}.out" \
      "${ROOT}/logs/olmo-patch-plot-${PLOT_JOB}.err" \
      "${RESULT_ROOT}/logs/activation_patching.log" \
      "${RESULT_ROOT}/logs/tpe_activation_patching.log" \
      "${RESULT_ROOT}/activation_patching/summary.json" \
      "${RESULT_ROOT}/tpe_activation_patching/summary.json" \
      "${RESULT_ROOT}/interleaved/activation_vs_tpe_matplotlib.png" \
      "${RESULT_ROOT}/interleaved/activation_vs_tpe_matplotlib.pdf"; do
      if [[ -e "${path}" ]]; then
        stat -c '%y %s %n' "${path}"
      else
        echo "missing ${path}"
      fi
    done
  } >> "${MONITOR_LOG}" 2>&1

  if ! squeue -j "${ACT_JOB},${TPE_JOB},${PLOT_JOB}" -h >/dev/null 2>&1; then
    break
  fi
  sleep 300
done
