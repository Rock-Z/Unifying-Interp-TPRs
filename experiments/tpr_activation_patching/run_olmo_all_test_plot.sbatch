#!/bin/bash
#SBATCH --job-name=olmo-patch-plot
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/olmo-patch-plot-%j.out
#SBATCH --error=logs/olmo-patch-plot-%j.err

set -euo pipefail

module reset

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_ROOT="results/olmo-2-1124-13b-all-layers-all-test"
ACTIVATION_OUT="${RESULT_ROOT}/activation_patching"
TPE_OUT="${RESULT_ROOT}/tpe_activation_patching"
PLOT_OUT="${RESULT_ROOT}/interleaved"

mkdir -p "${PLOT_OUT}" logs

echo "[$(date)] Validating completed patching summaries"
test -f "${ACTIVATION_OUT}/summary.json"
test -f "${TPE_OUT}/summary.json"

uv run --no-sync python - <<'PY' "${ACTIVATION_OUT}/summary.json" "${TPE_OUT}/summary.json"
import json
import sys
from pathlib import Path

for label, path in [("activation", Path(sys.argv[1])), ("tpe", Path(sys.argv[2]))]:
    summary = json.loads(path.read_text())
    print(
        f"{label}: num_trials={summary.get('num_trials')} "
        f"heatmap_shape={summary.get('heatmap_shape')} "
        f"positions={summary.get('token_positions')}"
    )
PY

echo "[$(date)] Generating interleaved matplotlib plot"
uv run --no-sync python experiments/tpe_activation_patching/interleaved_patch_plot_matplotlib.py \
  --activation-summary "${ACTIVATION_OUT}/summary.json" \
  --tpe-summary "${TPE_OUT}/summary.json" \
  --output-stem "${PLOT_OUT}/activation_vs_tpe_matplotlib" \
  --y-tick-step 5

echo "[$(date)] Plot generation finished"
