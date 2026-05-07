#!/bin/bash
#SBATCH --job-name=olmo-full-patch
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=512G
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=logs/olmo-full-patch-%j.out
#SBATCH --error=logs/olmo-full-patch-%j.err

set -euo pipefail

module reset

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

MODEL_ID="allenai/OLMo-2-1124-13B"
RESULT_ROOT="results/olmo-2-1124-13b-all-layers-all-test"
ACTIVATION_OUT="${RESULT_ROOT}/activation_patching"
TPE_OUT="${RESULT_ROOT}/tpe_activation_patching"
PLOT_OUT="${RESULT_ROOT}/interleaved"
LOG_OUT="${RESULT_ROOT}/logs"

mkdir -p "${ACTIVATION_OUT}" "${TPE_OUT}" "${PLOT_OUT}" "${LOG_OUT}" logs

echo "[$(date)] Starting full-test OLMo activation/TPE patching flow"
echo "[$(date)] Result root: ${RESULT_ROOT}"
echo "[$(date)] Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "[$(date)] Node list: ${SLURM_NODELIST:-unknown}"

run_activation() {
  echo "[$(date)] Activation patching started on $(hostname)"
  uv run --no-sync python experiments/activation_patching/activation_patching.py \
    experiments/activation_patching/configs/activation_patching_olmo13b_all_tokens.gin \
    --activation_patching.run_activation_patching.output_dir="'${ACTIVATION_OUT}'" \
    --activation_patching.run_activation_patching.model_id="'${MODEL_ID}'" \
    --activation_patching.run_activation_patching.max_trials=None \
    --activation_patching.run_activation_patching.token_focus="'all'" \
    --activation_patching.run_activation_patching.layer_indices=None \
    --activation_patching.run_activation_patching.explicit_token_positions=None \
    --activation_patching.run_activation_patching.random_seed=7 \
    --activation_patching.run_activation_patching.dtype="'bfloat16'" \
    --activation_patching.run_activation_patching.device="'cuda'"
  echo "[$(date)] Activation patching finished"
}

run_tpe() {
  echo "[$(date)] TPE patching started on $(hostname)"
  uv run --no-sync python experiments/tpe_activation_patching/tpe_activation_patching.py \
    experiments/tpe_activation_patching/configs/tpe_activation_patching_olmo13b_all_layers.gin \
    --tpe_activation_patching.run_tpe_activation_patching.output_dir="'${TPE_OUT}'" \
    --tpe_activation_patching.run_tpe_activation_patching.model_id="'${MODEL_ID}'" \
    --tpe_activation_patching.run_tpe_activation_patching.tpe_checkpoint_path="'experiments/layerwise_decoder_tpe/checkpoints/olmo-2-1124-13b_layers_all_modelbatch'" \
    --tpe_activation_patching.run_tpe_activation_patching.max_trials=None \
    --tpe_activation_patching.run_tpe_activation_patching.token_focus="'all'" \
    --tpe_activation_patching.run_tpe_activation_patching.layer_indices="'all'" \
    --tpe_activation_patching.run_tpe_activation_patching.explicit_token_positions=None \
    --tpe_activation_patching.run_tpe_activation_patching.random_seed=7 \
    --tpe_activation_patching.run_tpe_activation_patching.dtype="'bfloat16'" \
    --tpe_activation_patching.run_tpe_activation_patching.device="'cuda'" \
    --tpe_activation_patching.run_tpe_activation_patching.tpe_dtype="'float32'" \
    --tpe_activation_patching.run_tpe_activation_patching.tpe_device="'cuda'"
  echo "[$(date)] TPE patching finished"
}

export -f run_activation
export -f run_tpe
export ROOT MODEL_ID RESULT_ROOT ACTIVATION_OUT TPE_OUT PLOT_OUT LOG_OUT UV_CACHE_DIR PYTHONPATH TOKENIZERS_PARALLELISM

echo "[$(date)] Launching activation and TPE patching in parallel"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=4 \
  bash -lc 'run_activation' > "${LOG_OUT}/activation_patching.log" 2>&1 &
ACT_PID=$!

srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=4 \
  bash -lc 'run_tpe' > "${LOG_OUT}/tpe_activation_patching.log" 2>&1 &
TPE_PID=$!

FAIL=0
wait "${ACT_PID}" || FAIL=1
wait "${TPE_PID}" || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[$(date)] One or both patching jobs failed; skipping plot generation"
  exit 1
fi

echo "[$(date)] Both patching jobs finished; validating summaries"
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

echo "[$(date)] Completed full-test OLMo activation/TPE patching flow"
