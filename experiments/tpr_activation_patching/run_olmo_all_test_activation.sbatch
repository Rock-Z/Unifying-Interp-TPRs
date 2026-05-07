#!/bin/bash
#SBATCH --job-name=olmo-act-full
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=512G
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=logs/olmo-act-full-%j.out
#SBATCH --error=logs/olmo-act-full-%j.err

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

mkdir -p "${ACTIVATION_OUT}" "${RESULT_ROOT}/logs" logs

echo "[$(date)] Activation patching started on $(hostname)"
echo "[$(date)] Output: ${ACTIVATION_OUT}"

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
  --activation_patching.run_activation_patching.device="'cuda'" \
  2>&1 | tee "${RESULT_ROOT}/logs/activation_patching.log"

echo "[$(date)] Activation patching finished"
