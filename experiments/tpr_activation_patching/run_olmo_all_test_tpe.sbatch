#!/bin/bash
#SBATCH --job-name=olmo-tpe-full
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=512G
#SBATCH --constraint="a100|h100|h200"
#SBATCH --output=logs/olmo-tpe-full-%j.out
#SBATCH --error=logs/olmo-tpe-full-%j.err

set -euo pipefail

module reset

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

MODEL_ID="allenai/OLMo-2-1124-13B"
RESULT_ROOT="results/olmo-2-1124-13b-all-layers-all-test"
TPE_OUT="${RESULT_ROOT}/tpe_activation_patching"

mkdir -p "${TPE_OUT}" "${RESULT_ROOT}/logs" logs

echo "[$(date)] TPE patching started on $(hostname)"
echo "[$(date)] Output: ${TPE_OUT}"

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
  --tpe_activation_patching.run_tpe_activation_patching.tpe_device="'cuda'" \
  2>&1 | tee "${RESULT_ROOT}/logs/tpe_activation_patching.log"

echo "[$(date)] TPE patching finished"
