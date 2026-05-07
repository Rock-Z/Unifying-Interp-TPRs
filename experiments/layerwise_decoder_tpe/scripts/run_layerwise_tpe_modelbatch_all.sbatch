#!/bin/bash
#SBATCH --job-name=layerwise-tpe-modelbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --output=logs/layerwise_tpe/modelbatch-%x-%A_%a.out

set -euo pipefail
mkdir -p logs/layerwise_tpe

cd "$(git rev-parse --show-toplevel)"

export TQDM_DISABLE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Map array index to gin config and a readable name.
GINS=(
  "experiments/layerwise_decoder_tpe/configs/train_layerwise_tpe_qwen3_8b_alllayers_modelbatch.gin"
  # "experiments/layerwise_decoder_tpe/configs/train_layerwise_tpe_olmo13b_alllayers_modelbatch.gin"
  # "experiments/layerwise_decoder_tpe/configs/train_layerwise_tpe_gptoss20b_alllayers_modelbatch.gin"
)
NAMES=("qwen3-8b") # "olmo-13b" "gpt-oss-20b")

IDX="${SLURM_ARRAY_TASK_ID}"
GIN_PATH="${GINS[$IDX]}"
MODEL_NAME="${NAMES[$IDX]}"

echo "Running ModelBatch layerwise TPE for ${MODEL_NAME} using ${GIN_PATH}"

START_TIME=$(date +%s)
uv run experiments/layerwise_decoder_tpe/train_layerwise_tpe_modelbatch.py "${GIN_PATH}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "ModelBatch run for %s completed in %02d:%02d:%02d (hh:mm:ss)\n" \
  "${MODEL_NAME}" \
  $((ELAPSED/3600)) \
  $(((ELAPSED%3600)/60)) \
  $((ELAPSED%60))
