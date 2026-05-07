#!/bin/bash
#SBATCH --job-name=layerwise-tpe-qwen3
#SBATCH --partition=day
#SBATCH --array=0-35
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/layerwise_tpe/%x-%A_%a.out

set -euo pipefail
mkdir -p logs/layerwise_tpe

cd "$(git rev-parse --show-toplevel)"

export TQDM_DISABLE=1
LAYER_INDEX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_ROOT="checkpoints/layerwise_tpe/qwen3-8b"

uv run python experiments/layerwise_decoder_tpe/train_layerwise_tpe.py \
  experiments/layerwise_decoder_tpe/configs/layerwise_tpe_base.gin \
  --main.layer_indices="[${LAYER_INDEX}]" \
  --tpe/TrainingArguments.output_dir=\"${OUTPUT_ROOT}/layer${LAYER_INDEX}\" \
  --tpe/TrainingArguments.per_device_train_batch_size=256 \
  "$@"
