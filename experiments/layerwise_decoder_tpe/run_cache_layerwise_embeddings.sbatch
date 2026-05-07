#!/bin/bash
#SBATCH --job-name=cache-layerwise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/layerwise_cache/%x-%j.out

set -euo pipefail
mkdir -p logs/layerwise_cache


cd "$(git rev-parse --show-toplevel)"

export TQDM_DISABLE=1

MODELS=(
  "Qwen/Qwen3-8B"
  "openai/gpt-oss-20b"
  "allenai/OLMo-2-1124-13B"
)

for model in "${MODELS[@]}"; do
  echo "[INFO] Caching per-layer embeddings for ${model}"
  uv run python experiments/layerwise_decoder_tpe/cache_layerwise_embeddings.py \
    --sentences-path data/sentences \
    --embedding-model-name "${model}" \
    --embedding-cache-path data/sentences \
    --device cuda \
    "$@"
done
