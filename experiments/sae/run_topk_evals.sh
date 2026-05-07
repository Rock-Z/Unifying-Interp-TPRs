#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${ROOT_DIR}"

DATA_PATH="data/sentences_multiple"
CACHE_PATH="data/sentences_multiple"

declare -a runs=(
  "modernbert:experiments/sae_sentences_topk/checkpoints/modernbert-embed-base/run-w4gfyb30:nomic-ai/modernbert-embed-base"
  "embeddinggemma:experiments/sae_sentences_topk/checkpoints/embeddinggemma-300m/run-utxy0zcc:google/embeddinggemma-300m"
  "qwen3_8b:experiments/sae_sentences_topk/checkpoints/qwen3-embedding-8b/run-kbx35i08:Qwen/Qwen3-Embedding-8B"
)

for run in "${runs[@]}"; do
  IFS=":" read -r name checkpoint embedding_model <<<"${run}"
  output_dir="experiments/sae/results/topk_${name}"
  mkdir -p "${output_dir}"
  uv run scripts/sae/eval.py \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --metrics-dir "${output_dir}" \
    --dataset-type sentences \
    --data-path "${DATA_PATH}" \
    --embedding-model-name "${embedding_model}" \
    --embedding-cache-path "${CACHE_PATH}" \
    --role-scheme svo \
    --label-mode filler_role &
done

wait
