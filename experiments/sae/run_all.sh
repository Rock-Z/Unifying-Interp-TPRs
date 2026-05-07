#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${ROOT_DIR}"

CONFIG_DIR="experiments/sae/configs"

runs=(
  "tpr_sae_modernbert.gin:experiments/sae/results/modernbert"
  "tpr_sae_embeddinggemma.gin:experiments/sae/results/embeddinggemma"
  "tpr_sae_qwen3_8b.gin:experiments/sae/results/qwen3_8b"
)

for run in "${runs[@]}"; do
  IFS=":" read -r config output_dir <<<"${run}"
  mkdir -p "${output_dir}"
  uv run src/tpr_sae.py "${CONFIG_DIR}/${config}"
done
