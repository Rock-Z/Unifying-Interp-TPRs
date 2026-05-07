#!/usr/bin/env bash
set -euo pipefail

FORCE_ANALOGY="${FORCE_ANALOGY:-0}"

ROOT_DIR="experiments/sentences/filler_role_holdout/analogy"
CONFIG_DIR="${ROOT_DIR}/configs"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

run_eval() {
  local name="$1"
  local config_path="${CONFIG_DIR}/${name}_tpe_generalization.gin"
  local output_path="${RESULTS_DIR}/${name}_tpe_generalization.json"

  if [[ "${FORCE_ANALOGY}" -eq 1 || ! -f "${output_path}" ]]; then
    uv run experiments/sentences/filler_role_holdout/analogy/evaluate_holdout_analogies.py "${config_path}"
  fi
}

run_eval "modernbert"
run_eval "embeddinggemma"
run_eval "qwen3_8b"
