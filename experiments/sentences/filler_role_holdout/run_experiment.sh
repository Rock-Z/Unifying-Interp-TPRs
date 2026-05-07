#!/usr/bin/env bash
set -euo pipefail

FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

ROOT_DIR="experiments/sentences/filler_role_holdout"
CONFIG_DIR="${ROOT_DIR}/configs"
DATA_DIR="data/sentences_multiple_filler_role_holdout"
DATA_PREFIX="${DATA_DIR}/data"

ensure_dataset() {
  if [[ "${FORCE_DATASET}" -eq 1 || ! -f "${DATA_PREFIX}.train" || ! -f "${DATA_PREFIX}.valid" || ! -f "${DATA_PREFIX}.test" || ! -f "${DATA_PREFIX}.generalization" || ! -f "${DATA_PREFIX}.holdout_metadata.json" ]]; then
    uv run src/generate_sentences.py \
      --verb_set multiple_verbs \
      --prefix "${DATA_PREFIX}" \
      --holdout_step 11 \
      --subject_holdout_offset 0 \
      --object_holdout_offset 5 \
      --seed 0
  fi
}

run_experiment() {
  local model="$1"
  local train_config="${CONFIG_DIR}/${model}.gin"
  local eval_config="${CONFIG_DIR}/evaluate_${model}.gin"
  local checkpoint_dir
  local force_probe_retrain="False"

  case "${model}" in
    modernbert) checkpoint_dir="${ROOT_DIR}/checkpoints/modernbert/tpe" ;;
    embeddinggemma) checkpoint_dir="${ROOT_DIR}/checkpoints/embeddinggemma/tpe" ;;
    qwen3_8b_embedding) checkpoint_dir="${ROOT_DIR}/checkpoints/qwen3-8B/tpe" ;;
    *) echo "Unknown model: ${model}" >&2; return 1 ;;
  esac

  if [[ "${FORCE_RETRAIN}" -eq 1 || ! -f "${checkpoint_dir}/best_model/config.json" ]]; then
    uv run src/train_sentences.py "${train_config}" \
      --tpe/TrainingArguments.overwrite_output_dir=True
  fi

  if [[ "${FORCE_EVAL}" -eq 1 || "${FORCE_RETRAIN}" -eq 1 ]]; then
    force_probe_retrain="True"
  fi

  if [[ "${FORCE_EVAL}" -eq 1 || "${FORCE_RETRAIN}" -eq 1 || ! -f "${checkpoint_dir}/holdout_eval_metrics.json" || ! -f "${checkpoint_dir}/probe_compare_results_svo.json" ]]; then
    uv run experiments/sentences/filler_role_holdout/evaluate_holdout.py "${eval_config}" \
      --holdout_eval.force_trainable_probe_retrain="${force_probe_retrain}"
  fi
}

ensure_dataset

run_experiment "modernbert"
run_experiment "embeddinggemma"
run_experiment "qwen3_8b_embedding"
