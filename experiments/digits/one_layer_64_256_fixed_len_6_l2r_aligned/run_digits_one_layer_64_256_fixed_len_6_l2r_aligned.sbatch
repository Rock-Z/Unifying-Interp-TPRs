#!/bin/bash
#SBATCH --job-name=digits-1l-64-256-len6-l2r
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-12
#SBATCH --output=logs/digits/one_layer_64_256_fixed_len_6_l2r_regularized/%x-%A_%a.out

set -euo pipefail

export TQDM_DISABLE=1
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_PROBES="${FORCE_PROBES:-0}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

mkdir -p logs/digits/one_layer_64_256_fixed_len_6_l2r_regularized

ensure_dataset() {
  local task="$1"
  local train_path="data/digits/${task}_vocab_20_length_6_fixed.train"
  local valid_path="data/digits/${task}_vocab_20_length_6_fixed.valid"
  local test_path="data/digits/${task}_vocab_20_length_6_fixed.test"

  if [[ ! -f "${train_path}" || ! -f "${valid_path}" || ! -f "${test_path}" ]]; then
    uv run src/generate_digits.py "configs/digits_fixed_len_6/${task}.gin"
  fi
}

ensure_dataset "copy"
ensure_dataset "reverse"

BASE_DIR="experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned"
CONFIG_DIR="${BASE_DIR}/configs"
PROBE_CONFIG_DIR="experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized"
PROBE_RESULTS_DIR="experiments/digits_probe/results/1le64h256_fixed_len_6_l2r_regularized"

TRAIN_TASKS=(copy copy copy reverse reverse reverse)
TRAIN_ARCHS=(rnn gru lstm rnn gru lstm)
PROBE_TASKS=(copy copy copy reverse reverse reverse)
PROBE_ARCHS=(rnn gru lstm rnn gru lstm)

TRAIN_COUNT=${#TRAIN_TASKS[@]}
PROBE_COUNT=${#PROBE_TASKS[@]}
PLOT_TASK_ID=$((TRAIN_COUNT + PROBE_COUNT))

if [[ "${SLURM_ARRAY_TASK_ID}" -lt "${TRAIN_COUNT}" ]]; then
  TASK="${TRAIN_TASKS[$SLURM_ARRAY_TASK_ID]}"
  ARCH="${TRAIN_ARCHS[$SLURM_ARRAY_TASK_ID]}"
  CONFIG="${CONFIG_DIR}/${TASK}_${ARCH}.gin"
  SEQ_EVAL="${BASE_DIR}/checkpoints/seq2seq/${TASK}/${ARCH}/eval_results.json"
  TPE_SUB="${BASE_DIR}/checkpoints/tpe/${TASK}/${ARCH}/substitution_eval_results.json"

  if [[ "${FORCE_RETRAIN}" -ne 1 && -f "${TPE_SUB}" ]]; then
    exit 0
  fi

  if [[ "${FORCE_RETRAIN}" -ne 1 && -f "${SEQ_EVAL}" ]]; then
    PYTHONPATH=src uv run --isolated --with transformers==4.51.1 \
      src/train.py "${CONFIG}" \
      --main.skip_seq2seq=True \
      --tpe/TrainingArguments.overwrite_output_dir=True
  else
    PYTHONPATH=src uv run --isolated --with transformers==4.51.1 \
      src/train.py "${CONFIG}" \
      --seq2seq/Seq2SeqTrainingArguments.overwrite_output_dir=True \
      --tpe/TrainingArguments.overwrite_output_dir=True
  fi
elif [[ "${SLURM_ARRAY_TASK_ID}" -lt "${PLOT_TASK_ID}" ]]; then
  INDEX=$((SLURM_ARRAY_TASK_ID - TRAIN_COUNT))
  TASK="${PROBE_TASKS[$INDEX]}"
  ARCH="${PROBE_ARCHS[$INDEX]}"
  CONFIG="${PROBE_CONFIG_DIR}/${TASK}_${ARCH}_probe_all_pos.gin"
  RESULT="${PROBE_RESULTS_DIR}/${TASK}_${ARCH}/probe_compare_results.json"

  if [[ "${FORCE_PROBES}" -ne 1 && -f "${RESULT}" ]]; then
    exit 0
  fi

  uv run src/invert_tpr.py "${CONFIG}"
else
  EXPECTED_RESULTS=(
    "${PROBE_RESULTS_DIR}/copy_rnn/probe_compare_results.json"
    "${PROBE_RESULTS_DIR}/copy_gru/probe_compare_results.json"
    "${PROBE_RESULTS_DIR}/copy_lstm/probe_compare_results.json"
    "${PROBE_RESULTS_DIR}/reverse_rnn/probe_compare_results.json"
    "${PROBE_RESULTS_DIR}/reverse_gru/probe_compare_results.json"
    "${PROBE_RESULTS_DIR}/reverse_lstm/probe_compare_results.json"
  )

  while :; do
    MISSING=0
    for RESULT in "${EXPECTED_RESULTS[@]}"; do
      if [[ ! -f "${RESULT}" ]]; then
        MISSING=1
        break
      fi
    done

    if [[ "${MISSING}" -eq 0 ]]; then
      break
    fi

    sleep 30
  done

  uv run experiments/digits_probe/plot_probe_grid.py \
    --results_dir "${PROBE_RESULTS_DIR}"
fi
