#!/bin/bash
# Role-dim sweep runner for fixed-len-6 L2R-aligned digits (no config cloning).
# Usage:
#   ROLE_DIM=16 ROLE_SUFFIX=role16 MODE=train sbatch --array=0-5 <this file>
#   ROLE_DIM=16 ROLE_SUFFIX=role16 MODE=probe sbatch --array=0-5 --dependency=afterok:<train_job_id> <this file>
#   ROLE_DIM=16 ROLE_SUFFIX=role16 MODE=plot sbatch --array=0 --dependency=afterok:<probe_job_id> <this file>
# Writes checkpoints under experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}
# and probe results/plots under experiments/digits_probe/results/1le64h256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}.
#SBATCH --job-name=digits-1l-64-256-len6-l2r-role
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/digits/one_layer_64_256_fixed_len_6_l2r_aligned_role_dim/%x-%A_%a.out

set -euo pipefail

export TQDM_DISABLE=1
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_PROBES="${FORCE_PROBES:-0}"
ROLE_DIM="${ROLE_DIM:?ROLE_DIM is required (e.g., 16 or 64)}"
ROLE_SUFFIX="${ROLE_SUFFIX:-role${ROLE_DIM}}"
MODE="${MODE:-train}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

mkdir -p logs/digits/one_layer_64_256_fixed_len_6_l2r_aligned_role_dim

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

BASE_DIR="experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}"
CONFIG_DIR="experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned/configs"
PROBE_CONFIG_DIR="experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_regularized"
PROBE_RESULTS_DIR="experiments/digits_probe/results/1le64h256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}"
ANALYTIC_BASE_DIR="experiments/digits_probe/checkpoints/analytic/1le64h256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}"
TRAINABLE_BASE_DIR="experiments/digits_probe/checkpoints/trainable_probe/1le64h256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}"
TRAINABLE_LOG_BASE_DIR="experiments/digits_probe/logs/trainable_probe/1le64h256_fixed_len_6_l2r_aligned_${ROLE_SUFFIX}"
SHARED_SEQ2SEQ_DIR="experiments/digits/one_layer_64_256_fixed_len_6_l2r_aligned/checkpoints/seq2seq"

TRAIN_TASKS=(copy copy copy reverse reverse reverse)
TRAIN_ARCHS=(rnn gru lstm rnn gru lstm)
PROBE_TASKS=(copy copy copy reverse reverse reverse)
PROBE_ARCHS=(rnn gru lstm rnn gru lstm)

TRAIN_COUNT=${#TRAIN_TASKS[@]}
PROBE_COUNT=${#PROBE_TASKS[@]}

if [[ "${MODE}" == "train" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TRAIN_COUNT}" ]]; then
    echo "Training task id ${SLURM_ARRAY_TASK_ID} out of range (0..$((TRAIN_COUNT - 1)))."
    exit 1
  fi

  TASK="${TRAIN_TASKS[$SLURM_ARRAY_TASK_ID]}"
  ARCH="${TRAIN_ARCHS[$SLURM_ARRAY_TASK_ID]}"
  CONFIG="${CONFIG_DIR}/${TASK}_${ARCH}.gin"
  SEQ2SEQ_SHARED="${SHARED_SEQ2SEQ_DIR}/${TASK}/${ARCH}"
  SEQ2SEQ_DIR="${BASE_DIR}/checkpoints/seq2seq/${TASK}/${ARCH}"
  SEQ_EVAL_SHARED="${SEQ2SEQ_SHARED}/eval_results.json"
  SEQ_EVAL_NEW="${SEQ2SEQ_DIR}/eval_results.json"
  TPE_DIR="${BASE_DIR}/checkpoints/tpe/${TASK}/${ARCH}"
  TPE_SUB="${TPE_DIR}/substitution_eval_results.json"

  if [[ "${FORCE_RETRAIN}" -ne 1 && -f "${TPE_SUB}" ]]; then
    exit 0
  fi

  SKIP_SEQ2SEQ=0
  SEQ2SEQ_PATH="${SEQ2SEQ_DIR}"

  if [[ "${FORCE_RETRAIN}" -ne 1 ]]; then
    if [[ -f "${SEQ_EVAL_NEW}" ]]; then
      SKIP_SEQ2SEQ=1
    elif [[ -f "${SEQ_EVAL_SHARED}" ]]; then
      SEQ2SEQ_PATH="${SEQ2SEQ_SHARED}"
      SKIP_SEQ2SEQ=1
    fi
  fi

  case "${ARCH}" in
    lstm)
      TPE_HIDDEN=512
      ;;
    rnn|gru)
      TPE_HIDDEN=256
      ;;
    *)
      echo "Unknown arch: ${ARCH}"
      exit 1
      ;;
  esac

  TEMP_CONFIG="$(mktemp)"
  cat > "${TEMP_CONFIG}" <<EOF
include '${CONFIG}'
main.tpe_config = {
    'hidden_size': ${TPE_HIDDEN},
    'n_roles': 9,
    'filler_dim': 64,
    'role_dim': ${ROLE_DIM},
    'role_scheme': 'l2r',
}
EOF
  trap 'rm -f "${TEMP_CONFIG}"' EXIT

  COMMON_ARGS=(
    "${TEMP_CONFIG}"
    "--tpe/TrainingArguments.output_dir=${TPE_DIR}"
    "--tpe/TrainingArguments.overwrite_output_dir=True"
    "--tpe_eval/Seq2SeqTrainingArguments.output_dir=${TPE_DIR}"
    "--seq2seq/Seq2SeqTrainingArguments.output_dir=${SEQ2SEQ_PATH}"
  )

  if [[ "${SKIP_SEQ2SEQ}" -eq 1 ]]; then
    PYTHONPATH=src uv run --isolated --with transformers==4.51.1 \
      src/train.py "${COMMON_ARGS[@]}" \
      --main.skip_seq2seq=True
  else
    PYTHONPATH=src uv run --isolated --with transformers==4.51.1 \
      src/train.py "${COMMON_ARGS[@]}" \
      --seq2seq/Seq2SeqTrainingArguments.overwrite_output_dir=True
  fi
elif [[ "${MODE}" == "probe" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${PROBE_COUNT}" ]]; then
    echo "Probe task id ${SLURM_ARRAY_TASK_ID} out of range (0..$((PROBE_COUNT - 1)))."
    exit 1
  fi

  INDEX="${SLURM_ARRAY_TASK_ID}"
  TASK="${PROBE_TASKS[$INDEX]}"
  ARCH="${PROBE_ARCHS[$INDEX]}"
  CONFIG="${PROBE_CONFIG_DIR}/${TASK}_${ARCH}_probe_all_pos.gin"
  RESULT="${PROBE_RESULTS_DIR}/${TASK}_${ARCH}/probe_compare_results.json"
  SEQ2SEQ_SHARED="${SHARED_SEQ2SEQ_DIR}/${TASK}/${ARCH}"
  SEQ2SEQ_DIR="${BASE_DIR}/checkpoints/seq2seq/${TASK}/${ARCH}"
  SEQ2SEQ_PATH="${SEQ2SEQ_SHARED}"
  TPE_DIR="${BASE_DIR}/checkpoints/tpe/${TASK}/${ARCH}"
  ANALYTIC_DIR="${ANALYTIC_BASE_DIR}/${TASK}_${ARCH}"
  TRAINABLE_DIR="${TRAINABLE_BASE_DIR}/${TASK}_${ARCH}"
  TRAINABLE_LOG_DIR="${TRAINABLE_LOG_BASE_DIR}/${TASK}_${ARCH}"

  if [[ -f "${SEQ2SEQ_DIR}/eval_results.json" ]]; then
    SEQ2SEQ_PATH="${SEQ2SEQ_DIR}"
  fi

  if [[ "${FORCE_PROBES}" -ne 1 && -f "${RESULT}" ]]; then
    exit 0
  fi

  while [[ ! -d "${TPE_DIR}/best_model" ]]; do
    sleep 30
  done

  uv run src/invert_tpr.py "${CONFIG}" \
    --main.seq2seq_path="${SEQ2SEQ_PATH}" \
    --main.tpe_path="${TPE_DIR}" \
    --main.results_dir="${PROBE_RESULTS_DIR}/${TASK}_${ARCH}" \
    --analytic/TrainingArguments.output_dir="${ANALYTIC_DIR}" \
    --trainable/TrainingArguments.output_dir="${TRAINABLE_DIR}" \
    --trainable/TrainingArguments.logging_dir="${TRAINABLE_LOG_DIR}"
elif [[ "${MODE}" == "plot" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]]; then
    exit 0
  fi

  uv run experiments/digits_probe/plot_probe_grid.py \
    --results_dir "${PROBE_RESULTS_DIR}"
else
  echo "Unknown MODE=${MODE} (expected train|probe|plot)."
  exit 1
fi
