#!/usr/bin/env bash
set -euo pipefail

export TQDM_DISABLE=1
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_PROBES="${FORCE_PROBES:-0}"

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

run_training() {
  local task="$1"
  local arch="$2"
  local config="${CONFIG_DIR}/${task}_${arch}.gin"
  local seq_eval="${BASE_DIR}/checkpoints/seq2seq/${task}/${arch}/eval_results.json"
  local tpe_sub="${BASE_DIR}/checkpoints/tpe/${task}/${arch}/substitution_eval_results.json"

  if [[ "${FORCE_RETRAIN}" -ne 1 && -f "${tpe_sub}" ]]; then
    return
  fi

  if [[ "${FORCE_RETRAIN}" -ne 1 && -f "${seq_eval}" ]]; then
    uv run src/train.py "${config}" \
      --main.skip_seq2seq=True \
      --tpe/TrainingArguments.overwrite_output_dir=True
  else
    uv run src/train.py "${config}" \
      --seq2seq/Seq2SeqTrainingArguments.overwrite_output_dir=True \
      --tpe/TrainingArguments.overwrite_output_dir=True
  fi
}

for task in copy reverse; do
  for arch in rnn gru lstm; do
    run_training "${task}" "${arch}"
  done
done

PROBE_CONFIG_DIR="experiments/digits_probe/configs/1le64h256_fixed_len_6_l2r_aligned"
run_probe() {
  local task="$1"
  local arch="$2"
  local config="${PROBE_CONFIG_DIR}/${task}_${arch}_probe_all_pos.gin"
  local result="experiments/digits_probe/results/1le64h256_fixed_len_6_l2r_aligned/${task}_${arch}/probe_compare_results.json"

  if [[ "${FORCE_PROBES}" -ne 1 && -f "${result}" ]]; then
    return
  fi

  uv run src/invert_tpr.py "${config}"
}

for task in copy reverse; do
  for arch in rnn gru lstm; do
    run_probe "${task}" "${arch}"
  done
done

uv run experiments/digits_probe/plot_probe_grid.py \
  --results_dir "experiments/digits_probe/results/1le64h256_fixed_len_6_l2r_aligned"
