#!/bin/bash
#SBATCH --job-name=llm-punct-probe-seed
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-8%3
#SBATCH --output=experiments/llm_sentences_last_layer/logs/probe_seed_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/probe_seed_%A_%a.err

set -euo pipefail

module reset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/probe_seed_checks/manifest.tsv}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r RUN_ID MODEL_SLUG CONFIG_PATH SEED RESULTS_DIR CHECKPOINT_DIR < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${RUN_ID:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p "experiments/llm_sentences_last_layer/logs" "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "[$(date)] Starting ${RUN_ID} on ${SLURM_JOB_NODELIST:-unknown}"
echo "model=${MODEL_SLUG} seed=${SEED}"

uv run python src/invert_svo.py "${CONFIG_PATH}" \
  --main.results_dir="${RESULTS_DIR}" \
  --main.cache_trained_results_path="${RESULTS_DIR}/trained_probe_results_svo.json" \
  --analytic/TrainingArguments.output_dir="${CHECKPOINT_DIR}/analytic" \
  --trainable/TrainingArguments.output_dir="${CHECKPOINT_DIR}/trainable" \
  --trainable/TrainingArguments.seed="${SEED}" \
  --trainable/TrainingArguments.data_seed="${SEED}"

echo "[$(date)] Finished ${RUN_ID}"
