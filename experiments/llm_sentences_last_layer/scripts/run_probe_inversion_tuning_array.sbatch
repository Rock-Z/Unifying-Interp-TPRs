#!/bin/bash
#SBATCH --job-name=llm-probe-tune
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-2%3
#SBATCH --output=experiments/llm_sentences_last_layer/logs/probe_tune_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/probe_tune_%A_%a.err

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

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/probe_inversion_tuning/manifest.tsv}"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r MODEL_SLUG MODEL_NAME TPE_PATH OUTPUT_PATH < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${MODEL_SLUG:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

if [[ ! -d "${TPE_PATH}/best_model" ]]; then
  echo "Missing TPE checkpoint: ${TPE_PATH}/best_model" >&2
  exit 3
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")" experiments/llm_sentences_last_layer/logs

echo "[$(date)] Starting probe inversion tuning for ${MODEL_SLUG} on ${SLURM_JOB_NODELIST:-unknown}"

uv run --no-sync python experiments/llm_sentences_last_layer/scripts/evaluate_probe_inversion_grid.py \
  --model-slug "${MODEL_SLUG}" \
  --model-name "${MODEL_NAME}" \
  --tpe-path "${TPE_PATH}" \
  --output-path "${OUTPUT_PATH}"

echo "[$(date)] Finished probe inversion tuning for ${MODEL_SLUG}"
