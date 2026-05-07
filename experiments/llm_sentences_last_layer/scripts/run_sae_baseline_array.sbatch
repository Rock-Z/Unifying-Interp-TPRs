#!/bin/bash
#SBATCH --job-name=llm-punct-sae-base
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-5%3
#SBATCH --output=experiments/llm_sentences_last_layer/logs/sae_baseline_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/sae_baseline_%A_%a.err

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

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/sae_baselines/manifest.tsv}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r RUN_ID BASELINE_TYPE MODEL_SLUG MODEL_NAME CONFIG_PATH OUTPUT_DIR SEED < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${RUN_ID:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p "experiments/llm_sentences_last_layer/logs" "${OUTPUT_DIR}"

echo "[$(date)] Starting ${RUN_ID} on ${SLURM_JOB_NODELIST:-unknown}"
echo "baseline=${BASELINE_TYPE} model=${MODEL_SLUG} seed=${SEED}"

uv run python src/train_sae.py "${CONFIG_PATH}" \
  --main.embedding_model_name="${MODEL_NAME}" \
  --main.random_seed="${SEED}" \
  --transformers/TrainingArguments.output_dir="${OUTPUT_DIR}" \
  --transformers/TrainingArguments.logging_dir="${OUTPUT_DIR}/logs"

echo "[$(date)] Finished ${RUN_ID}"
