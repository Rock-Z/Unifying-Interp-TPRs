#!/bin/bash
#SBATCH --job-name=llm-punct-eval
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-8%3
#SBATCH --output=experiments/llm_sentences_last_layer/logs/downstream_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/downstream_%A_%a.err

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

MANIFEST="experiments/llm_sentences_last_layer/configs/downstream_manifest.tsv"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r TASK MODEL_SLUG CONFIG_PATH REQUIRED_TPE_DIR < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${TASK:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

if [[ ! -d "${REQUIRED_TPE_DIR}" ]]; then
  echo "Required TPE checkpoint missing: ${REQUIRED_TPE_DIR}" >&2
  exit 3
fi

echo "[$(date)] Starting ${TASK}/${MODEL_SLUG} on ${SLURM_JOB_NODELIST:-unknown}"

case "${TASK}" in
  analogy)
    mkdir -p "experiments/llm_sentences_last_layer/results/analogy/${MODEL_SLUG}"
    uv run python scripts/analogy/evaluate_analogy.py "${CONFIG_PATH}"
    ;;
  probe)
    mkdir -p "experiments/llm_sentences_last_layer/results/probe/${MODEL_SLUG}"
    uv run python src/invert_svo.py "${CONFIG_PATH}"
    ;;
  sae)
    mkdir -p "experiments/llm_sentences_last_layer/results/sae/${MODEL_SLUG}"
    uv run python src/tpr_sae.py "${CONFIG_PATH}"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 4
    ;;
esac

echo "[$(date)] Finished ${TASK}/${MODEL_SLUG}"
