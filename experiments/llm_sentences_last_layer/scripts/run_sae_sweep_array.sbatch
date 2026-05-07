#!/bin/bash
#SBATCH --job-name=llm-punct-sae
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-17%6
#SBATCH --output=experiments/llm_sentences_last_layer/logs/sae_sweep_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/sae_sweep_%A_%a.err

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

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/sae_sweeps/manifest_ridge.tsv}"
CONSTRUCTION_CALIBRATION_SAMPLES="${CONSTRUCTION_CALIBRATION_SAMPLES:-2048}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r RUN_ID MODEL_SLUG CONFIG_PATH DECODER_REFINEMENT DECODER_REFINEMENT_L2 DECODER_BIAS_SOURCE OUTPUT_DIR < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${RUN_ID:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p "experiments/llm_sentences_last_layer/logs" "${OUTPUT_DIR}"

echo "[$(date)] Starting ${RUN_ID} on ${SLURM_JOB_NODELIST:-unknown}"
echo "model=${MODEL_SLUG} refinement=${DECODER_REFINEMENT} l2=${DECODER_REFINEMENT_L2} bias=${DECODER_BIAS_SOURCE}"

uv run python src/tpr_sae.py "${CONFIG_PATH}" \
  --main.sae_output_dir="${OUTPUT_DIR}" \
  --main.decoder_refinement="${DECODER_REFINEMENT}" \
  --main.decoder_refinement_l2="${DECODER_REFINEMENT_L2}" \
  --main.decoder_bias_source="${DECODER_BIAS_SOURCE}" \
  --main.construction_calibration_samples="${CONSTRUCTION_CALIBRATION_SAMPLES}"

echo "[$(date)] Finished ${RUN_ID}"
