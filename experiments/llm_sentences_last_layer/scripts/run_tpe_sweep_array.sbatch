#!/bin/bash
#SBATCH --job-name=llm-punct-tpe
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-8%3
#SBATCH --output=experiments/llm_sentences_last_layer/logs/tpe_sweep_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/tpe_sweep_%A_%a.err

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

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/tpe_sweeps/manifest.tsv}"
BASE_CONFIG="${BASE_CONFIG:-experiments/llm_sentences_last_layer/configs/tpe_sweeps/base_decoder_punct_tpe.gin}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r RUN_ID MODEL_SLUG MODEL_NAME LR WD WARMUP BATCH_SIZE EPOCHS SEED OUTPUT_DIR < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${RUN_ID:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p "experiments/llm_sentences_last_layer/logs" "${OUTPUT_DIR}"

echo "[$(date)] Starting ${RUN_ID} on ${SLURM_JOB_NODELIST:-unknown}"
echo "model=${MODEL_NAME} lr=${LR} wd=${WD} warmup=${WARMUP} batch=${BATCH_SIZE} epochs=${EPOCHS} seed=${SEED}"

uv run python src/train_sentences.py "${BASE_CONFIG}" \
  --main.embedding_model_name="${MODEL_NAME}" \
  --main.random_seed="${SEED}" \
  --tpe/TrainingArguments.output_dir="${OUTPUT_DIR}" \
  --tpe/TrainingArguments.learning_rate="${LR}" \
  --tpe/TrainingArguments.weight_decay="${WD}" \
  --tpe/TrainingArguments.warmup_ratio="${WARMUP}" \
  --tpe/TrainingArguments.per_device_train_batch_size="${BATCH_SIZE}" \
  --tpe/TrainingArguments.per_device_eval_batch_size="${BATCH_SIZE}" \
  --tpe/TrainingArguments.num_train_epochs="${EPOCHS}" \
  --analytic_probe_eval/TrainingArguments.output_dir="${OUTPUT_DIR}/analytic_probe" \
  --trainable_probe/TrainingArguments.output_dir="${OUTPUT_DIR}/trainable_probe"

echo "[$(date)] Finished ${RUN_ID}"
