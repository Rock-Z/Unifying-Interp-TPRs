#!/bin/bash
#SBATCH --job-name=llm-topk-tune
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-17%4
#SBATCH --output=experiments/llm_sentences_last_layer/logs/sae_topk_tune_%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/logs/sae_topk_tune_%A_%a.err

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

MANIFEST="${MANIFEST:-experiments/llm_sentences_last_layer/configs/sae_baselines/topk_tuning_manifest.tsv}"
CONFIG_PATH="${CONFIG_PATH:-experiments/llm_sentences_last_layer/configs/sae_baselines/train_topk.gin}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r RUN_ID MODEL_SLUG MODEL_NAME HIDDEN_DIM K LEARNING_RATE BATCH_SIZE NUM_TRAIN_EPOCHS RESAMPLE_TIMES SEED OUTPUT_DIR < <(sed -n "${LINE_NUMBER}p" "${MANIFEST}")

if [[ -z "${RUN_ID:-}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

mkdir -p "experiments/llm_sentences_last_layer/logs" "${OUTPUT_DIR}"

echo "[$(date)] Starting ${RUN_ID} on ${SLURM_JOB_NODELIST:-unknown}"
echo "model=${MODEL_SLUG} hidden_dim=${HIDDEN_DIM} k=${K} lr=${LEARNING_RATE} epochs=${NUM_TRAIN_EPOCHS}"

uv run python src/train_sae.py "${CONFIG_PATH}" \
  --main.embedding_model_name="${MODEL_NAME}" \
  --main.random_seed="${SEED}" \
  --main.resample_times="${RESAMPLE_TIMES}" \
  --main.sae_config="{'hidden_dim': ${HIDDEN_DIM}, 'k': ${K}, 'dead_latent_threshold': 1e-6}" \
  --transformers/TrainingArguments.output_dir="${OUTPUT_DIR}" \
  --transformers/TrainingArguments.logging_dir="${OUTPUT_DIR}/logs" \
  --transformers/TrainingArguments.learning_rate="${LEARNING_RATE}" \
  --transformers/TrainingArguments.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  --transformers/TrainingArguments.per_device_train_batch_size="${BATCH_SIZE}" \
  --transformers/TrainingArguments.per_device_eval_batch_size="${BATCH_SIZE}"

echo "[$(date)] Finished ${RUN_ID}"
