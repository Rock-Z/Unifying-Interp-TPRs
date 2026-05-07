#!/bin/bash
#SBATCH --job-name=llm-topk-wandb
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-2
#SBATCH --output=experiments/llm_sentences_last_layer/sweeps/logs/%x-%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/sweeps/logs/%x-%A_%a.err

set -euo pipefail

module reset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

# Cached hidden states make these CPU jobs viable. Use SWEEP_COUNT=20 to match
# the embedding-model sweep budget; the default is larger for fuller LLM tuning.
# Each agent gets 32 CPU threads because the SAE matrices are small enough that
# CPU BLAS parallelism can be competitive with queued GPU jobs.
SWEEP_COUNT="${SWEEP_COUNT:-40}"

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIGS=(
  "experiments/llm_sentences_last_layer/sweeps/sae_topk_qwen3_8b.yaml"
  "experiments/llm_sentences_last_layer/sweeps/sae_topk_olmo_13b.yaml"
  "experiments/llm_sentences_last_layer/sweeps/sae_topk_gpt_oss_20b.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p experiments/llm_sentences_last_layer/sweeps/logs

echo "[$(date)] Creating W&B sweep from ${CONFIG}"
SWEEP_OUTPUT="$(uv run wandb sweep "${CONFIG}" 2>&1)"
echo "${SWEEP_OUTPUT}"

SWEEP_PATH="$(echo "${SWEEP_OUTPUT}" | awk '/Run sweep agent with: wandb agent/{print $NF; exit}')"
if [[ -z "${SWEEP_PATH}" ]]; then
  echo "[SWEEP] Failed to parse sweep path from wandb output." >&2
  exit 1
fi

echo "[$(date)] Running ${SWEEP_COUNT} trials for ${SWEEP_PATH}"
uv run wandb agent --count "${SWEEP_COUNT}" "${SWEEP_PATH}"
echo "[$(date)] Finished ${SWEEP_PATH}"
