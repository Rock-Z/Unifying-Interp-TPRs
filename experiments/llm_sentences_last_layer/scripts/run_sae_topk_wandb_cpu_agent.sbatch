#!/bin/bash
#SBATCH --job-name=llm-topk-cpu-agent
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=experiments/llm_sentences_last_layer/sweeps/logs/agents/%x-%A_%a.out
#SBATCH --error=experiments/llm_sentences_last_layer/sweeps/logs/agents/%x-%A_%a.err

set -euo pipefail

module reset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

if [[ -z "${SWEEP_PATH:-}" ]]; then
  echo "Missing SWEEP_PATH for W&B agent." >&2
  exit 2
fi

RUNS_PER_AGENT="${RUNS_PER_AGENT:-1}"
MODEL_SLUG="${MODEL_SLUG:-unknown}"
SWEEP_CONFIG="${SWEEP_CONFIG:-unknown}"

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p experiments/llm_sentences_last_layer/sweeps/logs/agents

echo "[$(date)] Starting CPU W&B agent"
echo "model_slug=${MODEL_SLUG}"
echo "sweep_config=${SWEEP_CONFIG}"
echo "sweep_path=${SWEEP_PATH}"
echo "runs_per_agent=${RUNS_PER_AGENT}"
echo "slurm_job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-0}"
echo "node=${SLURM_JOB_NODELIST:-unknown} cpus=${SLURM_CPUS_PER_TASK}"

uv run wandb agent --count "${RUNS_PER_AGENT}" "${SWEEP_PATH}"
echo "[$(date)] Finished CPU W&B agent for ${MODEL_SLUG}"
