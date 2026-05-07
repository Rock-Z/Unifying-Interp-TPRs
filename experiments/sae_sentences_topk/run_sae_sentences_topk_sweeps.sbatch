#!/bin/bash
#SBATCH --job-name=sae-sentences-topk-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=experiments/sae_sentences_topk/sweeps/logs/%x-%A_%a.out

# Launch one W&B sweep per embedding model and run a fixed number of agents.
# Override SWEEP_COUNT at submit time: SWEEP_COUNT=40 sbatch ...

# Quiet progress bars and set CPU thread count to the allocated cores.
export TQDM_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Work from the repo root so relative paths resolve correctly.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../.."

SWEEP_COUNT=${SWEEP_COUNT:-20}

CONFIGS=(
  "experiments/sae_sentences_topk/sweeps/sae_topk_modernbert.yaml"
  "experiments/sae_sentences_topk/sweeps/sae_topk_qwen3_8b.yaml"
  "experiments/sae_sentences_topk/sweeps/sae_topk_e5_small.yaml"
  "experiments/sae_sentences_topk/sweeps/sae_topk_embeddinggemma.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p experiments/sae_sentences_topk/sweeps/logs

SWEEP_OUTPUT="$(uv run wandb sweep "$CONFIG" 2>&1)"
echo "$SWEEP_OUTPUT"
SWEEP_PATH="$(echo "$SWEEP_OUTPUT" | awk '/Run sweep agent with: wandb agent/{print $NF; exit}')"
if [[ -z "$SWEEP_PATH" ]]; then
  echo "[SWEEP] Failed to parse sweep path from wandb output."
  exit 1
fi

uv run wandb agent --count "$SWEEP_COUNT" "$SWEEP_PATH"
