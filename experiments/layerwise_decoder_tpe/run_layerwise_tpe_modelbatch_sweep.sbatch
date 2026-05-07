#!/bin/bash
#SBATCH --job-name=layerwise-tpe-modelbatch-sweep
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/layerwise_decoder_tpe/sweeps/logs/%x-%A.out

set -euo pipefail

# Launch the ModelBatch sweep and fan out agent jobs.
# Usage: sbatch run_layerwise_tpe_modelbatch_sweep.sbatch <sweep_yaml>
# Override at submit time:
#   SWEEP_COUNT=24 MAX_CONCURRENT=4 sbatch ...

# Quiet progress bars and set CPU thread count to the allocated cores.
export TQDM_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Work from the repo root so relative paths resolve correctly.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../.."

if [[ "${1:-}" == "--agent" ]]; then
  if [[ -z "${SWEEP_PATH:-}" ]]; then
    echo "[SWEEP] Missing SWEEP_PATH for agent."
    exit 1
  fi
  uv run wandb agent --count 1 "$SWEEP_PATH"
  exit 0
fi

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  echo "Usage: sbatch $0 <sweep_yaml>"
  exit 1
fi

SWEEP_COUNT=${SWEEP_COUNT:-24}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
if [[ "$SWEEP_COUNT" -lt 1 ]]; then
  echo "[SWEEP] SWEEP_COUNT must be >= 1."
  exit 1
fi
if [[ "$MAX_CONCURRENT" -lt 1 ]]; then
  echo "[SWEEP] MAX_CONCURRENT must be >= 1."
  exit 1
fi

AGENT_PARTITION=${AGENT_PARTITION:-gpu}
AGENT_GRES=${AGENT_GRES:-gpu:1}
AGENT_CONSTRAINT=${AGENT_CONSTRAINT:-a100|h100}
AGENT_CPUS=${AGENT_CPUS:-8}
AGENT_MEM=${AGENT_MEM:-256G}
AGENT_TIME=${AGENT_TIME:-4:00:00}

mkdir -p experiments/layerwise_decoder_tpe/sweeps/logs

SWEEP_OUTPUT="$(uv run wandb sweep "$CONFIG" 2>&1)"
echo "$SWEEP_OUTPUT"
SWEEP_PATH="$(echo "$SWEEP_OUTPUT" | awk '/Run sweep agent with: wandb agent/{print $NF; exit}')"
if [[ -z "$SWEEP_PATH" ]]; then
  echo "[SWEEP] Failed to parse sweep path from wandb output."
  exit 1
fi

CONFIG_NAME="$(basename "$CONFIG" .yaml)"
SWEEP_RECORD="experiments/layerwise_decoder_tpe/sweeps/logs/${CONFIG_NAME}-${SLURM_JOB_ID}-sweep.txt"
echo "$SWEEP_PATH" > "$SWEEP_RECORD"
echo "[SWEEP] Recorded sweep path to ${SWEEP_RECORD}"

sbatch \
  --job-name=layerwise-tpe-modelbatch-agent \
  --partition="$AGENT_PARTITION" \
  --gres="$AGENT_GRES" \
  --constraint="$AGENT_CONSTRAINT" \
  --cpus-per-task="$AGENT_CPUS" \
  --mem="$AGENT_MEM" \
  --time="$AGENT_TIME" \
  --array=0-$((SWEEP_COUNT - 1))%${MAX_CONCURRENT} \
  --output=experiments/layerwise_decoder_tpe/sweeps/logs/%x-%A_%a.out \
  --export=ALL,SWEEP_PATH="$SWEEP_PATH" \
  "$0" --agent
