#!/bin/bash
#SBATCH --job-name=llm-topk-fanout
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=experiments/llm_sentences_last_layer/sweeps/logs/%x-%A.out
#SBATCH --error=experiments/llm_sentences_last_layer/sweeps/logs/%x-%A.err

set -euo pipefail

module reset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

TOTAL_RUNS_PER_SWEEP="${TOTAL_RUNS_PER_SWEEP:-40}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-1}"
MAX_PARALLEL_AGENTS="${MAX_PARALLEL_AGENTS:-4}"
AGENT_SCRIPT="experiments/llm_sentences_last_layer/scripts/run_sae_topk_wandb_cpu_agent.sbatch"

if (( TOTAL_RUNS_PER_SWEEP < 1 )); then
  echo "TOTAL_RUNS_PER_SWEEP must be >= 1." >&2
  exit 2
fi
if (( RUNS_PER_AGENT < 1 )); then
  echo "RUNS_PER_AGENT must be >= 1." >&2
  exit 2
fi
if (( MAX_PARALLEL_AGENTS < 1 )); then
  echo "MAX_PARALLEL_AGENTS must be >= 1." >&2
  exit 2
fi

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

CONFIGS=(
  "qwen3_8b experiments/llm_sentences_last_layer/sweeps/sae_topk_qwen3_8b.yaml"
  "olmo_13b experiments/llm_sentences_last_layer/sweeps/sae_topk_olmo_13b.yaml"
  "gpt_oss_20b experiments/llm_sentences_last_layer/sweeps/sae_topk_gpt_oss_20b.yaml"
)

mkdir -p experiments/llm_sentences_last_layer/sweeps/logs experiments/llm_sentences_last_layer/sweeps/logs/agents

AGENTS_PER_SWEEP=$(((TOTAL_RUNS_PER_SWEEP + RUNS_PER_AGENT - 1) / RUNS_PER_AGENT))
ARRAY_SPEC="0-$((AGENTS_PER_SWEEP - 1))%${MAX_PARALLEL_AGENTS}"
SWEEP_RECORD="experiments/llm_sentences_last_layer/sweeps/logs/fanout-${SLURM_JOB_ID}-sweeps.tsv"
echo -e "model_slug\tsweep_path\tconfig\tagent_submit" > "${SWEEP_RECORD}"

for ENTRY in "${CONFIGS[@]}"; do
  read -r MODEL_SLUG CONFIG <<< "${ENTRY}"
  echo "[$(date)] Creating W&B sweep for ${MODEL_SLUG} from ${CONFIG}"
  SWEEP_OUTPUT="$(uv run wandb sweep "${CONFIG}" 2>&1)"
  echo "${SWEEP_OUTPUT}"
  SWEEP_PATH="$(echo "${SWEEP_OUTPUT}" | awk '/Run sweep agent with: wandb agent/{print $NF; exit}')"
  if [[ -z "${SWEEP_PATH}" ]]; then
    echo "[SWEEP] Failed to parse sweep path from wandb output for ${MODEL_SLUG}." >&2
    exit 1
  fi

  SUBMIT_OUTPUT="$(
    sbatch \
      --array="${ARRAY_SPEC}" \
      --export=ALL,SWEEP_PATH="${SWEEP_PATH}",RUNS_PER_AGENT="${RUNS_PER_AGENT}",MODEL_SLUG="${MODEL_SLUG}",SWEEP_CONFIG="${CONFIG}" \
      "${AGENT_SCRIPT}"
  )"
  echo "${MODEL_SLUG}: ${SUBMIT_OUTPUT}"
  printf '%s\t%s\t%s\t%s\n' "${MODEL_SLUG}" "${SWEEP_PATH}" "${CONFIG}" "${SUBMIT_OUTPUT}" >> "${SWEEP_RECORD}"
done

echo "[$(date)] Recorded fan-out sweeps to ${SWEEP_RECORD}"
