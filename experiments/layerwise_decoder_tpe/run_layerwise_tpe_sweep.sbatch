#!/bin/bash
#SBATCH --job-name=layerwise-tpe-sweep
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/layerwise_tpe/%x-%j.out

set -euo pipefail
mkdir -p logs/layerwise_tpe

cd "$(git rev-parse --show-toplevel)"

# Create a sweep (prints the wandb agent command). Does not run the agent automatically.
uv run python experiments/layerwise_decoder_tpe/run_layerwise_tpe_sweep.py \
  --model Qwen/Qwen3-8B \
  --base_config experiments/layerwise_decoder_tpe/configs/layerwise_tpe_base.gin \
  --sentences_path data/sentences \
  --cache_path data/sentences \
  --output_root checkpoints/layerwise_tpe \
  --project_prefix layerwise-tpe \
  "$@"

echo "To launch an agent on this node, set SWEEP_ID and run:"
echo "  wandb agent \$SWEEP_ID"
