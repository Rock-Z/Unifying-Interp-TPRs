#!/bin/bash
#SBATCH --job-name=digits-filler-role-holdout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/digits/filler_role_holdout/logs/%x-%j.out
#SBATCH --error=experiments/digits/filler_role_holdout/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../.."

export UV_CACHE_DIR="/tmp/uv-cache"

mkdir -p experiments/digits/filler_role_holdout/logs

bash experiments/digits/filler_role_holdout/run_experiment.sh
