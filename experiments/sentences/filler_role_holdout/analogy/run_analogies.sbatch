#!/bin/bash
#SBATCH --job-name=sentences-holdout-analogy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/sentences/filler_role_holdout/analogy/logs/%x-%j.out
#SBATCH --error=experiments/sentences/filler_role_holdout/analogy/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../../../.."

export UV_CACHE_DIR="/tmp/uv-cache"

mkdir -p experiments/sentences/filler_role_holdout/analogy/logs

bash experiments/sentences/filler_role_holdout/analogy/run_analogies.sh
