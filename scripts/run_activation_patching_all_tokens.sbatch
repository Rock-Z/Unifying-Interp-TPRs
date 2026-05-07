#!/bin/bash
#SBATCH --job-name=patch-all-tokens
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gpus=h100:4
#SBATCH --array=0-3

set -euo pipefail

MODELS=(
  "openai/gpt-oss-20b"
  "openai/gpt-oss-120b"
  "zai-org/GLM-4.5-Air"
  "allenai/OLMo-2-1124-13B"
)

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${#MODELS[@]} ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range"
  exit 1
fi

MODEL_ID="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SAFE_MODEL=$(echo "${MODEL_ID}" | tr '/:' '__')
BASE_OUT="results/activation_patching/${SAFE_MODEL}/all_tokens"
mkdir -p "${BASE_OUT}"

echo "[$(date)] Starting activation patching for ${MODEL_ID}"

split_and_run() {
  local split_label="$1"
  local split_positions="$2"
  local out_dir="${BASE_OUT}/${split_label}"
  mkdir -p "${out_dir}"

  echo "[$(date)] Running ${MODEL_ID} for positions ${split_positions} -> ${out_dir}"
  uv run python experiments/activation_patching/activation_patching.py experiments/activation_patching/configs/activation_patching_sentences.gin \
    --activation_patching.run_activation_patching.model_id="'${MODEL_ID}'" \
    --activation_patching.run_activation_patching.output_dir="'${out_dir}'" \
    --activation_patching.run_activation_patching.token_focus="'all'" \
    --activation_patching.run_activation_patching.explicit_token_positions="[${split_positions}]" \
    --activation_patching.run_activation_patching.max_trials=12 \
    --activation_patching.run_activation_patching.dtype="'bfloat16'" \
    --activation_patching.run_activation_patching.device="'cuda'"
}

# Run the first (active clause) chunk and the second (passive clause) chunk.
split_and_run "split_0_6" "0,1,2,3,4,5,6"
split_and_run "split_7_13" "7,8,9,10,11,12,13"

echo "[$(date)] Combining splits for ${MODEL_ID}"
PYTHONPATH=src uv run python - <<'PY' "${BASE_OUT}/split_0_6" "${BASE_OUT}/split_7_13" "${BASE_OUT}"
import json
import sys
from pathlib import Path
import numpy as np
from activation_patching.plotting import plot_activation_patching_results

split_dirs = [Path(sys.argv[1]), Path(sys.argv[2])]
out_dir = Path(sys.argv[3])

heatmaps = []
token_labels = []
token_positions = []
layer_labels = None
num_trials = 0
clean_means = []
corrupt_means = []

for split in split_dirs:
    summary = json.loads((split / "summary.json").read_text())
    heatmaps.append(np.load(split / "token_heatmap.npy"))
    token_labels.extend(summary["token_labels"])
    token_positions.extend(summary.get("token_positions", []))
    if layer_labels is None:
        layer_labels = summary["layer_labels"]
    num_trials += summary.get("num_trials", 0)
    clean_means.append(summary.get("mean_clean_logit_diff", 0.0))
    corrupt_means.append(summary.get("mean_corrupted_logit_diff", 0.0))

full_heatmap = np.concatenate(heatmaps, axis=1)
np.save(out_dir / "token_heatmap.npy", full_heatmap)

plot_activation_patching_results(
    full_heatmap,
    token_labels=token_labels,
    layer_labels=layer_labels,
    token_positions=token_positions,
    plot_title="Token-level activation patching (full prompt)",
).write_html(out_dir / "token_heatmap.html")

with (out_dir / "token_heatmap.csv").open("w") as fh:
    fh.write("layer," + ",".join(token_labels) + "\n")
    for label, row in zip(layer_labels, full_heatmap):
        fh.write(label + "," + ",".join(f"{val:.6f}" for val in row) + "\n")

with (out_dir / "trial_summaries.jsonl").open("w") as out_fh:
    for split in split_dirs:
        with (split / "trial_summaries.jsonl").open() as in_fh:
            for line in in_fh:
                out_fh.write(line)

summary = {
    "component_runs": [str(split) for split in split_dirs],
    "num_trials": num_trials,
    "layer_labels": layer_labels,
    "token_labels": token_labels,
    "token_positions": token_positions,
    "mean_clean_logit_diff": float(np.mean(clean_means) if clean_means else 0.0),
    "mean_corrupted_logit_diff": float(np.mean(corrupt_means) if corrupt_means else 0.0),
    "heatmap_shape": list(full_heatmap.shape),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
PY

echo "[$(date)] Completed activation patching for ${MODEL_ID}"
