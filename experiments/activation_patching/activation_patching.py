from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, Optional, Sequence

import gin
from gin import config as gin_config
import numpy as np
import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

# Allow running directly without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from utils import gin_config_to_readable_dictionary, parse_args_for_gin  # noqa: E402
from activation_patching.patching import TrialPatchResult, run_trial_patch  # noqa: E402
from activation_patching.plotting import plot_heatmap, write_heatmap_csv  # noqa: E402
from activation_patching.trials import load_sentence_trials  # noqa: E402
from activation_patching.utils import (  # noqa: E402
    get_layers,
    pretty_token,
    resolve_device,
    resolve_token_positions,
    torch_dtype_from_str,
)


@gin.configurable(module="activation_patching")
def run_activation_patching(
    sentences_path: str = "data/sentences/data.test",
    output_dir: str = "results/activation_patching",
    model_id: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_trials: Optional[int] = 8,
    random_seed: int = 0,
    layer_indices: Optional[Sequence[int]] = None,
    explicit_token_positions: Optional[Sequence[int]] = None,
    token_focus: Literal["all", "active_clause", "passive_clause"] = "passive_clause",
    normalization_eps: float = 1e-9,
) -> None:
    """Run token-level activation patching over SVO sentence pairs.

    Loads a tokenizer/model, samples trials via `load_sentence_trials` (clean prompts differ from
    corrupted prompts by subject only), selects layer indices and token positions (from
    `explicit_token_positions` or `token_focus`), then patches residual stream inputs with
    `run_trial_patch` to compute normalized restoration heatmaps.

    Outputs written under `output_dir`: `token_heatmap.npy`, `token_heatmap.csv`,
    `token_heatmap.html`, `trial_summaries.jsonl`, and `summary.json`.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Sampling sentence pairs from {sentences_path}")
    trials = load_sentence_trials(
        tokenizer,
        sentences_path,
        max_trials,
        random_seed,
    )
    seq_lens = {len(trial.token_ids) for trial in trials}
    if len(seq_lens) != 1:
        raise ValueError(
            f"Tokenized prompt lengths differ across trials: {sorted(seq_lens)}. "
            "Ensure nouns are single tokens or set explicit_token_positions."
        )

    sample_len = len(trials[0].token_ids)
    resolved_positions = resolve_token_positions(
        trials[0].token_strings, explicit_token_positions, token_focus
    )
    if not resolved_positions:
        raise ValueError("No token positions selected for patching")
    print(
        f"Patching over {len(trials)} trials | sequence length {sample_len} | "
        f"token positions: {resolved_positions}"
    )

    print(f"Loading {model_id} with dtype={dtype} on device={device}")
    torch_dtype = torch_dtype_from_str(dtype)
    lm_kwargs: dict[str, object] = {"dtype": torch_dtype, "tokenizer": tokenizer}
    device_lower = device.lower()
    if device_lower.startswith("cuda"):
        lm_kwargs["device_map"] = "auto" if device_lower == "cuda" else device
    elif device_lower == "auto":
        lm_kwargs["device_map"] = "auto"
    model = LanguageModel(model_id, **lm_kwargs)  # type: ignore[arg-type]
    layers = get_layers(model)
    target_device = resolve_device(device_lower)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    else:
        layer_indices = [idx for idx in layer_indices if 0 <= idx < len(layers)]
    if not layer_indices:
        raise ValueError("layer_indices resolved to an empty set")

    layer_labels = [f"Layer {idx}" for idx in layer_indices]
    token_labels = [pretty_token(trials[0].token_strings[pos]) for pos in resolved_positions]

    results: list[TrialPatchResult] = []
    for trial in tqdm(trials, desc="Activation patching"):
        result = run_trial_patch(
            model,
            layers,
            trial,
            layer_indices,
            resolved_positions,
            normalization_eps=normalization_eps,
            target_device=target_device,
        )
        results.append(result)

    stacked = np.stack([res.heatmap for res in results], axis=0)
    avg_heatmap = np.mean(stacked, axis=0)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    heatmap_npy = output_dir_path / "token_heatmap.npy"
    heatmap_csv = output_dir_path / "token_heatmap.csv"
    heatmap_html = output_dir_path / "token_heatmap.html"
    trials_jsonl = output_dir_path / "trial_summaries.jsonl"
    summary_json = output_dir_path / "summary.json"

    np.save(heatmap_npy, avg_heatmap)
    write_heatmap_csv(heatmap_csv, avg_heatmap, layer_labels, token_labels)
    plot_heatmap(
        heatmap_html,
        avg_heatmap,
        layer_labels,
        token_labels,
        token_positions=resolved_positions,
    )

    with trials_jsonl.open("w") as fh:
        for res in results:
            record = {
                "meta": res.trial.meta,
                "clean_logit_diff": res.clean_logit_diff,
                "corrupted_logit_diff": res.corrupted_logit_diff,
                "mean_restoration": float(res.heatmap.mean()),
                "max_restoration": float(res.heatmap.max()),
                "min_restoration": float(res.heatmap.min()),
            }
            fh.write(json.dumps(record) + "\n")

    clean_vals = np.array([res.clean_logit_diff for res in results])
    corrupt_vals = np.array([res.corrupted_logit_diff for res in results])
    summary = {
        "config": gin_config_to_readable_dictionary(gin_config._OPERATIVE_CONFIG),
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "num_trials": len(results),
        "layer_indices": layer_indices,
        "token_positions": resolved_positions,
        "layer_labels": layer_labels,
        "token_labels": token_labels,
        "mean_clean_logit_diff": float(clean_vals.mean()),
        "mean_corrupted_logit_diff": float(corrupt_vals.mean()),
        "heatmap_shape": list(avg_heatmap.shape),
    }
    with summary_json.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Mean clean logit difference: {clean_vals.mean():.3f}")
    print(f"Mean corrupted logit difference: {corrupt_vals.mean():.3f}")
    print(f"Saved aggregated heatmap to {heatmap_csv} and {heatmap_html}")


def main() -> None:
    """CLI entry point for token-level activation patching experiments."""

    parse_args_for_gin()
    run_activation_patching()


if __name__ == "__main__":
    main()
