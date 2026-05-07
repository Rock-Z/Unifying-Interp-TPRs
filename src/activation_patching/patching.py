from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from nnsight import LanguageModel

from .trials import TokenPatchTrial
from .utils import maybe_empty_cache, proxy_to_float


@dataclass
class TrialPatchResult:
    """Stores heatmap and logit stats for a trial."""

    trial: TokenPatchTrial
    heatmap: np.ndarray
    clean_logit_diff: float
    corrupted_logit_diff: float
    layer_indices: list[int]
    token_positions: list[int]


def run_trial_patch(
    model: LanguageModel,
    layers: Sequence,
    trial: TokenPatchTrial,
    layer_indices: Sequence[int],
    token_positions: Sequence[int],
    *,
    normalization_eps: float,
    target_device: torch.device,
) -> TrialPatchResult:
    """Patch corrupted prompt activations with cached clean inputs."""

    if not layer_indices:
        raise ValueError("layer_indices must contain at least one entry")
    if not token_positions:
        raise ValueError("token_positions must contain at least one entry")

    clean_cache_proxies: dict[int, object] = {}

    # Clean pass -- build cache of clean layer inputs
    with model.trace() as tracer:
        with tracer.invoke(trial.clean_prompt):
            logits = model.lm_head.output
            clean_correct_proxy = logits[0, -1, trial.correct_token_id].save()
            clean_comp_proxy = logits[0, -1, trial.competitor_token_id].save()
            for layer_idx in layer_indices:
                clean_cache_proxies[layer_idx] = layers[layer_idx].input.save()
    clean_cache: dict[int, torch.Tensor] = {}
    for idx, proxy in clean_cache_proxies.items():
        clean_cache[idx] = proxy.value.detach().to("cpu")
    clean_correct = proxy_to_float(clean_correct_proxy)
    clean_comp = proxy_to_float(clean_comp_proxy)

    # Corrupted baseline
    with model.trace() as tracer:
        with tracer.invoke(trial.corrupted_prompt):
            logits = model.lm_head.output
            corrupt_correct_proxy = logits[0, -1, trial.correct_token_id].save()
            corrupt_comp_proxy = logits[0, -1, trial.competitor_token_id].save()
    # Store the corrupted logit differences
    corrupt_correct = proxy_to_float(corrupt_correct_proxy)
    corrupt_comp = proxy_to_float(corrupt_comp_proxy)

    clean_diff = clean_correct - clean_comp
    corrupt_diff = corrupt_correct - corrupt_comp
    denom = clean_diff - corrupt_diff

    heatmap = np.zeros((len(layer_indices), len(token_positions)), dtype=np.float32)
    # Patched passes
    for row, layer_idx in enumerate(layer_indices):
        for col, pos in enumerate(token_positions):
            with model.trace() as tracer:
                with tracer.invoke(trial.corrupted_prompt):
                    layer_input = layers[layer_idx].input
                    patched = layer_input.clone()
                    clean_tensor = clean_cache[layer_idx].to(target_device)
                    patched[:, pos, :] = clean_tensor[:, pos, :]
                    layers[layer_idx].input = patched
                    logits = model.lm_head.output
                    patched_correct = logits[0, -1, trial.correct_token_id].save()
                    patched_comp = logits[0, -1, trial.competitor_token_id].save()
            patched_diff = proxy_to_float(patched_correct) - proxy_to_float(patched_comp)
            if abs(denom) < normalization_eps:
                restoration = 0.0
            else:
                restoration = (patched_diff - corrupt_diff) / denom
            heatmap[row, col] = restoration
            maybe_empty_cache(target_device)

    return TrialPatchResult(
        trial=trial,
        heatmap=heatmap,
        clean_logit_diff=clean_diff,
        corrupted_logit_diff=corrupt_diff,
        layer_indices=list(layer_indices),
        token_positions=list(token_positions),
    )
