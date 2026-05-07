from __future__ import annotations

import sys
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence
import ast

import gin
import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from activation_patching.plotting import plot_heatmap, write_heatmap_csv  # noqa: E402
from activation_patching.trials import TokenPatchTrial, load_sentence_trials  # noqa: E402
from activation_patching.utils import (  # noqa: E402
    get_layers,
    maybe_empty_cache,
    pretty_token,
    proxy_to_float,
    resolve_device,
    resolve_token_positions,
    torch_dtype_from_str,
)
from model import TensorProductEncoderForPretraining  # noqa: E402
from sentences import SVORoleAssigner, filter_words_single_token, load_sentences  # noqa: E402
from utils import gin_config_to_readable_dictionary, parse_args_for_gin  # noqa: E402


@dataclass
class TPESynthResult:
    """Stores TPE-patched trial outputs."""

    trial: TokenPatchTrial
    heatmap: np.ndarray
    clean_logit_diff: float
    corrupted_logit_diff: float
    layer_indices: list[int]
    token_positions: list[int]


def _load_role_assigner(sentences_path: str, tokenizer, role_scheme: str) -> SVORoleAssigner:
    """Recreate the role assigner used in TPE training with the same vocab layout."""

    _, assigner = load_sentences(sentences_path, role_scheme=role_scheme)
    assigner.nouns_sg = filter_words_single_token(tokenizer, assigner.nouns_sg)
    return assigner


def _build_filler_role_ids_for_triple(
    subject: str, dobj: str, verb: str, assigner: SVORoleAssigner
) -> tuple[list[int], list[int]]:
    noun_vocab = len(assigner.noun_idx2filler)
    filler_ids = [
        int(assigner.noun_filler2idx[subject]),
        int(assigner.noun_filler2idx[dobj]),
        int(noun_vocab + assigner.verb_filler2idx[verb]),
    ]
    role_ids = [
        int(assigner.role2idx["subject"]),
        int(assigner.role2idx["object"]),
        int(assigner.role2idx["verb"]),
    ]
    return filler_ids, role_ids


def _build_single_role_ids(
    filler: str,
    role_name: str,
    assigner: SVORoleAssigner,
) -> tuple[list[int], list[int]]:
    if role_name not in assigner.role2idx:
        raise ValueError(f"Role {role_name!r} not found in role2idx.")
    if role_name == "verb":
        filler_id = int(len(assigner.noun_idx2filler) + assigner.verb_filler2idx[filler])
    else:
        filler_id = int(assigner.noun_filler2idx[filler])
    role_id = int(assigner.role2idx[role_name])
    return [filler_id], [role_id]


def _synthesize_tpe_hidden(
    tpe_model: TensorProductEncoderForPretraining,
    filler_ids: Sequence[int],
    role_ids: Sequence[int],
    *,
    target_sequence_length: int,
    per_token_hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a TPE-predicted hidden state and reshape to (1, seq_len, hidden)."""

    with torch.no_grad():
        inputs = {
            "filler_ids": torch.tensor([filler_ids], device=device, dtype=torch.long),
            "role_ids": torch.tensor([role_ids], device=device, dtype=torch.long),
        }
        outputs = tpe_model(**inputs)
        flat = outputs.encoder_hidden_states.squeeze(1)
        shaped = flat.view(1, target_sequence_length, per_token_hidden_size)
    return shaped


def _run_trial_with_tpe(
    model: LanguageModel,
    layers,
    trial: TokenPatchTrial,
    layer_indices: Sequence[int],
    token_positions: Sequence[int],
    *,
    tpe_hidden_by_layer: dict[int, torch.Tensor],
    target_device: torch.device,
    normalization_eps: float,
) -> TPESynthResult:
    """Patch corrupted prompt with TPE hidden states and compute restoration."""

    with model.trace() as tracer:
        with tracer.invoke(trial.clean_prompt):
            logits = model.lm_head.output
            clean_correct_proxy = logits[0, -1, trial.correct_token_id].save()
            clean_comp_proxy = logits[0, -1, trial.competitor_token_id].save()
    maybe_empty_cache(target_device)
    clean_correct = proxy_to_float(clean_correct_proxy)
    clean_comp = proxy_to_float(clean_comp_proxy)

    with model.trace() as tracer:
        with tracer.invoke(trial.corrupted_prompt):
            logits = model.lm_head.output
            corrupt_correct_proxy = logits[0, -1, trial.correct_token_id].save()
            corrupt_comp_proxy = logits[0, -1, trial.competitor_token_id].save()
    corrupt_correct = proxy_to_float(corrupt_correct_proxy)
    corrupt_comp = proxy_to_float(corrupt_comp_proxy)
    maybe_empty_cache(target_device)

    clean_diff = clean_correct - clean_comp
    corrupt_diff = corrupt_correct - corrupt_comp
    denom = clean_diff - corrupt_diff

    heatmap = np.zeros((len(layer_indices), len(token_positions)), dtype=np.float32)
    for row, layer_idx in enumerate(layer_indices):
        for col, pos in enumerate(token_positions):
            with model.trace() as tracer:
                with tracer.invoke(trial.corrupted_prompt):
                    layer_output = layers[layer_idx].output
                    patched = layer_output.clone()
                    tpe_slice = tpe_hidden_by_layer[layer_idx].to(target_device)
                    patched[:, pos, :] = tpe_slice[:, pos, :]
                    layers[layer_idx].output = patched
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

    return TPESynthResult(
        trial=trial,
        heatmap=heatmap,
        clean_logit_diff=clean_diff,
        corrupted_logit_diff=corrupt_diff,
        layer_indices=list(layer_indices),
        token_positions=list(token_positions),
    )


@gin.configurable(module="tpe_activation_patching")
def run_tpe_activation_patching(
    sentences_path: str = "data/sentences/data.test",
    output_dir: str = "results/tpe_activation_patching",
    model_id: str = "Qwen/Qwen3-8B",
    tpe_checkpoint_path: str = "checkpoints/layerwise_tpe/qwen3-8b/layer0/best_model",
    device: str = "cuda",
    dtype: str = "bfloat16",
    tpe_device: str = "cuda",
    tpe_dtype: str = "float32",
    max_trials: Optional[int] = 8,
    random_seed: int = 0,
    layer_indices: Optional[Sequence[int]] = None,
    explicit_token_positions: Optional[Sequence[int]] = None,
    token_focus: str = "passive_clause",
    role_scheme: str = "svo",
    normalization_eps: float = 1e-9,
    binding_role_name: str = "subject",
) -> None:
    """Patch LM activations with TPE-synthesized hidden states."""

    tpe_checkpoint_root = Path(tpe_checkpoint_path)
    tpe_root_has_layers = tpe_checkpoint_root.is_dir() and (tpe_checkpoint_root / "layer0").exists()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assigner = _load_role_assigner(Path(sentences_path).parent, tokenizer, role_scheme=role_scheme)

    print(f"Sampling sentence pairs from {sentences_path}")
    trials = load_sentence_trials(
        tokenizer,
        sentences_path,
        max_trials,
        random_seed,
    )

    resolved_positions = resolve_token_positions(trials[0].token_strings, explicit_token_positions, token_focus)

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
    tpe_models: dict[int, TensorProductEncoderForPretraining] = {}
    seq_len: Optional[int] = None
    per_token_hidden: Optional[int] = None

    def get_tpe_model(layer_idx: int) -> TensorProductEncoderForPretraining:
        """Load a TPE model for a given layer index."""
        if layer_idx in tpe_models:
            return tpe_models[layer_idx]
        tpe_path = Path(tpe_checkpoint_path)
        if tpe_root_has_layers:
            layer_dir = tpe_checkpoint_root / f"layer{layer_idx}"
            best_model = layer_dir / "best_model"
            tpe_path = best_model if best_model.exists() else layer_dir
        tpe_model = TensorProductEncoderForPretraining.from_pretrained(tpe_path)
        tpe_model = tpe_model.to(device=tpe_device, dtype=torch_dtype_from_str(tpe_dtype))
        tpe_model.eval()
        target_seq_len = int(tpe_model.config.target_sequence_length)
        target_hidden = int(tpe_model.config.per_token_hidden_size)
        if seq_len is not None and per_token_hidden is not None:
            if target_seq_len != seq_len or target_hidden != per_token_hidden:
                raise ValueError(
                    "TPE checkpoints disagree on target sequence length or hidden size."
                )
        tpe_models[layer_idx] = tpe_model
        return tpe_model

    if layer_indices is None:
        if tpe_root_has_layers:
            raise ValueError(
                "layer_indices must be set when tpe_checkpoint_path points to a root "
                "directory containing multiple layer checkpoints."
            )
        print(f"Loading TPE checkpoint from {tpe_checkpoint_path}")
        tpe = TensorProductEncoderForPretraining.from_pretrained(tpe_checkpoint_path)
        tpe = tpe.to(device=tpe_device, dtype=torch_dtype_from_str(tpe_dtype))
        tpe.eval()
        seq_len = int(tpe.config.target_sequence_length)
        per_token_hidden = int(tpe.config.per_token_hidden_size)
        tpe_layer = int(tpe.config.layer_id)
        layer_indices = [tpe_layer]
        tpe_models[tpe_layer] = tpe
    else:
        if isinstance(layer_indices, str):
            if layer_indices.lower() == "all":
                layer_indices = list(range(len(layers)))
            else:
                parsed = ast.literal_eval(layer_indices)
                assert isinstance(parsed, (list, tuple)), "layer_indices must be a list of ints or 'all'."
                layer_indices = list(parsed)
        layer_indices = [idx for idx in layer_indices if 0 <= idx < len(layers)]
        assert layer_indices, "layer_indices resolved to an empty set."
        # Load one TPE to define seq_len/per_token_hidden for reshaping patches.
        first_tpe = get_tpe_model(layer_indices[0])
        seq_len = int(first_tpe.config.target_sequence_length)
        per_token_hidden = int(first_tpe.config.per_token_hidden_size)

    assert seq_len is not None
    assert per_token_hidden is not None
    print(f"Patching over {len(trials)} trials | sequence length {seq_len} | token positions: {resolved_positions}")

    layer_labels = [f"Layer {idx}" for idx in layer_indices]
    token_labels = [pretty_token(trials[0].token_strings[pos]) for pos in resolved_positions]

    if binding_role_name not in assigner.role2idx:
        raise ValueError(f"binding_role_name {binding_role_name!r} not found in role2idx.")

    results: list[TPESynthResult] = []
    for trial in trials:
        tpe_hidden_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            tpe_model = get_tpe_model(layer_idx)
            clean_subject = trial.meta["subject"]
            corrupted_subject = trial.meta["corrupted_subject"]
            dobj = trial.meta["object"]
            verb = trial.meta["verb"]
            base_filler_ids, base_role_ids = _build_filler_role_ids_for_triple(
                corrupted_subject,
                dobj,
                verb,
                assigner,
            )
            base_hidden = _synthesize_tpe_hidden(
                tpe_model=tpe_model,
                filler_ids=base_filler_ids,
                role_ids=base_role_ids,
                target_sequence_length=seq_len,
                per_token_hidden_size=per_token_hidden,
                device=torch.device(tpe_device),
            )
            if str(tpe_model.config.aggregation) == "mean":
                raise NotImplementedError(
                    "Binding arithmetic currently assumes sum aggregation; "
                    "mean aggregation is not supported."
                )
            from_ids, from_roles = _build_single_role_ids(
                corrupted_subject, binding_role_name, assigner
            )
            to_ids, to_roles = _build_single_role_ids(
                clean_subject, binding_role_name, assigner
            )
            binding_from = _synthesize_tpe_hidden(
                tpe_model=tpe_model,
                filler_ids=from_ids,
                role_ids=from_roles,
                target_sequence_length=seq_len,
                per_token_hidden_size=per_token_hidden,
                device=torch.device(tpe_device),
            )
            binding_to = _synthesize_tpe_hidden(
                tpe_model=tpe_model,
                filler_ids=to_ids,
                role_ids=to_roles,
                target_sequence_length=seq_len,
                per_token_hidden_size=per_token_hidden,
                device=torch.device(tpe_device),
            )
            tpe_hidden_by_layer[layer_idx] = base_hidden - binding_from + binding_to
        result = _run_trial_with_tpe(
            model,
            layers,
            trial,
            layer_indices,
            resolved_positions,
            tpe_hidden_by_layer=tpe_hidden_by_layer,
            target_device=target_device,
            normalization_eps=normalization_eps,
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
        "config": gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG),
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "tpe_checkpoint": tpe_checkpoint_path,
        "tpe_device": tpe_device,
        "tpe_dtype": tpe_dtype,
        "binding_role_name": binding_role_name,
        "num_trials": len(results),
        "layer_indices": list(layer_indices),
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


def main():
    parse_args_for_gin()
    run_tpe_activation_patching()


if __name__ == "__main__":
    main()
