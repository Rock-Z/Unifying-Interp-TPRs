#!/usr/bin/env python
"""Evaluate analytic SVO probe construction parameters for LLM TPE checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from model import TensorProductEncoderForPretraining
from probing import construct_unbinding_vectors
from sentences import load_sentences
from utils import load_dataset_with_embeddings


ROOT = Path("experiments/llm_sentences_last_layer")
DEFAULT_OUTPUT_L2 = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6)
DEFAULT_ROLE_L2 = (1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2)
DEFAULT_FILLER_L2 = (1e-4, 1e-2)
ROLE_NAMES = ("subj", "verb", "obj")


def parse_float_list(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse comma-separated floats, preserving a default when unset."""
    if value is None or value.strip() == "":
        return default
    return tuple(float(part) for part in value.split(",") if part.strip())


def add_labels_for_role(dataset_split, role_id: int):
    """Add labels for one SVO role to a dataset split."""
    labels: list[int] = []
    for current_roles, current_fillers in zip(dataset_split["role_ids"], dataset_split["filler_ids"]):
        label = -1
        for rid, fid in zip(current_roles, current_fillers):
            if int(rid) == int(role_id):
                label = int(fid)
                break
        labels.append(label)
    if "labels" in dataset_split.column_names:
        dataset_split = dataset_split.remove_columns("labels")
    return dataset_split.add_column("labels", labels)


def remap_role_slice(dataset_split, label_offset: int, label_count: int):
    """Filter to a label slice and remap labels to `[0, label_count)`."""
    max_label = label_offset + label_count
    dataset_split = dataset_split.filter(lambda x: label_offset <= x["labels"] < max_label)
    labels = [int(label) - label_offset for label in dataset_split["labels"]]
    dataset_split = dataset_split.remove_columns("labels")
    return dataset_split.add_column("labels", labels)


def role_metadata(role_assigner) -> dict[str, dict[str, int]]:
    """Return role ids and label slices for SVO roles."""
    noun_count = len(role_assigner.noun_filler2idx)
    verb_count = len(role_assigner.verb_filler2idx)
    return {
        "subj": {
            "role_id": int(role_assigner.role2idx["subject"]),
            "label_offset": 0,
            "label_count": noun_count,
        },
        "verb": {
            "role_id": int(role_assigner.role2idx["verb"]),
            "label_offset": noun_count,
            "label_count": verb_count,
        },
        "obj": {
            "role_id": int(role_assigner.role2idx["object"]),
            "label_offset": 0,
            "label_count": noun_count,
        },
    }


def tensorize_role_split(dataset, role_info: dict[str, int], split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build hidden-state and remapped-label tensors for one role/split."""
    split = add_labels_for_role(dataset[split_name], role_info["role_id"])
    split = remap_role_slice(split, role_info["label_offset"], role_info["label_count"])
    hidden = torch.tensor(split["hidden_states"], dtype=torch.float32)
    labels = torch.tensor(split["labels"], dtype=torch.long)
    return hidden, labels


def output_inverse_from_svd(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    output_bias: torch.Tensor,
    lambda_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct Tikhonov output inverse weights from a cached SVD."""
    damped = s / (s.square() + float(lambda_value))
    w_inv = (vh.T * damped.unsqueeze(0)) @ u.T
    bias = -w_inv @ output_bias
    return w_inv.contiguous(), bias.contiguous()


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute argmax accuracy."""
    if labels.numel() == 0:
        return float("nan")
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def evaluate_grid(
    *,
    tpe,
    dataset,
    role_assigner,
    output_l2_values: tuple[float, ...],
    role_l2_values: tuple[float, ...],
    filler_l2_values: tuple[float, ...],
    selection_split: str,
    eval_split: str,
) -> dict[str, Any]:
    """Evaluate all analytic-construction variants for a model."""
    if tpe.output_layer is None:
        raise ValueError("Probe inversion tuning expects a TPE output_layer.")

    role_info = role_metadata(role_assigner)
    output_weight = tpe.output_layer.weight.detach().float().cpu()
    output_bias = tpe.output_layer.bias.detach().float().cpu()
    u, s, vh = torch.linalg.svd(output_weight, full_matrices=False)

    output_variants: dict[str, dict[str, Any]] = {}
    for lambda_value in output_l2_values:
        name = f"out_l2_{lambda_value:.0e}"
        w_inv, bias = output_inverse_from_svd(u, s, vh, output_bias, lambda_value)
        output_variants[name] = {
            "lambda": float(lambda_value),
            "w_inv": w_inv,
            "bias": bias,
        }

    unbinding_variants: list[dict[str, Any]] = [
        {
            "name": "role_norm__filler_norm",
            "role_unbinding": "norm",
            "role_pinv_regularization": "none",
            "role_pinv_l2_lambda": None,
            "filler_unbinding": "norm",
            "filler_pinv_regularization": "none",
            "filler_pinv_l2_lambda": None,
        }
    ]
    for role_l2 in role_l2_values:
        unbinding_variants.append(
            {
                "name": f"role_l2_{role_l2:.0e}__filler_norm",
                "role_unbinding": "pinv",
                "role_pinv_regularization": "l2",
                "role_pinv_l2_lambda": float(role_l2),
                "filler_unbinding": "norm",
                "filler_pinv_regularization": "none",
                "filler_pinv_l2_lambda": None,
            }
        )
    for role_l2 in role_l2_values:
        for filler_l2 in filler_l2_values:
            unbinding_variants.append(
                {
                    "name": f"role_l2_{role_l2:.0e}__filler_l2_{filler_l2:.0e}",
                    "role_unbinding": "pinv",
                    "role_pinv_regularization": "l2",
                    "role_pinv_l2_lambda": float(role_l2),
                    "filler_unbinding": "pinv",
                    "filler_pinv_regularization": "l2",
                    "filler_pinv_l2_lambda": float(filler_l2),
                }
            )

    selected: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for role_name in ROLE_NAMES:
        info = role_info[role_name]
        select_hidden, select_labels = tensorize_role_split(dataset, info, selection_split)
        eval_hidden, eval_labels = tensorize_role_split(dataset, info, eval_split)
        best_row: dict[str, Any] | None = None

        for output_name, output_variant in output_variants.items():
            select_tpr = select_hidden @ output_variant["w_inv"].T + output_variant["bias"]
            eval_tpr = eval_hidden @ output_variant["w_inv"].T + output_variant["bias"]

            for unbinding in unbinding_variants:
                w_probe = construct_unbinding_vectors(
                    tpencoder=tpe,
                    role_id=info["role_id"],
                    role_unbinding=unbinding["role_unbinding"],
                    filler_unbinding=unbinding["filler_unbinding"],
                    role_pinv_regularization=unbinding["role_pinv_regularization"],
                    role_pinv_l2_lambda=unbinding["role_pinv_l2_lambda"],
                    filler_pinv_regularization=unbinding["filler_pinv_regularization"],
                    filler_pinv_l2_lambda=unbinding["filler_pinv_l2_lambda"],
                    mode="classification",
                    device=torch.device("cpu"),
                ).float()
                label_offset = info["label_offset"]
                label_count = info["label_count"]
                w_probe = w_probe[label_offset : label_offset + label_count, :]

                select_logits = select_tpr @ w_probe.T
                eval_logits = eval_tpr @ w_probe.T
                select_acc = accuracy_from_logits(select_logits, select_labels)
                eval_acc = accuracy_from_logits(eval_logits, eval_labels)
                row = {
                    "role_name": role_name,
                    "selection_split": selection_split,
                    "eval_split": eval_split,
                    "selection_accuracy": select_acc,
                    "eval_accuracy": eval_acc,
                    "output_variant": output_name,
                    "output_l2_lambda": output_variant["lambda"],
                    **{k: v for k, v in unbinding.items() if k != "name"},
                    "unbinding_variant": unbinding["name"],
                }
                all_rows.append(row)
                if best_row is None or (select_acc, eval_acc) > (
                    best_row["selection_accuracy"],
                    best_row["eval_accuracy"],
                ):
                    best_row = row

        if best_row is None:
            raise RuntimeError(f"No probe tuning rows evaluated for role {role_name}.")
        selected[role_name] = best_row

    return {
        "selection_split": selection_split,
        "eval_split": eval_split,
        "output_l2_values": list(output_l2_values),
        "role_l2_values": list(role_l2_values),
        "filler_l2_values": list(filler_l2_values),
        "selected": selected,
        "rows": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tpe-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sentences-path", default="data/sentences")
    parser.add_argument("--embedding-cache-path", default="data/sentences")
    parser.add_argument("--output-l2-values", default=None)
    parser.add_argument("--role-l2-values", default=None)
    parser.add_argument("--filler-l2-values", default=None)
    parser.add_argument("--selection-split", default="valid")
    parser.add_argument("--eval-split", default="test")
    args = parser.parse_args()

    output_l2_values = parse_float_list(args.output_l2_values, DEFAULT_OUTPUT_L2)
    role_l2_values = parse_float_list(args.role_l2_values, DEFAULT_ROLE_L2)
    filler_l2_values = parse_float_list(args.filler_l2_values, DEFAULT_FILLER_L2)

    dataset, role_assigner = load_sentences(args.sentences_path, role_scheme="svo")
    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=args.sentences_path,
        embedding_model_name=args.model_name,
        embedding_cache_path=args.embedding_cache_path,
        embedding_column_name="target_embeddings",
        encoder_model_type="decoder-only-punct",
        device="cpu",
    )
    for split in dataset:
        if "target_embeddings" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("target_embeddings", "hidden_states")

    tpr_model = TensorProductEncoderForPretraining.from_pretrained(Path(args.tpe_path) / "best_model")
    tpe = tpr_model.encoder

    summary = evaluate_grid(
        tpe=tpe,
        dataset=dataset,
        role_assigner=role_assigner,
        output_l2_values=output_l2_values,
        role_l2_values=role_l2_values,
        filler_l2_values=filler_l2_values,
        selection_split=args.selection_split,
        eval_split=args.eval_split,
    )
    summary.update(
        {
            "model_slug": args.model_slug,
            "model_name": args.model_name,
            "tpe_path": args.tpe_path,
            "embedding_dim": int(embedding_dim),
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    for role_name, row in summary["selected"].items():
        print(
            f"{args.model_slug} {role_name}: "
            f"valid={row['selection_accuracy']:.4f} test={row['eval_accuracy']:.4f} "
            f"out={row['output_l2_lambda']:.0e} {row['unbinding_variant']}"
        )


if __name__ == "__main__":
    main()
