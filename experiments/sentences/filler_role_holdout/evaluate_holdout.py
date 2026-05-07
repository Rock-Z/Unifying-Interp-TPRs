"""Evaluate SVO filler-role holdout checkpoints on test and generalization splits.

Usage:
    uv run experiments/sentences/filler_role_holdout/evaluate_holdout.py \
        experiments/sentences/filler_role_holdout/configs/evaluate_modernbert.gin
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import gin
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from invert_svo import add_labels_for_role, collate_embeddings, compute_metrics, remap_labels_for_role
from model import TensorProductEncoderForPretraining
from probing import LinearProbe, LinearProbeConfig, auto_select_role_pinv_l2_lambda
from sae import compute_r2
from sentences import load_sentences
from utils import load_dataset_with_embeddings, parse_args_for_gin


def _load_holdout_metadata(sentences_path: str) -> Optional[dict]:
    """Load dataset holdout metadata when present."""

    metadata_path = os.path.join(sentences_path, "data.holdout_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def _load_reg_param_from_tpe_dir(tpe_path: str) -> float:
    """Reuse the existing sentence probe rule: reg_param = sqrt(eval_loss)."""

    eval_json_candidates = [
        os.path.join(tpe_path, "eval_results_tpe.json"),
        os.path.join(tpe_path, "eval_results.json"),
    ]
    eval_loss_val = None
    for path in eval_json_candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        if "eval_loss" in data:
            eval_loss_val = float(data["eval_loss"])
            break
    if eval_loss_val is None:
        raise FileNotFoundError(
            "Could not infer reg_param from eval_results_tpe.json or eval_results.json."
        )
    return float(max(0.0, eval_loss_val) ** 0.5)


def _collate_reconstruction(batch: list[dict]) -> dict:
    """Tensorize filler/role ids and target embeddings for TPE evaluation."""

    def _stack(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(values, dim=0)
        return torch.tensor(values)

    return {
        key: [row[key] for row in batch] if key == "sentence" else _stack([row[key] for row in batch])
        for key in batch[0].keys()
    }


def _compute_reconstruction_metrics(
    model: TensorProductEncoderForPretraining,
    dataset_split,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Compare TPE predictions against cached target embeddings on one split."""

    dataloader = DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_reconstruction,
    )

    all_targets = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                filler_ids=batch["filler_ids"].to(device),
                role_ids=batch["role_ids"].to(device),
            )
            predictions = outputs.encoder_hidden_states.reshape(outputs.encoder_hidden_states.shape[0], -1)
            targets = batch["target_embeddings"].to(device).reshape(batch["target_embeddings"].shape[0], -1)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    targets = torch.cat(all_targets, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    target_mean = targets.mean(dim=0, keepdim=True)
    ss_total = torch.sum((targets - target_mean) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    ss_regression = torch.sum((predictions - target_mean) ** 2)

    return {
        "mse": float(F.mse_loss(predictions, targets).item()),
        "r2": float(compute_r2(targets, predictions)),
        "cosine_similarity": float(F.cosine_similarity(predictions, targets, dim=1).mean().item()),
        "ss_total": float(ss_total.item()),
        "ss_residual": float(ss_residual.item()),
        "ss_regression": float(ss_regression.item()),
        "explained_variance_ratio": float((ss_regression / ss_total).item()) if float(ss_total.item()) != 0.0 else float("nan"),
    }


def _build_eval_datasets(hidden_dataset, role_assigner, roles_to_eval: Sequence[str], split_names: Sequence[str]):
    """Prepare train/eval splits for subject, verb, and object probes."""

    name_to_roleidx = {}
    if "subject" in role_assigner.role2idx:
        name_to_roleidx["subj"] = role_assigner.role2idx["subject"]
    if "verb" in role_assigner.role2idx:
        name_to_roleidx["verb"] = role_assigner.role2idx["verb"]
    if "object" in role_assigner.role2idx:
        name_to_roleidx["obj"] = role_assigner.role2idx["object"]

    noun_count = len(role_assigner.noun_filler2idx)
    verb_count = len(role_assigner.verb_filler2idx)
    results = {}
    for role_name in roles_to_eval:
        if role_name not in name_to_roleidx:
            continue
        role_id = name_to_roleidx[role_name]
        label_offset = noun_count if role_name == "verb" else 0
        label_count = verb_count if role_name == "verb" else noun_count

        train_ds = add_labels_for_role(hidden_dataset["train"], role_id)
        train_ds = train_ds.filter(lambda x: x["labels"] != -1)
        train_ds = remap_labels_for_role(train_ds, label_offset, label_count)

        eval_splits = {}
        for split_name in split_names:
            split_ds = add_labels_for_role(hidden_dataset[split_name], role_id)
            split_ds = split_ds.filter(lambda x: x["labels"] != -1)
            eval_splits[split_name] = remap_labels_for_role(split_ds, label_offset, label_count)

        results[role_name] = {
            "role_id": role_id,
            "label_offset": label_offset,
            "label_count": label_count,
            "train": train_ds,
            "eval_splits": eval_splits,
        }
    return results


def _build_analytic_probe(
    tpe,
    *,
    role_id: int,
    label_offset: int,
    label_count: int,
    regularization: str,
    reg_param: Optional[float],
    role_unbinding: str,
    role_pinv_regularization: str,
    role_pinv_l2_lambda: Optional[float],
    role_pinv_atol: Optional[float],
    role_pinv_topk: Optional[int],
    embedding_model_name: str,
):
    """Construct the analytic probe and slice it down to the role-specific label set."""

    analytic_probe = LinearProbe.from_tpencoder(
        tpencoder=tpe,
        encoder=None,
        role_id=role_id,
        regularization=regularization if regularization in ("l2", "atol", "topk") else "l2",
        l2_lambda=reg_param if regularization == "l2" else None,
        atol=reg_param if regularization == "atol" else None,
        topk=int(reg_param) if regularization == "topk" and reg_param is not None else None,
        role_unbinding=role_unbinding,
        role_pinv_regularization=role_pinv_regularization,
        role_pinv_l2_lambda=role_pinv_l2_lambda,
        role_pinv_atol=role_pinv_atol,
        role_pinv_topk=role_pinv_topk,
        embedding_model_name=embedding_model_name,
    )
    classifier = analytic_probe.classifier[-1]
    max_label = label_offset + label_count
    if max_label > classifier.out_features:
        raise ValueError("Requested label slice exceeds probe output dimension.")

    new_layer = torch.nn.Linear(
        classifier.in_features,
        label_count,
        dtype=classifier.weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.copy_(classifier.weight[label_offset:max_label, :])
        new_layer.bias.copy_(classifier.bias[label_offset:max_label])
    analytic_probe.classifier[-1] = new_layer
    analytic_probe.config.num_labels = label_count
    return analytic_probe


def _evaluate_probe_on_splits(
    model,
    eval_splits: Dict[str, object],
    training_args: TrainingArguments,
) -> Dict[str, float]:
    """Evaluate one probe model on all requested splits."""

    metrics = {}
    for split_name, eval_ds in eval_splits.items():
        eval_args = copy.deepcopy(training_args)
        eval_args.output_dir = os.path.join(str(training_args.output_dir), split_name)
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_ds,
            data_collator=collate_embeddings,
            compute_metrics=compute_metrics,
        )
        results = trainer.evaluate()
        metrics[split_name] = float(results.get("eval_accuracy", float("nan")))
    return metrics


@gin.configurable("holdout_eval")
def main(
    sentences_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str],
    tpe_path: str,
    split_names: Sequence[str] = ("test", "generalization"),
    roles_to_eval: Sequence[str] = ("subj", "verb", "obj"),
    regularization: str = "l2",
    reg_param: Optional[float] = None,
    role_unbinding: str = "pinv",
    role_pinv_regularization: str = "l2",
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    results_output_path: Optional[str] = None,
    probe_results_output_path: Optional[str] = None,
    trained_probe_cache_path: Optional[str] = None,
    force_trainable_probe_retrain: bool = False,
    *,
    analytic_training_args: TrainingArguments,
    trainable_training_args: TrainingArguments,
):
    """Evaluate reconstruction and split-aware probes for the SVO holdout experiment."""

    dataset, role_assigner = load_sentences(sentences_path, role_scheme="svo")
    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(f"Dataset split '{split_name}' not found. Available: {list(dataset.keys())}")

    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=sentences_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="target_embeddings",
        add_prefix="search_query: " if embedding_model_name.startswith("nomic-ai") else "",
    )

    tpr_model = TensorProductEncoderForPretraining.from_pretrained(
        os.path.join(tpe_path, "best_model")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tpr_model = tpr_model.to(device)
    try:
        tpe = tpr_model.encoder
    except Exception:
        tpe = tpr_model

    if reg_param is None:
        reg_param = _load_reg_param_from_tpe_dir(tpe_path)

    if role_unbinding == "pinv" and role_pinv_regularization == "l2" and role_pinv_l2_lambda is None:
        batch_subset = dataset["train"].select(range(min(128, len(dataset["train"]))))
        filler_ids_tensor = torch.tensor(batch_subset["filler_ids"], dtype=torch.long).to(device)
        role_ids_tensor = torch.tensor(batch_subset["role_ids"], dtype=torch.long).to(device)
        reg_lambda, _, _ = auto_select_role_pinv_l2_lambda(
            tpe,
            filler_ids=filler_ids_tensor,
            role_ids=role_ids_tensor,
            device=device,
        )
        role_pinv_l2_lambda = float(reg_lambda)

    reconstruction_results = {}
    for split_name in split_names:
        reconstruction_results[split_name] = {
            "num_examples": len(dataset[split_name]),
            **_compute_reconstruction_metrics(
                tpr_model,
                dataset[split_name],
                batch_size=int(analytic_training_args.per_device_eval_batch_size),
                device=device,
            ),
        }

    hidden_dataset = dataset
    for split_name in hidden_dataset:
        if "target_embeddings" in hidden_dataset[split_name].column_names:
            hidden_dataset[split_name] = hidden_dataset[split_name].rename_column(
                "target_embeddings", "hidden_states"
            )

    role_datasets = _build_eval_datasets(hidden_dataset, role_assigner, roles_to_eval, split_names)

    if trained_probe_cache_path is None:
        trained_probe_cache_path = os.path.join(tpe_path, "trained_probe_results_svo_holdout.json")
    trained_cache = {}
    if not force_trainable_probe_retrain and os.path.exists(trained_probe_cache_path):
        try:
            with open(trained_probe_cache_path, "r") as f:
                trained_cache = json.load(f)
        except Exception:
            trained_cache = {}

    probe_results = []
    for role_name, role_info in role_datasets.items():
        analytic_probe = _build_analytic_probe(
            tpe,
            role_id=role_info["role_id"],
            label_offset=role_info["label_offset"],
            label_count=role_info["label_count"],
            regularization=regularization,
            reg_param=reg_param,
            role_unbinding=role_unbinding,
            role_pinv_regularization=role_pinv_regularization,
            role_pinv_l2_lambda=role_pinv_l2_lambda,
            role_pinv_atol=role_pinv_atol,
            role_pinv_topk=role_pinv_topk,
            embedding_model_name=embedding_model_name,
        )
        analytic_args = copy.deepcopy(analytic_training_args)
        analytic_args.output_dir = os.path.join(str(analytic_training_args.output_dir), role_name)
        analytic_scores = _evaluate_probe_on_splits(
            analytic_probe,
            role_info["eval_splits"],
            analytic_args,
        )

        cached_scores = trained_cache.get(role_name)
        if (
            not force_trainable_probe_retrain
            and isinstance(cached_scores, dict)
            and all(split_name in cached_scores for split_name in split_names)
        ):
            trained_scores = {split_name: float(cached_scores[split_name]) for split_name in split_names}
        else:
            trainable_probe = LinearProbe(
                LinearProbeConfig(
                    encoder_model_type="sentence-transformers",
                    encoder_hidden_size=embedding_dim,
                    num_labels=role_info["label_count"],
                ),
                None,
            )
            trainable_args = copy.deepcopy(trainable_training_args)
            trainable_args.output_dir = os.path.join(str(trainable_training_args.output_dir), role_name)
            trainer = Trainer(
                model=trainable_probe,
                args=trainable_args,
                train_dataset=role_info["train"],
                eval_dataset=next(iter(role_info["eval_splits"].values())),
                data_collator=collate_embeddings,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trained_scores = _evaluate_probe_on_splits(
                trainable_probe,
                role_info["eval_splits"],
                trainable_args,
            )
            trained_cache[role_name] = trained_scores
            with open(trained_probe_cache_path, "w") as f:
                json.dump(trained_cache, f, indent=2)

        probe_results.append(
            {
                "role_name": role_name,
                "analytic_accuracy": analytic_scores,
                "trained_accuracy": trained_scores,
            }
        )

    results = {
        "sentences_path": sentences_path,
        "embedding_model_name": embedding_model_name,
        "tpe_path": tpe_path,
        "split_names": list(split_names),
        "holdout_metadata": _load_holdout_metadata(sentences_path),
        "reconstruction": reconstruction_results,
        "probe_results": probe_results,
    }
    probe_results_payload = {
        "splits": list(split_names),
        "results": probe_results,
    }

    if results_output_path is None:
        results_output_path = os.path.join(tpe_path, "holdout_eval_metrics.json")
    if probe_results_output_path is None:
        probe_results_output_path = os.path.join(tpe_path, "probe_compare_results_svo.json")

    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    with open(results_output_path, "w") as f:
        json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(probe_results_output_path), exist_ok=True)
    with open(probe_results_output_path, "w") as f:
        json.dump(probe_results_payload, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {results_output_path}")
    print(f"Saved split-aware probe metrics to {probe_results_output_path}")


if __name__ == "__main__":
    gin.external_configurable(TrainingArguments)
    parse_args_for_gin()
    main()
