"""Evaluate fixed-length digits holdout experiments on eval and generalization splits.

Usage:
    uv run experiments/digits/filler_role_holdout/evaluate_holdout.py \
        experiments/digits/filler_role_holdout/configs/copy_rnn.gin
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import gin
import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits import load_digits, tokenize_function
from model import RecurrentEncoderDecoderModel, TensorProductEncoderForPretraining
from sae import compute_r2
from utils import parse_args_for_gin


def compute_metrics(eval_pred, tokenizer):
    """Compute token and sequence accuracy for autoregressive predictions."""
    predictions, labels = eval_pred
    predictions = predictions[:, 1:]
    labels = labels[:, :-1]
    accuracy = (
        (predictions == labels) & (labels != tokenizer.pad_token_id)
    ).sum() / (labels != tokenizer.pad_token_id).sum()
    correct_sequences = (
        ((predictions != labels) & (labels != tokenizer.pad_token_id)).sum(axis=1) == 0
    )
    return {
        "token_accuracy": float(accuracy),
        "sequence_accuracy": float(correct_sequences.mean()),
    }


@gin.configurable
def seq2seq_init(tokenizer=None, encoder_config=None, decoder_config=None):
    """Placeholder to satisfy shared gin includes during evaluation."""
    del tokenizer, encoder_config, decoder_config
    return None


@gin.configurable
def evaluate_tpe(tpe_training_args=None):
    """Placeholder to expose eval args through gin."""
    return tpe_training_args


def _flatten_hidden_state(hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Flatten recurrent hidden states to `(batch, features)`."""
    if isinstance(hidden_state, tuple):
        hidden_state = torch.cat([hidden_state[0], hidden_state[1]], dim=-1)
    return hidden_state.reshape(hidden_state.shape[0], -1)


def _load_holdout_metadata(data_paths_dict: dict[str, str]) -> Optional[dict]:
    """Load dataset holdout metadata when available."""
    train_path = data_paths_dict.get("train")
    if not train_path:
        return None
    prefix, suffix = os.path.splitext(train_path)
    if suffix != ".train":
        return None
    metadata_path = prefix + ".holdout_metadata.json"
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def _normalize_metrics(metrics: dict, prefix: str) -> dict:
    """Strip a split prefix from trainer metrics for easier JSON output."""
    normalized = {}
    prefix_with_sep = f"{prefix}_"
    for key, value in metrics.items():
        if key.startswith(prefix_with_sep):
            normalized[key[len(prefix_with_sep):]] = value
        else:
            normalized[key] = value
    return normalized


def _evaluate_substitution_accuracy(
    model: RecurrentEncoderDecoderModel,
    dataset_split,
    tokenizer,
    eval_args: Seq2SeqTrainingArguments,
    split_name: str,
    *,
    role_scheme: Optional[str] = None,
) -> dict:
    """Run autoregressive substitution evaluation on one dataset split."""
    if role_scheme is None:
        collator = lambda examples: tokenize_function(examples, tokenizer)
    else:
        collator = lambda examples: tokenize_function(
            examples,
            tokenizer,
            format="tpe_eval",
            role_scheme=role_scheme,
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=copy.deepcopy(eval_args),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        eval_dataset=dataset_split,
        data_collator=collator,
    )
    metrics = trainer.evaluate(metric_key_prefix=split_name)
    return _normalize_metrics(metrics, split_name)


def _compute_reconstruction_metrics(
    dataset_split,
    tokenizer,
    seq2seq_model: RecurrentEncoderDecoderModel,
    tpencoder,
    *,
    batch_size: int,
    role_scheme: str,
    device: torch.device,
) -> dict:
    """Compute reconstruction metrics between frozen seq2seq and TPE embeddings."""
    dataloader = DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda examples: tokenize_function(
            examples,
            tokenizer,
            format="tpe",
            role_scheme=role_scheme,
        ),
    )

    all_targets = []
    all_predictions = []

    seq2seq_model.eval()
    tpencoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            target_hidden = seq2seq_model.encoder(
                input_ids=batch["embedding_model_input_ids"].to(device),
                input_lengths=batch["embedding_model_input_lengths"].to(device),
            ).last_hidden_state
            predicted_hidden = tpencoder(
                filler_ids=batch["filler_ids"].to(device),
                role_ids=batch["role_ids"].to(device),
            ).last_hidden_state

            all_targets.append(_flatten_hidden_state(target_hidden).cpu())
            all_predictions.append(_flatten_hidden_state(predicted_hidden).cpu())

    targets = torch.cat(all_targets, dim=0)
    predictions = torch.cat(all_predictions, dim=0)

    return {
        "mse": float(torch.nn.functional.mse_loss(predictions, targets).item()),
        "r2": float(compute_r2(targets, predictions)),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(predictions, targets, dim=1).mean().item()
        ),
    }


@gin.configurable
def main(
    data_paths_dict,
    seq2seq_training_args: Seq2SeqTrainingArguments,
    tpe_training_args: TrainingArguments,
    split_names: Sequence[str] = ("test", "generalization"),
    results_output_path: Optional[str] = None,
    **unused_kwargs,
):
    """Evaluate seq2seq and TPE models on configured splits."""
    dataset, tokenizer = load_digits(file_paths=data_paths_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq2seq_model = RecurrentEncoderDecoderModel.from_pretrained(
        os.path.join(seq2seq_training_args.output_dir, "best_model")
    ).to(device)

    tpe_model = TensorProductEncoderForPretraining.from_pretrained(
        os.path.join(tpe_training_args.output_dir, "best_model")
    )
    tpencoder = tpe_model.encoder.to(device)
    role_scheme = getattr(tpencoder.config, "role_scheme", None)
    if role_scheme is None:
        raise ValueError("role_scheme must be specified in the TPE config")

    tpr_encoder_decoder = RecurrentEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder=tpencoder,
        decoder=seq2seq_model.decoder,
    ).to(device)

    eval_args = gin.get_bindings(evaluate_tpe).get("tpe_training_args")
    if not isinstance(eval_args, Seq2SeqTrainingArguments):
        raise TypeError("evaluate_tpe.tpe_training_args must resolve to Seq2SeqTrainingArguments")

    results = {
        "seq2seq_checkpoint_dir": os.path.join(seq2seq_training_args.output_dir, "best_model"),
        "tpe_checkpoint_dir": os.path.join(tpe_training_args.output_dir, "best_model"),
        "split_names": list(split_names),
        "holdout_metadata": _load_holdout_metadata(data_paths_dict),
        "splits": {},
    }

    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(f"Dataset split '{split_name}' not found. Available: {list(dataset.keys())}")
        split = dataset[split_name]
        results["splits"][split_name] = {
            "num_examples": len(split),
            "seq2seq_substitution": _evaluate_substitution_accuracy(
                seq2seq_model,
                split,
                tokenizer,
                eval_args,
                split_name,
            ),
            "tpe_substitution": _evaluate_substitution_accuracy(
                tpr_encoder_decoder,
                split,
                tokenizer,
                eval_args,
                split_name,
                role_scheme=role_scheme,
            ),
            "reconstruction": _compute_reconstruction_metrics(
                split,
                tokenizer,
                seq2seq_model,
                tpencoder,
                batch_size=int(eval_args.per_device_eval_batch_size),
                role_scheme=role_scheme,
                device=device,
            ),
        }

    output_path = results_output_path or os.path.join(
        tpe_training_args.output_dir,
        "holdout_eval_metrics.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {output_path}")


if __name__ == "__main__":
    gin.external_configurable(Seq2SeqTrainingArguments, module="transformers")
    gin.external_configurable(TrainingArguments, module="transformers")
    parse_args_for_gin()
    main()
