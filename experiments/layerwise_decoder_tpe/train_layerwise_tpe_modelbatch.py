import copy
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gin
import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.utils.generic import ModelOutput

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import TensorProductEncoderConfig, TensorProductEncoderForPretraining  # noqa: E402
from sentences import (  # noqa: E402
    SVORoleAssigner,
    build_active_passive_prompts_from_svo,
    filter_words_single_token,
    load_sentences,
)
from utils import (  # noqa: E402
    calculate_variance_explained,
    gin_config_to_readable_dictionary,
    parse_args_for_gin,
    set_random_seed,
    load_dataset_with_embeddings,
)
from vocabulary import PASSIVE_PARTICIPLES  # noqa: E402
from modelbatch.huggingface_integration import HFModelBatch  # noqa: E402


def _load_vocab(path: Path) -> tuple[list[str], list[str]]:
    nouns = (path / "data.nouns").read_text().splitlines()
    verbs = (path / "data.verbs").read_text().splitlines()
    verbs = [v for v in verbs if v in PASSIVE_PARTICIPLES]
    return nouns, verbs


def _build_active_passive_dataset(
    sentences_path: str,
    embedding_model_name: str,
    role_scheme: str,
    seed: int,
    max_examples_per_split: Optional[int],
) -> tuple[DatasetDict, SVORoleAssigner]:
    base = Path(sentences_path)
    nouns, verbs = _load_vocab(base)

    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    nouns = filter_words_single_token(tokenizer, nouns)
    dataset, assigner = load_sentences(sentences_path, role_scheme=role_scheme)
    assigner.nouns_sg = nouns  # keep filtered nouns
    dataset = build_active_passive_prompts_from_svo(
        dataset=dataset,
        role_assigner=assigner,
        tokenizer=tokenizer,
        max_examples_per_split=max_examples_per_split,
    )
    return dataset, assigner


def _collate_factory(num_layers: int, flat_dim: int):
    def _collate(batch):
        filler_ids = torch.stack([ex["filler_ids"] for ex in batch]).long()
        role_ids = torch.stack([ex["role_ids"] for ex in batch]).long()
        targets = torch.stack([ex["target_embeddings"] for ex in batch], dim=0)
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)
        if targets.shape[1:] != (num_layers, flat_dim):
            raise ValueError(f"Expected targets shape (batch, {num_layers}, {flat_dim}), got {targets.shape}")
        targets = targets.transpose(0, 1).contiguous()
        return {
            "filler_ids": filler_ids,
            "role_ids": role_ids,
            "target_embeddings_per_model": targets,
        }

    return _collate


class LayerwiseDataset(Dataset):
    """Torch dataset that slices flattened per-sentence embeddings into per-layer targets."""

    def __init__(
        self,
        split_ds,
        base_layer_indices: Sequence[int],
        flat_dim: int,
        selected_layers: Optional[Sequence[int]] = None,
        embedding_column_prefix: str = "target_embeddings",
    ):
        self.split_ds = split_ds
        self.base_layer_indices = list(base_layer_indices)
        self.flat_dim = int(flat_dim)
        self.selected_layers = list(selected_layers) if selected_layers is not None else self.base_layer_indices
        self.layer_pos = {layer: idx for idx, layer in enumerate(self.base_layer_indices)}
        self.embedding_column_prefix = embedding_column_prefix

    def __len__(self) -> int:
        return len(self.split_ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.split_ds[int(idx)]
        per_layer = []
        for layer in self.selected_layers:
            col_name = f"{self.embedding_column_prefix}_layer{layer}"
            vec = np.asarray(row[col_name], dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != self.flat_dim:
                vec = vec.reshape(-1)
            per_layer.append(vec)
        target = torch.tensor(np.stack(per_layer, axis=0), dtype=torch.float32)
        return {
            "filler_ids": torch.tensor(row["filler_ids"], dtype=torch.long),
            "role_ids": torch.tensor(row["role_ids"], dtype=torch.long),
            "target_embeddings": target,
        }


class LayerwiseHFModelBatch(HFModelBatch):
    """HFModelBatch wrapper that feeds per-model targets and tracks per-model loss."""

    def __init__(self, models: list[TensorProductEncoderForPretraining]):
        super().__init__(models, shared_input=True)
        # Keep stacked params trainable so optimizer updates the shared storage.
        self.compute_loss_inside_forward = True

    def forward(
        self,
        target_embeddings_per_model: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        kwargs.pop("num_items_in_batch", None)

        outputs = []
        losses = []
        hidden_states = []

        for idx, model in enumerate(self.models):
            model_kwargs = dict(kwargs)
            if target_embeddings_per_model is not None:
                model_kwargs["target_embeddings"] = target_embeddings_per_model[idx]
            out = model(**model_kwargs)
            outputs.append(out)
            if getattr(out, "loss", None) is not None:
                losses.append(out.loss)
            if getattr(out, "encoder_hidden_states", None) is not None:
                hidden_states.append(out.encoder_hidden_states)

        stacked_hidden = torch.stack(hidden_states) if hidden_states else None
        loss = None
        if losses:
            per_model_losses = torch.stack(losses)
            # Sum (instead of mean) so each model's gradients match standalone training.
            loss = per_model_losses.sum()
            self.latest_losses = per_model_losses.detach()

        return ModelOutput(loss=loss, encoder_hidden_states=stacked_hidden)


def _compute_eval_loss(model: TensorProductEncoderForPretraining, dataset: Dataset, batch_size: int = 64) -> float:
    """Compute mean eval loss for a single layer dataset."""

    def _collate(batch):
        return {
            "filler_ids": torch.stack([row["filler_ids"] for row in batch], dim=0),
            "role_ids": torch.stack([row["role_ids"] for row in batch], dim=0),
            "target_embeddings": torch.stack([row["target_embeddings"] for row in batch], dim=0),
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            filler_ids = batch["filler_ids"].to(model.device)
            role_ids = batch["role_ids"].to(model.device)
            targets = batch["target_embeddings"].to(model.device)
            if targets.ndim == 3 and targets.shape[1] == 1:
                targets = targets[:, 0, :]
            outputs = model(
                filler_ids=filler_ids,
                role_ids=role_ids,
                target_embeddings=targets,
            )
            if outputs.loss is None:
                continue
            total_loss += float(outputs.loss.item()) * filler_ids.shape[0]
            total_items += filler_ids.shape[0]
    return total_loss / total_items if total_items > 0 else float("nan")


@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str],
    tpe_config: Dict[str, Any],
    tpe_training_args: TrainingArguments,
    layer_indices: Sequence[int] | str = "all",
    role_scheme: str = "svo",
    random_seed: int = 0,
    max_examples_per_split: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
) -> None:
    set_random_seed(random_seed)

    base_dataset, role_assigner = _build_active_passive_dataset(
        sentences_path=sentences_path,
        embedding_model_name=embedding_model_name,
        role_scheme=role_scheme,
        seed=random_seed,
        max_examples_per_split=max_examples_per_split,
    )
    print("[DEBUG] Built base dataset")

    if len(base_dataset["train"]) == 0:
        raise ValueError("No training examples available after filtering.")

    decoder_config = AutoConfig.from_pretrained(embedding_model_name, trust_remote_code=True)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_layers = int(
        getattr(decoder_config, "num_hidden_layers", getattr(decoder_config, "n_layer", 0))
    )
    if num_layers <= 0:
        raise ValueError("Could not determine decoder layer count from the model config.")
    print(f"[DEBUG] Loaded decoder config with {num_layers} layers, hidden size {hidden_size}")

    if isinstance(layer_indices, str):
        if layer_indices == "all":
            layer_indices = list(range(num_layers))
        else:
            # Accept bracketed lists from gin (e.g., "[0,1,2]")
            layer_indices = ast.literal_eval(layer_indices)
    if layer_indices == "all" or layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = list(layer_indices)
    if not layer_indices:
        raise ValueError("At least one layer index must be provided.")

    for layer_id in layer_indices:
        if int(layer_id) < 0 or int(layer_id) >= num_layers:
            raise ValueError(f"Invalid layer index {layer_id}; must be in [0, {num_layers}).")

    # Compute/load concatenated embeddings for all requested layers in one pass
    base_dataset, total_emb_dim = load_dataset_with_embeddings(
        base_dataset,
        sentences_path,
        embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="target_embeddings",
        encoder_model_type="decoder-only-full",
        decoder_layer_indices=layer_indices,
        create_combined_column=False,
    )
    embedding_cols = [f"target_embeddings_layer{layer}" for layer in layer_indices]
    for split in base_dataset:
        base_dataset[split] = base_dataset[split].with_format(
            type="numpy",
            columns=["filler_ids", "role_ids"] + embedding_cols,
            output_all_columns=True,
        )

    if total_emb_dim % len(layer_indices) != 0:
        raise ValueError(f"Embedding dim {total_emb_dim} not divisible by {len(layer_indices)} layers")
    flat_dim = int(total_emb_dim // len(layer_indices))
    seq_len = flat_dim // hidden_size
    if seq_len * hidden_size != flat_dim:
        raise ValueError("Cached embeddings do not align with decoder hidden size.")

    train_dataset = LayerwiseDataset(base_dataset["train"], layer_indices, flat_dim)
    eval_dataset = LayerwiseDataset(base_dataset["valid"], layer_indices, flat_dim)
    test_dataset = LayerwiseDataset(base_dataset["test"], layer_indices, flat_dim)

    models: list[TensorProductEncoderForPretraining] = []
    for layer_id in layer_indices:
        cfg = dict(tpe_config)
        cfg["hidden_size"] = flat_dim
        cfg["n_fillers"] = len(role_assigner.noun_filler2idx) + len(role_assigner.verb_filler2idx) + 1
        cfg["n_roles"] = len(role_assigner.role2idx)
        cfg["layer_id"] = int(layer_id)
        cfg["target_sequence_length"] = int(seq_len)
        cfg["per_token_hidden_size"] = int(hidden_size)
        cfg.setdefault("has_linear_layer", True)
        models.append(TensorProductEncoderForPretraining(TensorProductEncoderConfig(**cfg)))
    print(f"[DEBUG] Instantiated {len(models)} TPE models")

    model_batch = LayerwiseHFModelBatch(models)
    print("[DEBUG] Wrapped models in LayerwiseHFModelBatch")

    batch_training_args = copy.deepcopy(tpe_training_args)
    base_output_dir = str(batch_training_args.output_dir)
    batch_training_args.output_dir = base_output_dir
    batch_training_args.save_strategy = "no"
    batch_training_args.load_best_model_at_end = False
    batch_training_args.save_total_limit = 0
    batch_training_args.remove_unused_columns = False
    batch_training_args.label_names = ["target_embeddings_per_model"]
    if use_wandb:
        batch_training_args.report_to = "wandb"
    else:
        batch_training_args.report_to = []

    if use_wandb:
        import wandb

        init_kwargs: Dict[str, Any] = {
            "project": wandb_project or "layerwise_tpe",
            "entity": wandb_entity,
            "name": wandb_run_name,
            "group": wandb_group,
            "tags": list(wandb_tags) if wandb_tags is not None else None,
            "config": gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG),
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        wandb.init(**init_kwargs)

    trainer = Trainer(
        model=model_batch,
        args=batch_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collate_factory(len(layer_indices), flat_dim),
    )

    print(f"[INFO] Training {len(layer_indices)} layers in a single ModelBatch run")
    trainer.train()

    gin_config_str = gin.operative_config_str()
    metrics_by_layer: dict[int, dict[str, float]] = {}

    for model, layer_id in zip(model_batch.models, layer_indices):
        print(f"[INFO] Saving artifacts for layer {layer_id}", flush=True)
        layer_output_dir = os.path.join(base_output_dir, f"layer{layer_id}")
        os.makedirs(layer_output_dir, exist_ok=True)

        best_model_dir = os.path.join(layer_output_dir, "best_model")
        model.save_pretrained(best_model_dir)

        gin_config_path = os.path.join(layer_output_dir, "config.gin")
        with open(gin_config_path, "w") as f:
            f.write(gin_config_str)

        layer_test_dataset = LayerwiseDataset(
            base_dataset["test"],
            layer_indices,
            flat_dim,
            selected_layers=[layer_id],
        )
        try:
            metrics = calculate_variance_explained(model, layer_test_dataset)
        except Exception:
            import traceback

            traceback.print_exc()
            metrics = {"Explained_Variance_Ratio": float("nan")}
        metrics["eval_loss"] = _compute_eval_loss(model, layer_test_dataset)
        metrics_by_layer[int(layer_id)] = metrics
        metrics_path = os.path.join(layer_output_dir, "eval_results_tpe.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics for layer {layer_id} to {metrics_path}", flush=True)

    if use_wandb:
        import wandb
        eval_losses = [m.get("eval_loss") for m in metrics_by_layer.values() if m.get("eval_loss") is not None]
        eval_loss_sum = float(np.nansum(eval_losses)) if eval_losses else float("nan")
        eval_loss_mean = float(np.nanmean(eval_losses)) if eval_losses else float("nan")
        wandb.log(
            {
                "eval_loss_sum": eval_loss_sum,
                "eval_loss_mean": eval_loss_mean,
                **{
                    f"layer{lid}_explained_variance": m.get("Explained_Variance_Ratio", float("nan"))
                    for lid, m in metrics_by_layer.items()
                },
                **{
                    f"layer{lid}_eval_loss": m.get("eval_loss", float("nan"))
                    for lid, m in metrics_by_layer.items()
                },
            }
        )
        wandb.finish()

    eval_losses = [m.get("eval_loss") for m in metrics_by_layer.values() if m.get("eval_loss") is not None]
    summary = {
        "eval_loss_sum": float(np.nansum(eval_losses)) if eval_losses else float("nan"),
        "eval_loss_mean": float(np.nanmean(eval_losses)) if eval_losses else float("nan"),
    }
    summary_path = os.path.join(base_output_dir, "eval_results_tpe_modelbatch.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved modelbatch summary metrics to {summary_path}", flush=True)


if __name__ == "__main__":
    gin.external_configurable(TrainingArguments, module="transformers")
    gin.external_configurable(
        AutoModelForCausalLM.from_pretrained,
        name="AutoModelForCausalLM.from_pretrained",
        module="transformers",
    )
    parse_args_for_gin()
    main()
