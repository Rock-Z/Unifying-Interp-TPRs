import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gin
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import TensorProductEncoderConfig, TensorProductEncoderForPretraining  # noqa: E402
from sentences import (
    SVORoleAssigner,
    build_active_passive_prompts_from_svo,
    filter_words_single_token,
    load_sentences,
)  # noqa: E402
from utils import (  # noqa: E402
    calculate_variance_explained,
    gin_config_to_readable_dictionary,
    load_dataset_with_embeddings,
    parse_args_for_gin,
    set_random_seed,
)
from vocabulary import PASSIVE_PARTICIPLES  # noqa: E402


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


def _collate(batch):
    filler_ids = torch.tensor([ex["filler_ids"] for ex in batch], dtype=torch.long)
    role_ids = torch.tensor([ex["role_ids"] for ex in batch], dtype=torch.long)
    targets = torch.tensor([ex["target_embeddings"] for ex in batch], dtype=torch.float32)
    return {"filler_ids": filler_ids, "role_ids": role_ids, "target_embeddings": targets}


@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str],
    tpe_config: Dict[str, Any],
    tpe_training_args: TrainingArguments,
    layer_indices: Sequence[int],
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

    if len(base_dataset["train"]) == 0:
        raise ValueError("No training examples available after filtering.")

    decoder_config = AutoConfig.from_pretrained(embedding_model_name, trust_remote_code=True)
    hidden_size = int(getattr(decoder_config, "hidden_size"))
    num_layers = int(
        getattr(decoder_config, "num_hidden_layers", getattr(decoder_config, "n_layer", 0))
    )
    if num_layers <= 0:
        raise ValueError("Could not determine decoder layer count from the model config.")

    base_output_dir = str(tpe_training_args.output_dir)
    layer_indices = list(layer_indices)
    for layer_id in layer_indices:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(f"Invalid layer index {layer_id}; must be in [0, {num_layers}).")

        layer_dataset = copy.deepcopy(base_dataset)
        layer_dataset, _ = load_dataset_with_embeddings(
            dataset=layer_dataset,
            dataset_path=sentences_path,
            embedding_model_name=embedding_model_name,
            embedding_cache_path=embedding_cache_path,
            embedding_column_name="target_embeddings",
            encoder_model_type="decoder-only-full",
            device=None,
            decoder_layer_indices=[int(layer_id)],
        )

        sample_flat = np.asarray(layer_dataset["train"][0]["target_embeddings"], dtype=np.float32)
        flat_dim = int(sample_flat.size)
        seq_len = flat_dim // hidden_size
        if seq_len * hidden_size != flat_dim:
            raise ValueError("Cached embeddings do not align with decoder hidden size.")

        cfg = dict(tpe_config)
        cfg["hidden_size"] = seq_len * hidden_size
        cfg["n_fillers"] = len(role_assigner.noun_filler2idx) + len(role_assigner.verb_filler2idx) + 1
        cfg["n_roles"] = len(role_assigner.role2idx)
        cfg["layer_id"] = int(layer_id)
        cfg["target_sequence_length"] = int(seq_len)
        cfg["per_token_hidden_size"] = int(hidden_size)
        cfg.setdefault("has_linear_layer", True)

        model = TensorProductEncoderForPretraining(TensorProductEncoderConfig(**cfg))

        layer_args = copy.deepcopy(tpe_training_args)
        layer_output_dir = os.path.join(base_output_dir, f"layer{layer_id}")
        layer_args.output_dir = layer_output_dir
        layer_args.label_names = ["target_embeddings", "filler_ids", "role_ids"]
        if use_wandb:
            layer_args.report_to = "wandb"

        os.makedirs(layer_output_dir, exist_ok=True)

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
            if wandb.run is not None:
                base_dir = layer_output_dir
                run_suffix = wandb.run.id or wandb.run.name
                if run_suffix:
                    layer_output_dir = os.path.join(base_dir, f"run-{run_suffix}")
                    os.makedirs(layer_output_dir, exist_ok=True)
                    layer_args.output_dir = layer_output_dir

        trainer = Trainer(
            model=model,
            args=layer_args,
            train_dataset=layer_dataset["train"],
            eval_dataset=layer_dataset["valid"],
            data_collator=_collate,
            callbacks=[EarlyStoppingCallback()],
        )

        print(f"[INFO] Training layer {layer_id} TPE on {len(layer_dataset['train'])} examples")
        trainer.train()

        best_model_dir = os.path.join(layer_output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.model.save_pretrained(best_model_dir)

        gin_config_path = os.path.join(layer_output_dir, "config.gin")
        with open(gin_config_path, "w") as f:
            f.write(gin.operative_config_str())

        metrics = calculate_variance_explained(trainer.model, layer_dataset["test"])
        metrics_path = os.path.join(layer_output_dir, "eval_results_tpe.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics for layer {layer_id} to {metrics_path}")
        if use_wandb:
            import wandb

            wandb.log({f"layer{layer_id}_explained_variance": metrics.get("Explained_Variance_Ratio", float('nan'))})
            wandb.finish()


if __name__ == "__main__":
    gin.external_configurable(TrainingArguments, module="transformers")
    gin.external_configurable(EarlyStoppingCallback, module="transformers")
    parse_args_for_gin()
    main()
