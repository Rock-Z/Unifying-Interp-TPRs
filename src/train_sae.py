import os
import json
import numpy as np
from typing import Dict, Any, Optional, Literal

import gin
import torch
from torch.utils.data import DataLoader
from transformers.training_args import TrainingArguments
from transformers import AutoModel

from sae import (
    SparseAutoencoder,
    SAEConfig,
    SAETrainer,
    DeadLatentResampler,
    sae_trainer_embedding_collator,
    evaluate_sae,
    SupervisedSparseAutoencoder,
    TopKSparseAutoencoder,
    TopKSAEConfig,
)
from utils import parse_args_for_gin, set_random_seed, load_dataset_with_embeddings

# Sentences imports
from sentences import load_sentences

# Digits imports
from digits import load_digits, tokenize_function, get_roles
from model import RecurrentEncoderDecoderModel


def _load_seq2seq_encoder(seq2seq_checkpoint_dir: str):
    """Best-effort loader for encoder from a seq2seq checkpoint directory."""
    candidates = [seq2seq_checkpoint_dir.rstrip("/"), os.path.join(seq2seq_checkpoint_dir.rstrip("/"), "best_model")]
    last_error = None
    for cand in candidates:
        try:
            model = RecurrentEncoderDecoderModel.from_pretrained(cand)
            return model.get_encoder()
        except Exception as e:
            last_error = e
            # Try to directly load an encoder checkpoint
            try:
                return AutoModel.from_pretrained(cand)
            except Exception:
                enc_sub = os.path.join(cand, "encoder")
                try:
                    return AutoModel.from_pretrained(enc_sub)
                except Exception as e3:
                    last_error = e3
                    continue
    raise RuntimeError(f"Failed to load encoder from '{seq2seq_checkpoint_dir}': {last_error}")


@gin.configurable
def main(
    dataset_type: Literal["sentences", "digits"],
    training_args: TrainingArguments,
    # Sentences branch
    sentences_path: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    embedding_cache_path: Optional[str] = None,
    role_scheme: str = "svo",
    dataset_loader=load_sentences,
    # Digits branch
    data_paths_dict: Optional[Dict[str, str]] = None,
    seq2seq_checkpoint_dir: Optional[str] = None,
    # SAE config and extras
    sae_config: Dict[str, Any] = {},
    random_seed: Optional[int] = None,
    resample_threshold: float = 1.0,
    resample_times: int = 0,
    dead_latent_threshold: float = 1e-6,
    supervised: bool = False,
    feature_mode: Literal["filler", "filler_role"] = "filler",
) -> Dict[str, Any]:
    """Unified entry point to train SAE on sentences or digits embeddings."""
    if random_seed is not None:
        set_random_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_type == "sentences":
        if sentences_path is None or embedding_model_name is None:
            raise ValueError("sentences_path and embedding_model_name are required for dataset_type='sentences'")
        dataset, role_assigner = dataset_loader(sentences_path, role_scheme=role_scheme)
        dataset, emb_dim = load_dataset_with_embeddings(
            dataset=dataset,
            dataset_path=sentences_path,
            embedding_model_name=embedding_model_name,
            embedding_cache_path=embedding_cache_path,
        )
        input_dim = emb_dim
    elif dataset_type == "digits":
        if data_paths_dict is None or seq2seq_checkpoint_dir is None:
            raise ValueError("data_paths_dict and seq2seq_checkpoint_dir are required for dataset_type='digits'")
        dataset, tokenizer = load_digits(file_paths=data_paths_dict)
        encoder = _load_seq2seq_encoder(seq2seq_checkpoint_dir).to(device).eval()

        @torch.no_grad()
        def add_target_embeddings(batch: Dict[str, Any]) -> Dict[str, Any]:
            inputs = batch["input"]
            tokenized_input = tokenizer(
                inputs,
                padding="longest",
                return_token_type_ids=False,
                return_tensors="pt",
            )
            filler_ids, role_ids = get_roles(
                tokenized_input["input_ids"],
                tokenized_input["attention_mask"].clone(),
                role_scheme=role_scheme,
            )
            input_ids = tokenized_input["input_ids"].to(device)
            input_lengths = torch.sum(tokenized_input["attention_mask"], dim=1).to(device)
            enc_out = encoder(input_ids=input_ids, input_lengths=input_lengths).last_hidden_state

            if isinstance(enc_out, tuple):
                h, c = enc_out
                enc_vec = torch.cat([h[:, -1, :], c[:, -1, :]], dim=-1)
            else:
                enc_vec = enc_out[:, -1, :] if enc_out.dim() == 3 else enc_out
            return {
                "target_embeddings": enc_vec.cpu().numpy().tolist(),
                "filler_ids": filler_ids.cpu().tolist(),
                "role_ids": role_ids.cpu().tolist(),
            }

        for split in list(dataset.keys()):
            dataset[split] = dataset[split].map(
                add_target_embeddings,
                batched=True,
                batch_size=256,
            )
        # infer input dim from a sample
        input_dim = len(dataset["train"][0]["target_embeddings"]) if len(dataset["train"]) > 0 else None
        if input_dim is None:
            raise ValueError("Empty training split; cannot infer input_dim")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # If given supervision of activations, also track the feature map in the config
    idx_to_feature = {}
    if supervised:
        feature_set = set()
        for split in dataset.keys():
            for fs, rs in zip(dataset[split]["filler_ids"], dataset[split]["role_ids"]):
                for f, r in zip(fs, rs):
                    feature_set.add(f if feature_mode == "filler" else (f, r))
        feature_to_idx = {feat: i for i, feat in enumerate(sorted(feature_set))}

        for feat, idx in feature_to_idx.items():
            if feature_mode == "filler":
                idx_to_feature[idx] = {"filler_id": feat}
            else:
                idx_to_feature[idx] = {
                    "filler_id": feat[0],
                    "role_id": feat[1],
                }

        def add_labels(example: Dict[str, Any]) -> Dict[str, Any]:
            labels = [0.0] * len(feature_to_idx)
            for f, r in zip(example["filler_ids"], example["role_ids"]):
                key = f if feature_mode == "filler" else (f, r)
                labels[feature_to_idx[key]] = 1.0
            example["feature_labels"] = labels
            return example

        for split in dataset.keys():
            dataset[split] = dataset[split].map(add_labels)

        # set the hidden dim to the number of features
        sae_config.setdefault("hidden_dim", len(feature_to_idx))
        assert (
            sae_config["hidden_dim"] == len(feature_to_idx)
        ), (
            "Supervised SAE requires hidden_dim to match the number of supervision features; "
            f"got hidden_dim={sae_config['hidden_dim']} but {len(feature_to_idx)} feature labels."
        )
        # attach feature map to config so it is saved with the model
        sae_config.setdefault("feature_map", {str(k): v for k, v in idx_to_feature.items()})

    # Determine the type of SAE to use
    sae_config.setdefault("input_dim", input_dim)
    sae_config.setdefault("dead_latent_threshold", dead_latent_threshold)
    # Store metadata about embedding model and role scheme
    if dataset_type == "sentences":
        if embedding_model_name is not None:
            sae_config.setdefault("embedding_model_name", embedding_model_name)
    elif dataset_type == "digits":
        # For digits, use seq2seq checkpoint as embedding model identifier
        if seq2seq_checkpoint_dir is not None:
            sae_config.setdefault("embedding_model_name", seq2seq_checkpoint_dir)
        elif embedding_model_name is not None:
            # Fallback to embedding_model_name if provided
            sae_config.setdefault("embedding_model_name", embedding_model_name)
    sae_config.setdefault("role_scheme", role_scheme)
    # Store feature_map_scheme based on feature_mode
    sae_config.setdefault("feature_map_scheme", feature_mode)
    if sae_config.get("k") is not None:
        print("[INFO] 'k' is set in SAE config, using TopKSparseAutoencoder")
        top_k = sae_config.pop("k")
        model = TopKSparseAutoencoder(TopKSAEConfig(k=top_k, **sae_config))
    else:
        if supervised:
            print("[INFO] instantiating SupervisedSparseAutoencoder")
            model = SupervisedSparseAutoencoder(SAEConfig(**sae_config))
        else:
            print("[INFO] instantiating SparseAutoencoder")
            model = SparseAutoencoder(SAEConfig(**sae_config))

    def compute_metrics(eval_pred):
        """Aggregate auxiliary losses from model outputs during eval."""
        preds = eval_pred.predictions
        if preds is None:
            return {}
        if not isinstance(preds, (tuple, list)):
            preds = (preds,)

        # With ignored inference keys, predictions only include loss scalars.
        if len(preds) == 2:
            names = ("reconstruction_loss", "sparsity_loss")
        elif len(preds) == 3:
            names = ("reconstruction_loss", "sparsity_loss", "supervision_loss")
        else:
            return {}

        metrics = {name: float(np.mean(values)) for name, values in zip(names, preds)}
        activation_threshold = getattr(model.config, "activation_threshold", None)
        if activation_threshold is not None:
            metrics["activation_threshold"] = float(activation_threshold)
        return metrics

    trainer = SAETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=sae_trainer_embedding_collator,
        callbacks=[DeadLatentResampler(threshold=resample_threshold, resample_times=resample_times)],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    # Also collect auxiliary losses on the test split, if available.
    test_aux_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    eval_metrics.update(test_aux_metrics)

    output_dir = training_args.output_dir or ("./checkpoints/sae_sentences" if dataset_type == "sentences" else "./checkpoints/sae_digits")
    trainer.save_model(os.path.join(output_dir, "best_model"))
    print(f"Saved SAE model to {os.path.join(output_dir, 'best_model')}")
    
    # Evaluate on test split with comprehensive metrics
    test_split = dataset["test"]
    if "filler_ids" in test_split.column_names and "role_ids" in test_split.column_names:
        eval_batch_size = (
            getattr(training_args, "per_device_eval_batch_size", None)
            or getattr(training_args, "per_device_train_batch_size", None)
            or 32
        )
        # Collect forward-pass tensors here so evaluation stays metric-only.
        eval_loader = DataLoader(
            test_split, batch_size=eval_batch_size, collate_fn=sae_trainer_embedding_collator
        )
        label_loader = DataLoader(
            dataset["train"], batch_size=eval_batch_size, collate_fn=sae_trainer_embedding_collator
        )
        eval_embeddings = []
        eval_reconstructions = []
        eval_activations = []
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch["inputs_embeds"].to(device)
                encoded = model.encode(inputs)
                decoded = model.decode(encoded)
                eval_embeddings.append(inputs.detach().cpu())
                eval_reconstructions.append(decoded.detach().cpu())
                eval_activations.append(encoded.detach().cpu())
        label_activations = []
        with torch.no_grad():
            for batch in label_loader:
                inputs = batch["inputs_embeds"].to(device)
                encoded = model.encode(inputs)
                label_activations.append(encoded.detach().cpu())
        eval_embeddings = torch.cat(eval_embeddings, dim=0)
        eval_reconstructions = torch.cat(eval_reconstructions, dim=0)
        eval_activations = torch.cat(eval_activations, dim=0)
        label_activations = torch.cat(label_activations, dim=0)
        test_metrics = evaluate_sae(
            sae=model,
            label_dataset=dataset["train"],
            eval_dataset=test_split,
            eval_embeddings=eval_embeddings,
            eval_reconstructions=eval_reconstructions,
            eval_activations=eval_activations,
            label_activations=label_activations,
            sae_output_dir=None,  # Don't save here, we'll save merged metrics below
            label_mode=feature_mode,
        )
        # Merge test metrics into eval_metrics
        eval_metrics.update(test_metrics)
    
    out_dir = os.path.join(output_dir, "sae_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    try:
        import wandb

        if wandb.run is not None:
            # Emit final eval/test metrics so sweeps can optimize them.
            wandb.log(eval_metrics, commit=True)
    except Exception:
        pass
    print(eval_metrics)
    return eval_metrics


if __name__ == "__main__":
    gin.external_configurable(TrainingArguments, module="transformers")
    gin.external_configurable(load_sentences, module="sentences")
    gin.external_configurable(load_digits, module="digits")
    gin.external_configurable(tokenize_function, module="digits")
    parse_args_for_gin()
    main()  # type: ignore[call-arg]
