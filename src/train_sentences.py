import os
import gin
import json
import math
import copy 
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Callable, Tuple

import torch
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sentence_transformers import SentenceTransformer

from sentences import load_sentences
from paired_sentences import load_paired_sentences
from model import (
    TensorProductEncoderForPretraining, 
    TensorProductEncoderWithDecodingLoss, 
    TensorProductEncoderWithBackProjection,
    TensorProductEncoderConfig
)
from utils import (
    load_dataset_with_embeddings,
    parse_args_for_gin,
    gin_config_to_readable_dictionary,
    calculate_variance_explained,
    set_random_seed,
)
from probing import (
    LinearProbe,
    LinearProbeConfig,
    invert_output_layer,
    auto_select_role_pinv_l2_lambda,
    auto_select_tpe_output_l2_lambda,
)

try:
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
        torch._dynamo.config.suppress_errors = True
except (AttributeError, Exception):
    print("[INFO] torch._dynamo.config not found, skipping suppress_errors configuration.")

def compute_metrics_probe(pred: Any) -> dict:
    """Compute probe accuracy.

    Args:
        pred (Any): Trainer prediction output.

    Returns:
        dict: Dictionary with key ``score`` holding accuracy.
    """
    labels = pred.label_ids
    preds_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = preds_logits.argmax(-1)
    return {'score': (labels == preds).mean()}



# Added from train_sentences_probe.py
def add_labels_for_role(dataset_split: Any, role_id: int, allow_missing: bool = True) -> Any:
    """Adds a 'labels' column to the dataset_split, where each label is the filler corresponding to the given role_id.

    Args:
        dataset_split (Any): The dataset split to modify.
        role_id (int): The role ID to extract labels for.
        allow_missing (bool): If True, appends -1 for missing roles; if False, raises ValueError.

    Returns:
        Any: Modified dataset split with the added 'labels' column.
    """
    dataset_split_copy = copy.copy(dataset_split)
    if "labels" in dataset_split_copy.column_names:
        dataset_split_copy = dataset_split_copy.remove_columns("labels")

    # Get role_ids and filler_ids data (handle ragged arrays by processing individually)
    role_ids_data = dataset_split_copy['role_ids']
    filler_ids_data = dataset_split_copy['filler_ids']

    labels_list = []
    for i in range(len(role_ids_data)):
        found_role = False
        current_roles = role_ids_data[i]
        current_fillers = filler_ids_data[i]
        # Handle both list and numpy array inputs
        for j in range(len(current_roles)):
            if current_roles[j] == role_id:
                labels_list.append(current_fillers[j])
                found_role = True
                break
        if not found_role:
            if allow_missing:
                print(f"[WARNING] Role {role_id} not found in example {i}. Appending -1 as label.")
                labels_list.append(-1)
            else:
                raise ValueError(f"Role ID {role_id} not found in example {i} of the dataset split.")
    dataset_split_copy = dataset_split_copy.add_column("labels", np.array(labels_list))
    return dataset_split_copy

def tpe_data_collator(features):
    """Custom collator for training TPE with decoding loss"""
    
    batch = {}
    for k in features[0]:
        if k == 'sentence':  # Skip tensorizing string inputs
            batch[k] = [f[k] for f in features]
            continue
        # Handle nested lists (like filler_ids and role_ids)
        values = [f[k] for f in features]
        if isinstance(values[0], list):
            batch[k] = torch.tensor(values)
        elif isinstance(values[0], torch.Tensor):
            batch[k] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            batch[k] = torch.tensor(values)
        else:
            batch[k] = values
    
    # Add probe_labels if decoding loss is enabled
    if 'labels' in batch:
        batch['probe_labels'] = batch['labels'].clone().detach()
    else:
        raise ValueError("Expected 'labels' in batch for TPE decoding loss.")

    return batch

@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str, # Used for both TPE target and Probe input features
    embedding_cache_path: str,
    tpe_config: Dict[str, Any],             # TensorProductEncoderConfig kwargs
    tpe_training_args: TrainingArguments, # For TPE
    tpe_probe_config: Dict[str, Any] = {}, # For inverted TPE probe
    role_scheme: str = "svo", # Role scheme to use, can be "svo"/"bow"
    skip_tpe: bool = False, # If True, skips TPE training and only runs probe
    skip_trainable_probe: bool = False, # If True, skips probe training and evaluation
    skip_analytic_probe: bool = False, # If True, skips analytic probe evaluation
    role_for_probe: int = 0,
    probe_config_input: Optional[Dict[str, Any]] = None, # LinearProbeConfig kwargs
    probe_training_args: Optional[TrainingArguments] = None, # For trainable probe
    regularization_probe: str = 'l2',
    reg_param_probe: Optional[float] = None,
    role_unbinding: str = "pinv",
    role_pinv_regularization: str = "l2",
    role_pinv_l2_lambda: Optional[float] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    analytic_training_args: Optional[TrainingArguments] = None, # For analytic probe
    tpe_decoding_loss: bool = False,  # Decoding loss (probe classification) when training TPE
    tpe_back_projection: bool = False,  # Back-projection loss when training TPE
    use_wandb: bool = False, # Whether to use wandb for logging
    dataset_loader=load_sentences,  # Function used to load the dataset
    random_seed: Optional[int] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_param_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """
    Combined script to:
    1. Train a Tensor Product Encoder (TPE).
    2. Train a linear probe and compare with an analytically constructed probe.

    Args:
        skip_tpe (bool): If True, skips TPE training and only runs probe.
        skip_trained_probe (bool): If True, skips probe training and evaluation.
        skip_analytic_probe (bool): If True, skips analytic probe evaluation.
        sentences_path (str): Path to the sentences dataset.
        embedding_model_name (str): Name of the embedding model.
        embedding_cache_path (str): Path to the embedding cache.
        tpe_config (Dict[str, Any]): Configuration for TensorProductEncoder.
        tpe_training_args (TrainingArguments): Training arguments for TPE.
        role_for_probe (int, optional): Role ID for probing. Defaults to 0.
        probe_config_input (Optional[Dict[str, Any]], optional): Probe config. Defaults to None.
        probe_training_args (Optional[TrainingArguments], optional): Training args for probe. Defaults to None.
        regularization_probe (str, optional): Regularization type. Defaults to 'l2'.
        reg_param_probe (Optional[float], optional): Regularization parameter. Defaults to None.
        role_unbinding (str, optional): Role unbinding method. Defaults to 'pinv'.
        role_pinv_regularization (str, optional): Regularization for role pinv. Defaults to 'l2'.
        role_pinv_l2_lambda (Optional[float], optional): Role pinv l2 lambda. Defaults to None.
        role_pinv_atol (Optional[float], optional): Role pinv atol. Defaults to None.
        role_pinv_topk (Optional[int], optional): Role pinv top-k. Defaults to None.
        analytic_training_args (Optional[TrainingArguments], optional): Training args for analytic probe. Defaults to None.
        tpe_decoding_loss (bool, optional): Use TensorProductEncoderWithDecodingLoss. Defaults to False.
        tpe_back_projection (bool, optional): Use TensorProductEncoderWithBackProjection. Defaults to False.
        dataset_loader (callable): Function returning a ``DatasetDict`` and role assigner given ``sentences_path``.
        random_seed (int, optional): Seed used to make training deterministic.
        wandb_param_mapping (Optional[Dict[str, str]], optional): Mapping from W&B sweep keys to gin parameter names.
    Returns:
        None
    """

    if use_wandb:
        import wandb
        init_kwargs: Dict[str, Any] = {
            "project": wandb_project or "sentences_tpe",
            "entity": wandb_entity,
            "name": wandb_run_name,
            "group": wandb_group,
            "config": gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG),
        }
        if wandb_tags is not None:
            init_kwargs["tags"] = list(wandb_tags)
        # Strip None values so wandb.init ignores unspecified fields
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        wandb.init(**init_kwargs)

        if hasattr(tpe_training_args, "output_dir") and wandb.run is not None:
            base_output_dir = str(tpe_training_args.output_dir)
            run_suffix = wandb.run.id or wandb.run.name
            if run_suffix:
                tpe_training_args.output_dir = os.path.join(base_output_dir, f"run-{run_suffix}")
                os.makedirs(tpe_training_args.output_dir, exist_ok=True)
        
        # Load sweep parameters from wandb if wandb.config has entries
        if wandb.config:
            sweep_params = dict(wandb.config)
            param_map = dict(wandb_param_mapping or {})
            with gin.unlock_config():
                if sweep_params:
                    print(f"[INFO] Loading {len(sweep_params)} sweep parameters from wandb.config")
                    for param_name, param_value in sweep_params.items():
                        if param_name.startswith("_"):
                            continue
                        target_param = param_map.get(param_name, param_name)
                        print(f"[INFO] Binding gin parameter: {target_param} = {param_value}")
                        try:
                            if target_param.startswith("main.tpe_config."):
                                subkey = target_param.split(".", 2)[2]
                                current_cfg = dict(gin.query_parameter("main.tpe_config"))
                                current_cfg[subkey] = param_value
                                gin.bind_parameter("main.tpe_config", current_cfg)
                            else:
                                gin.bind_parameter(target_param, param_value)
                        except ValueError as exc:
                            print(f"[WARNING] Failed to bind gin parameter {target_param}: {exc}")
        
        # make sure run also logs to wandb
        tpe_training_args.report_to = "wandb"
    else:
        wandb = None

    if random_seed is not None:
        print(f"[INFO] Setting random seed to {random_seed}")
        set_random_seed(random_seed)
    else:
        # generate a random seed
        random_seed = np.random.randint(0, 1000)
        print(f"[INFO] No random seed provided, generated random seed {random_seed} for reproducibility")
        set_random_seed(random_seed)

    # === Part 1: Train or Load Tensor Product Encoder ===
    
    dataset, sentence_role_assigner = dataset_loader(sentences_path, role_scheme=role_scheme)

    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=sentences_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="target_embeddings",
        add_prefix="search_query: " if embedding_model_name.startswith("nomic-ai") else "",
    )

    # Detect dataset type and adjust probe behavior accordingly
    is_paired_sentences = hasattr(sentence_role_assigner, 'occupations')  # PairedRoleAssigner has 'occupations' attribute
    
    # Determine dataset type and validate role_for_probe
    if is_paired_sentences:
        print(f"[INFO] Detected paired sentences dataset with {len(sentence_role_assigner.occupations)} occupations and {len(sentence_role_assigner.verbs)} verbs.")
    else:
        print(f"[INFO] Detected regular sentences dataset with role scheme '{role_scheme}'.")
    
    # Validate and correct role_for_probe if needed
    available_roles = list(sentence_role_assigner.role2idx.values())
    if role_for_probe not in available_roles:
        dataset_type = "paired sentences" if is_paired_sentences else "sentences"
        raise ValueError(f"role_for_probe={role_for_probe} not valid for {dataset_type} dataset. Available roles: {available_roles}.")

    # Add role as labels for the specified role
    # Some examples might not have the specified role, so we use allow_missing=True
    for split_name in dataset:
        dataset[split_name] = add_labels_for_role(dataset[split_name], role_for_probe, allow_missing=True)

    # Model type selection (mutually exclusive) - used for both training and loading
    if tpe_decoding_loss and tpe_back_projection:
        raise ValueError("Cannot use both tpe_decoding_loss and tpe_back_projection simultaneously. Choose one.")
    
    if tpe_decoding_loss:
        tpe_model_class = TensorProductEncoderWithDecodingLoss
    elif tpe_back_projection:
        tpe_model_class = TensorProductEncoderWithBackProjection
        # Ensure has_linear_layer is True for back projection
        if not tpe_config.get("has_linear_layer", True):
            print("[WARNING] Setting has_linear_layer=True as required for TensorProductEncoderWithBackProjection")
            tpe_config["has_linear_layer"] = True
    else:
        tpe_model_class = TensorProductEncoderForPretraining

    
    eval_results_tpe_path = os.path.join(str(tpe_training_args.output_dir), "eval_results_tpe.json")
    if skip_tpe:
        print("\n[INFO] Skipping TPE training. Loading model from checkpoint...")
        best_tpe_model_dir = os.path.join(str(tpe_training_args.output_dir), "best_model")
        
        print(f"[INFO] Loading {tpe_model_class.__name__}")
        tpe_model_instance = tpe_model_class.from_pretrained(best_tpe_model_dir)
    else:
        print("\n[INFO] Part 1: Training Tensor Product Encoder...")
        print("Dataset loaded and processed.")

        # Set up TPE configuration
        tpe_config["hidden_size"] = embedding_dim
        if is_paired_sentences:
            # For paired sentences: fillers are occupations, roles are verbs
            tpe_config["n_fillers"] = len(sentence_role_assigner.occupations)
            tpe_config["n_roles"] = len(sentence_role_assigner.verbs)
        else:
            # For regular sentences: original logic
            tpe_config["n_fillers"] = len(sentence_role_assigner.noun_filler2idx) + len(sentence_role_assigner.verb_filler2idx) + 1
            tpe_config["n_roles"] = len(sentence_role_assigner.role2idx)
        
        print(f"[INFO] Using {tpe_model_class.__name__}")
        tpe_model_instance = tpe_model_class(TensorProductEncoderConfig(**tpe_config))

        tpe_trainer = Trainer(
            model=tpe_model_instance,
            args=tpe_training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=tpe_data_collator if tpe_decoding_loss else None,
        )

        print("Starting TPE training...")
        tpe_trainer.train()
        print("TPE training finished.")
        
        best_tpe_model_dir = os.path.join(str(tpe_training_args.output_dir), "best_model")
        os.makedirs(str(best_tpe_model_dir), exist_ok=True)
        tpe_trainer.model.save_pretrained(best_tpe_model_dir)
        print(f"Saved best TPE model to {best_tpe_model_dir}")
        gin_config_path_tpe = os.path.join(str(tpe_training_args.output_dir), "config.gin")
        with open(gin_config_path_tpe, "w") as f:
            f.write(gin.operative_config_str())
        print(f"Saved TPE part gin config to {gin_config_path_tpe}")

        # Calculate variance explained metrics for TPE
        for split_name, split_data in [("Train", dataset["train"]), ("Valid", dataset["valid"]), ("Test", dataset["test"])]:
            metrics = calculate_variance_explained(tpe_model_instance, split_data)
            if split_name == "Valid":
                eval_results_tpe = tpe_trainer.evaluate()
                if "eval_loss" in eval_results_tpe:
                    metrics["eval_loss"] = float(eval_results_tpe["eval_loss"])
                with open(eval_results_tpe_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                if wandb is not None:
                    wandb.log({
                        "tpe_explained_variance_ratio": metrics["Explained_Variance_Ratio"],
                        "tpe_eval_loss": metrics.get("eval_loss", float('nan')),
                    })
            print(f"\nTPE Variance Explained Metrics ({split_name.title()} Set):")
            for metric_name, value in metrics.items(): 
                print(f"{metric_name}: {value:.4f}")

    # === Part 2: Train and Evaluate Linear Probe ===

    print("\n[INFO] Part 2: Training and Evaluating Linear Probe...")

    for split_name in dataset:
        if "target_embeddings" in dataset[split_name].column_names:
            dataset[split_name] = dataset[split_name].rename_column("target_embeddings", "hidden_states")
    print("Probe Dataset prepared from loaded embeddings.")

    # Set up evaluation datasets based on dataset type
    if is_paired_sentences:
        # For simplified paired sentences, we can evaluate on both roles since they're always [0, 1]
        print(f"[INFO] Setting up evaluation for both roles in paired sentences.")
        test_role0_dataset = add_labels_for_role(dataset['test'], 0, allow_missing=True)  # First part of pair
        test_role1_dataset = add_labels_for_role(dataset['test'], 1, allow_missing=True)  # Second part of pair
        eval_datasets = {
            'first_part': test_role0_dataset.filter(lambda x: x['labels'] != -1),
            'second_part': test_role1_dataset.filter(lambda x: x['labels'] != -1)
        }
    else:
        # For regular sentences, use the original subject/object evaluation
        if hasattr(sentence_role_assigner, 'role2idx') and 'object' in sentence_role_assigner.role2idx and 'subject' in sentence_role_assigner.role2idx:
            test_obj_dataset = add_labels_for_role(dataset['test'], sentence_role_assigner.role2idx['object'], allow_missing=True)
            test_subj_dataset = add_labels_for_role(dataset['test'], sentence_role_assigner.role2idx['subject'], allow_missing=True)
            eval_datasets = {
                'subj': test_subj_dataset.filter(lambda x: x['labels'] != -1), 
                'obj': test_obj_dataset.filter(lambda x: x['labels'] != -1)
            }
        else:
            # Fallback for BOW or other schemes
            test_dataset_for_role = add_labels_for_role(dataset['test'], role_for_probe, allow_missing=True)
            eval_datasets = {
                f'role_{role_for_probe}': test_dataset_for_role.filter(lambda x: x['labels'] != -1)
            }

    # --- Analytic probe logic (inversion of TPE) ---
    analytic_results = {}
    reg_param_objective_used: Optional[str] = None
    if not skip_analytic_probe:
        print("\nSetting up analytic probe...")
        loaded_tpe_pretrain_model = TensorProductEncoderForPretraining.from_pretrained(best_tpe_model_dir)
        tpe_encoder_for_analytic_probe = loaded_tpe_pretrain_model.encoder
        if role_unbinding == "pinv" and role_pinv_regularization == "l2" and role_pinv_l2_lambda is None:
            batch_subset = dataset["train"].select(range(min(128, len(dataset["train"]))))
            filler_ids_tensor = torch.tensor(batch_subset["filler_ids"], dtype=torch.long)
            role_ids_tensor = torch.tensor(batch_subset["role_ids"], dtype=torch.long)
            reg_lambda, best_value, (log_lo, log_hi) = auto_select_role_pinv_l2_lambda(
                tpe_encoder_for_analytic_probe,
                filler_ids=filler_ids_tensor,
                role_ids=role_ids_tensor,
                device=tpe_encoder_for_analytic_probe.filler_embedding.weight.device,
            )
            role_pinv_l2_lambda = float(reg_lambda)
            print(
                f"[INFO] Auto-selected role unbinding l2 ≈ {role_pinv_l2_lambda:.5g} "
                f"(val_mse≈{best_value:.4e}; window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
            )
        if reg_param_probe is None:
            first_eval_dataset = next(iter(eval_datasets.values()))
            probe_batch_size = min(128, len(first_eval_dataset))
            batch_subset = first_eval_dataset.select(range(probe_batch_size))

            hidden_states_batch = torch.tensor(batch_subset["hidden_states"], dtype=torch.float32) if "hidden_states" in batch_subset.column_names else None
            filler_ids_batch = torch.tensor(batch_subset["filler_ids"], dtype=torch.long) if "filler_ids" in batch_subset.column_names else None
            role_ids_batch = torch.tensor(batch_subset["role_ids"], dtype=torch.long) if "role_ids" in batch_subset.column_names else None

            device = tpe_encoder_for_analytic_probe.filler_embedding.weight.device
            target_hidden = hidden_states_batch.to(device)
            filler_ids_tensor = filler_ids_batch.to(device)
            role_ids_tensor = role_ids_batch.to(device)

            reg_param_objective_used = "mse"
            reg_lambda, best_value, (log_lo, log_hi) = auto_select_tpe_output_l2_lambda(
                tpe_encoder_for_analytic_probe,
                target_hidden,
                filler_ids_tensor,
                role_ids_tensor,
                device=device,
            )
            reg_param_probe = reg_lambda
            detail_tokens: List[str] = []
            if math.isfinite(best_value):
                detail_tokens.append(f"val_mse≈{best_value:.4e}")
            detail_tokens.append(f"search window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}]")
            print(f"[INFO] Auto-selected reg_param ≈ {reg_param_probe:.5g} (objective={reg_param_objective_used}; {', '.join(detail_tokens)})")
        else:
            reg_param_objective_used = "configured"
            print(f"[INFO] Using config-provided reg_param = {reg_param_probe}")

        analytic_probe = LinearProbe.from_tpencoder(
            tpencoder=tpe_encoder_for_analytic_probe,
            encoder=None,
            role_id=role_for_probe,
            regularization=regularization_probe if regularization_probe in ('l2', 'atol', 'topk') else 'l2',
            l2_lambda=reg_param_probe if regularization_probe == 'l2' else None,
            atol=reg_param_probe if regularization_probe == 'atol' else None,
            topk=int(reg_param_probe) if regularization_probe == 'topk' and reg_param_probe is not None and not np.isnan(reg_param_probe) else None,
            role_unbinding=role_unbinding,
            role_pinv_regularization=role_pinv_regularization,
            role_pinv_l2_lambda=role_pinv_l2_lambda,
            role_pinv_atol=role_pinv_atol,
            role_pinv_topk=role_pinv_topk,
            embedding_model_name=embedding_model_name,
            **tpe_probe_config
        )
        analytic_trainer = Trainer(
            model=analytic_probe,
            args=analytic_training_args,
            eval_dataset=eval_datasets,
            compute_metrics=compute_metrics_probe
        )
        print("Evaluating analytic probe...")
        analytic_results = analytic_trainer.evaluate()
        
        # Log results for each evaluation dataset
        for eval_name in eval_datasets.keys():
            score_key = f'eval_{eval_name}_score'
            score_value = analytic_results.get(score_key, float('nan'))
            print(f"[RESULT] Analytic probe score ({eval_name}): {score_value:.4f}")

        if wandb is not None:
            log_dict = {}
            analytic_scores = []
            for eval_name in eval_datasets.keys():
                score_key = f'eval_{eval_name}_score'
                score_value = analytic_results.get(score_key, float('nan'))
                log_dict[f"analytic_probe_{eval_name}_score"] = score_value
                if isinstance(score_value, (int, float)) and not np.isnan(score_value):
                    analytic_scores.append(float(score_value))
            if analytic_scores:
                log_dict["analytic_probe_average_score"] = float(np.mean(analytic_scores))
            wandb.log(log_dict)
    else:
        print("[INFO] Skipping analytic probe evaluation.")

    # --- Trainable probe logic ---
    current_probe_config_dict = dict(probe_config_input) if probe_config_input is not None else {}
    current_probe_config_dict.setdefault('encoder_model_type', 'sentence-transformers') 
    
    # Set num_labels based on dataset type
    if is_paired_sentences:
        current_probe_config_dict['num_labels'] = len(sentence_role_assigner.occupations)
    else:
        current_probe_config_dict['num_labels'] = len(sentence_role_assigner.noun_filler2idx)
        
    current_probe_config_dict['encoder_hidden_size'] = embedding_dim
    trainable_probe = LinearProbe(LinearProbeConfig(**current_probe_config_dict), None)

    trained_results = {}
    if not skip_trainable_probe:
        probe_trainer = Trainer(
            model=trainable_probe,
            args=probe_training_args,
            train_dataset=dataset['train'].filter(lambda x: x['labels'] != -1),
            eval_dataset=eval_datasets,
            compute_metrics=compute_metrics_probe
        )
        print("Starting trainable probe training...")
        probe_trainer.train()
        print("Trainable probe training finished.")
        trained_results = probe_trainer.evaluate()
        
        # Log results for each evaluation dataset
        for eval_name in eval_datasets.keys():
            score_key = f'eval_{eval_name}_score'
            score_value = trained_results.get(score_key, float('nan'))
            print(f"[RESULT] Trained probe score ({eval_name}): {score_value:.4f}")
    else:
        print("[INFO] Skipping trainable probe training and evaluation.")

    # --- Save Combined Probe Results ---
    probe_results_path = os.path.join(str(tpe_training_args.output_dir), f"probe_results_role{role_for_probe}.json")
    final_probe_results = {
        'role_probed': role_for_probe,
        'dataset_type': 'paired_sentences' if is_paired_sentences else 'regular_sentences',
        'reg_param_used': reg_param_probe if reg_param_probe is not None else float('nan'),
        'regularization_type': regularization_probe,
    }
    if reg_param_objective_used is not None:
        final_probe_results['reg_param_objective_used'] = reg_param_objective_used
    
    # Add results for each evaluation dataset
    for eval_name in eval_datasets.keys():
        final_probe_results[f'trained_probe_{eval_name}_score'] = trained_results.get(f'eval_{eval_name}_score', float('nan'))
        final_probe_results[f'analytic_probe_{eval_name}_score'] = analytic_results.get(f'eval_{eval_name}_score', float('nan'))
    
    if analytic_results:
        analytic_scores = []
        for eval_name in eval_datasets.keys():
            score_key = f'eval_{eval_name}_score'
            score_value = analytic_results.get(score_key, float('nan'))
            if isinstance(score_value, (int, float)) and not np.isnan(score_value):
                analytic_scores.append(float(score_value))
        if analytic_scores:
            final_probe_results['analytic_probe_average_score'] = float(np.mean(analytic_scores))

    with open(probe_results_path, 'w') as f:
        json.dump(final_probe_results, f, indent=4)
    print("\n[INFO] Done! Key results saved in TPE checkpoint directory.")

if __name__ == "__main__":
    gin.external_configurable(TrainingArguments, module="transformers")
    gin.external_configurable(load_paired_sentences, module="paired_sentences")
    gin.external_configurable(load_sentences, module="sentences")
    parse_args_for_gin()
    main()
