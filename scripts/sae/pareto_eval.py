#!/usr/bin/env python3
"""Script to evaluate SAE pareto frontier by sweeping activation thresholds.

This script loads an SAE checkpoint and evaluates the pareto frontier by:
1. Sweeping activation thresholds from 0 up to a high-percentile activation (default 99.9th)
2. For each threshold, refitting the decoder on train split
3. Computing the mean number of active units and reconstruction loss on the test split
4. Plotting mean active units vs reconstruction loss

Example:
    uv run scripts/sae/pareto_eval.py --checkpoint checkpoints/sae/tpr_sae_modernbert
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

# Script directory for locating eval.py
_HERE = os.path.abspath(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import importlib.util

from sae import SparseAutoencoder
from sentences import load_sentences
from digits import load_digits, tokenize_function
from utils import load_dataset_with_embeddings

# Import shared functions and constants from eval.py
# Use importlib to avoid conflict with built-in eval()
eval_module_path = os.path.join(_HERE, "eval.py")
spec = importlib.util.spec_from_file_location("eval_module", eval_module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load eval.py from {eval_module_path}")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

infer_dataset_type = eval_module.infer_dataset_type
resolve_checkpoint_path = eval_module.resolve_checkpoint_path
SENTENCES_DATA_PATH = eval_module.SENTENCES_DATA_PATH
DIGITS_DATA_PATH_BASE = eval_module.DIGITS_DATA_PATH_BASE

DEFAULT_THRESHOLD_PERCENTILE = 99.9
PLOT_MAX_MEAN_ACTIVE_UNITS = 4.0
TARGET_MAX_MEAN_ACTIVE_UNITS = 4.0


def _eval_collate(batch):
    embeddings = torch.tensor([ex["target_embeddings"] for ex in batch], dtype=torch.float32)
    return {"target_embeddings": embeddings}


def encode_with_threshold(sae: SparseAutoencoder, x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Encode input with activation threshold applied.
    
    Args:
        sae: SparseAutoencoder model
        x: Input tensor [B, input_dim]
        threshold: Activation threshold (activations below this are zeroed)
        
    Returns:
        Encoded features [B, hidden_dim] with threshold applied
    """
    encoded = sae.encoder(x)
    encoded = torch.relu(encoded)
    # Apply threshold: zero out activations below threshold
    encoded = torch.where(encoded > threshold, encoded, torch.zeros_like(encoded))
    return encoded


def compute_activation_threshold_cap(
    sae: SparseAutoencoder,
    loaders: Iterable[DataLoader],
    device: torch.device,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
) -> Tuple[float, float]:
    """Compute a percentile-based activation cap (and record the absolute max).

    Args:
        sae: SparseAutoencoder model
        loaders: Iterable of dataloaders whose activations will be analysed
        device: Device to run on
        percentile: Activation percentile (0-100] used as the sweep upper bound

    Returns:
        Tuple (percentile_value, max_activation) where percentile_value is the
        requested percentile across all activations and max_activation is the
        observed absolute maximum. If no activations are found, both are 0.0.
    """
    sae.eval()
    max_activation = 0.0
    activation_batches: List[torch.Tensor] = []

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                h = batch["target_embeddings"].to(device)
                encoded = sae.encode(h)
                if encoded.numel() == 0:
                    continue
                max_activation = max(max_activation, encoded.max().item())
                activation_batches.append(encoded.detach().cpu().reshape(-1))

    if not activation_batches:
        return 0.0, max_activation

    all_activations = torch.cat(activation_batches).to(torch.float32)
    quantile = max(0.0, min(1.0, percentile / 100.0))
    percentile_value = float(torch.quantile(all_activations, quantile).item())
    return percentile_value, max_activation


def refit_decoder(
    sae: SparseAutoencoder,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device
) -> nn.Linear:
    """Refit decoder layer using least squares regression.
    
    Args:
        sae: SparseAutoencoder model (for architecture reference)
        features: Feature activations [N, hidden_dim]
        targets: Target embeddings [N, input_dim]
        device: Device to run on
        
    Returns:
        Refitted decoder layer
    """
    features_np = features.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Solve least squares: features @ W.T = targets
    # W.T = pinv(features) @ targets
    # W = (pinv(features) @ targets).T
    
    # Handle bias: add column of ones to features
    features_with_bias = np.column_stack([features_np, np.ones(features_np.shape[0])])
    
    # Solve: features_with_bias @ [W.T; b] = targets
    # [W.T; b] = pinv(features_with_bias) @ targets
    solution = np.linalg.pinv(features_with_bias) @ targets_np
    
    # Extract weight and bias
    W_T = solution[:-1, :]  # [hidden_dim, input_dim]
    b = solution[-1, :]  # [input_dim]
    
    # Create new decoder layer
    decoder = nn.Linear(sae.config.hidden_dim, sae.config.input_dim).to(device)
    decoder.weight.data = torch.tensor(W_T.T, dtype=torch.float32, device=device)
    decoder.bias.data = torch.tensor(b, dtype=torch.float32, device=device)
    
    return decoder


def evaluate_threshold(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    threshold: float,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate SAE at a given activation threshold.
    
    Args:
        sae: SparseAutoencoder model
        train_loader: DataLoader for train split (used to refit decoder)
        test_loader: DataLoader for test split (used for evaluation)
        threshold: Activation threshold
        device: Device to run on
        
    Returns:
        Tuple of (mean_active_units, reconstruction_loss) on test split
    """
    sae.eval()
    
    # Collect features and targets from train split
    train_features_list = []
    train_targets_list = []
    
    with torch.no_grad():
        for batch in train_loader:
            h = batch["target_embeddings"].to(device)
            features = encode_with_threshold(sae, h, threshold)
            train_features_list.append(features.cpu())
            train_targets_list.append(h.cpu())
    
    train_features = torch.cat(train_features_list, dim=0)
    train_targets = torch.cat(train_targets_list, dim=0)
    
    # Refit decoder on train split
    refitted_decoder = refit_decoder(sae, train_features, train_targets, device)
    
    # Evaluate on test split
    mse_total = 0.0
    n_items = 0
    total_active_units = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            h = batch["target_embeddings"].to(device)
            features = encode_with_threshold(sae, h, threshold)
            reconstructed = refitted_decoder(features)
            
            batch_mse = nn.functional.mse_loss(reconstructed, h).item()
            mse_total += batch_mse * h.size(0)
            n_items += h.size(0)
            active_counts = (features > 0).sum(dim=1)
            total_active_units += active_counts.sum().item()
    
    reconstruction_loss = mse_total / n_items if n_items > 0 else float('nan')
    mean_active_units = total_active_units / n_items if n_items > 0 else float('nan')
    
    return mean_active_units, reconstruction_loss


def make_cached_threshold_evaluator(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
):
    """Create a cached evaluator to avoid redundant recomputation of thresholds."""
    cache: Dict[float, Tuple[float, float]] = {}

    def evaluate(threshold: float) -> Tuple[float, float]:
        key = round(float(threshold), 12)
        if key not in cache:
            cache[key] = evaluate_threshold(sae, train_loader, test_loader, float(threshold), device)
        return cache[key]

    return evaluate, cache


def find_threshold_for_target_mean_active(
    evaluate_fn,
    target_mean: float,
    max_threshold: float,
    tolerance: float = 1e-4,
    max_iters: int = 20,
) -> Tuple[float, Tuple[float, float]]:
    """Binary search to find smallest threshold whose mean active units <= target."""
    baseline_mean, baseline_loss = evaluate_fn(0.0)
    if baseline_mean <= target_mean:
        return 0.0, (baseline_mean, baseline_loss)

    max_mean, max_loss = evaluate_fn(max_threshold)
    if max_mean > target_mean:
        print(
            f"[WARNING] Even at sweep cap mean active units ({max_mean:.3f}) exceed target "
            f"{target_mean}. Using cap threshold."
        )
        return max_threshold, (max_mean, max_loss)

    low = 0.0
    high = max_threshold
    best_threshold = max_threshold
    best_metrics = (max_mean, max_loss)

    for _ in range(max_iters):
        mid = 0.5 * (low + high)
        mean_mid, loss_mid = evaluate_fn(mid)
        if mean_mid <= target_mean:
            best_threshold = mid
            best_metrics = (mean_mid, loss_mid)
            high = mid
        else:
            low = mid
        if high - low <= tolerance:
            break

    return best_threshold, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE pareto frontier by sweeping activation thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to SAE checkpoint directory (will auto-append 'best_model' if needed)"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["sentences", "digits"],
        default=None,
        help="Type of dataset to evaluate on (auto-inferred from checkpoint path if not specified)"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to dataset (defaults to hardcoded paths: data/sentences or data/digits/<task>)"
    )
    parser.add_argument(
        "--role-scheme",
        default=None,
        help="Role scheme for the dataset (default: 'svo' for sentences, 'l2r' for digits)"
    )
    parser.add_argument(
        "--embedding-model-name",
        default=None,
        help="Embedding model name for sentences datasets (required if dataset has no pre-computed embeddings)"
    )
    parser.add_argument(
        "--embedding-cache-path",
        default=None,
        help="Cache path for embeddings (defaults to data-path if not specified)"
    )
    parser.add_argument(
        "--seq2seq-checkpoint-dir",
        default=None,
        help="Path to seq2seq checkpoint for digits datasets (required if using embeddings)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save evaluation results (defaults to checkpoint directory)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=50,
        help="Number of threshold values to evaluate (default: 50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"[INFO] Loading SAE from: {checkpoint_path}")
    
    # Infer dataset type if not provided
    dataset_type = args.dataset_type or infer_dataset_type(args.checkpoint)
    print(f"[INFO] Using dataset type: {dataset_type}")
    
    # Set default data path if not provided
    if args.data_path is None:
        if dataset_type == "sentences":
            data_path = SENTENCES_DATA_PATH
        else:  # digits
            # Try to infer task name from checkpoint path
            checkpoint_lower = args.checkpoint.lower()
            if "copy" in checkpoint_lower:
                task = "copy"
            elif "reverse" in checkpoint_lower:
                task = "reverse"
            elif "sort" in checkpoint_lower:
                task = "sort_ascending"  # default
            else:
                task = "copy"  # default
            data_path = f"{DIGITS_DATA_PATH_BASE}/{task}_vocab_20_length_6"
        print(f"[INFO] Using default data path: {data_path}")
    else:
        data_path = args.data_path
    
    # Set default role scheme if not provided
    if args.role_scheme is None:
        if dataset_type == "sentences":
            role_scheme = "svo"
        else:  # digits
            role_scheme = "l2r"
        print(f"[INFO] Using default role scheme: {role_scheme}")
    else:
        role_scheme = args.role_scheme
    
    # Set output directory
    output_dir = args.output_dir or checkpoint_path
    results_dir = os.path.join(output_dir, "pareto_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load SAE model
    sae = SparseAutoencoder.from_pretrained(checkpoint_path).to(device)
    sae.eval()
    print(f"[INFO] Loaded SAE with input_dim={sae.config.input_dim}, hidden_dim={sae.config.hidden_dim}")
    
    # Load metadata from SAE config if available
    config_embedding_model_name = getattr(sae.config, "embedding_model_name", None)
    config_role_scheme = getattr(sae.config, "role_scheme", None)
    
    # Use embedding_model_name from config if not provided via args
    embedding_model_name = args.embedding_model_name or config_embedding_model_name
    if config_embedding_model_name and not args.embedding_model_name:
        print(f"[INFO] Using embedding_model_name from SAE config: {config_embedding_model_name}")
    
    # Use role_scheme from config if not provided via args
    if config_role_scheme and args.role_scheme is None:
        role_scheme = config_role_scheme
        print(f"[INFO] Using role_scheme from SAE config: {config_role_scheme}")
    
    # Load dataset
    if dataset_type == "sentences":
        print(f"[INFO] Loading sentences dataset from: {data_path}")
        dataset, _ = load_sentences(data_path, role_scheme=role_scheme)

        if embedding_model_name is not None:
            cache_path = args.embedding_cache_path or data_path
            print(f"[INFO] Loading embeddings from model: {embedding_model_name}")
            dataset, _ = load_dataset_with_embeddings(
                dataset=dataset,
                dataset_path=data_path,
                embedding_model_name=embedding_model_name,
                embedding_cache_path=cache_path,
            )
        elif "target_embeddings" not in dataset["test"].column_names:
            raise ValueError(
                "Dataset has no target_embeddings and --embedding-model-name not provided. "
                "Either provide pre-computed embeddings or specify --embedding-model-name"
            )
        else:
            print(f"[INFO] Using pre-computed embeddings from dataset")
    
    elif dataset_type == "digits":
        print(f"[INFO] Loading digits dataset from: {data_path}")
        file_paths = {
            "train": f"{data_path}.train",
            "valid": f"{data_path}.valid",
            "test": f"{data_path}.test"
        }
        dataset, tokenizer = load_digits(file_paths)
        
        # Convert to TPE format
        if role_scheme not in ["l2r", "r2l", "l2r_content", "r2l_content", "bow", "bidirectional"]:
            raise ValueError(f"Invalid role_scheme for digits: {role_scheme}")
        
        def process_digits_batch(batch):
            examples = [{"input": inp, "label": lbl} for inp, lbl in zip(batch["input"], batch["label"])]
            result = tokenize_function(examples, tokenizer, format="tpe", role_scheme=role_scheme)
            return result
        
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                process_digits_batch,
                batched=True,
                remove_columns=["input", "label"]
            )

        # Add target embeddings if seq2seq checkpoint provided
        seq2seq_checkpoint = args.seq2seq_checkpoint_dir or embedding_model_name
        if seq2seq_checkpoint is not None:
            from model import RecurrentEncoderDecoderModel
            if embedding_model_name and not args.seq2seq_checkpoint_dir:
                print(f"[INFO] Using seq2seq checkpoint from SAE config: {embedding_model_name}")
            print(f"[INFO] Loading seq2seq encoder from: {seq2seq_checkpoint}")
            seq2seq_model = RecurrentEncoderDecoderModel.from_pretrained(seq2seq_checkpoint)
            encoder_model = seq2seq_model.get_encoder()
            
            if encoder_model is None:
                raise ValueError("No encoder found in seq2seq model")
            
            encoder_model = encoder_model.to(device)
            encoder_model.eval()
            
            def add_target_embeddings(batch):
                input_ids = torch.tensor(batch["embedding_model_input_ids"], dtype=torch.long).to(device)
                input_lengths = torch.tensor(batch["embedding_model_input_lengths"], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    encoder_output = encoder_model(input_ids=input_ids, input_lengths=input_lengths)
                    target_embeddings = encoder_output.last_hidden_state
                    
                    if isinstance(target_embeddings, tuple):
                        target_embeddings = torch.cat(target_embeddings, dim=-1)
                    
                    if target_embeddings.dim() == 3 and target_embeddings.shape[1] == 1:
                        target_embeddings = target_embeddings.squeeze(1)
                    elif target_embeddings.dim() == 3:
                        batch_size = target_embeddings.shape[0]
                        last_embeddings = []
                        for i in range(batch_size):
                            seq_len = input_lengths[i].item()
                            last_embedding = target_embeddings[i, seq_len-1, :]
                            last_embeddings.append(last_embedding)
                        target_embeddings = torch.stack(last_embeddings)
                    batch["target_embeddings"] = target_embeddings.cpu().numpy()
                
                return batch
            
            for split in dataset.keys():
                dataset[split] = dataset[split].map(add_target_embeddings, batched=True)
            
            print(f"[INFO] Added target embeddings using seq2seq encoder")
        elif "target_embeddings" not in dataset["test"].column_names:
            raise ValueError(
                "Dataset has no target_embeddings and seq2seq checkpoint not provided. "
                "Either provide pre-computed embeddings or specify --seq2seq-checkpoint-dir, "
                "or ensure embedding_model_name is stored in SAE config"
            )
        else:
            print(f"[INFO] Using pre-computed embeddings from dataset")
    
    # Check that splits have required columns
    for split_name in ["train", "test"]:
        if split_name not in dataset:
            raise ValueError(f"Dataset must have '{split_name}' split")
        split = dataset[split_name]
        if "target_embeddings" not in split.column_names:
            raise ValueError(f"Dataset split '{split_name}' must have 'target_embeddings' column")
    
    # Create data loaders
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=_eval_collate)
    test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=_eval_collate)
    
    # Determine percentile-based threshold cap
    percentile = DEFAULT_THRESHOLD_PERCENTILE
    print(f"[INFO] Estimating activation threshold cap at {percentile}th percentile...")
    percentile_value, max_activation = compute_activation_threshold_cap(
        sae, [train_loader, test_loader], device, percentile=percentile
    )
    if not np.isfinite(percentile_value) or percentile_value <= 0.0:
        threshold_cap = max_activation
        print(
            f"[WARNING] Percentile activation non-positive (value={percentile_value:.6f}). "
            "Falling back to absolute max activation."
        )
    else:
        threshold_cap = min(percentile_value, max_activation) if max_activation > 0 else percentile_value
    print(
        f"[INFO] Activation statistics — max: {max_activation:.6f}, "
        f"{percentile}th percentile: {percentile_value:.6f}, sweep cap: {threshold_cap:.6f}"
    )

    evaluate_threshold_cached, _ = make_cached_threshold_evaluator(
        sae, train_loader, test_loader, device
    )

    start_threshold = 0.0
    if TARGET_MAX_MEAN_ACTIVE_UNITS is not None:
        start_threshold, start_metrics = find_threshold_for_target_mean_active(
            evaluate_threshold_cached,
            TARGET_MAX_MEAN_ACTIVE_UNITS,
            threshold_cap,
        )
    else:
        start_metrics = evaluate_threshold_cached(start_threshold)

    # Generate threshold values
    if threshold_cap <= start_threshold:
        thresholds = np.array([start_threshold], dtype=float)
    elif args.num_thresholds <= 1:
        thresholds = np.array([threshold_cap], dtype=float)
    else:
        thresholds = np.linspace(start_threshold, threshold_cap, args.num_thresholds)
    print(
        f"[INFO] Evaluating {len(thresholds)} thresholds from {thresholds[0]:.6f} "
        f"to {thresholds[-1]:.6f}"
    )
    
    # Evaluate at each threshold
    results = []
    for i, threshold in enumerate(thresholds):
        print(f"[INFO] Evaluating threshold {i+1}/{len(thresholds)}: {threshold:.6f}")
        if i == 0 and TARGET_MAX_MEAN_ACTIVE_UNITS is not None:
            mean_active_units, reconstruction_loss = start_metrics
        else:
            mean_active_units, reconstruction_loss = evaluate_threshold_cached(threshold)
        results.append({
            "threshold": float(threshold),
            "mean_active_units": float(mean_active_units),
            "reconstruction_loss": float(reconstruction_loss)
        })
        print(f"  Mean active units: {mean_active_units:.6f}, Reconstruction loss: {reconstruction_loss:.6e}")
    
    # Save results
    results_file = os.path.join(results_dir, "pareto_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to: {results_file}")
    
    # Plot pareto frontier
    mean_active_values = [r["mean_active_units"] for r in results]
    loss_values = [r["reconstruction_loss"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_active_values, loss_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Mean Active Units', fontsize=12)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
    plt.title('SAE Pareto Frontier: Mean Active Units vs Reconstruction Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    if PLOT_MAX_MEAN_ACTIVE_UNITS is not None:
        plt.xlim(left=0.0, right=PLOT_MAX_MEAN_ACTIVE_UNITS)
    plt.tight_layout()
    
    plot_file = os.path.join(results_dir, "pareto_frontier.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot to: {plot_file}")
    plt.close()
    
    print(f"[INFO] Pareto frontier evaluation complete!")
    return results


if __name__ == "__main__":
    main()
