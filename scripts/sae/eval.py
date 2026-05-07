"""Minimal script to evaluate a trained SAE checkpoint.

This script loads an SAE checkpoint and evaluates it on a test dataset.
It automatically infers dataset type from checkpoint path and uses hardcoded data paths.

Example:
    uv run scripts/sae/eval.py --checkpoint checkpoints/sae/tpr_sae_modernbert
    
    uv run scripts/sae/eval.py \
        --checkpoint checkpoints/sae_digits/reverse_v20l6_supervised_filler \
        --role-scheme r2l
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from sae import (
    SparseAutoencoder,
    evaluate_sae,
    compute_feature_well_rankedness_per_feature,
    select_feature_labels_by_well_rankedness,
)
from sentences import load_sentences
from digits import load_digits, tokenize_function
from utils import load_dataset_with_embeddings

# Hardcoded data paths
SENTENCES_DATA_PATH = "data/sentences"
DIGITS_DATA_PATH_BASE = "data/digits"


def infer_dataset_type(checkpoint_path: str) -> str:
    """Infer dataset type from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        "sentences" or "digits"
    """
    path_lower = checkpoint_path.lower()
    if "sentences" in path_lower or "sentence" in path_lower:
        return "sentences"
    elif "digits" in path_lower or "digit" in path_lower:
        return "digits"
    else:
        print(f"[WARNING] Could not infer dataset type from path '{checkpoint_path}', defaulting to 'sentences'")
        return "sentences"


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve checkpoint path, appending 'best_model' if needed.
    
    Args:
        checkpoint_path: Base checkpoint directory path
        
    Returns:
        Resolved checkpoint path that exists
    """
    path = Path(checkpoint_path).expanduser().resolve()
    
    # Try the path as-is first
    if path.exists() and (path / "config.json").exists():
        return str(path)
    
    # Try appending 'best_model'
    best_model_path = path / "best_model"
    if best_model_path.exists() and (best_model_path / "config.json").exists():
        return str(best_model_path)
    
    # If neither exists, return the original (will fail with a clear error)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            f"Tried: {path} and {best_model_path}"
        )
    
    raise FileNotFoundError(
        f"Checkpoint directory exists but missing config.json: {checkpoint_path}\n"
        f"Tried: {path} and {best_model_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAE checkpoint",
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
        help="Directory to save evaluation metrics (defaults to checkpoint directory)"
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory to write metrics.json and eval_command logs (defaults to <output-dir>/eval_results)"
    )
    # evaluate_sae arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--label-mode",
        choices=["filler", "filler_role"],
        default="filler",
        help="Label mode for feature quality and well-rankedness calculation (default: 'filler')"
    )
    parser.add_argument(
        "--ignore-singleton-verb",
        action="store_true",
        default=True,
        help="Ignore singleton verbs in feature quality calculation (default: True)"
    )
    parser.add_argument(
        "--no-ignore-singleton-verb",
        dest="ignore_singleton_verb",
        action="store_false",
        help="Do not ignore singleton verbs"
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=None,
        help="Override activation threshold used for feature metrics. "
             "Defaults to value stored in checkpoint (or 0.0 if unavailable)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # =========================================================================
    # CHECKPOINT AND DATASET SETUP
    # =========================================================================
    
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"[INFO] Loading SAE from: {checkpoint_path}")
    
    # Infer dataset type if not provided
    dataset_type = args.dataset_type or infer_dataset_type(args.checkpoint)
    print(f"[INFO] Using dataset type: {dataset_type}")
    
    # Set default data path based on dataset type
    if args.data_path is None:
        if dataset_type == "sentences":
            data_path = SENTENCES_DATA_PATH
        else:
            # Infer task name from checkpoint path for digits
            checkpoint_lower = args.checkpoint.lower()
            if "copy" in checkpoint_lower:
                task = "copy"
            elif "reverse" in checkpoint_lower:
                task = "reverse"
            elif "sort" in checkpoint_lower:
                task = "sort_ascending"
            else:
                task = "copy"
            data_path = f"{DIGITS_DATA_PATH_BASE}/{task}_vocab_20_length_6"
        print(f"[INFO] Using default data path: {data_path}")
    else:
        data_path = args.data_path
    
    # Set default role scheme based on dataset type
    if args.role_scheme is None:
        role_scheme = "svo" if dataset_type == "sentences" else "l2r"
        print(f"[INFO] Using default role scheme: {role_scheme}")
    else:
        role_scheme = args.role_scheme
    
    # =========================================================================
    # OUTPUT DIRECTORY AND COMMAND LOGGING
    # =========================================================================
    
    output_dir = args.output_dir or checkpoint_path
    metrics_dir = args.metrics_dir or os.path.join(output_dir, "eval_results")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Build run tag from checkpoint path (e.g., "sae_digits__reverse_v20l6_supervised_filler")
    path = Path(checkpoint_path)
    if path.name == "best_model":
        path = path.parent
    run_tag_parts = [path.name]
    if path.parent.name:
        run_tag_parts.insert(0, path.parent.name)
    run_tag = "__".join(run_tag_parts)
    
    command_file = os.path.join(metrics_dir, f"eval_command_{run_tag}.txt")
    command_json_file = os.path.join(metrics_dir, f"eval_command_{run_tag}.json")
    
    # Save raw command for reproducibility
    raw_command = " ".join(sys.argv)
    with open(command_file, "w") as f:
        f.write(f"# Evaluation command\n")
        f.write(f"{raw_command}\n\n")
        f.write(f"# Parsed arguments\n")
        for key, value in vars(args).items():
            f.write(f"--{key.replace('_', '-')}={value}\n")
    
    # Save command as JSON for programmatic access
    with open(command_json_file, "w") as f:
        json.dump({
            "raw_command": raw_command,
            "script": sys.argv[0],
            "arguments": vars(args)
        }, f, indent=2)
    
    print(f"[INFO] Evaluation command saved to: {command_file}")
    
    # =========================================================================
    # LOAD SAE MODEL
    # =========================================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    sae = SparseAutoencoder.from_pretrained(checkpoint_path).to(device)
    sae.eval()
    print(f"[INFO] Loaded SAE with input_dim={sae.config.input_dim}, hidden_dim={sae.config.hidden_dim}")
    
    # Load metadata from SAE config if available
    config_embedding_model_name = getattr(sae.config, "embedding_model_name", None)
    config_role_scheme = getattr(sae.config, "role_scheme", None)
    config_activation_threshold = getattr(sae.config, "activation_threshold", None)
    
    # Use embedding_model_name from config if not provided via args
    embedding_model_name = args.embedding_model_name or config_embedding_model_name
    if config_embedding_model_name and not args.embedding_model_name:
        print(f"[INFO] Using embedding_model_name from SAE config: {config_embedding_model_name}")
    
    # Use role_scheme from config if not provided via args
    if config_role_scheme and args.role_scheme is None:
        role_scheme = config_role_scheme
        print(f"[INFO] Using role_scheme from SAE config: {config_role_scheme}")
    
    # Infer label_mode from feature_map_scheme in config if available
    inferred_label_mode = args.label_mode
    config_feature_map_scheme = getattr(sae.config, "feature_map_scheme", None)
    if config_feature_map_scheme and args.label_mode == "filler":
        if config_feature_map_scheme in ["filler", "filler_role"]:
            inferred_label_mode = config_feature_map_scheme
            print(f"[INFO] Inferred label_mode='{inferred_label_mode}' from SAE config feature_map_scheme")
    
    # Determine activation threshold (args > config > default 0.0)
    if args.activation_threshold is not None:
        activation_threshold = float(args.activation_threshold)
    elif config_activation_threshold is not None:
        activation_threshold = config_activation_threshold
    else:
        activation_threshold = 0.0
    sae.config.activation_threshold = activation_threshold
    print(f"[INFO] Using activation_threshold: {activation_threshold}")
    
    # =========================================================================
    # LOAD DATASET
    # =========================================================================
    
    role_assigner = None
    tokenizer = None
    
    if dataset_type == "sentences":
        print(f"[INFO] Loading sentences dataset from: {data_path}")
        dataset, role_assigner = load_sentences(data_path, role_scheme=role_scheme)

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
        
        if role_scheme not in ["l2r", "r2l", "l2r_content", "r2l_content", "bow", "bidirectional"]:
            raise ValueError(f"Invalid role_scheme for digits: {role_scheme}")
        
        # Convert to TPE format with role assignments
        def process_digits_batch(batch):
            examples = [{"input": inp, "label": lbl} for inp, lbl in zip(batch["input"], batch["label"])]
            return tokenize_function(examples, tokenizer, format="tpe", role_scheme=role_scheme)
        
        for split in dataset.keys():
            dataset[split] = dataset[split].map(process_digits_batch, batched=True)

        # Add target embeddings using seq2seq encoder if checkpoint provided
        # For digits, embedding_model_name from config is the seq2seq checkpoint path
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
                    
                    # Handle different encoder output formats
                    if isinstance(target_embeddings, tuple):
                        target_embeddings = torch.cat(target_embeddings, dim=-1)
                    
                    # Extract final hidden state per sequence
                    if target_embeddings.dim() == 3 and target_embeddings.shape[1] == 1:
                        target_embeddings = target_embeddings.squeeze(1)
                    elif target_embeddings.dim() == 3:
                        batch_size = target_embeddings.shape[0]
                        last_embeddings = []
                        for i in range(batch_size):
                            seq_len = input_lengths[i].item()
                            last_embeddings.append(target_embeddings[i, seq_len-1, :])
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
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    test_split = dataset["test"]
    if "target_embeddings" not in test_split.column_names:
        raise ValueError("Test dataset must have 'target_embeddings' column")
    
    if "filler_ids" not in test_split.column_names or "role_ids" not in test_split.column_names:
        print("[WARNING] Test dataset missing filler_ids/role_ids. Feature quality metrics will be skipped.")
    
    print(f"[INFO] Evaluating SAE on test split ({len(test_split)} examples)...")
    
    # Collate function: extract embeddings from batch
    def collate_embeddings(batch):
        embeddings = torch.tensor([ex["target_embeddings"] for ex in batch], dtype=torch.float32)
        return {"target_embeddings": embeddings}
    
    eval_loader = torch.utils.data.DataLoader(
        test_split, batch_size=args.batch_size, collate_fn=collate_embeddings
    )
    label_loader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=args.batch_size, collate_fn=collate_embeddings
    )
    
    # -------------------------------------------------------------------------
    # Collect forward-pass tensors from eval dataset
    # -------------------------------------------------------------------------
    eval_embeddings_list = []
    eval_reconstructions_list = []
    eval_activations_list = []
    sae.eval()
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["target_embeddings"].to(device)
            encoded = sae.encode(inputs)
            decoded = sae.decode(encoded)
            eval_embeddings_list.append(inputs.detach().cpu())
            eval_reconstructions_list.append(decoded.detach().cpu())
            eval_activations_list.append(encoded.detach().cpu())
    
    eval_embeddings = torch.cat(eval_embeddings_list, dim=0)
    eval_reconstructions = torch.cat(eval_reconstructions_list, dim=0)
    eval_activations = torch.cat(eval_activations_list, dim=0)
    
    # -------------------------------------------------------------------------
    # Collect activations from training dataset (for feature labeling)
    # -------------------------------------------------------------------------
    label_activations_list = []
    with torch.no_grad():
        for batch in label_loader:
            inputs = batch["target_embeddings"].to(device)
            encoded = sae.encode(inputs)
            label_activations_list.append(encoded.detach().cpu())
    
    label_activations = torch.cat(label_activations_list, dim=0)
    
    # -------------------------------------------------------------------------
    # Compute feature labels and scores
    # -------------------------------------------------------------------------
    feature_labels = select_feature_labels_by_well_rankedness(
        dataset_split=dataset["train"],
        activations=label_activations,
        label_mode=inferred_label_mode,
        ignore_singleton_verb=args.ignore_singleton_verb,
    )
    feature_scores = compute_feature_well_rankedness_per_feature(
        dataset_split=test_split,
        activations=eval_activations,
        feature_labels=feature_labels,
        label_mode=inferred_label_mode,
        ignore_singleton_verb=args.ignore_singleton_verb,
    )
    
    # Run main evaluation
    metrics = evaluate_sae(
        sae=sae,
        label_dataset=dataset["train"],
        eval_dataset=test_split,
        eval_embeddings=eval_embeddings,
        eval_reconstructions=eval_reconstructions,
        eval_activations=eval_activations,
        label_activations=label_activations,
        sae_output_dir=None,
        label_mode=inferred_label_mode,
        ignore_singleton_verb=args.ignore_singleton_verb,
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    metrics_path = os.path.join(metrics_dir, f"metrics_{run_tag}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # Build feature details: top activations and labels for each feature
    # -------------------------------------------------------------------------
    sentences = test_split["sentence"] if "sentence" in test_split.column_names else None
    features_payload = []
    num_features = eval_activations.shape[-1]
    
    for feature_idx in range(num_features):
        acts = eval_activations[:, feature_idx]
        
        # Skip features that never activate
        if torch.max(acts).item() <= 0:
            continue
        
        # Get top-k activating examples
        topk = min(5, acts.shape[0])
        top_vals, top_idx = torch.topk(acts, k=topk)
        examples = []
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
            entry = {"rank": rank, "index": int(idx), "activation": float(val)}
            if sentences is not None:
                entry["text"] = sentences[int(idx)]
            examples.append(entry)

        label = feature_labels[feature_idx]
        score_val = feature_scores[feature_idx].item()
        score = float(score_val) if torch.isfinite(feature_scores[feature_idx]) else None
        
        # Format label for human readability
        if label is None:
            formatted_label = None
            label_raw = None
        elif inferred_label_mode == "filler":
            # Single filler ID
            if dataset_type == "sentences" and role_assigner is not None:
                formatted_label = role_assigner.noun_idx2filler.get(int(label), f"filler_{label}")
            elif dataset_type == "digits" and tokenizer is not None:
                try:
                    token = tokenizer.convert_ids_to_tokens([int(label)], skip_special_tokens=False)[0]
                    formatted_label = token.strip()
                except Exception:
                    formatted_label = f"token_{label}"
            else:
                formatted_label = str(label)
            label_raw = int(label)
        else:
            # Filler-role pair
            filler_id, role_id = int(label[0]), int(label[1])
            if dataset_type == "sentences" and role_assigner is not None:
                verb_role = role_assigner.role2idx.get("verb")
                noun_vocab = len(role_assigner.noun_idx2filler)
                if role_id == verb_role:
                    filler_idx = filler_id - noun_vocab
                    filler = role_assigner.verb_idx2filler.get(filler_idx, f"verb_{filler_idx}")
                else:
                    filler = role_assigner.noun_idx2filler.get(filler_id, f"filler_{filler_id}")
                role = role_assigner.idx2role.get(role_id, f"role_{role_id}")
                formatted_label = f"{filler} x {role}"
            else:
                formatted_label = f"({filler_id}, {role_id})"
            label_raw = [filler_id, role_id]
        
        features_payload.append({
            "feature_index": feature_idx,
            "label": formatted_label,
            "label_raw": label_raw,
            "score": score,
            "top_activations": examples,
        })

    feature_path = os.path.join(metrics_dir, f"feature_details_{run_tag}.json")
    with open(feature_path, "w") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "dataset_type": dataset_type,
            "label_mode": inferred_label_mode,
            "features": features_payload,
        }, f, indent=2)

    print(f"[INFO] Evaluation complete. Metrics saved to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
