from typing import Dict, Any, Optional, Literal, Tuple, Union

import gin
import torch
from torch.utils.data import DataLoader

from sentences import load_sentences
from digits import load_digits, tokenize_function
from model import TensorProductEncoderForPretraining, RecurrentEncoderDecoderModel
from sae import SparseAutoencoder, evaluate_sae
from probing import auto_select_role_pinv_l2_lambda, auto_select_tpe_output_l2_lambda
from utils import parse_args_for_gin, load_dataset_with_embeddings


def _sentence_feature_semantics(assigner) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Resolve sentence fillers and roles to descriptive labels.

    Args:
        assigner: Role assigner produced by ``load_sentences``. Provides the
            noun/verb vocabularies and the role name mapping.

    Returns:
        Tuple containing two dictionaries: the first maps filler ids to token
        strings, and the second maps role ids to their semantic names.
    """
    filler_names: Dict[int, str] = {}
    noun_vocab = len(assigner.nouns_sg)
    for noun, idx in assigner.noun_filler2idx.items():
        filler_names[int(idx)] = noun
    for verb, idx in assigner.verb_filler2idx.items():
        filler_names[int(idx) + noun_vocab] = verb
    role_names = {int(idx): name for idx, name in assigner.idx2role.items()}
    return filler_names, role_names


def find_legal_pairs(assigner) -> list[tuple[int, int]]:
    """Return legal filler-role pairs for sentence datasets.

    Noun fillers are paired with non-verb roles, and verb fillers are paired only
    with the verb role.
    """
    noun_vocab = len(assigner.noun_idx2filler)
    verb_vocab = len(assigner.verb_idx2filler)
    verb_role = assigner.role2idx.get("verb")
    if verb_role is None:
        raise ValueError("Role scheme must define a verb role for sentence TPR SAEs.")

    role_ids = list(assigner.idx2role.keys())
    allowed_pairs: list[tuple[int, int]] = []
    for filler_id in range(noun_vocab):
        for role_id in role_ids:
            if role_id == verb_role:
                continue
            allowed_pairs.append((filler_id, role_id))
    for verb_idx in range(verb_vocab):
        filler_id = noun_vocab + verb_idx
        allowed_pairs.append((filler_id, verb_role))
    return allowed_pairs


def find_observed_filler_role_pairs(dataset_split) -> list[tuple[int, int]]:
    """Collect legal filler-role pairs observed in a dataset split.

    Args:
        dataset_split: Hugging Face dataset split containing ``filler_ids`` and
            ``role_ids`` columns.

    Returns:
        Sorted list of unique ``(filler_id, role_id)`` pairs observed in the
        split.
    """
    if "filler_ids" not in dataset_split.column_names or "role_ids" not in dataset_split.column_names:
        raise ValueError("Dataset split must contain 'filler_ids' and 'role_ids' columns.")

    observed_pairs: set[tuple[int, int]] = set()
    filler_batches = dataset_split["filler_ids"]
    role_batches = dataset_split["role_ids"]
    for filler_ids, role_ids in zip(filler_batches, role_batches):
        if len(filler_ids) != len(role_ids):
            raise ValueError("Each example must have matching filler_ids and role_ids lengths.")
        for filler_id, role_id in zip(filler_ids, role_ids):
            observed_pairs.add((int(filler_id), int(role_id)))

    return sorted(observed_pairs, key=lambda pair: (pair[1], pair[0]))


def filter_filler_role_pairs_by_presence(
    dataset_split,
    pairs: list[tuple[int, int]],
    *,
    max_presence: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Filter pairs that appear in at least ``max_presence`` fraction of examples.

    Presence is computed per-example (a pair counts at most once per row).
    Pairs with empirical presence >= ``max_presence`` are removed.
    """
    if max_presence <= 0.0 or max_presence > 1.0:
        raise ValueError(f"max_presence must be in (0, 1], got {max_presence}")
    if not pairs:
        return [], []

    pair_set = set((int(f), int(r)) for f, r in pairs)
    counts = {pair: 0 for pair in pair_set}
    n_examples = len(dataset_split)
    if n_examples == 0:
        return pairs, []

    filler_batches = dataset_split["filler_ids"]
    role_batches = dataset_split["role_ids"]
    for filler_ids, role_ids in zip(filler_batches, role_batches):
        seen_in_example = {
            (int(filler_id), int(role_id))
            for filler_id, role_id in zip(filler_ids, role_ids)
            if (int(filler_id), int(role_id)) in pair_set
        }
        for pair in seen_in_example:
            counts[pair] += 1

    kept: list[tuple[int, int]] = []
    removed: list[tuple[int, int]] = []
    for pair in pairs:
        presence = counts.get((int(pair[0]), int(pair[1])), 0) / float(n_examples)
        if presence >= max_presence:
            removed.append(pair)
        else:
            kept.append(pair)
    return kept, removed


def _digits_feature_semantics(tokenizer, tpe_config, role_scheme: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Infer human-readable labels for digit fillers and roles.

    Args:
        tokenizer: Tokenizer returned by ``load_digits``. Used to map filler ids
            back to their original symbol if available.
        tpe_config: Configuration object for the tensor product encoder; must
            expose filler and role vocabulary sizes.
        role_scheme: Role scheme identifier (e.g., ``"l2r"``,
            ``"l2r_content"``, ``"bidirectional"``)
            used to build deterministic role names when no tokenizer metadata is
            available.

    Returns:
        Tuple with two dictionaries: filler id -> token name and role id ->
        textual description (including padding where present).
    """
    filler_names: Dict[int, str] = {}
    role_names: Dict[int, str] = {}

    vocab_size = int(getattr(tpe_config, "n_fillers", 0) or 0)
    if tokenizer is not None and vocab_size:
        for idx in range(vocab_size):
            try:
                token = tokenizer.convert_ids_to_tokens([idx], skip_special_tokens=False)[0]
            except (KeyError, IndexError, TypeError):
                token = None
            label = token.strip() if isinstance(token, str) else f"token_{idx}"
            filler_names[idx] = label
    else:
        filler_names = {idx: f"filler_{idx}" for idx in range(vocab_size)}

    n_roles = int(getattr(tpe_config, "n_roles", 0) or 0)
    pad_role_id = getattr(tpe_config, "role_pad_token_id", None)
    if pad_role_id is not None:
        role_names[int(pad_role_id)] = "pad"

    for idx in range(n_roles):
        if pad_role_id is not None and idx == pad_role_id:
            continue
        if role_scheme == "bow":
            role_names[idx] = "bag_of_words"
        elif role_scheme in ("l2r", "l2r_content"):
            role_names[idx] = f"position_{idx}"
        elif role_scheme in ("r2l", "r2l_content"):
            role_names[idx] = f"reverse_position_{idx}"
        elif role_scheme == "bidirectional":
            offset = n_roles // 2
            if pad_role_id is not None and offset <= pad_role_id:
                offset = (n_roles - 1) // 2
            direction = "l2r" if idx <= offset else "r2l"
            base = idx if idx <= offset else max(idx - offset, 1)
            role_names[idx] = f"{direction}_position_{base}"
        else:
            role_names[idx] = f"role_{idx}"

    return filler_names, role_names


def _collate(batch):
    """Collate heterogeneous samples from sentence or digit datasets.

    Args:
        batch: Sequence of dataset examples containing filler/role ids and
            optional target embeddings.

    Returns:
        Dictionary assembled into tensor batches for filler ids, role ids, and
        any cached target embeddings.
    """
    # Extract filler_ids and role_ids (present in both dataset types after processing)
    filler_ids = torch.stack([torch.tensor(x["filler_ids"], dtype=torch.long) for x in batch])
    role_ids = torch.stack([torch.tensor(x["role_ids"], dtype=torch.long) for x in batch])
    collated = {"filler_ids": filler_ids, "role_ids": role_ids}
    
    # Add target embeddings if present (sentences dataset)
    if "target_embeddings" in batch[0]:
        embeds = torch.tensor([x["target_embeddings"] for x in batch], dtype=torch.float)
        collated["target_embeddings"] = embeds
    
    return collated


@gin.configurable
def main(
    tpe_checkpoint_dir: str,
    sae_output_dir: str,
    sae_config: Dict[str, Any],
    data_path: str,
    dataset_type: Literal["sentences", "digits"] = "sentences",
    role_scheme: str = "svo",
    batch_size: int = 32,
    embedding_model_name: Optional[str] = None,
    embedding_cache_path: Optional[str] = None,
    seq2seq_checkpoint_dir: Optional[str] = None,
    tpe_output_layer_regularization: Literal["l2", "atol", "topk"] = "l2",
    tpe_output_layer_regularization_value: Optional[Union[float, str]] = None,
    first_layer_construction: Literal["unbinding", "pinv-tpencoding"] = "unbinding",
    second_layer_construction: Literal["transpose-unbinding", "pinv-unbinding", "tpencoding"] = "pinv-unbinding",
    filler_unbinding: Literal["pinv", "norm"] = "pinv",
    role_unbinding: Literal["pinv", "norm"] = "pinv",
    role_pinv_regularization: Literal["none", "atol", "l2", "topk"] = "l2",
    role_pinv_l2_lambda: Optional[Union[float, str]] = None,
    role_pinv_atol: Optional[float] = None,
    role_pinv_topk: Optional[int] = None,
    role_invariant: bool = True,
    eval_label_mode: Literal["filler", "filler_role"] = "filler",
    gating_strategy: Literal["none", "quantile", "mad"] = "none",
    gating_target_sparsity: float = 0.02,
    gating_mad_scale: float = 3.0,
    gating_calibration_split: Literal["train", "valid", "test"] = "valid",
    gating_calibration_samples: int = 256,
    use_observed_pairs_for_digits: bool = True,
    digits_exclude_pairs_with_presence_ge: Optional[float] = None,
    decoder_bias_source: Literal["tpe_output_bias", "train_mean_target_embedding"] = "tpe_output_bias",
    construction_calibration_split: Literal["train", "valid", "test"] = "train",
    construction_calibration_samples: int = 2048,
    decoder_pinv_whiten: bool = False,
    decoder_pinv_regularization: Literal["none", "atol", "l2", "topk"] = "none",
    decoder_pinv_l2_lambda: Optional[float] = None,
    decoder_pinv_atol: Optional[float] = None,
    decoder_pinv_topk: Optional[int] = None,
    feature_rescale_strategy: Literal["none", "inv_std", "inv_mad"] = "none",
    feature_rescale_eps: float = 1e-6,
    decoder_refinement: Literal["none", "ridge"] = "none",
    decoder_refinement_l2: float = 1e-4,
) -> None:
    """Construct and evaluate an SAE derived from a trained TPE.
    
    Args:
        tpe_checkpoint_dir: Path to the trained TPE checkpoint
        sae_output_dir: Directory to save the SAE and evaluation results
        sae_config: Configuration dict for the SAE
        dataset_type: Type of dataset to use ("sentences" or "digits")
        data_path: Path to dataset. If None, uses defaults:
                  - "sentences" for sentences dataset
                  - "data/digits/copy_vocab_20_length_6" for digits dataset
        role_scheme: Role scheme for TPE ("svo" for sentences, and
            "l2r"/"r2l"/"l2r_content"/"r2l_content"/"bow"/"bidirectional"
            for digits)
        batch_size: Batch size for evaluation
        embedding_model_name: Optional embedding model for sentences datasets.
                          For digits datasets, any non-None value triggers using seq2seq encoder embeddings
        embedding_cache_path: Cache path for embeddings
        seq2seq_checkpoint_dir: Path to seq2seq model checkpoint for digits datasets (used with embedding_model_name)
        tpe_output_layer_regularization: Strategy for inverting the TPE output layer when constructing the SAE.
        tpe_output_layer_regularization_value: Hyperparameter associated with the chosen inversion strategy.
            Set to "auto" (or None) to select an l2 value automatically.
        filler_unbinding: Method for computing filler unbinding vectors.
        role_unbinding: Method for computing role unbinding vectors.
        role_pinv_regularization: Regularization method for role unbinding pseudoinverse.
        role_pinv_l2_lambda: Role unbinding l2 lambda or "auto" to select.
        role_pinv_atol: Role unbinding atol value.
        role_pinv_topk: Role unbinding top-k value.
        role_invariant: If True, construct features per filler only; otherwise per filler-role pair.
        eval_label_mode: Label mode for feature quality and well-rankedness ("filler" or "filler_role").
        use_observed_pairs_for_digits: If True, build pairwise digits features
            only for (filler, role) combinations observed in the dataset.
        digits_exclude_pairs_with_presence_ge: Optional threshold used only for
            digits pairwise features. Observed pairs appearing in at least this
            fraction of train examples are removed.
        decoder_bias_source: Reconstruction anchor used when setting encoder and
            decoder biases.
        construction_calibration_split: Split used for whitening/rescaling and
            decoder refinement calibration.
        construction_calibration_samples: Max number of calibration samples.
        decoder_pinv_whiten: If True, use whitening-aware pseudoinverse for
            decoder construction.
        decoder_pinv_regularization: Regularization mode for decoder
            pseudoinverse construction.
        decoder_pinv_l2_lambda: Tikhonov lambda for decoder pseudoinverse when
            ``decoder_pinv_regularization='l2'``.
        decoder_pinv_atol: Absolute tolerance for decoder pseudoinverse when
            ``decoder_pinv_regularization='atol'``.
        decoder_pinv_topk: Number of singular values for decoder pseudoinverse
            when ``decoder_pinv_regularization='topk'``.
        feature_rescale_strategy: Optional post-construction feature rescaling.
        feature_rescale_eps: Numerical epsilon for calibration statistics.
        decoder_refinement: Optional closed-form decoder refinement mode.
        decoder_refinement_l2: Ridge coefficient for decoder refinement.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    role_assigner = None
    tokenizer = None

    # Load dataset based on type
    if dataset_type == "sentences":
        dataset, role_assigner = load_sentences(data_path, role_scheme=role_scheme)
        if embedding_model_name is not None:
            dataset, _ = load_dataset_with_embeddings(
                dataset=dataset,
                dataset_path=data_path,
                embedding_model_name=embedding_model_name,
                embedding_cache_path=embedding_cache_path,
            )
        tokenizer = None  # sentences already have filler_ids and role_ids
    elif dataset_type == "digits":
        # Build file paths dict for digits dataset
        file_paths = {
            "train": f"{data_path}.train",
            "valid": f"{data_path}.valid", 
            "test": f"{data_path}.test"
        }
        
        # Load digits dataset and tokenizer
        dataset, tokenizer = load_digits(file_paths)
        
        # Convert digits dataset to TPE format using batch processing
        def process_digits_batch(batch):
            examples = [{"input": inp, "label": lbl} for inp, lbl in zip(batch["input"], batch["label"])]
            # For digits datasets, ensure role_scheme is compatible with tokenize_function
            if role_scheme in ["l2r", "r2l", "l2r_content", "r2l_content", "bow", "bidirectional"]:
                digits_role_scheme = role_scheme
            else:
                raise ValueError(f"Unknown role scheme: {role_scheme}")
            result = tokenize_function(examples, tokenizer, format="tpe", role_scheme=digits_role_scheme)
            return result
        
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                process_digits_batch,
                batched=True,
                remove_columns=["input", "label"]
            )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # For digits datasets, use the seq2seq model's encoder to generate target embeddings
    if dataset_type == "digits" and embedding_model_name is not None:
        if seq2seq_checkpoint_dir is None:
            raise ValueError("seq2seq_checkpoint_dir must be provided when using embeddings with digits datasets")
            
        print("[INFO] Using seq2seq model's encoder to generate target embeddings for digits dataset")
        
        # Load the seq2seq model to access its encoder
        seq2seq_model = RecurrentEncoderDecoderModel.from_pretrained(seq2seq_checkpoint_dir)
        encoder_model = seq2seq_model.get_encoder()
        
        if encoder_model is None:
            raise ValueError("No encoder found in seq2seq model")
        
        # Generate embeddings for the digits dataset  
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
        
        def add_target_embeddings(batch):
            """Add target embeddings generated from the seq2seq encoder model."""
            # Get input_ids and lengths from the tokenized data
            input_ids = torch.tensor(batch["embedding_model_input_ids"], dtype=torch.long).to(device)
            input_lengths = torch.tensor(batch["embedding_model_input_lengths"], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Generate embeddings using the seq2seq encoder model
                encoder_output = encoder_model(input_ids=input_ids, input_lengths=input_lengths)
                target_embeddings = encoder_output.last_hidden_state
                
                # Handle tuple outputs (e.g., LSTM returns (hidden, cell))
                if isinstance(target_embeddings, tuple):
                    target_embeddings = torch.cat(target_embeddings, dim=-1)
                
                # The encoder returns [batch_size, 1, hidden_size] so we squeeze the middle dimension
                # target_embeddings shape: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
                if target_embeddings.dim() == 3 and target_embeddings.shape[1] == 1:
                    target_embeddings = target_embeddings.squeeze(1)
                elif target_embeddings.dim() == 3:
                    # If we have multiple timesteps, take the last non-padding token
                    batch_size = target_embeddings.shape[0]
                    last_embeddings = []
                    for i in range(batch_size):
                        seq_len = input_lengths[i].item()
                        # Take the embedding of the last non-padding token
                        last_embedding = target_embeddings[i, seq_len-1, :]  # -1 because seq_len is 1-indexed
                        last_embeddings.append(last_embedding)
                    target_embeddings = torch.stack(last_embeddings)  # [batch_size, hidden_size]
                batch["target_embeddings"] = target_embeddings.cpu().numpy()
            
            return batch
        
        # Add target embeddings to all splits
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                add_target_embeddings,
                batched=True
            )
        
        print(f"[INFO] Added target embeddings to digits dataset using seq2seq encoder from {seq2seq_checkpoint_dir}")

    loader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=_collate)

    tpe = TensorProductEncoderForPretraining.from_pretrained(tpe_checkpoint_dir).to(device)

    # Use the trained base encoder weights when building the analytic SAE. The
    # wrapper class exposes a fresh TensorProductEncoder at `tpe.encoder`; the
    # top-level embeddings belong to the pretraining head and remain
    # uninitialised for checkpoints saved mid-training. Passing the wrapper
    # directly therefore reconstructs the SAE from random weights. Unwrap the
    # inner encoder when available so we always use the trained parameters.
    tpe_for_sae = tpe.encoder if getattr(tpe, "encoder", None) is not None else tpe

    if isinstance(tpe_output_layer_regularization_value, str):
        if tpe_output_layer_regularization_value.lower() == "auto":
            tpe_output_layer_regularization_value = None
        else:
            raise ValueError(
                "tpe_output_layer_regularization_value must be a float, None, or 'auto'. "
                f"Got {tpe_output_layer_regularization_value!r}."
            )

    if isinstance(role_pinv_l2_lambda, str):
        if role_pinv_l2_lambda.lower() == "auto":
            role_pinv_l2_lambda = None
        else:
            raise ValueError(
                "role_pinv_l2_lambda must be a float, None, or 'auto'. "
                f"Got {role_pinv_l2_lambda!r}."
            )

    # Auto-select l2 lambda for output layer inversion when not provided
    if (
        tpe_output_layer_regularization == "l2" and tpe_output_layer_regularization_value is None
    ):
        batch_subset = dataset["test"].select(range(min(128, len(dataset["test"]))))

        if "target_embeddings" in batch_subset.column_names:
            target_hidden = torch.tensor(batch_subset["target_embeddings"], dtype=torch.float32).to(device)
        else:
            with torch.no_grad():
                out = tpe_for_sae(
                    filler_ids=torch.tensor(batch_subset["filler_ids"], dtype=torch.long).to(device),
                    role_ids=torch.tensor(batch_subset["role_ids"], dtype=torch.long).to(device),
                )
                h = out.encoder_hidden_states if hasattr(out, "encoder_hidden_states") else out.hidden_states
                if h.dim() == 3:
                    h = h.squeeze(1)
                target_hidden = h

        filler_ids_tensor = torch.tensor(batch_subset["filler_ids"], dtype=torch.long).to(device)
        role_ids_tensor = torch.tensor(batch_subset["role_ids"], dtype=torch.long).to(device)

        reg_lambda, best_value, (log_lo, log_hi) = auto_select_tpe_output_l2_lambda(
            tpe_for_sae,
            target_hidden,
            filler_ids_tensor,
            role_ids_tensor,
            device=device,
        )
        tpe_output_layer_regularization_value = float(reg_lambda)
        print(
            f"[INFO] Auto-selected TPE output-layer l2 ≈ {tpe_output_layer_regularization_value:.5g} "
            f"(val_mse≈{best_value:.4e}; window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
        )

    if (
        role_unbinding == "pinv"
        and role_pinv_regularization == "l2"
        and role_pinv_l2_lambda is None
    ):
        batch_subset = dataset["test"].select(range(min(128, len(dataset["test"]))))
        filler_ids_tensor = torch.tensor(batch_subset["filler_ids"], dtype=torch.long).to(device)
        role_ids_tensor = torch.tensor(batch_subset["role_ids"], dtype=torch.long).to(device)
        reg_lambda, best_value, (log_lo, log_hi) = auto_select_role_pinv_l2_lambda(
            tpe_for_sae,
            filler_ids=filler_ids_tensor,
            role_ids=role_ids_tensor,
            device=device,
        )
        role_pinv_l2_lambda = float(reg_lambda)
        print(
            f"[INFO] Auto-selected role unbinding l2 ≈ {role_pinv_l2_lambda:.5g} "
            f"(val_mse≈{best_value:.4e}; window [{max(1e-12, 10.0 ** log_lo):.3e}, {min(1e12, 10.0 ** log_hi):.3e}])"
        )

    # Add metadata to sae_config before constructing SAE
    sae_config.setdefault("tpe_output_layer_regularization", tpe_output_layer_regularization)
    sae_config.setdefault("tpe_output_layer_regularization_value", tpe_output_layer_regularization_value)
    sae_config.setdefault("first_layer_construction", first_layer_construction)
    sae_config.setdefault("second_layer_construction", second_layer_construction)
    sae_config.setdefault("filler_unbinding", filler_unbinding)
    sae_config.setdefault("role_unbinding", role_unbinding)
    sae_config.setdefault("role_pinv_regularization", role_pinv_regularization)
    sae_config.setdefault("role_pinv_l2_lambda", role_pinv_l2_lambda)
    sae_config.setdefault("role_pinv_atol", role_pinv_atol)
    sae_config.setdefault("role_pinv_topk", role_pinv_topk)
    sae_config.setdefault("role_invariant", role_invariant)
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
    # Set feature_map_scheme based on role_invariant: True -> "filler", False -> "filler_role"
    sae_config.setdefault("feature_map_scheme", "filler" if role_invariant else "filler_role")
    # Store gating config for reproducibility
    sae_config.setdefault("gating_strategy", gating_strategy)
    sae_config.setdefault("gating_target_sparsity", gating_target_sparsity)
    sae_config.setdefault("gating_mad_scale", gating_mad_scale)
    
    def _collect_embeddings(split_name: str, max_samples: int) -> torch.Tensor:
        split = dataset[split_name]
        n_samples = min(max_samples, len(split))
        subset = split.select(range(n_samples))
        if "target_embeddings" in subset.column_names:
            return torch.tensor(subset["target_embeddings"], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = tpe_for_sae(
                filler_ids=torch.tensor(subset["filler_ids"], dtype=torch.long, device=device),
                role_ids=torch.tensor(subset["role_ids"], dtype=torch.long, device=device),
            )
            h = out.encoder_hidden_states if hasattr(out, "encoder_hidden_states") else out.hidden_states
            if h.dim() == 3:
                h = h.squeeze(1)
            return h

    gating_calibration_embeddings = None
    if gating_strategy != "none":
        gating_calibration_embeddings = _collect_embeddings(gating_calibration_split, gating_calibration_samples)
        print(
            f"[INFO] Using {gating_calibration_embeddings.shape[0]} samples from "
            f"'{gating_calibration_split}' for gating calibration"
        )

    construction_calibration_embeddings = None
    needs_construction_calibration = (
        decoder_pinv_whiten
        or feature_rescale_strategy != "none"
        or decoder_refinement != "none"
        or decoder_bias_source == "train_mean_target_embedding"
    )
    if needs_construction_calibration:
        construction_calibration_embeddings = _collect_embeddings(
            construction_calibration_split,
            construction_calibration_samples,
        )
        print(
            f"[INFO] Using {construction_calibration_embeddings.shape[0]} samples from "
            f"'{construction_calibration_split}' for construction calibration"
        )

    bias_anchor = None
    if decoder_bias_source == "train_mean_target_embedding":
        bias_anchor = construction_calibration_embeddings.mean(dim=0)
        print("[INFO] Decoder bias anchor set to train-mean calibration embedding")

    allowed_filler_role_pairs = None
    if dataset_type == "sentences" and not role_invariant:
        allowed_filler_role_pairs = find_legal_pairs(role_assigner)
    elif dataset_type == "digits" and not role_invariant and use_observed_pairs_for_digits:
        allowed_filler_role_pairs = find_observed_filler_role_pairs(dataset["train"])
        print(f"[INFO] Observed digits filler-role pairs: {len(allowed_filler_role_pairs)}")
        if digits_exclude_pairs_with_presence_ge is not None:
            kept_pairs, removed_pairs = filter_filler_role_pairs_by_presence(
                dataset["train"],
                allowed_filler_role_pairs,
                max_presence=digits_exclude_pairs_with_presence_ge,
            )
            print(
                "[INFO] Pair filtering by presence >= "
                f"{digits_exclude_pairs_with_presence_ge:.3f}: "
                f"{len(allowed_filler_role_pairs)} -> {len(kept_pairs)} "
                f"(removed {len(removed_pairs)})"
            )
            allowed_filler_role_pairs = kept_pairs

    sae_config.setdefault("digits_exclude_pairs_with_presence_ge", digits_exclude_pairs_with_presence_ge)
    sae_config.setdefault("decoder_bias_source", decoder_bias_source)
    sae_config.setdefault("construction_calibration_split", construction_calibration_split)
    sae_config.setdefault("construction_calibration_samples", construction_calibration_samples)
    sae_config.setdefault("decoder_pinv_whiten", decoder_pinv_whiten)
    sae_config.setdefault("decoder_pinv_regularization", decoder_pinv_regularization)
    sae_config.setdefault("decoder_pinv_l2_lambda", decoder_pinv_l2_lambda)
    sae_config.setdefault("decoder_pinv_atol", decoder_pinv_atol)
    sae_config.setdefault("decoder_pinv_topk", decoder_pinv_topk)
    sae_config.setdefault("feature_rescale_strategy", feature_rescale_strategy)
    sae_config.setdefault("feature_rescale_eps", feature_rescale_eps)
    sae_config.setdefault("decoder_refinement", decoder_refinement)
    sae_config.setdefault("decoder_refinement_l2", decoder_refinement_l2)
    
    sae = SparseAutoencoder.from_tensor_product_encoder(
        tpe_for_sae,
        sae_config,
        tpe_output_layer_regularization=tpe_output_layer_regularization,
        tpe_output_layer_regularization_value=tpe_output_layer_regularization_value,
        filler_unbinding=filler_unbinding,
        role_unbinding=role_unbinding,
        role_pinv_regularization=role_pinv_regularization,
        role_pinv_l2_lambda=role_pinv_l2_lambda,
        role_pinv_atol=role_pinv_atol,
        role_pinv_topk=role_pinv_topk,
        role_invariant=role_invariant,
        first_layer_construction=first_layer_construction,
        second_layer_construction=second_layer_construction,
        allowed_filler_role_pairs=allowed_filler_role_pairs,
        bias_anchor=bias_anchor,
        construction_calibration_embeddings=construction_calibration_embeddings,
        decoder_pinv_whiten=decoder_pinv_whiten,
        decoder_pinv_regularization=decoder_pinv_regularization,
        decoder_pinv_l2_lambda=decoder_pinv_l2_lambda,
        decoder_pinv_atol=decoder_pinv_atol,
        decoder_pinv_topk=decoder_pinv_topk,
        feature_rescale_strategy=feature_rescale_strategy,
        feature_rescale_eps=feature_rescale_eps,
        decoder_refinement=decoder_refinement,
        decoder_refinement_l2=decoder_refinement_l2,
        gating_calibration_embeddings=gating_calibration_embeddings,
        gating_strategy=gating_strategy,
        gating_target_sparsity=gating_target_sparsity,
        gating_mad_scale=gating_mad_scale,
    )

    filler_names: Dict[int, str] = {}
    role_names: Dict[int, str] = {}
    if dataset_type == "sentences":
        filler_names, role_names = _sentence_feature_semantics(role_assigner)
    elif dataset_type == "digits":
        filler_names, role_names = _digits_feature_semantics(tokenizer, tpe.config, role_scheme)
    feature_map = getattr(sae.config, "feature_map", None)
    if feature_map:
        # Convert raw filler/role ids into human-readable names for the stored config
        named_map: Dict[str, str] = {}
        for feature_idx, meta in feature_map.items():
            filler_id = meta.get("filler_id") if isinstance(meta, dict) else None
            role_id = meta.get("role_id") if isinstance(meta, dict) else None
            filler_label = filler_names.get(int(filler_id)) if filler_id is not None else None
            if filler_label is None and filler_id is not None:
                filler_label = f"filler_{int(filler_id)}"
            if role_id is None:
                named_map[feature_idx] = filler_label or f"feature_{feature_idx}"
            else:
                role_label = role_names.get(int(role_id), f"role_{int(role_id)}")
                filler_label = filler_label or f"filler_{int(filler_id)}"
                named_map[feature_idx] = f"{filler_label} x {role_label}"
        sae.config.feature_map = named_map

    sae = sae.to(device)
    sae.save_pretrained(sae_output_dir)
    print(f"[INFO] Saved SAE model to {sae_output_dir}")

    # Evaluate the SAE
    # Collect forward-pass tensors here so sae.py stays metric-only.
    eval_embeddings = []
    eval_reconstructions = []
    eval_activations = []
    sae.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["target_embeddings"].to(device)
            encoded = sae.encode(inputs)
            decoded = sae.decode(encoded)
            eval_embeddings.append(inputs.detach().cpu())
            eval_reconstructions.append(decoded.detach().cpu())
            eval_activations.append(encoded.detach().cpu())
    label_loader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=_collate)
    label_activations = []
    with torch.no_grad():
        for batch in label_loader:
            inputs = batch["target_embeddings"].to(device)
            encoded = sae.encode(inputs)
            label_activations.append(encoded.detach().cpu())
    eval_embeddings = torch.cat(eval_embeddings, dim=0)
    eval_reconstructions = torch.cat(eval_reconstructions, dim=0)
    eval_activations = torch.cat(eval_activations, dim=0)
    label_activations = torch.cat(label_activations, dim=0)
    evaluate_sae(
        sae=sae,
        label_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        eval_embeddings=eval_embeddings,
        eval_reconstructions=eval_reconstructions,
        eval_activations=eval_activations,
        label_activations=label_activations,
        sae_output_dir=sae_output_dir,
        label_mode=eval_label_mode,
    )


if __name__ == "__main__":
    gin.external_configurable(load_sentences, module="sentences")
    gin.external_configurable(load_digits, module="digits")
    gin.external_configurable(SparseAutoencoder, module="sae")
    parse_args_for_gin()
    main()
