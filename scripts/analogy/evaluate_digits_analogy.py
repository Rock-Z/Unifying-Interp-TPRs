import json
import os
import gin
import numpy as np
from typing import Optional, Literal, Dict, Any, Callable
from tqdm import tqdm
import torch
import random

from digits import load_digits, tokenize_function, get_roles
from model import RecurrentEncoderDecoderModel, TensorProductEncoderForPretraining
from utils import parse_args_for_gin, set_random_seed
from analogy_utils import (
    cosine_similarity,
    compute_analogy_stats,
    evaluate_analogy,
    print_results,
    print_sanity_examples,
    digits_without_special_tokens,
)


def create_digit_analogy_quadruples(
    sequences: list[str],
    max_analogies: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    """Create analogy quadruples for digit sequences by dynamically generating intermediate sentences."""

    if not max_analogies:
        max_analogies = len(sequences * 4)

    parsed = [digits_without_special_tokens(s).split() for s in sequences]
    if not parsed:
        raise ValueError("No sequences provided")
    
    # Group sequences by length
    sequences_by_length = {}
    for seq in parsed:
        length = len(seq)
        if length not in sequences_by_length:
            sequences_by_length[length] = []
        sequences_by_length[length].append(seq)

    distinct_lengths = [length for length in sequences_by_length.keys() if len(sequences_by_length[length]) >= 2]
    vocab = set()
    for seq in sequences_by_length.values():
        for s in seq:
            vocab.update(s)
    
    quadruples = []
    generated_sentences = set()  # Track generated sentences to add to evaluation set
    original_sentences = []
    if random_seed is not None:
        random.seed(random_seed)

    # First sample by length
    samples_each_length = max_analogies // (len(distinct_lengths))
    # Sample by length
    for length in distinct_lengths:
        seqs = sequences_by_length[length]
        if len(seqs[0]) < 2:
            continue # with less than 2 digits, we can't create an analogy
        seqs = random.sample(seqs, min(samples_each_length, len(seqs)))
        original_sentences.extend(seqs)

    # Generate analogies for each original sentence
    for seq_A in original_sentences:
        seq_len = len(seq_A)
        seq_B = seq_A.copy()
        seq_C = seq_A.copy()
        seq_D = seq_A.copy()
        
        # Sample random positions to change
        n_changes = random.randint(1, seq_len // 2) if seq_len > 2 else 1
        change_positions = random.sample(range(seq_len), n_changes)
        kept_positions = [pos for pos in range(seq_len) if pos not in change_positions]
        for pos in kept_positions:
            seq_B[pos] = random.choice([v for v in vocab if v != seq_A[pos]])
        for pos in change_positions:
            seq_D[pos] = seq_C[pos] = random.choice([v for v in vocab if v != seq_A[pos]])
        for pos in kept_positions:
            seq_C[pos] = seq_B[pos]     
        
        # Convert to strings
        def to_string(seq):
            # Ensure BOS is separated so digits_without_special_tokens keeps the first digit.
            return "<bos> " + " ".join(seq) + " <sep>"

        str_A = to_string(seq_A)
        str_B = to_string(seq_B)
        str_C = to_string(seq_C)
        str_D = to_string(seq_D)
        
        # Add generated sentences to the set
        generated_sentences = generated_sentences.union({str_B, str_C, str_D})
        
        quadruples.append({
            "A": str_A,
            "B": str_B,
            "C": str_C,
            "D": str_D,
            "analogy_type": f"position_{'_'.join(map(str, change_positions))}",
            "different_positions": change_positions,
        })
    
    # Return quadruples and generated sentences
    return quadruples, list(generated_sentences)


def _parse_digit_sequence(sequence: str) -> list[int]:
    return [int(token) for token in digits_without_special_tokens(sequence).split()]


def _contains_any_holdout_pair(sequence: str, holdout_pairs: list[tuple[int, int]]) -> bool:
    digits = _parse_digit_sequence(sequence)
    heldout_lookup = {(int(filler), int(position)) for filler, position in holdout_pairs}
    return any((digit, position) in heldout_lookup for position, digit in enumerate(digits, start=1))


def _quadruple_contains_any_holdout_pair(
    quadruple: dict[str, Any],
    holdout_pairs: list[tuple[int, int]],
) -> bool:
    return any(
        _contains_any_holdout_pair(quadruple[key], holdout_pairs)
        for key in ("A", "B", "C", "D")
    )


def _quadruple_has_changed_holdout_pair(
    quadruple: dict[str, Any],
    holdout_pairs: list[tuple[int, int]],
) -> bool:
    digits_A = _parse_digit_sequence(quadruple["A"])
    digits_B = _parse_digit_sequence(quadruple["B"])
    digits_C = _parse_digit_sequence(quadruple["C"])
    digits_D = _parse_digit_sequence(quadruple["D"])
    heldout_by_position = {int(position): int(filler) for filler, position in holdout_pairs}

    for zero_indexed_position in quadruple["different_positions"]:
        position = int(zero_indexed_position) + 1
        filler = heldout_by_position.get(position)
        if filler is None:
            continue
        if (digits_A[zero_indexed_position] == filler and digits_B[zero_indexed_position] == filler) or (
            digits_C[zero_indexed_position] == filler and digits_D[zero_indexed_position] == filler
        ):
            return True
    return False


def create_filtered_digit_analogy_quadruples(
    sequences: list[str],
    accept_quadruple: Callable[[dict[str, Any]], bool],
    max_analogies: Optional[int] = None,
    random_seed: Optional[int] = None,
    batch_size_multiplier: int = 4,
    max_attempts_multiplier: int = 50,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Reuse the standard quartet generator and keep only accepted quartets."""
    if not max_analogies:
        max_analogies = len(sequences * 4)

    accepted_quadruples = []
    generated_sentences = set()
    seen_quadruples = set()
    attempts = 0
    max_attempts = max(max_analogies * max_attempts_multiplier, 1)

    while len(accepted_quadruples) < max_analogies and attempts < max_attempts:
        remaining = max_analogies - len(accepted_quadruples)
        batch_size = max(remaining * batch_size_multiplier, remaining)
        batch_seed = None if random_seed is None else random_seed + attempts
        candidate_quadruples, _ = create_digit_analogy_quadruples(
            sequences,
            max_analogies=batch_size,
            random_seed=batch_seed,
        )
        attempts += len(candidate_quadruples)
        for quadruple in candidate_quadruples:
            key = (quadruple["A"], quadruple["B"], quadruple["C"], quadruple["D"])
            if key in seen_quadruples or not accept_quadruple(quadruple):
                continue
            accepted_quadruples.append(quadruple)
            seen_quadruples.add(key)
            generated_sentences.update([quadruple["B"], quadruple["C"], quadruple["D"]])
            if len(accepted_quadruples) >= max_analogies:
                break

    if len(accepted_quadruples) < max_analogies:
        raise RuntimeError(
            f"Could only collect {len(accepted_quadruples)} accepted analogies out of requested {max_analogies}"
        )

    return accepted_quadruples, list(generated_sentences), attempts


def _encode_sequences(dataset, model, tokenizer):
    """Return embeddings and mapping from sequence string to embedding."""
    embeddings = []
    seq_to_emb = {}
    model.eval()
    for sequence in dataset:
        # Handle both string sequences and record dictionaries
        if isinstance(sequence, str):
            inp = sequence
        else:
            inp = sequence["input"]
        
        # Manually tokenize the input sequence
        tokenized = tokenizer(inp, return_tensors="pt", padding=False, truncation=True)
        input_ids = tokenized["input_ids"]
        input_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long)
        
        with torch.no_grad():
            model.eval()
            out = model.encoder(input_ids=input_ids, input_lengths=input_lengths)
            hidden = out.last_hidden_state
            if isinstance(hidden, tuple):
                hidden = torch.cat(list(hidden), dim=-1)
            hidden = hidden.reshape(hidden.size(0), -1)
        key = inp
        emb = hidden.squeeze(0).cpu().numpy()
        embeddings.append(emb)
        seq_to_emb[key] = emb
    return np.stack(embeddings), seq_to_emb


def evaluate_analogy_tpe_digits(tpe_model, quadruples, test_sequences, test_embeddings, seq_to_emb, tokenizer, verbose=True):
    results = []
    # role scheme is typically stored in the model config
    role_scheme = tpe_model.config.role_scheme
    if verbose:
        print(f"Inferred TPE model role scheme: {role_scheme}")
    for quad in tqdm(quadruples, disable=not verbose):
        A, B, C, D, analogy_type = quad["A"], quad["B"], quad["C"], quad["D"], quad["analogy_type"]
        
        # Tokenize sequences B and C to get their role assignments
        tokenized_B = tokenizer(B, return_tensors="pt", padding=False, truncation=True)
        tokenized_C = tokenizer(C, return_tensors="pt", padding=False, truncation=True)
        
        # Get role assignments for B and C using get_roles function
        filler_ids_B, role_ids_B = get_roles(tokenized_B["input_ids"], tokenized_B["attention_mask"], role_scheme=role_scheme)
        filler_ids_C, role_ids_C = get_roles(tokenized_C["input_ids"], tokenized_C["attention_mask"], role_scheme=role_scheme)
        
        # Collect all changed positions for batch processing
        changed_positions = quad["different_positions"]

        # +1 because of <bos> token
        # for bidirectional role scheme, we need to add the second half of the role ids
        changed_role_positions = [pos + 1 for pos in changed_positions]
        if role_scheme == "bidirectional":
            changed_role_positions += [pos + filler_ids_B.shape[1] // 2 for pos in changed_role_positions]

        filler_ids_B_batch = [filler_ids_B[0, pos].item() for pos in changed_role_positions]
        filler_ids_C_batch = [filler_ids_C[0, pos].item() for pos in changed_role_positions]
        role_ids_batch = [role_ids_B[0, pos].item() for pos in changed_role_positions] 
        
        # Create batch tensors for single forward pass
        fid_B_batch = torch.tensor([filler_ids_B_batch], dtype=torch.long)  # [1, num_positions]
        fid_C_batch = torch.tensor([filler_ids_C_batch], dtype=torch.long)  # [1, num_positions]
        rid_batch = torch.tensor([role_ids_batch], dtype=torch.long)        # [1, num_positions]
        
        with torch.no_grad():
            # Single forward pass for all changed positions
            b_out = tpe_model(filler_ids=fid_B_batch, role_ids=rid_batch)
            bindings_B = b_out.encoder_hidden_states.squeeze(0).cpu().numpy()  # [num_positions, hidden_dim]
            
            c_out = tpe_model(filler_ids=fid_C_batch, role_ids=rid_batch)
            bindings_C = c_out.encoder_hidden_states.squeeze(0).cpu().numpy()  # [num_positions, hidden_dim]
            
            # Sum binding differences across all changed positions
            total_binding_diff = np.sum(bindings_C - bindings_B, axis=0)
        
        emb_A = seq_to_emb[A]
        emb_D = seq_to_emb[D]
        analogy_vec = emb_A + total_binding_diff
        #analogy_vec = emb_A
        
        similarities = np.array([cosine_similarity(analogy_vec, emb) for emb in test_embeddings])
        target_sim = cosine_similarity(analogy_vec, emb_D)
        rank = np.sum(similarities > target_sim) + 1
        top_indices = np.argsort(similarities)[::-1]
        results.append({
            "rank": rank,
            "target_similarity": target_sim,
            "top_1_correct": rank <= 1,
            "top_3_correct": rank <= 3,
            "top_5_correct": rank <= 5,
            "top_10_correct": rank <= 10,
            "top_predictions": [(test_sequences[i], similarities[i]) for i in top_indices[:10]],
            "total_candidates": len(test_sequences),
            "analogy_type": analogy_type,
            "sentence_A": A,
            "sentence_B": B,
            "sentence_C": C,
            "sentence_D": D,
        })
    return results


@gin.configurable
def main(
    seq2seq_checkpoint_dir: str,
    tpe_checkpoint_dir: str,
    digits_prefix: Optional[str] = None,
    data_paths_dict: Optional[dict[str, str]] = None,
    max_analogies: Optional[int] = None,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
    print_examples: int = 3,
    evaluation_mode: Literal["nn", "tpe", "both"] = "both",
    heldout_pairs: Optional[list[tuple[int, int]]] = None,
    iid_split_name: str = "test",
    generalization_split_name: str = "generalization",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate analogy performance on the digits task."""
    
    set_random_seed(random_seed)
    if data_paths_dict is not None:
        paths = data_paths_dict
    elif digits_prefix is not None:
        paths = {
            "train": f"{digits_prefix}.train",
            "valid": f"{digits_prefix}.valid",
            "test": f"{digits_prefix}.test",
        }
    else:
        raise ValueError("Either digits_prefix or data_paths_dict must be provided")
    dataset, tokenizer = load_digits(paths)

    seq2seq_model = RecurrentEncoderDecoderModel.from_pretrained(seq2seq_checkpoint_dir)
    tpe_model = None
    if evaluation_mode in {"tpe", "both"}:
        tpe_model = TensorProductEncoderForPretraining.from_pretrained(tpe_checkpoint_dir)
        tpe_model.eval()

    def run_single_evaluation(
        split_name: str,
        quadruples: list[dict[str, Any]],
        generated_sentences: list[str],
        attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        if verbose:
            print(f"Created {len(quadruples)} analogy quadruples for {split_name}")
            print(f"Generated {len(generated_sentences)} additional sentences for {split_name}")

        all_analogy_sequences = set()
        for quad in quadruples:
            all_analogy_sequences.update([quad["A"], quad["B"], quad["C"], quad["D"]])
        all_analogy_sequences = list(all_analogy_sequences)
        test_embeddings, seq_to_emb = _encode_sequences(all_analogy_sequences, seq2seq_model, tokenizer)
        test_sequences = all_analogy_sequences

        results = []
        overall_stats = {}
        if evaluation_mode in {"nn", "both"}:
            if verbose:
                print(f"Starting Seq2Seq evaluation for {split_name}...")
            for quad in tqdm(quadruples, disable=not verbose):
                A, B, C, D, atype = quad["A"], quad["B"], quad["C"], quad["D"], quad["analogy_type"]
                res = evaluate_analogy(
                    seq_to_emb[A], seq_to_emb[B], seq_to_emb[C], seq_to_emb[D],
                    test_embeddings, test_sequences
                )
                res.update({
                    "analogy_type": atype,
                    "sentence_A": A,
                    "sentence_B": B,
                    "sentence_C": C,
                    "sentence_D": D,
                })
                results.append(res)
            overall_stats = compute_analogy_stats(results)
            if verbose:
                print_results(overall_stats, f"Seq2Seq Encoder ({split_name})")
            if print_examples:
                print_sanity_examples(results, print_examples, f"{split_name} SANITY CHECK EXAMPLES")

        tpe_results = []
        tpe_overall_stats = {}
        if tpe_model is not None:
            tpe_results = evaluate_analogy_tpe_digits(
                tpe_model, quadruples, test_sequences, test_embeddings, seq_to_emb, tokenizer, verbose=verbose
            )
            tpe_overall_stats = compute_analogy_stats(tpe_results)
            if verbose:
                print_results(tpe_overall_stats, f"TPE Model ({split_name})", prefix="TPE ")
            if print_examples:
                print_sanity_examples(tpe_results, print_examples, f"TPE {split_name} SANITY CHECK EXAMPLES")

        output = {
            "quadruple_count": len(quadruples),
            "candidate_sequence_count": len(test_sequences),
            "generated_sequence_count": len(generated_sentences),
            "nn_embeddings": {
                "overall_statistics": overall_stats,
                "detailed_results": results,
            } if evaluation_mode in {"nn", "both"} else {},
            "tpe_embeddings": {
                "overall_statistics": tpe_overall_stats,
                "detailed_results": tpe_results,
            } if evaluation_mode in {"tpe", "both"} else {},
        }
        if attempts is not None:
            output["sampling_attempts"] = attempts
        return output

    if heldout_pairs is None:
        quadruples, generated_sentences = create_digit_analogy_quadruples(dataset["test"]["input"], max_analogies, random_seed)
        output = run_single_evaluation("test", quadruples, generated_sentences)
    else:
        normalized_holdout_pairs = [(int(filler), int(position)) for filler, position in heldout_pairs]
        iid_quadruples, iid_generated_sentences, iid_attempts = create_filtered_digit_analogy_quadruples(
            dataset[iid_split_name]["input"],
            accept_quadruple=lambda quad: not _quadruple_contains_any_holdout_pair(quad, normalized_holdout_pairs),
            max_analogies=max_analogies,
            random_seed=random_seed,
        )
        generalization_quadruples, generalization_generated_sentences, generalization_attempts = create_filtered_digit_analogy_quadruples(
            dataset[generalization_split_name]["input"],
            accept_quadruple=lambda quad: _quadruple_has_changed_holdout_pair(quad, normalized_holdout_pairs),
            max_analogies=max_analogies,
            random_seed=None if random_seed is None else random_seed + 100000,
        )
        output = {
            "heldout_pairs": [
                {"filler": int(filler), "position": int(position)}
                for filler, position in normalized_holdout_pairs
            ],
            "splits": {
                "test_iid_clean": run_single_evaluation(
                    "test_iid_clean",
                    iid_quadruples,
                    iid_generated_sentences,
                    attempts=iid_attempts,
                ),
                "generalization_changed_holdout": run_single_evaluation(
                    "generalization_changed_holdout",
                    generalization_quadruples,
                    generalization_generated_sentences,
                    attempts=generalization_attempts,
                ),
            },
        }
    if output_path:
        def _json_default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True, default=_json_default)
    return output


if __name__ == "__main__":
    parse_args_for_gin()
    main()
