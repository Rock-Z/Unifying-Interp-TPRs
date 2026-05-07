import os
import gin
import json
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Literal
from tqdm import tqdm
import torch
import einops

from sentences import load_sentences
from utils import load_dataset_with_embeddings, parse_args_for_gin, set_random_seed
from model import TensorProductEncoderForPretraining
from analogy_utils import cosine_similarity, compute_analogy_stats, evaluate_analogy, print_results, print_sanity_examples


def batch_rank_analogies_np(
    analogy_vectors: np.ndarray,
    target_indices: list[int],
    all_sentences: list[str],
    all_embeddings: np.ndarray,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Compute rank and top-k predictions for analogy vectors against a candidate pool.

    This is a vectorized equivalent of per-analogy cosine ranking:
    for each analogy vector, compute cosine similarity against all candidate
    embeddings, determine the target's rank, and collect the top-k predictions.

    Args:
        analogy_vectors: Array of shape [B, D] with analogy vectors.
        target_indices: List of length B with the target sentence indices.
        all_sentences: Candidate sentence list aligned with all_embeddings.
        all_embeddings: Array of shape [N, D] for candidate embeddings.
        batch_size: Number of analogy vectors to score per batch.

    Returns:
        A list of result dicts (one per analogy) containing rank, target
        similarity, top-k correctness flags, and top-10 predictions.
    """
    embeddings = all_embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_matrix = embeddings / (norms + 1e-12)
    total_candidates = embedding_matrix.shape[0]
    results: list[dict[str, Any]] = []
    target_indices_arr = np.array(target_indices, dtype=int)

    for start in range(0, len(analogy_vectors), batch_size):
        batch_vecs = analogy_vectors[start:start + batch_size].astype(np.float32)
        vec_norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
        batch_vecs = batch_vecs / (vec_norms + 1e-12)
        sims = embedding_matrix @ batch_vecs.T  # [N, B]

        batch_targets = target_indices_arr[start:start + batch_size]
        batch_ids = np.arange(batch_vecs.shape[0])
        target_sim = sims[batch_targets, batch_ids]
        ranks = (sims > target_sim).sum(axis=0) + 1

        k = min(10, total_candidates)
        top_idx = np.argsort(sims, axis=0)[-k:][::-1]
        top_vals = np.take_along_axis(sims, top_idx, axis=0)

        for i in range(batch_vecs.shape[0]):
            predictions = [
                (all_sentences[int(idx)], float(top_vals[j, i]))
                for j, idx in enumerate(top_idx[:, i])
            ]
            rank = int(ranks[i])
            results.append(
                {
                    "rank": rank,
                    "target_similarity": float(target_sim[i]),
                    "top_1_correct": rank <= 1,
                    "top_3_correct": rank <= 3,
                    "top_5_correct": rank <= 5,
                    "top_10_correct": rank <= 10,
                    "top_predictions": predictions,
                    "total_candidates": total_candidates,
                }
            )

    return results


def get_filler_for_role(roles_dict, role_name, role_assigner):
    """
    Get the filler ID for a specific role name from a roles dictionary.
    
    This function safely extracts the filler ID for a given role name by:
    1. Looking up the role index from role_assigner.role2idx
    2. Finding the position in role_ids where that role index appears
    3. Returning the corresponding filler_id at that position
    
    This approach is more robust than hardcoded positions because it works
    regardless of the order of roles in the tuples.
    
    Args:
        roles_dict: Dictionary with 'filler_ids' and 'role_ids' tuples
        role_name: Name of the role (e.g., 'subject', 'object', 'verb')
        role_assigner: SVORoleAssigner object containing role2idx mapping
        
    Returns:
        The filler ID corresponding to the specified role
        
    Raises:
        ValueError: If the role name is not found in role_assigner.role2idx or role_ids
    """
    role2idx = role_assigner.role2idx
    if role_name not in role2idx:
        raise ValueError(f"Role '{role_name}' not found in role2idx: {list(role2idx.keys())}")
    
    role_idx = role2idx[role_name]
    filler_ids = roles_dict['filler_ids']
    role_ids = roles_dict['role_ids']
    
    # Find the position where this role appears
    for pos, role_id in enumerate(role_ids):
        if role_id == role_idx:
            return filler_ids[pos]
    
    raise ValueError(f"Role '{role_name}' (idx={role_idx}) not found in role_ids: {role_ids}")


def group_by_fixed_roles(sentences, sentence_role_assigner, fixed_role_names):
    """Group sentences by the values of fixed roles.
    
    This function groups sentences that have the same values for the specified
    fixed roles, enabling systematic analogy creation.
    
    Example:
        For subject analogies (vary subject, fix verb+object):
        Input sentences:
            "the doctor will see the patient ."  # roles: (doctor, see, patient)
            "the lawyer will see the patient ."  # roles: (lawyer, see, patient)  
            "the doctor will see the student ."  # roles: (doctor, see, student)
            "the lawyer will see the student ."  # roles: (lawyer, see, student)
        
        With fixed_role_names=['verb', 'object']:
        Returns:
            {
                (see, patient): ["the doctor will see the patient .", "the lawyer will see the patient ."],
                (see, student): ["the doctor will see the student .", "the lawyer will see the student ."]
            }
    
    Args:
        sentences: List of sentences to group
        sentence_role_assigner: SVORoleAssigner object to assign roles to sentences
        fixed_role_names: List of role names to use for grouping (e.g., ['verb', 'object'])
        
    Returns:
        Dictionary where keys are tuples of filler IDs for fixed roles,
        and values are lists of sentences with those fixed role values
    """
    groups = {}
    role2idx = sentence_role_assigner.role2idx
    
    # Infer role_ids_structure from a sample sentence instead of hardcoding
    if not sentences:
        return groups
    
    sample_sentence = sentences[0]
    sample_roles = sentence_role_assigner.get_roles(sample_sentence)
    role_ids_structure = sample_roles['role_ids']
    
    # Create mapping from role_id to position
    role_to_position = {}
    for pos, role_id in enumerate(role_ids_structure):
        role_to_position[role_id] = pos
    
    fixed_positions = []
    for role_name in fixed_role_names:
        if role_name not in role2idx:
            raise ValueError(f"Role '{role_name}' not found in role2idx: {list(role2idx.keys())}")
        role_idx = role2idx[role_name]
        if role_idx not in role_to_position:
            raise ValueError(f"Role index {role_idx} not found in role_ids: {role_ids_structure}")
        position = role_to_position[role_idx]
        fixed_positions.append(position)
    
    for s in sentences:
        roles = sentence_role_assigner.get_roles(s)
        key = tuple(roles['filler_ids'][i] for i in fixed_positions)
        groups.setdefault(key, []).append(s)
    return groups


def create_analogy_quadruples(dataset, sentence_role_assigner, max_analogies: Optional[int] = None, random_seed: Optional[int] = None):
    """
    Create analogy quadruples of the form A - B + C = D.
    For sentences like "the X will Y the Z", we create analogies by changing one role while keeping others constant.
    
    Args:
        dataset: Dataset containing sentences
        sentence_role_assigner: Object to assign roles to sentences
        max_analogies: Maximum number of analogies to create (None for all possible)
        random_seed: Random seed for reproducibility when sampling
        
    Returns:
        List of dictionaries with keys 'A', 'B', 'C', 'D', 'analogy_type'
    """
    import random
    if random_seed is not None:
        random.seed(random_seed)
    
    quadruples = []
    sentences = dataset['test']['sentence']
    role2idx = sentence_role_assigner.role2idx

    def create_role_analogies(sentences, sentence_role_assigner, role_to_vary, fixed_role_names, analogy_type, quadruples):
        """Create analogies for a specific role by varying one role and fixing others."""
        try:
            grouped = group_by_fixed_roles(sentences, sentence_role_assigner, fixed_role_names)
            for fixed_vals, sent_list in grouped.items():
                if len(sent_list) < 2:
                    continue
                # Map each value of the varied role to its sentence
                value_to_sent = {get_filler_for_role(sentence_role_assigner.get_roles(s), role_to_vary, sentence_role_assigner): s for s in sent_list}
                values = list(value_to_sent.keys())
                for val_A in values:
                    for val_C in values:
                        if val_A == val_C:
                            continue
                        sent_A, sent_D = value_to_sent[val_A], value_to_sent[val_C]
                        # Find another group with both val_A and val_C present
                        for other_vals, other_sents in grouped.items():
                            if other_vals == fixed_vals:
                                continue
                            other_value_to_sent = {get_filler_for_role(sentence_role_assigner.get_roles(s), role_to_vary, sentence_role_assigner): s for s in other_sents}
                            if val_A in other_value_to_sent and val_C in other_value_to_sent:
                                sent_B, sent_C = other_value_to_sent[val_A], other_value_to_sent[val_C]
                                quadruples.append({
                                    "A": sent_A,
                                    "B": sent_B,
                                    "C": sent_C,
                                    "D": sent_D,
                                    "analogy_type": analogy_type
                                })
                                break
        except Exception as e:
            print(f"[ERROR] Failed to create {analogy_type} analogies: {e}")
            # Print some debug info
            sample_sent = list(sentences)[0]
            sample_roles = sentence_role_assigner.get_roles(sample_sent)
            print(f"[DEBUG] Sample sentence: {sample_sent}")
            print(f"[DEBUG] Sample roles: {sample_roles}")
            print(f"[DEBUG] role2idx: {role2idx}")
            print(f"[DEBUG] role_to_vary: {role_to_vary}")
            print(f"[DEBUG] fixed_role_names: {fixed_role_names}")
            raise

    # Create analogies based on available roles
    # Using role names instead of hardcoded positions makes this more robust
    # and less prone to errors when role order changes
    if 'subject' in role2idx and 'object' in role2idx and 'verb' in role2idx:
        # Subject analogies: vary subject, fix verb+object
        create_role_analogies(sentences, sentence_role_assigner, 'subject', ['verb', 'object'], 'subject_analogy', quadruples)
        # Object analogies: vary object, fix subject+verb  
        create_role_analogies(sentences, sentence_role_assigner, 'object', ['subject', 'verb'], 'object_analogy', quadruples)
    else:
        print(f"[WARNING] Role scheme does not support subject/object analogies. Available: {list(role2idx.keys())}")

    # Sample if needed
    if max_analogies and len(quadruples) > max_analogies:
        quadruples = random.sample(quadruples, max_analogies)

    return quadruples



def evaluate_analogy_tpe(tpe_model, sentence_role_assigner, quadruples, all_sentences, all_embeddings, sent_to_emb, verbose=True):
    """
    Evaluate analogies using TPE role/filler bindings for the manipulated role only, with A and D in sentence embedding space.
    For B and C, only the manipulated role/filler pair is encoded.
    Analogy: analogy_vec = emb_A - binding_B + binding_C, compared to all sentence embeddings.
    """
    results = []
    role2idx = sentence_role_assigner.role2idx
    analogy_vectors = []
    target_indices = []
    sent_to_index = {s: i for i, s in enumerate(all_sentences)}

    for quad in tqdm(quadruples, disable=not verbose):
        sent_A, sent_B, sent_C, sent_D, analogy_type = quad["A"], quad["B"], quad["C"], quad["D"], quad["analogy_type"]
        # Determine role name from analogy type
        if analogy_type == 'subject_analogy' and 'subject' in role2idx:
            role_name = 'subject'
        elif analogy_type == 'object_analogy' and 'object' in role2idx:
            role_name = 'object'
        else:
            continue

        # Get sentence embeddings for A and D
        emb_A = sent_to_emb[sent_A]
        emb_D = sent_to_emb[sent_D]

        # Get the manipulated filler for B and C using role names
        roles_B = sentence_role_assigner.get_roles(sent_B)
        roles_C = sentence_role_assigner.get_roles(sent_C)
        filler_id_B = get_filler_for_role(roles_B, role_name, sentence_role_assigner)
        filler_id_C = get_filler_for_role(roles_C, role_name, sentence_role_assigner)
        role_id = role2idx[role_name]  # always the same for both

        # Prepare tensors for the manipulated role/filler pair
        filler_id_B_tensor = einops.rearrange(torch.tensor(filler_id_B, dtype=torch.long), '-> 1 1')
        filler_id_C_tensor = einops.rearrange(torch.tensor(filler_id_C, dtype=torch.long), '-> 1 1')
        role_id_tensor = einops.rearrange(torch.tensor(role_id, dtype=torch.long), '-> 1 1')
        # Encode only the manipulated role/filler pair for B and C using the TPE model's forward method
        with torch.no_grad():
            out_B = tpe_model(filler_ids=filler_id_B_tensor, role_ids=role_id_tensor)
            binding_B = einops.rearrange(out_B.encoder_hidden_states, '1 1 d -> d').cpu().numpy()
            out_C = tpe_model(filler_ids=filler_id_C_tensor, role_ids=role_id_tensor)
            binding_C = einops.rearrange(out_C.encoder_hidden_states, '1 1 d -> d').cpu().numpy()

        # Compute analogy vector
        analogy_vec = emb_A - binding_B + binding_C
        analogy_vectors.append(analogy_vec)
        target_indices.append(sent_to_index[sent_D])

        results.append(
            {
                "analogy_type": analogy_type,
                "sentence_A": sent_A,
                "sentence_B": sent_B,
                "sentence_C": sent_C,
                "sentence_D": sent_D,
            }
        )

    batch_results = batch_rank_analogies_np(
        np.array(analogy_vectors),
        target_indices,
        all_sentences,
        all_embeddings,
    )

    for base, batch in zip(results, batch_results):
        base.update(batch)

    return results


@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str] = None,
    role_scheme: str = "svo",
    max_analogies: Optional[int] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
    print_examples: int = 3,
    tpe_checkpoint_dir: str = "experiments/sentences/checkpoints/modernbert/tpe/best_model",
    evaluation_mode: Literal["sentence", "tpe", "both"] = "both",
) -> Dict[str, Any]:
    """
    Evaluate analogy performance on sentence embeddings.
    
    Args:
        sentences_path: Path to the sentences dataset directory
        embedding_model_name: Name of the sentence transformer model
        embedding_cache_path: Path to cache embeddings (optional)
        role_scheme: Role scheme to use ("svo" or "bow")
        max_analogies: Maximum number of analogies to evaluate (None for 4x dataset size)
        output_path: Path to save detailed results (optional, currently disabled)
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress information
        print_examples: Number of sanity check examples to print (0 for none)
        tpe_checkpoint_dir: Path to TPE model checkpoint directory
        evaluation_mode: Which embeddings to evaluate ("sentence", "tpe", or "both")
        
    Returns:
        Dictionary with evaluation results including overall statistics,
        breakdown by analogy type, and TPE results (if applicable)
    """
    
    # Validate evaluation mode
    valid_modes = {"sentence", "tpe", "both"}
    if evaluation_mode not in valid_modes:
        raise ValueError(f"evaluation_mode must be one of {valid_modes}, got '{evaluation_mode}'")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        try:
            import torch
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        except ImportError:
            pass
        if verbose:
            print(f"[INFO] Set random seed to {random_seed}")
            print(f"[INFO] Evaluation mode: {evaluation_mode}")
    
    # Load dataset and embeddings
    if verbose:
        print(f"[INFO] Loading dataset from {sentences_path} with role scheme '{role_scheme}'")
    
    dataset, sentence_role_assigner = load_sentences(sentences_path, role_scheme=role_scheme)
    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=sentences_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="embeddings",
        add_prefix="search_query: " if embedding_model_name.startswith("nomic-ai") else "",
    )
    
    if verbose:
        print(f"[INFO] Loaded {len(dataset['test'])} test sentences, embedding dim: {embedding_dim}")

    # Set default max_analogies
    if max_analogies is None:
        max_analogies = 4 * len(dataset['test'])

    # Create analogy quadruples
    if verbose:
        print("[INFO] Creating analogy quadruples...")
    quadruples = create_analogy_quadruples(dataset, sentence_role_assigner, max_analogies, random_seed)
    if verbose:
        print(f"[INFO] Created {len(quadruples)} analogy quadruples")

    # Prepare data for evaluations
    test_sentences = dataset['test']['sentence']
    # Candidate pool includes all splits (train/valid/test) for ranking.
    all_sentences = []
    all_embeddings = []
    for split_name in ["train", "valid", "test"]:
        if split_name not in dataset:
            continue
        all_sentences.extend(dataset[split_name]["sentence"])
        all_embeddings.extend(dataset[split_name]["embeddings"])
    all_embeddings = np.array(all_embeddings)
    sent_to_emb = dict(zip(all_sentences, all_embeddings))
    
    # Initialize results variables
    results = []
    overall_stats = {}
    analogy_stats = {}
    analogy_type_results = {"subject_analogy": [], "object_analogy": []}
    
    # Evaluate standard embeddings (if requested)
    if evaluation_mode in {"sentence", "both"}:
        if verbose:
            print("[INFO] Evaluating standard embeddings...")

        analogy_vectors = []
        target_indices = []
        sent_to_index = {s: i for i, s in enumerate(all_sentences)}
        base_payload = []

        for quad in tqdm(quadruples, disable=not verbose):
            sent_A, sent_B, sent_C, sent_D, analogy_type = quad["A"], quad["B"], quad["C"], quad["D"], quad["analogy_type"]
            analogy_vectors.append(sent_to_emb[sent_A] - sent_to_emb[sent_B] + sent_to_emb[sent_C])
            target_indices.append(sent_to_index[sent_D])
            base_payload.append(
                {
                    "analogy_type": analogy_type,
                    "sentence_A": sent_A,
                    "sentence_B": sent_B,
                    "sentence_C": sent_C,
                    "sentence_D": sent_D,
                }
            )

        batch_results = batch_rank_analogies_np(
            np.array(analogy_vectors),
            target_indices,
            all_sentences,
            all_embeddings,
        )

        for base, batch in zip(base_payload, batch_results):
            base.update(batch)
            results.append(base)
            analogy_type_results[base["analogy_type"]].append(base)

        # Compute standard embedding statistics
        overall_stats = compute_analogy_stats(results)
        analogy_stats = {k: compute_analogy_stats(v) for k, v in analogy_type_results.items()}

        if verbose:
            print_results(overall_stats, embedding_model_name, prefix="Sentence Embeddings ")

        # Print sanity check examples
        if print_examples > 0:
            print_sanity_examples(results, print_examples)

    # Initialize TPE results variables
    tpe_results = []
    tpe_overall_stats = {}
    tpe_analogy_stats = {}
    
    # Evaluate TPE embeddings (if requested)
    if evaluation_mode in {"tpe", "both"}:
        if verbose:
            print(f"\n[INFO] Loading TPE model from {tpe_checkpoint_dir}")
        
        tpe_model = TensorProductEncoderForPretraining.from_pretrained(
            tpe_checkpoint_dir
        )
        tpe_model.eval()
        
        tpe_results = evaluate_analogy_tpe(
            tpe_model, sentence_role_assigner, quadruples, all_sentences, all_embeddings, sent_to_emb, verbose=verbose
        )
        
        # Compute TPE statistics
        tpe_overall_stats = compute_analogy_stats(tpe_results)
        tpe_analogy_stats = {k: compute_analogy_stats([r for r in tpe_results if r['analogy_type'] == k]) 
                            for k in analogy_type_results.keys()}
        
        if verbose:
            print_results(tpe_overall_stats, "TPE Model", "TPE ")
        
        # Print TPE sanity check examples
        if print_examples > 0:
            print_sanity_examples(tpe_results, print_examples, "TPE SANITY CHECK EXAMPLES")

    # Compile results
    final_results = {
        'metadata': {
            'sentences_path': sentences_path,
            'embedding_model_name': embedding_model_name,
            'role_scheme': role_scheme,
            'total_analogies_evaluated': len(quadruples),
            'evaluation_mode': evaluation_mode,
        }
    }
    
    # Add sentence embedding results if evaluated
    if evaluation_mode in {"sentence", "both"}:
        final_results['sentence_embeddings'] = {
            'overall_statistics': overall_stats,
            'by_analogy_type': analogy_stats,
            'detailed_results': results,
        }
    
    # Add TPE embedding results if evaluated
    if evaluation_mode in {"tpe", "both"}:
        final_results['tpe_embeddings'] = {
            'overall_statistics': tpe_overall_stats,
            'by_analogy_type': tpe_analogy_stats,
            'detailed_results': tpe_results,
        }

    # Save results to JSON if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        if verbose:
            print(f"[INFO] Results saved to {output_path}")

    return final_results


if __name__ == "__main__":
    parse_args_for_gin()
    main() 
