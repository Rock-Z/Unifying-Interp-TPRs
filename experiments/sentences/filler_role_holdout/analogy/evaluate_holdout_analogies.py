"""Evaluate TPE additive analogies on the SVO filler-role holdout split.

Usage:
    uv run experiments/sentences/filler_role_holdout/analogy/evaluate_holdout_analogies.py \
        experiments/sentences/filler_role_holdout/analogy/configs/modernbert_tpe_generalization.gin
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gin
import einops
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analogy_utils import compute_analogy_stats, print_results, print_sanity_examples
from model import TensorProductEncoderForPretraining
from sentences import load_sentences
from utils import load_dataset_with_embeddings, parse_args_for_gin


def batch_rank_analogies_np(
    analogy_vectors: np.ndarray,
    target_indices: list[int],
    all_sentences: list[str],
    all_embeddings: np.ndarray,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Rank analogy vectors against a candidate sentence embedding pool."""

    embeddings = all_embeddings.astype(np.float32)
    embedding_matrix = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    total_candidates = embedding_matrix.shape[0]
    target_indices_arr = np.array(target_indices, dtype=int)
    results: list[dict[str, Any]] = []

    for start in range(0, len(analogy_vectors), batch_size):
        batch_vecs = analogy_vectors[start:start + batch_size].astype(np.float32)
        batch_vecs = batch_vecs / (np.linalg.norm(batch_vecs, axis=1, keepdims=True) + 1e-12)
        sims = embedding_matrix @ batch_vecs.T

        batch_targets = target_indices_arr[start:start + batch_size]
        batch_ids = np.arange(batch_vecs.shape[0])
        target_sim = sims[batch_targets, batch_ids]
        ranks = (sims > target_sim).sum(axis=0) + 1

        k = min(10, total_candidates)
        top_idx = np.argsort(sims, axis=0)[-k:][::-1]
        top_vals = np.take_along_axis(sims, top_idx, axis=0)

        for i in range(batch_vecs.shape[0]):
            results.append(
                {
                    "rank": int(ranks[i]),
                    "target_similarity": float(target_sim[i]),
                    "top_1_correct": int(ranks[i]) <= 1,
                    "top_3_correct": int(ranks[i]) <= 3,
                    "top_5_correct": int(ranks[i]) <= 5,
                    "top_10_correct": int(ranks[i]) <= 10,
                    "top_predictions": [
                        (all_sentences[int(idx)], float(top_vals[j, i]))
                        for j, idx in enumerate(top_idx[:, i])
                    ],
                    "total_candidates": total_candidates,
                }
            )
    return results


def get_filler_for_role(roles_dict: dict, role_name: str, role_assigner) -> int:
    """Return the filler id assigned to a named SVO role."""

    role_idx = role_assigner.role2idx[role_name]
    for pos, role_id in enumerate(roles_dict["role_ids"]):
        if role_id == role_idx:
            return roles_dict["filler_ids"][pos]
    raise ValueError(f"Role '{role_name}' not found in role_ids: {roles_dict['role_ids']}")


def group_by_fixed_roles(sentences: Sequence[str], sentence_role_assigner, fixed_role_names: Sequence[str]) -> dict:
    """Group sentences by the filler ids of the fixed roles."""

    groups: dict[tuple[int, ...], list[str]] = {}
    if not sentences:
        return groups

    sample_roles = sentence_role_assigner.get_roles(sentences[0])
    role_to_position = {role_id: pos for pos, role_id in enumerate(sample_roles["role_ids"])}
    fixed_positions = [
        role_to_position[sentence_role_assigner.role2idx[role_name]]
        for role_name in fixed_role_names
    ]

    for sentence in sentences:
        roles = sentence_role_assigner.get_roles(sentence)
        key = tuple(roles["filler_ids"][pos] for pos in fixed_positions)
        groups.setdefault(key, []).append(sentence)
    return groups


def create_holdout_analogy_quadruples(
    dataset,
    sentence_role_assigner,
    *,
    analogy_split: str,
    max_analogies: Optional[int],
    random_seed: Optional[int],
) -> list[dict[str, str]]:
    """Create SVO analogies from a configurable split, typically generalization."""

    if analogy_split not in dataset:
        raise ValueError(f"Analogy split '{analogy_split}' not found. Available: {list(dataset.keys())}")

    if random_seed is not None:
        random.seed(random_seed)

    quadruples: list[dict[str, str]] = []
    sentences = dataset[analogy_split]["sentence"]
    role2idx = sentence_role_assigner.role2idx

    def create_role_analogies(role_to_vary: str, fixed_role_names: list[str], analogy_type: str) -> None:
        grouped = group_by_fixed_roles(sentences, sentence_role_assigner, fixed_role_names)
        for fixed_vals, sent_list in grouped.items():
            if len(sent_list) < 2:
                continue

            value_to_sent = {
                get_filler_for_role(sentence_role_assigner.get_roles(sentence), role_to_vary, sentence_role_assigner): sentence
                for sentence in sent_list
            }
            values = list(value_to_sent.keys())
            for val_a in values:
                for val_c in values:
                    if val_a == val_c:
                        continue
                    sent_a = value_to_sent[val_a]
                    sent_d = value_to_sent[val_c]
                    for other_vals, other_sents in grouped.items():
                        if other_vals == fixed_vals:
                            continue
                        other_value_to_sent = {
                            get_filler_for_role(
                                sentence_role_assigner.get_roles(sentence),
                                role_to_vary,
                                sentence_role_assigner,
                            ): sentence
                            for sentence in other_sents
                        }
                        if val_a in other_value_to_sent and val_c in other_value_to_sent:
                            quadruples.append(
                                {
                                    "A": sent_a,
                                    "B": other_value_to_sent[val_a],
                                    "C": other_value_to_sent[val_c],
                                    "D": sent_d,
                                    "analogy_type": analogy_type,
                                }
                            )
                            break

    if {"subject", "object", "verb"}.issubset(role2idx):
        create_role_analogies("subject", ["verb", "object"], "subject_analogy")
        create_role_analogies("object", ["subject", "verb"], "object_analogy")
    else:
        raise ValueError(f"SVO analogies require subject/object/verb roles. Available: {list(role2idx.keys())}")

    if max_analogies and len(quadruples) > max_analogies:
        quadruples = random.sample(quadruples, max_analogies)
    return quadruples


def evaluate_holdout_analogy_tpe(
    tpe_model,
    sentence_role_assigner,
    quadruples: Sequence[dict[str, str]],
    all_sentences: list[str],
    all_embeddings: np.ndarray,
    sent_to_emb: dict[str, np.ndarray],
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Evaluate heldout analogies by replacing one TPE filler-role binding."""

    role2idx = sentence_role_assigner.role2idx
    sent_to_index = {sentence: i for i, sentence in enumerate(all_sentences)}
    analogy_vectors = []
    target_indices = []
    results = []

    for quad in tqdm(quadruples, disable=not verbose):
        sent_a = quad["A"]
        sent_b = quad["B"]
        sent_c = quad["C"]
        sent_d = quad["D"]
        analogy_type = quad["analogy_type"]
        if analogy_type == "subject_analogy":
            role_name = "subject"
        elif analogy_type == "object_analogy":
            role_name = "object"
        else:
            continue

        roles_b = sentence_role_assigner.get_roles(sent_b)
        roles_c = sentence_role_assigner.get_roles(sent_c)
        filler_id_b = get_filler_for_role(roles_b, role_name, sentence_role_assigner)
        filler_id_c = get_filler_for_role(roles_c, role_name, sentence_role_assigner)
        role_id = role2idx[role_name]

        filler_id_b_tensor = einops.rearrange(torch.tensor(filler_id_b, dtype=torch.long), "-> 1 1")
        filler_id_c_tensor = einops.rearrange(torch.tensor(filler_id_c, dtype=torch.long), "-> 1 1")
        role_id_tensor = einops.rearrange(torch.tensor(role_id, dtype=torch.long), "-> 1 1")
        with torch.no_grad():
            out_b = tpe_model(filler_ids=filler_id_b_tensor, role_ids=role_id_tensor)
            binding_b = einops.rearrange(out_b.encoder_hidden_states, "1 1 d -> d").cpu().numpy()
            out_c = tpe_model(filler_ids=filler_id_c_tensor, role_ids=role_id_tensor)
            binding_c = einops.rearrange(out_c.encoder_hidden_states, "1 1 d -> d").cpu().numpy()

        analogy_vectors.append(sent_to_emb[sent_a] - binding_b + binding_c)
        target_indices.append(sent_to_index[sent_d])
        results.append(
            {
                "analogy_type": analogy_type,
                "sentence_A": sent_a,
                "sentence_B": sent_b,
                "sentence_C": sent_c,
                "sentence_D": sent_d,
            }
        )

    rank_results = batch_rank_analogies_np(
        np.array(analogy_vectors),
        target_indices,
        all_sentences,
        all_embeddings,
    )
    for result, rank_result in zip(results, rank_results):
        result.update(rank_result)
    return results


@gin.configurable
def main(
    sentences_path: str,
    embedding_model_name: str,
    tpe_checkpoint_dir: str,
    embedding_cache_path: Optional[str] = None,
    role_scheme: str = "svo",
    analogy_split: str = "generalization",
    candidate_splits: Sequence[str] = ("train", "valid", "test", "generalization"),
    max_analogies: Optional[int] = None,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
    print_examples: int = 3,
    evaluation_mode: str = "tpe",
) -> Dict[str, Any]:
    """Evaluate heldout TPE additive analogies and save JSON metrics."""

    if evaluation_mode != "tpe":
        raise ValueError("evaluate_holdout_analogies.py only supports evaluation_mode='tpe'.")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    dataset, sentence_role_assigner = load_sentences(sentences_path, role_scheme=role_scheme)
    missing_splits = [split for split in (analogy_split, *candidate_splits) if split not in dataset]
    if missing_splits:
        raise ValueError(f"Missing split(s) {sorted(set(missing_splits))}. Available: {list(dataset.keys())}")

    dataset, embedding_dim = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=sentences_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
        embedding_column_name="embeddings",
        add_prefix="search_query: " if embedding_model_name.startswith("nomic-ai") else "",
    )

    if max_analogies is None:
        max_analogies = 4 * len(dataset[analogy_split])

    quadruples = create_holdout_analogy_quadruples(
        dataset,
        sentence_role_assigner,
        analogy_split=analogy_split,
        max_analogies=max_analogies,
        random_seed=random_seed,
    )

    all_sentences: list[str] = []
    all_embeddings: list[np.ndarray] = []
    for split_name in candidate_splits:
        all_sentences.extend(dataset[split_name]["sentence"])
        all_embeddings.extend(dataset[split_name]["embeddings"])
    all_embeddings_array = np.array(all_embeddings)
    sent_to_emb = dict(zip(all_sentences, all_embeddings_array))

    if verbose:
        print(f"[INFO] Embedding dim: {embedding_dim}")
        print(f"[INFO] Analogy split: {analogy_split} ({len(dataset[analogy_split])} sentences)")
        print(f"[INFO] Candidate splits: {list(candidate_splits)} ({len(all_sentences)} sentences)")
        print(f"[INFO] Created {len(quadruples)} heldout analogies")
        print(f"[INFO] Loading TPE model from {tpe_checkpoint_dir}")

    tpe_model = TensorProductEncoderForPretraining.from_pretrained(tpe_checkpoint_dir)
    tpe_model.eval()

    tpe_results = evaluate_holdout_analogy_tpe(
        tpe_model,
        sentence_role_assigner,
        quadruples,
        all_sentences,
        all_embeddings_array,
        sent_to_emb,
        verbose=verbose,
    )
    overall_stats = compute_analogy_stats(tpe_results)
    analogy_stats = {
        analogy_type: compute_analogy_stats([r for r in tpe_results if r["analogy_type"] == analogy_type])
        for analogy_type in ("subject_analogy", "object_analogy")
    }

    if verbose:
        print_results(overall_stats, "TPE Holdout Model", "Heldout TPE ")
    if print_examples > 0:
        print_sanity_examples(tpe_results, print_examples, "HELDOUT TPE SANITY CHECK EXAMPLES")

    results = {
        "metadata": {
            "sentences_path": sentences_path,
            "embedding_model_name": embedding_model_name,
            "role_scheme": role_scheme,
            "tpe_checkpoint_dir": tpe_checkpoint_dir,
            "analogy_split": analogy_split,
            "candidate_splits": list(candidate_splits),
            "total_analogies_evaluated": len(quadruples),
        },
        "tpe_embeddings": {
            "overall_statistics": overall_stats,
            "by_analogy_type": analogy_stats,
            "detailed_results": tpe_results,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"[INFO] Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parse_args_for_gin()
    main()
