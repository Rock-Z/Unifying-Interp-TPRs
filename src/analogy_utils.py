import numpy as np
from typing import Any, Optional

__all__ = [
    "cosine_similarity",
    "compute_analogy_stats",
    "evaluate_analogy",
    "print_results",
    "print_sanity_examples",
    "digits_without_special_tokens",
]


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two numpy arrays."""
    u = np.asarray(u)
    v = np.asarray(v)
    num = np.dot(u, v)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(num / denom)


def compute_analogy_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate accuracy statistics for analogy evaluation results."""
    if not results:
        return {
            k: 0.0
            for k in [
                "count",
                "mean_rank",
                "median_rank",
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "top_10_accuracy",
                "mean_target_similarity",
            ]
        }

    return {
        "count": len(results),
        "mean_rank": float(np.mean([r["rank"] for r in results])),
        "median_rank": float(np.median([r["rank"] for r in results])),
        "top_1_accuracy": float(np.mean([r["top_1_correct"] for r in results])),
        "top_3_accuracy": float(np.mean([r["top_3_correct"] for r in results])),
        "top_5_accuracy": float(np.mean([r["top_5_correct"] for r in results])),
        "top_10_accuracy": float(np.mean([r["top_10_correct"] for r in results])),
        "mean_target_similarity": float(
            np.mean([r["target_similarity"] for r in results])
        ),
    }


def evaluate_analogy(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
    emb_C: np.ndarray,
    target_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_items: list[str],
) -> dict[str, Any]:
    """Evaluate a single analogy: A - B + C ≈ D."""
    analogy_vec = emb_A - emb_B + emb_C
    similarities = np.array([
        cosine_similarity(analogy_vec, emb) for emb in all_embeddings
    ])
    target_sim = cosine_similarity(analogy_vec, target_emb)
    rank = int(np.sum(similarities > target_sim) + 1)
    top_indices = np.argsort(similarities)[::-1]
    return {
        "rank": rank,
        "target_similarity": target_sim,
        "top_1_correct": rank <= 1,
        "top_3_correct": rank <= 3,
        "top_5_correct": rank <= 5,
        "top_10_correct": rank <= 10,
        "top_predictions": [
            (all_items[i], float(similarities[i])) for i in top_indices[:10]
        ],
        "total_candidates": len(all_items),
    }


def print_sanity_examples(
    results: list[dict[str, Any]], num_examples: int = 3, title: str = "SANITY CHECK EXAMPLES"
) -> None:
    """Print random analogy examples with their top predictions."""
    if not results or num_examples <= 0:
        return

    import random

    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        atype = r.get("analogy_type", "unknown")
        by_type.setdefault(atype, []).append(r)

    # Stratified sampling: one per type, then fill up to num_examples
    examples: list[dict[str, Any]] = []
    types = list(by_type.keys())
    random.shuffle(types)
    for atype in types:
        if len(examples) < num_examples:
            examples.append(random.choice(by_type[atype]))
        else:
            break

    remaining = num_examples - len(examples)
    if remaining > 0:
        # Pool of all results not already chosen
        chosen_set = set(id(r) for r in examples)
        others = [r for r in results if id(r) not in chosen_set]
        if len(others) > 0:
            examples.extend(random.sample(others, min(len(others), remaining)))

    # If still not enough (e.g., not enough unique results), just use as many as possible
    examples = examples[:min(len(examples), num_examples)]

    print("\n" + "=" * 70)
    print(f"{title} (showing top 5 predictions for {len(examples)} analogies)")
    print("=" * 70)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}/{len(examples)} ({ex.get('analogy_type', 'unknown')})")
        print(
            f"Analogy: {ex.get('sentence_A', '[missing]')} - {ex.get('sentence_B', '[missing]')} + {ex.get('sentence_C', '[missing]')} = {ex.get('sentence_D', '[missing]')}"
        )
        top_preds = ex.get("top_predictions", [])
        target = ex.get("sentence_D", "[missing]")
        found_in_top5 = False
        for j, (pred, sim) in enumerate(top_preds[:5], 1):
            mark = "✓" if pred == target else " "
            if mark == "✓":
                found_in_top5 = True
            print(f"  {j:2d}. [{mark}] {pred} (sim: {sim:.4f})")
        if not found_in_top5:
            # Print the target's rank and similarity if not in top 5
            rank = ex.get('rank', 'unknown')
            target_sim = ex.get('target_similarity', 0.0)
            print("   ...")
            print(f"  {rank:2d}. [✓] {target} (sim: {target_sim:.4f})")


def print_results(stats: dict[str, float], model_name: str, prefix: str = "") -> None:
    """Pretty-print aggregated analogy results."""
    print(f"\n{prefix}Results for {model_name}")
    print("=" * 50)
    print(f"Total analogies: {stats['count']}")
    print(f"Mean rank: {stats['mean_rank']:.2f}")
    print(f"Top-1 accuracy: {stats['top_1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {stats['top_3_accuracy']:.3f}")
    print(f"Top-5 accuracy: {stats['top_5_accuracy']:.3f}")


def digits_without_special_tokens(seq: str) -> str:
    """Remove BOS/SEP/EOS tokens from a digit sequence string."""
    # Special tokens may be glued to the first digit (e.g., "<bos>11"), so strip
    # by substring replacement before splitting.
    cleaned = seq
    for token in ("<bos>", "<sep>", "<eos>", "<pad>"):
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())
