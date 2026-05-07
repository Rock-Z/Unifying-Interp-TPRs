#!/usr/bin/env python3
"""Build summary artifacts for LLM last-layer punctuation sentence experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
SUMMARY_DIR = ROOT / "results/summary"
RESULTS_MD = ROOT / "RESULTS.md"

MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "olmo_13b": "allenai/OLMo-2-1124-13B",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pick_stats(payload: dict[str, Any] | None, section: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    stats = payload.get(section, {}).get("overall_statistics")
    return stats if isinstance(stats, dict) else None


def summarize_probe(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    rows = payload.get("results")
    if not isinstance(rows, list):
        return None
    return {
        str(row.get("role_name")): {
            "analytic_accuracy": row.get("analytic_accuracy"),
            "trained_accuracy": row.get("trained_accuracy"),
        }
        for row in rows
    }


def apply_probe_tuning(
    roles: dict[str, Any] | None,
    selected_tuning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Overlay validation-selected analytic probe tuning results onto role rows."""
    if roles is None or not selected_tuning:
        return roles
    tuned_roles = dict(roles)
    for role_name, tuning_row in selected_tuning.items():
        if role_name not in tuned_roles:
            continue
        row = dict(tuned_roles[role_name])
        row["analytic_accuracy_untuned"] = row.get("analytic_accuracy")
        row["analytic_accuracy"] = tuning_row.get("eval_accuracy")
        row["analytic_tuning"] = {
            "selection_accuracy": tuning_row.get("selection_accuracy"),
            "selection_split": tuning_row.get("selection_split"),
            "eval_split": tuning_row.get("eval_split"),
            "output_l2_lambda": tuning_row.get("output_l2_lambda"),
            "role_unbinding": tuning_row.get("role_unbinding"),
            "role_pinv_l2_lambda": tuning_row.get("role_pinv_l2_lambda"),
            "filler_unbinding": tuning_row.get("filler_unbinding"),
            "filler_pinv_l2_lambda": tuning_row.get("filler_pinv_l2_lambda"),
            "unbinding_variant": tuning_row.get("unbinding_variant"),
        }
        tuned_roles[role_name] = row
    return tuned_roles


def summarize() -> dict[str, Any]:
    tpe_summary = load_json(SUMMARY_DIR / "tpe_sweep_summary.json")
    sae_baseline_summary = load_json(SUMMARY_DIR / "sae_baseline_summary.json")
    topk_tuning_summary = load_json(SUMMARY_DIR / "sae_topk_tuning_summary.json")
    probe_tuning_summary = load_json(SUMMARY_DIR / "probe_inversion_tuning_summary.json")
    sae_baselines = (sae_baseline_summary or {}).get("by_model", {})
    for model_slug, selected_topk in (topk_tuning_summary or {}).get("selected", {}).items():
        sae_baselines.setdefault(model_slug, {})["topk"] = selected_topk
    summary: dict[str, Any] = {
        "metadata": {
            "representation": "decoder-only-punct final-layer punctuation-token hidden state",
            "sentences_path": "data/sentences",
        },
        "tpe": {},
        "analogy": {},
        "probe": {},
        "sae": {},
        "sae_baselines": sae_baselines,
    }
    seed_check_summary = load_json(SUMMARY_DIR / "probe_seed_check_summary.json")
    if seed_check_summary is not None:
        summary["probe_seed_check"] = seed_check_summary.get("aggregates", {})
    if probe_tuning_summary is not None:
        summary["probe_inversion_tuning"] = probe_tuning_summary

    selected_tpes = (tpe_summary or {}).get("selected", {})
    selected_probe_tuning = (probe_tuning_summary or {}).get("selected", {})
    for slug, model_name in MODELS.items():
        selected = selected_tpes.get(slug)
        summary["tpe"][slug] = selected

        analogy = load_json(ROOT / f"results/analogy/{slug}/eval.json")
        summary["analogy"][slug] = {
            "model": model_name,
            "sentence_embeddings": pick_stats(analogy, "sentence_embeddings"),
            "tpe_embeddings": pick_stats(analogy, "tpe_embeddings"),
        }

        probe = load_json(ROOT / f"results/probe/{slug}/probe_compare_results_svo.json")
        probe_roles = summarize_probe(probe)
        summary["probe"][slug] = {
            "model": model_name,
            "roles": apply_probe_tuning(probe_roles, selected_probe_tuning.get(slug)),
        }

        sae = load_json(ROOT / f"checkpoints/sae/{slug}/metrics.json")
        summary["sae"][slug] = {
            "model": model_name,
            "metrics": sae,
        }

    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# LLM Last-Layer Sentence Results",
        "",
        "Representation: decoder-only punctuation-token hidden states on `data/sentences`.",
        "",
        "Baseline coverage: analogy compares nearest-neighbor retrieval in raw hidden-state space against TPE space; probes compare analytic TPE probes against trained linear probes; SAE compares analytic ridge construction against trained TopK and supervised filler-role SAEs.",
        "",
        "## TPE Selection",
        "",
        "| Model | Run | R2 | EV |",
        "| --- | --- | ---: | ---: |",
    ]
    for slug, model_name in MODELS.items():
        tpe = summary["tpe"].get(slug) or {}
        lines.append(
            f"| {model_name} | {fmt(tpe.get('run_id'))} | "
            f"{fmt(tpe.get('r_squared'))} | {fmt(tpe.get('explained_variance'))} |"
        )

    lines.extend(
        [
            "",
            "## Nearest-Neighbor Analogy",
            "",
            "| Model | Raw hidden NN top1 | Raw hidden NN top3 | TPE NN top1 | TPE NN top3 | Raw hidden mean rank | TPE mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slug, model_name in MODELS.items():
        row = summary["analogy"][slug]
        emb = row.get("sentence_embeddings") or {}
        tpe = row.get("tpe_embeddings") or {}
        lines.append(
            f"| {model_name} | {fmt(emb.get('top_1_accuracy'))} | "
            f"{fmt(emb.get('top_3_accuracy'))} | {fmt(tpe.get('top_1_accuracy'))} | "
            f"{fmt(tpe.get('top_3_accuracy'))} | {fmt(emb.get('mean_rank'))} | "
            f"{fmt(tpe.get('mean_rank'))} |"
        )

    lines.extend(
        [
            "",
            "## Linear Probes",
            "",
            "Analytic accuracies use validation-selected inversion parameters when `probe_inversion_tuning_summary.*` is present; untuned values are retained in `summary.json`.",
            "",
            "| Model | Role | Analytic acc | Trained acc |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for slug, model_name in MODELS.items():
        roles = summary["probe"][slug].get("roles") or {}
        if not roles:
            lines.append(f"| {model_name} | pending | pending | pending |")
            continue
        for role, vals in roles.items():
            lines.append(
                f"| {model_name} | {role} | {fmt(vals.get('analytic_accuracy'))} | "
                f"{fmt(vals.get('trained_accuracy'))} |"
            )

    seed_check = summary.get("probe_seed_check", {})
    if seed_check:
        lines.extend(
            [
                "",
                "## Probe Seed Check",
                "",
                "| Model | Role | N | Trained mean | Trained std | Trained min | Trained max |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for slug, model_name in MODELS.items():
            for role, vals in sorted((seed_check.get(slug) or {}).items()):
                lines.append(
                    f"| {model_name} | {role} | {fmt(vals.get('n'))} | "
                    f"{fmt(vals.get('trained_mean'))} | {fmt(vals.get('trained_std'))} | "
                    f"{fmt(vals.get('trained_min'))} | {fmt(vals.get('trained_max'))} |"
                )

    lines.extend(
        [
            "",
            "## SAE",
            "",
            "| Model | Method | MSE | R2 | Cosine | L0 | Purity | Accuracy | Well-rankedness |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slug, model_name in MODELS.items():
        metrics = summary["sae"][slug].get("metrics") or {}
        lines.append(
            f"| {model_name} | analytic ridge | {fmt(metrics.get('mse'))} | {fmt(metrics.get('r2'))} | "
            f"{fmt(metrics.get('cosine_similarity'))} | {fmt(metrics.get('l0_sparsity'))} | "
            f"{fmt(metrics.get('avg_feature_purity'))} | {fmt(metrics.get('avg_feature_accuracy'))} | "
            f"{fmt(metrics.get('avg_feature_well_rankedness'))} |"
        )
        baselines = summary.get("sae_baselines", {}).get(slug) or {}
        for baseline_name in ("topk", "supervised"):
            baseline = baselines.get(baseline_name) or {}
            if not baseline:
                continue
            lines.append(
                f"| {model_name} | trained {baseline_name} | {fmt(baseline.get('mse'))} | "
                f"{fmt(baseline.get('r2'))} | {fmt(baseline.get('cosine_similarity'))} | "
                f"{fmt(baseline.get('l0_sparsity'))} | {fmt(baseline.get('avg_feature_purity'))} | "
                f"{fmt(baseline.get('avg_feature_accuracy'))} | "
                f"{fmt(baseline.get('avg_feature_well_rankedness'))} |"
            )

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize()
    (SUMMARY_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(summary)


if __name__ == "__main__":
    main()
