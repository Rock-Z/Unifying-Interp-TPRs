#!/usr/bin/env python3
"""Summarize supervised SAE seed checks for LLM punctuation sentence runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/sae_baselines/supervised_seed_manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"


METRICS = (
    "mse",
    "r2",
    "cosine_similarity",
    "l0_sparsity",
    "avg_feature_well_rankedness",
    "avg_feature_accuracy",
)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MANIFEST.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metrics_path = Path(row["output_dir"]) / "sae_results/metrics.json"
            row["metrics_path"] = str(metrics_path)
            row["complete"] = metrics_path.exists()
            if metrics_path.exists():
                row.update(json.loads(metrics_path.read_text()))
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("complete"):
            grouped[str(row["model_slug"])].append(row)

    aggregates: dict[str, dict[str, Any]] = {}
    for model_slug, model_rows in sorted(grouped.items()):
        stats: dict[str, Any] = {
            "n": len(model_rows),
            "model_name": model_rows[0]["model_name"],
        }
        for metric in METRICS:
            values = [as_float(row.get(metric)) for row in model_rows]
            clean = [value for value in values if value is not None]
            if not clean:
                continue
            stats[f"{metric}_mean"] = mean(clean)
            stats[f"{metric}_std"] = pstdev(clean) if len(clean) > 1 else 0.0
            stats[f"{metric}_min"] = min(clean)
            stats[f"{metric}_max"] = max(clean)
        aggregates[model_slug] = stats
    return aggregates


def write_markdown(rows: list[dict[str, Any]], aggregates: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Supervised SAE Seed Check Summary",
        "",
        "## Aggregates",
        "",
        "| Model | N | R2 mean | R2 std | WR mean | WR std | MSE mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_slug in sorted(aggregates):
        stats = aggregates[model_slug]
        lines.append(
            f"| {stats['model_name']} | {stats['n']} | {fmt(stats.get('r2_mean'))} | "
            f"{fmt(stats.get('r2_std'))} | {fmt(stats.get('avg_feature_well_rankedness_mean'))} | "
            f"{fmt(stats.get('avg_feature_well_rankedness_std'))} | {fmt(stats.get('mse_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Complete | R2 | WR | MSE | L0 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['complete']} | {fmt(as_float(row.get('r2')))} | "
            f"{fmt(as_float(row.get('avg_feature_well_rankedness')))} | "
            f"{fmt(as_float(row.get('mse')))} | {fmt(as_float(row.get('l0_sparsity')))} |"
        )

    (SUMMARY_DIR / "sae_supervised_seed_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    aggregates = aggregate(rows)
    summary = {
        "manifest": str(MANIFEST),
        "rows": rows,
        "aggregates": aggregates,
    }
    (SUMMARY_DIR / "sae_supervised_seed_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(rows, aggregates)


if __name__ == "__main__":
    main()
