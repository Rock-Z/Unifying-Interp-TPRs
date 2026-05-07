#!/usr/bin/env python3
"""Summarize trained SAE baselines for LLM punctuation sentence runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/sae_baselines/manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_metrics(row: dict[str, str]) -> dict[str, Any]:
    metrics_path = Path(row["output_dir"]) / "sae_results/metrics.json"
    out: dict[str, Any] = dict(row)
    out["metrics_path"] = str(metrics_path)
    out["complete"] = metrics_path.exists()
    if metrics_path.exists():
        out.update(json.loads(metrics_path.read_text()))
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [load_metrics(row) for row in read_manifest()]

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_slug"]), {})[str(row["baseline_type"])] = row

    summary = {"runs": rows, "by_model": by_model}
    (SUMMARY_DIR / "sae_baseline_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# SAE Baseline Summary",
        "",
        "| Model | Baseline | Complete | MSE | R2 | Cosine | L0 | Purity | Accuracy | WR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (str(r["model_slug"]), str(r["baseline_type"]))):
        lines.append(
            f"| {row['model_slug']} | {row['baseline_type']} | {row.get('complete')} | "
            f"{fmt(row.get('mse'))} | {fmt(row.get('r2'))} | {fmt(row.get('cosine_similarity'))} | "
            f"{fmt(row.get('l0_sparsity'))} | {fmt(row.get('avg_feature_purity'))} | "
            f"{fmt(row.get('avg_feature_accuracy'))} | {fmt(row.get('avg_feature_well_rankedness'))} |"
        )
    (SUMMARY_DIR / "sae_baseline_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
