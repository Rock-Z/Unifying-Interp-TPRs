#!/usr/bin/env python3
"""Summarize and select tuned Top-k SAE baselines for LLM punctuation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/sae_baselines/topk_tuning_manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"


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


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            output_dir = Path(str(row["output_dir"]))
            metrics_path = output_dir / "sae_results/metrics.json"
            row["metrics_path"] = str(metrics_path)
            row["complete"] = metrics_path.exists()
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                row.update(metrics)
            rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], r2_gap: float) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("complete"):
            by_model.setdefault(str(row["model_slug"]), []).append(row)

    selected: dict[str, dict[str, Any]] = {}
    for model_slug, model_rows in by_model.items():
        scored = [row for row in model_rows if as_float(row.get("r2")) is not None]
        if not scored:
            continue
        best_r2 = max(as_float(row["r2"]) for row in scored)
        candidates = [
            row
            for row in scored
            if as_float(row.get("r2")) is not None
            and as_float(row["r2"]) >= best_r2 - r2_gap
        ]
        selected[model_slug] = max(
            candidates,
            key=lambda row: (
                as_float(row.get("avg_feature_well_rankedness")) or float("-inf"),
                as_float(row.get("r2")) or float("-inf"),
                -(as_float(row.get("l0_sparsity")) or float("inf")),
            ),
        )
    return selected


def write_markdown(rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]], r2_gap: float) -> None:
    lines = [
        "# SAE Top-k Tuning Summary",
        "",
        f"Selection: among complete runs within `{r2_gap}` R2 of the best reconstruction per model, choose highest well-rankedness.",
        "",
        "## Selected",
        "",
        "| Model | Run | Hidden | K | LR | R2 | L0 | Purity | Accuracy | Well-rankedness |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_slug in sorted(selected):
        row = selected[model_slug]
        lines.append(
            f"| {row['model_name']} | {row['run_id']} | {row['hidden_dim']} | {row['k']} | "
            f"{row['learning_rate']} | {fmt(as_float(row.get('r2')))} | "
            f"{fmt(as_float(row.get('l0_sparsity')))} | {fmt(as_float(row.get('avg_feature_purity')))} | "
            f"{fmt(as_float(row.get('avg_feature_accuracy')))} | "
            f"{fmt(as_float(row.get('avg_feature_well_rankedness')))} |"
        )

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| Run | Complete | R2 | MSE | L0 | Well-rankedness |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: str(item["run_id"])):
        lines.append(
            f"| {row['run_id']} | {row['complete']} | {fmt(as_float(row.get('r2')))} | "
            f"{fmt(as_float(row.get('mse')))} | {fmt(as_float(row.get('l0_sparsity')))} | "
            f"{fmt(as_float(row.get('avg_feature_well_rankedness')))} |"
        )

    (SUMMARY_DIR / "sae_topk_tuning_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--r2-gap", type=float, default=0.01)
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    selected = select_rows(rows, r2_gap=args.r2_gap)
    summary = {
        "manifest": str(args.manifest),
        "selection": {
            "primary": "highest well-rankedness among runs within R2 gap of best reconstruction",
            "r2_gap": args.r2_gap,
        },
        "rows": rows,
        "selected": selected,
    }
    (SUMMARY_DIR / "sae_topk_tuning_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(rows, selected, r2_gap=args.r2_gap)


if __name__ == "__main__":
    main()
