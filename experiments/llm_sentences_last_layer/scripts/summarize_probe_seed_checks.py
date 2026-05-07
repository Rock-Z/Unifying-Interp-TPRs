#!/usr/bin/env python3
"""Summarize trained-probe seed sanity checks for LLM punctuation runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/probe_seed_checks/manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_run(row: dict[str, str]) -> dict[str, Any]:
    result_path = Path(row["results_dir"]) / "probe_compare_results_svo.json"
    out: dict[str, Any] = dict(row)
    out["result_path"] = str(result_path)
    out["complete"] = result_path.exists()
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        out["results"] = payload.get("results", [])
    else:
        out["results"] = []
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_role: dict[tuple[str, str], list[float]] = defaultdict(list)
    analytic_by_model_role: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for result in row.get("results", []):
            role = str(result["role_name"])
            key = (str(row["model_slug"]), role)
            by_model_role[key].append(float(result["trained_accuracy"]))
            analytic_by_model_role[key].append(float(result["analytic_accuracy"]))

    aggregates: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model_slug, role), values in sorted(by_model_role.items()):
        analytic_values = analytic_by_model_role[(model_slug, role)]
        aggregates[model_slug][role] = {
            "n": len(values),
            "trained_mean": mean(values),
            "trained_std": pstdev(values) if len(values) > 1 else 0.0,
            "trained_min": min(values),
            "trained_max": max(values),
            "analytic_mean": mean(analytic_values),
        }
    return {"runs": rows, "aggregates": aggregates}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [load_run(row) for row in read_manifest()]
    summary = summarize(rows)
    (SUMMARY_DIR / "probe_seed_check_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Probe Seed Check Summary",
        "",
        "| Model | Role | N | Analytic mean | Trained mean | Trained std | Trained min | Trained max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_slug, roles in sorted(summary["aggregates"].items()):
        for role, stats in sorted(roles.items()):
            lines.append(
                f"| {model_slug} | {role} | {stats['n']} | {fmt(stats['analytic_mean'])} | "
                f"{fmt(stats['trained_mean'])} | {fmt(stats['trained_std'])} | "
                f"{fmt(stats['trained_min'])} | {fmt(stats['trained_max'])} |"
            )
    (SUMMARY_DIR / "probe_seed_check_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
