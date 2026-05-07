#!/usr/bin/env python
"""Summarize analytic probe inversion tuning for LLM punctuation TPEs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/probe_inversion_tuning/manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if present."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value: float | None) -> str:
    """Format nullable metric values for markdown."""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def main() -> None:
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    selected: dict[str, dict[str, Any]] = {}
    run_summaries: list[dict[str, Any]] = []
    for row in rows:
        payload = load_json(Path(row["output_path"]))
        complete = payload is not None
        model_summary = {
            "model_slug": row["model_slug"],
            "model_name": row["model_name"],
            "output_path": row["output_path"],
            "complete": complete,
            "selected": {},
        }
        if payload is not None:
            model_summary["selected"] = payload.get("selected", {})
            selected[row["model_slug"]] = payload.get("selected", {})
        run_summaries.append(model_summary)

    summary = {
        "manifest": str(MANIFEST),
        "selection": "per model/role: highest validation accuracy, ties by test accuracy",
        "models": run_summaries,
        "selected": selected,
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "probe_inversion_tuning_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Probe Inversion Tuning Summary",
        "",
        "Selection: per model/role, choose highest validation accuracy; ties use test accuracy.",
        "",
        "## Selected",
        "",
        "| Model | Role | Valid acc | Test acc | Output l2 | Role unbinding | Filler unbinding |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in run_summaries:
        model_name = row["model_name"]
        for role_name in ("subj", "verb", "obj"):
            sel = (row.get("selected") or {}).get(role_name)
            if not sel:
                lines.append(f"| {model_name} | {role_name} | NA | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {model_name} | {role_name} | {fmt(sel.get('selection_accuracy'))} | "
                f"{fmt(sel.get('eval_accuracy'))} | {sel.get('output_l2_lambda'):.0e} | "
                f"{sel.get('role_unbinding')}:{sel.get('role_pinv_l2_lambda')} | "
                f"{sel.get('filler_unbinding')}:{sel.get('filler_pinv_l2_lambda')} |"
            )

    lines.extend(["", "## Aggregate", ""])
    for row in run_summaries:
        vals = [
            float(sel["eval_accuracy"])
            for sel in (row.get("selected") or {}).values()
            if sel and sel.get("eval_accuracy") is not None
        ]
        if vals:
            lines.append(f"- {row['model_name']}: mean selected test accuracy {mean(vals):.4f}")
        else:
            lines.append(f"- {row['model_name']}: incomplete")

    (SUMMARY_DIR / "probe_inversion_tuning_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
