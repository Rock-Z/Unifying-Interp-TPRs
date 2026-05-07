#!/usr/bin/env python3
"""Write LaTeX tables for LLM punctuation sentence results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
SUMMARY_PATH = ROOT / "results/summary/summary.json"
OUT_DIR = ROOT / "results/summary/paper_tables"

MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "olmo_13b": "OLMo-2-13B",
    "gpt_oss_20b": "GPT-OSS-20B",
}


def pct(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100 * float(value):.1f}"


def num(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def write_tpe_table(summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Model & $R^2$ & EV \\\\",
        "\\midrule",
    ]
    for slug, label in MODEL_LABELS.items():
        row = summary["tpe"].get(slug) or {}
        lines.append(f"{label} & {num(row.get('r_squared'))} & {num(row.get('explained_variance'))} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(OUT_DIR / "llm_tpe_table.tex", lines)


def write_analogy_table(summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{NN Hidden} & \\multicolumn{2}{c}{TPE} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Model & Top-1 & Top-3 & Top-1 & Top-3 \\\\",
        "\\midrule",
    ]
    for slug, label in MODEL_LABELS.items():
        row = summary["analogy"].get(slug) or {}
        nn = row.get("sentence_embeddings") or {}
        tpe = row.get("tpe_embeddings") or {}
        lines.append(
            f"{label} & {pct(nn.get('top_1_accuracy'))} & {pct(nn.get('top_3_accuracy'))} & "
            f"{pct(tpe.get('top_1_accuracy'))} & {pct(tpe.get('top_3_accuracy'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(OUT_DIR / "llm_analogy_table.tex", lines)


def write_probe_table(summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Model & Role & Analytic & Trained \\\\",
        "\\midrule",
    ]
    role_labels = {"subj": "Subject", "verb": "Verb", "obj": "Object"}
    for slug, label in MODEL_LABELS.items():
        roles = (summary["probe"].get(slug) or {}).get("roles") or {}
        for role in ("subj", "verb", "obj"):
            vals = roles.get(role) or {}
            lines.append(
                f"{label} & {role_labels[role]} & {pct(vals.get('analytic_accuracy'))} & "
                f"{pct(vals.get('trained_accuracy'))} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(OUT_DIR / "llm_probe_table.tex", lines)


def write_sae_table(summary: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & SAE Type & $R^2$ & Quality & L0 \\\\",
        "\\midrule",
    ]
    for slug, label in MODEL_LABELS.items():
        analytic = ((summary["sae"].get(slug) or {}).get("metrics") or {})
        baselines = (summary.get("sae_baselines", {}).get(slug) or {})
        rows = [
            ("TPE-Constr.", analytic),
            ("Top-$k$", baselines.get("topk") or {}),
            ("Supervised", baselines.get("supervised") or {}),
        ]
        for method, metrics in rows:
            lines.append(
                f"{label} & {method} & {num(metrics.get('r2'))} & "
                f"{pct(metrics.get('avg_feature_well_rankedness'))} & "
                f"{num(metrics.get('l0_sparsity'))} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(OUT_DIR / "llm_sae_table.tex", lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    write_tpe_table(summary)
    write_analogy_table(summary)
    write_probe_table(summary)
    write_sae_table(summary)


if __name__ == "__main__":
    main()
