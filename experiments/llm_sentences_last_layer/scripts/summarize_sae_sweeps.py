#!/usr/bin/env python3
"""Summarize and optionally promote LLM punctuation analytic SAE sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/sae_sweeps/manifest_ridge.tsv"
SUMMARY_DIR = ROOT / "results/summary"
FINAL_CONFIG_DIR = ROOT / "configs/sae_final"
FINAL_CKPT_DIR = ROOT / "checkpoints/sae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary-stem", default="sae_sweep_summary")
    parser.add_argument(
        "--r2-tolerance",
        type=float,
        default=1e-3,
        help="Select by feature quality among runs within this absolute R2 gap from the best run.",
    )
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def read_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_metrics(run: dict[str, str]) -> dict[str, Any]:
    out_dir = Path(run["output_dir"])
    metrics_path = out_dir / "metrics.json"
    row: dict[str, Any] = dict(run)
    row["metrics_path"] = str(metrics_path)
    row["complete"] = metrics_path.exists()
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        row.update(metrics)
    return row


def as_float(value: object, default: float = float("-inf")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def write_promoted_config(row: dict[str, Any]) -> None:
    base = Path(str(row["config_path"])).read_text()
    additions = [
        "",
        "# Selected by experiments/llm_sentences_last_layer/scripts/summarize_sae_sweeps.py",
        f"# run_id = {row['run_id']}",
        f"main.sae_output_dir = 'experiments/llm_sentences_last_layer/checkpoints/sae/{row['model_slug']}'",
        f"main.decoder_refinement = '{row['decoder_refinement']}'",
        f"main.decoder_refinement_l2 = {row['decoder_refinement_l2']}",
        f"main.decoder_bias_source = '{row['decoder_bias_source']}'",
    ]
    FINAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_CONFIG_DIR / f"{row['model_slug']}.gin").write_text(base.rstrip() + "\n".join(additions) + "\n")


def promote(selected: dict[str, dict[str, Any]]) -> None:
    FINAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for model_slug, row in selected.items():
        src = Path(str(row["output_dir"]))
        dst = FINAL_CKPT_DIR / model_slug
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        write_promoted_config(row)


def fmt(value: object) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = [load_metrics(run) for run in read_manifest(args.manifest)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_slug"]), []).append(row)

    selected: dict[str, dict[str, Any]] = {}
    for model_slug, model_rows in grouped.items():
        complete_rows = [r for r in model_rows if r.get("complete")]
        if not complete_rows:
            continue
        best_r2 = max(as_float(r.get("r2")) for r in complete_rows)
        near_best_rows = [
            r for r in complete_rows if as_float(r.get("r2")) >= best_r2 - args.r2_tolerance
        ]
        selected[model_slug] = max(
            near_best_rows,
            key=lambda r: (
                as_float(r.get("avg_feature_accuracy")),
                as_float(r.get("avg_feature_purity")),
                as_float(r.get("avg_feature_well_rankedness")),
                as_float(r.get("r2")),
                -as_float(r.get("mse"), default=float("inf")),
            ),
        )

    summary = {
        "selection_policy": {
            "primary": "feature quality among near-best reconstruction runs",
            "r2_tolerance": args.r2_tolerance,
            "tie_breakers": [
                "avg_feature_accuracy",
                "avg_feature_purity",
                "avg_feature_well_rankedness",
                "r2",
                "mse",
            ],
        },
        "runs": rows,
        "selected": selected,
    }
    (SUMMARY_DIR / f"{args.summary_stem}.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# SAE Sweep Summary",
        "",
        "| Model | Run | Complete | R2 | MSE | Cosine | L0 | Purity | Accuracy | WR | Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_slug, model_rows in sorted(grouped.items()):
        for row in sorted(model_rows, key=lambda r: str(r["run_id"])):
            mark = "*" if selected.get(model_slug, {}).get("run_id") == row.get("run_id") else ""
            lines.append(
                f"| {model_slug} | {mark}`{row['run_id']}` | {row.get('complete')} | "
                f"{fmt(row.get('r2'))} | {fmt(row.get('mse'))} | {fmt(row.get('cosine_similarity'))} | "
                f"{fmt(row.get('l0_sparsity'))} | {fmt(row.get('avg_feature_purity'))} | "
                f"{fmt(row.get('avg_feature_accuracy'))} | {fmt(row.get('avg_feature_well_rankedness'))} | "
                f"{fmt(row.get('avg_feature_recall'))} |"
            )
    (SUMMARY_DIR / f"{args.summary_stem}.md").write_text("\n".join(lines) + "\n")

    if args.promote:
        promote(selected)


if __name__ == "__main__":
    main()
