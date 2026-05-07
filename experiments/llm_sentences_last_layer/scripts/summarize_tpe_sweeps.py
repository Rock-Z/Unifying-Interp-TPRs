#!/usr/bin/env python3
"""Summarize LLM punctuation TPE sweep metrics and write selected configs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path("experiments/llm_sentences_last_layer")
MANIFEST = ROOT / "configs/tpe_sweeps/manifest.tsv"
SUMMARY_DIR = ROOT / "results/summary"
FINAL_CONFIG_DIR = ROOT / "configs/tpe_final"
FINAL_CKPT_DIR = ROOT / "checkpoints/tpe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary-stem", default="tpe_sweep_summary")
    parser.add_argument("--promote", dest="promote", action="store_true", default=True)
    parser.add_argument("--no-promote", dest="promote", action="store_false")
    return parser.parse_args()


def read_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_metrics(run: dict[str, str]) -> dict[str, object]:
    out_dir = Path(run["output_dir"])
    metrics_path = out_dir / "eval_results_tpe.json"
    row: dict[str, object] = dict(run)
    row["metrics_path"] = str(metrics_path)
    row["complete"] = metrics_path.exists() and (out_dir / "best_model").exists()
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        row.update(
            {
                "explained_variance": metrics.get("Explained_Variance_Ratio"),
                "r_squared": metrics.get("R_Squared"),
                "eval_loss": metrics.get("eval_loss"),
                "mse_loss": metrics.get("MSE_Loss"),
            }
        )
    return row


def as_float(value: object, default: float = float("-inf")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if args.promote:
        FINAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [load_metrics(run) for run in read_manifest(args.manifest)]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_slug"]), []).append(row)

    selected = {}
    for model_slug, model_rows in grouped.items():
        complete_rows = [r for r in model_rows if r.get("complete")]
        if not complete_rows:
            continue
        best = max(complete_rows, key=lambda r: as_float(r.get("r_squared")))
        selected[model_slug] = best

        if args.promote:
            src = Path(str(best["output_dir"]))
            dst = FINAL_CKPT_DIR / model_slug
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

            config_src = dst / "config.gin"
            if config_src.exists():
                shutil.copy2(config_src, FINAL_CONFIG_DIR / f"{model_slug}.gin")

    summary = {"runs": rows, "selected": selected}
    (SUMMARY_DIR / f"{args.summary_stem}.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# TPE Sweep Summary", ""]
    for model_slug, model_rows in sorted(grouped.items()):
        lines.append(f"## {model_slug}")
        for row in sorted(model_rows, key=lambda r: str(r["run_id"])):
            mark = "*" if selected.get(model_slug, {}).get("run_id") == row.get("run_id") else "-"
            ev = row.get("explained_variance")
            r2 = row.get("r_squared")
            loss = row.get("eval_loss")
            complete = row.get("complete")
            lines.append(f"{mark} `{row['run_id']}` complete={complete} R2={r2} EV={ev} eval_loss={loss}")
        lines.append("")
    (SUMMARY_DIR / f"{args.summary_stem}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
