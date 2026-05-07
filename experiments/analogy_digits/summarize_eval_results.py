#!/usr/bin/env python
"""Summarize analogy eval JSONs into table-friendly JSON/CSV."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

FILE_RE = re.compile(
    r"^digits_(copy|reverse|sort_ascending)_(rnn|lstm|gru)(?:_repeat(\d+))?_eval\.json$"
)

TASK_LABELS = {
    "copy": "Copy",
    "reverse": "Reverse",
    "sort_ascending": "Sort",
}
MODEL_LABELS = {
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    metrics: dict[str, dict[str, float]] = {}

    for key in ("nn_embeddings", "tpe_embeddings"):
        block = data.get(key)
        if not block:
            continue
        results = block.get("detailed_results")
        if results:
            ranks = [entry["rank"] for entry in results]
            count = len(ranks)

            def acc(k: int) -> float:
                return sum(1 for r in ranks if r <= k) / count

            metrics[key] = {
                "top1": acc(1),
                "top2": acc(2),
                "top3": acc(3),
                "top5": acc(5),
            }
        else:
            stats = block.get("overall_statistics", {})
            if not stats:
                raise ValueError(f"No metrics found in {path}")
            metrics[key] = {
                "top1": stats.get("top_1_accuracy"),
                "top2": None,
                "top3": stats.get("top_3_accuracy"),
                "top5": stats.get("top_5_accuracy"),
            }
    return metrics


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "top1": mean(m["top1"] for m in items),
        "top2": mean(m["top2"] for m in items if m["top2"] is not None),
        "top3": mean(m["top3"] for m in items),
        "top5": mean(m["top5"] for m in items),
    }


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize digits analogy eval results into JSON/CSV."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("experiments/analogy_digits/results/1le64h256_fixedlen"),
        help="Directory containing eval result JSON files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON path (defaults to <results_dir>/summary.json).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path (defaults to <results_dir>/summary.csv).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if args.output_json is None:
        output_json = results_dir / "summary.json"
    else:
        output_json = args.output_json
    if args.output_csv is None:
        output_csv = results_dir / "summary.csv"
    else:
        output_csv = args.output_csv

    per_group: dict[tuple[str, str], list[dict[str, dict[str, float]]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        task, arch, _repeat = match.groups()
        per_group.setdefault((task, arch), []).append(load_metrics(path))

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    all_nn = []
    all_tpe = []

    for (task, arch), entries in sorted(per_group.items()):
        nn_metrics = [e["nn_embeddings"] for e in entries if "nn_embeddings" in e]
        tpe_metrics = [e["tpe_embeddings"] for e in entries if "tpe_embeddings" in e]

        if not nn_metrics or not tpe_metrics:
            raise ValueError(f"Missing embeddings for {task}/{arch}")

        nn_mean = mean_metrics(nn_metrics)
        tpe_mean = mean_metrics(tpe_metrics)

        summary.setdefault(TASK_LABELS[task], {})[MODEL_LABELS[arch]] = {
            "nn": nn_mean,
            "tpe": tpe_mean,
        }
        all_nn.append(nn_mean)
        all_tpe.append(tpe_mean)

    average = {
        "nn": mean_metrics(all_nn),
        "tpe": mean_metrics(all_tpe),
    }

    output_payload = {
        "results_dir": str(results_dir),
        "k_values": [1, 2, 3, 5],
        "tasks": summary,
        "average": average,
    }
    output_json.write_text(json.dumps(output_payload, indent=2))

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task",
                "model",
                "nn_top1",
                "nn_top2",
                "nn_top3",
                "nn_top5",
                "tpe_top1",
                "tpe_top2",
                "tpe_top3",
                "tpe_top5",
            ]
        )
        for task_name in ("Copy", "Reverse", "Sort"):
            for model_name in ("RNN", "LSTM", "GRU"):
                metrics = summary[task_name][model_name]
                row = [
                    task_name,
                    model_name,
                    pct(metrics["nn"]["top1"]),
                    pct(metrics["nn"]["top2"]),
                    pct(metrics["nn"]["top3"]),
                    pct(metrics["nn"]["top5"]),
                    pct(metrics["tpe"]["top1"]),
                    pct(metrics["tpe"]["top2"]),
                    pct(metrics["tpe"]["top3"]),
                    pct(metrics["tpe"]["top5"]),
                ]
                writer.writerow(row)
        writer.writerow(
            [
                "Average",
                "Average",
                pct(average["nn"]["top1"]),
                pct(average["nn"]["top2"]),
                pct(average["nn"]["top3"]),
                pct(average["nn"]["top5"]),
                pct(average["tpe"]["top1"]),
                pct(average["tpe"]["top2"]),
                pct(average["tpe"]["top3"]),
                pct(average["tpe"]["top5"]),
            ]
        )


if __name__ == "__main__":
    main()
