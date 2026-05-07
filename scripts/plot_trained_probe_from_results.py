import argparse
import json
import os
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_trained_only(results: List[Dict], *, title: str, x_label: str, out_dir: str, timestamp: str) -> str:
    # Filter out role_id == -1
    filtered = [r for r in results if r.get("role_id") != -1]
    role_ids = [r["role_id"] for r in filtered]
    trained_acc = [r["trained_accuracy"] for r in filtered]

    os.makedirs(out_dir, exist_ok=True)

    # Larger text
    plt.figure(figsize=(6, 4))
    plt.bar(role_ids, trained_acc, label="Trained Probe", width=0.6)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(role_ids, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.05)

    png_path = os.path.join(out_dir, f"probe_trained_only_{timestamp}.png")
    pdf_path = os.path.join(out_dir, f"probe_trained_only_{timestamp}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to {png_path} and {pdf_path}")
    return png_path


def plot_bars_trained_and_analytic(results: List[Dict], *, title: str, x_label: str, out_dir: str, timestamp: str, out_name: str | None = None) -> str:
    filtered = [r for r in results if r.get("role_id") != -1]
    role_ids = [r["role_id"] for r in filtered]
    trained_acc = [r["trained_accuracy"] for r in filtered]
    analytic_acc = [r["analytic_accuracy"] for r in filtered]

    x = np.arange(len(role_ids))
    width = 0.35

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    # Analytic: pastel orange; Trained: blue-ish
    plt.bar(x - width / 2, trained_acc, width=width, color="#4c78a8", label="Trained")
    plt.bar(x + width / 2, analytic_acc, width=width, color="#FFCC99", label="Analytic")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(x, role_ids, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=12, loc="lower right")

    base = out_name if out_name else "probe_both_bars"
    png_path = os.path.join(out_dir, f"{base}_{timestamp}.png")
    pdf_path = os.path.join(out_dir, f"{base}_{timestamp}.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to {png_path} and {pdf_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(description="Plot trained probe accuracy by position from saved results JSON.")
    parser.add_argument("--input", required=True, help="Path to results JSON produced by invert_tpr.py")
    parser.add_argument("--out_dir", default="results", help="Directory to save plots")
    parser.add_argument("--title", default="Probe Accuracy by Position", help="Plot title")
    parser.add_argument("--x_label", default="Position", help="X-axis label")
    parser.add_argument("--compare_bars", action="store_true", help="Plot trained and analytic accuracies side-by-side bars")
    parser.add_argument("--out_name", default=None, help="Base output filename (without timestamp extension)")
    args = parser.parse_args()

    payload = load_results(args.input)
    # Prefer embedded timestamp to match the original run; fallback to now.
    timestamp = payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.compare_bars:
        plot_bars_trained_and_analytic(
            payload["results"],
            title=args.title,
            x_label=args.x_label,
            out_dir=args.out_dir,
            timestamp=timestamp,
            out_name=args.out_name,
        )
    else:
        plot_trained_only(
            payload["results"],
            title=args.title,
            x_label=args.x_label,
            out_dir=args.out_dir,
            timestamp=timestamp,
        )


if __name__ == "__main__":
    main()


