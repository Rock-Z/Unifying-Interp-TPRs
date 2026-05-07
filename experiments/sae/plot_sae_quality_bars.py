"""
Plot per-feature well-rankedness for three SAE variants.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

TITLE_FONTSIZE = 8
LABEL_FONTSIZE = 8
TICK_FONTSIZE = 6

PANEL_COLORS = ["#4C78A8", "#F4C392", "#E69F00"]


def load_feature_scores(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    scores = []
    for feature in payload.get("features", []):
        score = feature.get("score")
        if score is None:
            continue
        score = float(score)
        if np.isfinite(score):
            scores.append(score)
    return np.array(scores, dtype=float)


def plot_quality_panels(panel_specs, output_path: Path, fig_size: tuple[float, float]):
    fig, axes = plt.subplots(1, len(panel_specs), figsize=fig_size, sharey=True)
    if len(panel_specs) == 1:
        axes = [axes]

    for ax, (title, scores, color) in zip(axes, panel_specs, strict=True):
        sorted_scores = np.sort(scores)[::-1]
        x = np.arange(sorted_scores.size)
        ax.bar(x, sorted_scores, color=color, width=1.0, linewidth=0.4, edgecolor="white")
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=5)
        ax.set_ylim(0.0, 1.0)
        ax.margins(y=0.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_xlim(-0.5, sorted_scores.size - 0.5)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=0.5, length=2, pad=2)
        ax.tick_params(axis="y", which="minor", width=0.3, length=1.5)
        for tick_value, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=False):
            if np.isclose(tick_value, 0):
                label.set_verticalalignment("bottom")
            elif np.isclose(tick_value, 1):
                label.set_verticalalignment("top")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

    fig.supylabel("Quality", fontsize=LABEL_FONTSIZE, x=0.022)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.0, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[INFO] Saved plot to {output_path.with_suffix('.png')} and {output_path.with_suffix('.pdf')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-feature well-rankedness for SAE variants."
    )
    parser.add_argument(
        "--tpe-feature-details",
        default="experiments/sae/results/qwen3_8b/feature_details_results__qwen3_8b.json",
        help="Feature details JSON for TPE-constructed SAE.",
    )
    parser.add_argument(
        "--supervised-feature-details",
        default=(
            "experiments/sae/results/supervised_qwen3_8b_v3/"
            "eval_results/feature_details_qwen3_8b__supervised.json"
        ),
        help="Feature details JSON for supervised SAE.",
    )
    parser.add_argument(
        "--topk-feature-details",
        default=(
            "experiments/sae/results/topk_qwen3_8b_v3/"
            "eval_results/feature_details_qwen3-embedding-8b__run-kbx35i08.json"
        ),
        help="Feature details JSON for Top-K SAE.",
    )
    parser.add_argument(
        "--output-path",
        default="experiments/sae/figures/sae_quality_bars_qwen3_8b",
        help="Output path (without extension) for plot files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tpe_scores = load_feature_scores(Path(args.tpe_feature_details))
    supervised_scores = load_feature_scores(Path(args.supervised_feature_details))
    topk_scores = load_feature_scores(Path(args.topk_feature_details))

    panels = [
        ("Supervised", supervised_scores, PANEL_COLORS[0]),
        ("TPE-constructed", tpe_scores, PANEL_COLORS[1]),
        (r"Top-$k$", topk_scores, PANEL_COLORS[2]),
    ]

    plot_quality_panels(
        panels,
        output_path=Path(args.output_path),
        fig_size=(6.75, 0.6),
    )


if __name__ == "__main__":
    main()
