"""Generate paper-style summary figures for LLM punctuation experiments.

Usage:
    uv run experiments/llm_sentences_last_layer/scripts/plot_llm_punctuation_figures.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


MODEL_ORDER = ("qwen3_8b", "olmo_13b", "gpt_oss_20b")
MODEL_LABELS = {
    "qwen3_8b": "Qwen",
    "olmo_13b": "OLMo",
    "gpt_oss_20b": "GPT-OSS",
}

ROLE_ORDER = ("subj", "verb", "obj")
ROLE_LABELS = {
    "subj": "Subject",
    "verb": "Verb",
    "obj": "Object",
}
ROLE_CHANCE = {
    "subj": 1 / 77,
    "verb": 1.0,
    "obj": 1 / 77,
}

TPE_COLOR = "#FFCC99"
TRAINED_COLOR = "#4c78a8"
RAW_COLOR = "#8C8C8C"
TOPK_COLOR = "#E69F00"
SUPERVISED_COLOR = "#4C78A8"
GRID_COLOR = "0.82"


@dataclass(frozen=True)
class FigureConfig:
    """Style constants shared across the LLM punctuation figure set."""

    dpi: int = 300
    tick_fontsize: float = 6.0
    label_fontsize: float = 7.0
    title_fontsize: float = 8.0
    legend_fontsize: float = 5.2
    bar_width: float = 0.32
    panel_height: float = 1.45
    three_panel_width: float = 6.2
    single_panel_width: float = 3.4
    y_grid_alpha: float = 0.42


DEFAULT_CONFIG = FigureConfig()


def load_summary(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def model_labels() -> list[str]:
    return [MODEL_LABELS[m] for m in MODEL_ORDER]


def style_axis(ax, cfg: FigureConfig, *, y_minor: float | None = 0.1) -> None:
    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize, pad=1.2, width=0.5, length=2)
    if y_minor is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
        ax.tick_params(axis="y", which="minor", width=0.3, length=1.2)
    ax.grid(True, axis="y", alpha=cfg.y_grid_alpha, linewidth=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def save_figure(fig, output_base: Path, cfg: FigureConfig) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {output_base.with_suffix('.png')}")
    print(f"[INFO] Saved {output_base.with_suffix('.pdf')}")


def add_value_labels(ax, xs: Iterable[float], ys: Iterable[float], cfg: FigureConfig, *, offset: float = 0.018) -> None:
    for x, y in zip(xs, ys, strict=True):
        if not np.isfinite(y):
            continue
        ax.text(
            x,
            y + offset,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=max(cfg.tick_fontsize - 1.0, 4.5),
            rotation=90,
        )


def plot_tpe_reconstruction(summary: dict, output_base: Path, cfg: FigureConfig) -> None:
    x = np.arange(len(MODEL_ORDER))
    r2 = [summary["tpe"][m]["r_squared"] for m in MODEL_ORDER]
    ev = [summary["tpe"][m]["explained_variance"] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(cfg.single_panel_width, cfg.panel_height))
    width = cfg.bar_width
    ax.bar(x - width / 2, r2, width=width, color=TPE_COLOR, label="$R^2$", zorder=2)
    ax.bar(x + width / 2, ev, width=width, color=TRAINED_COLOR, label="Expl. var.", zorder=2)
    add_value_labels(ax, x - width / 2, r2, cfg, offset=0.015)
    add_value_labels(ax, x + width / 2, ev, cfg, offset=0.015)
    ax.set_title("TPE Reconstruction", fontsize=cfg.title_fontsize, pad=4)
    ax.set_ylabel("Score", fontsize=cfg.label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels())
    ax.set_ylim(0.0, 0.9)
    ax.set_yticks([0.0, 0.4, 0.8])
    style_axis(ax, cfg)
    ax.legend(
        fontsize=cfg.legend_fontsize,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.15)
    save_figure(fig, output_base, cfg)


def plot_analogy_accuracy(summary: dict, output_base: Path, cfg: FigureConfig) -> None:
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(cfg.three_panel_width * 0.72, cfg.panel_height), sharey=True)
    metrics = (("top_1_accuracy", "Top-1"), ("top_3_accuracy", "Top-3"))
    width = cfg.bar_width

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        raw = [summary["analogy"][m]["sentence_embeddings"][metric] for m in MODEL_ORDER]
        tpe = [summary["analogy"][m]["tpe_embeddings"][metric] for m in MODEL_ORDER]
        ax.bar(x - width / 2, raw, width=width, color=RAW_COLOR, label="Raw hidden", zorder=2)
        ax.bar(x + width / 2, tpe, width=width, color=TPE_COLOR, label="TPE", zorder=2)
        ax.set_title(title, fontsize=cfg.title_fontsize, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels())
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        style_axis(ax, cfg)

    axes[0].set_ylabel("Analogy accuracy", fontsize=cfg.label_fontsize)
    axes[0].legend(
        fontsize=cfg.legend_fontsize,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.15, w_pad=0.35)
    save_figure(fig, output_base, cfg)


def plot_probe_grid(summary: dict, output_base: Path, cfg: FigureConfig) -> None:
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(cfg.three_panel_width, cfg.panel_height), sharey=True)
    width = cfg.bar_width

    for ax, role in zip(axes, ROLE_ORDER, strict=True):
        trained = [summary["probe"][m]["roles"][role]["trained_accuracy"] for m in MODEL_ORDER]
        constructed = [summary["probe"][m]["roles"][role]["analytic_accuracy"] for m in MODEL_ORDER]
        untuned = [summary["probe"][m]["roles"][role].get("analytic_accuracy_untuned", np.nan) for m in MODEL_ORDER]
        ax.bar(x - width / 2, trained, width=width, color=TRAINED_COLOR, label="Trained", zorder=2)
        ax.bar(x + width / 2, constructed, width=width, color=TPE_COLOR, label="Constructed", zorder=2)
        ax.scatter(
            x + width / 2,
            untuned,
            marker="_",
            s=55,
            linewidths=0.9,
            color="0.15",
            label="Untuned constructed",
            zorder=4,
        )
        if role != "verb":
            ax.axhline(ROLE_CHANCE[role], color="0.55", linestyle="--", linewidth=0.6, zorder=3)
        ax.set_title(ROLE_LABELS[role], fontsize=cfg.title_fontsize, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels())
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        style_axis(ax, cfg)

    axes[0].set_ylabel("Accuracy", fontsize=cfg.label_fontsize)
    axes[0].legend(
        fontsize=cfg.legend_fontsize,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.15, w_pad=0.3)
    save_figure(fig, output_base, cfg)


def sae_metric(summary: dict, model_slug: str, sae_type: str, metric: str) -> float:
    if sae_type == "constructed":
        return float(summary["sae"][model_slug]["metrics"][metric])
    return float(summary["sae_baselines"][model_slug][sae_type][metric])


def plot_sae_comparison(summary: dict, output_base: Path, cfg: FigureConfig) -> None:
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(cfg.three_panel_width, cfg.panel_height), sharey=False)
    sae_types = (
        ("constructed", "TPE", TPE_COLOR),
        ("topk", "Top-k", TOPK_COLOR),
        ("supervised", "Supervised", SUPERVISED_COLOR),
    )
    panels = (
        ("r2", "$R^2$", (0.88, 1.01), [0.9, 0.95, 1.0]),
        ("avg_feature_well_rankedness", "Feature quality", (0.82, 1.01), [0.85, 0.925, 1.0]),
        ("l0_sparsity", "L0", (0.0, 1.05), [0.0, 0.5, 1.0]),
    )
    width = cfg.bar_width * 0.72
    offsets = np.linspace(-width, width, len(sae_types))

    for ax, (metric, title, ylim, yticks) in zip(axes, panels, strict=True):
        for offset, (sae_type, label, color) in zip(offsets, sae_types, strict=True):
            vals = [sae_metric(summary, m, sae_type, metric) for m in MODEL_ORDER]
            ax.bar(x + offset, vals, width=width, color=color, label=label, zorder=2)
        ax.set_title(title, fontsize=cfg.title_fontsize, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels())
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        style_axis(ax, cfg)

    axes[0].set_ylabel("Metric value", fontsize=cfg.label_fontsize)
    axes[0].legend(
        fontsize=cfg.legend_fontsize,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.15, w_pad=0.35)
    save_figure(fig, output_base, cfg)


def plot_all(summary_path: Path, output_dir: Path, cfg: FigureConfig = DEFAULT_CONFIG) -> list[Path]:
    summary = load_summary(summary_path)
    outputs = [
        output_dir / "llm_tpe_reconstruction",
        output_dir / "llm_analogy_accuracy",
        output_dir / "llm_probe_accuracy",
        output_dir / "llm_sae_comparison",
    ]
    plot_tpe_reconstruction(summary, outputs[0], cfg)
    plot_analogy_accuracy(summary, outputs[1], cfg)
    plot_probe_grid(summary, outputs[2], cfg)
    plot_sae_comparison(summary, outputs[3], cfg)
    return [p.with_suffix(".png") for p in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM punctuation summary figures.")
    parser.add_argument(
        "--summary-path",
        default="experiments/llm_sentences_last_layer/results/summary/summary.json",
        help="Path to the generated LLM punctuation summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/llm_sentences_last_layer/results/summary/figures",
        help="Directory for PNG/PDF figure outputs.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_CONFIG.dpi)
    parser.add_argument("--panel-height", type=float, default=DEFAULT_CONFIG.panel_height)
    parser.add_argument("--three-panel-width", type=float, default=DEFAULT_CONFIG.three_panel_width)
    parser.add_argument("--single-panel-width", type=float, default=DEFAULT_CONFIG.single_panel_width)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FigureConfig(
        dpi=args.dpi,
        panel_height=args.panel_height,
        three_panel_width=args.three_panel_width,
        single_panel_width=args.single_panel_width,
    )
    plot_all(Path(args.summary_path), Path(args.output_dir), cfg)


if __name__ == "__main__":
    main()
