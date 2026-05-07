from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

TICK_FONTSIZE = 6
LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 6.5
FIGSIZE = (2.9, 1.8)


def _load_summary(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def _style_axes(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, pad=1, width=0.5, length=2)


def plot_interleaved_heatmap(
    *,
    activation_summary: Path,
    tpe_summary: Path,
    output_stem: Path,
    y_tick_step: int = 1,
) -> tuple[Path, Path]:
    activation = _load_summary(activation_summary)
    tpe = _load_summary(tpe_summary)

    layer_labels = activation.get("layer_labels", [])
    tpe_layer_labels = tpe.get("layer_labels", [])
    if not layer_labels:
        raise ValueError("Activation summary is missing layer_labels.")
    if layer_labels != tpe_layer_labels:
        raise ValueError("Layer labels differ between activation and TPE summaries.")

    positions = activation.get("token_positions", [])
    tpe_positions = tpe.get("token_positions", [])
    if not positions:
        raise ValueError("Activation summary is missing token_positions.")
    if positions != tpe_positions:
        raise ValueError("Token positions differ between activation and TPE summaries.")

    tokens = activation.get("token_labels", [])
    token_labels: list[str] = []
    for idx, pos in enumerate(positions):
        if idx < len(tokens):
            token_labels.append(f"{tokens[idx]} ({pos})")
        else:
            token_labels.append(f"pos {pos}")

    ap_heatmap = np.load(activation_summary.parent / "token_heatmap.npy")
    tpe_heatmap = np.load(tpe_summary.parent / "token_heatmap.npy")
    if ap_heatmap.shape != tpe_heatmap.shape:
        raise ValueError(
            f"Heatmap shapes differ: activation={ap_heatmap.shape} tpe={tpe_heatmap.shape}"
        )
    if ap_heatmap.shape[1] != len(token_labels):
        raise ValueError(
            f"Token label count ({len(token_labels)}) does not match heatmap width ({ap_heatmap.shape[1]})."
        )

    # Align: Std. patching layers 1..39 with TPE layers 0..38.
    layer_indices = activation.get("layer_indices", [])
    if not layer_indices:
        raise ValueError("Activation summary is missing layer_indices.")
    if len(layer_indices) != ap_heatmap.shape[0]:
        raise ValueError("Activation layer_indices length does not match heatmap height.")
    if ap_heatmap.shape[0] < 2:
        raise ValueError("Need at least 2 layers to slice (layers 1..).")

    ap_heatmap_plot = ap_heatmap[1:, :]
    tpe_heatmap_plot = tpe_heatmap[:-1, :]
    layer_indices_plot = layer_indices[1:]

    if ap_heatmap_plot.shape != tpe_heatmap_plot.shape:
        raise ValueError(
            f"Sliced heatmap shapes differ: activation={ap_heatmap_plot.shape} tpe={tpe_heatmap_plot.shape}"
        )

    num_layers, num_tokens = ap_heatmap_plot.shape
    ap_z = np.full((num_layers, num_tokens * 2), np.nan, dtype=np.float32)
    tpe_z = np.full((num_layers, num_tokens * 2), np.nan, dtype=np.float32)
    for idx in range(num_tokens):
        ap_z[:, idx * 2] = ap_heatmap_plot[:, idx]
        tpe_z[:, idx * 2 + 1] = tpe_heatmap_plot[:, idx]

    vmin = 0.0
    vmax = float(np.nanmax([ap_heatmap_plot.max(), tpe_heatmap_plot.max()]))

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.1],
        wspace=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_gs = gs[0, 1].subgridspec(nrows=1, ncols=2, wspace=0.0)
    cax_ap = fig.add_subplot(cbar_gs[0, 0])
    cax_tpe = fig.add_subplot(cbar_gs[0, 1])

    ap_masked = np.ma.masked_invalid(ap_z)
    tpe_masked = np.ma.masked_invalid(tpe_z)

    im_ap = ax.imshow(
        ap_masked,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    im_tpe = ax.imshow(
        tpe_masked,
        aspect="auto",
        interpolation="nearest",
        cmap="Oranges",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    ax.set_title("Mean Restoration Score, Activation vs TPE patching", fontsize=TITLE_FONTSIZE, pad=8, loc="left")
    ax.set_ylabel("Layer", fontsize=LABEL_FONTSIZE, rotation=90)
    ax.yaxis.label.set_horizontalalignment("left")
    ax.yaxis.label.set_verticalalignment("bottom")
    ax.yaxis.set_label_coords(-0.02, 0.0)

    tick_positions: list[float] = [2 * i + 0.5 for i in range(num_tokens)]
    tick_labels: list[str] = [str(positions[i] if i < len(positions) else i) for i in range(num_tokens)]
    token_tick_labels: list[str] = []
    for i in range(num_tokens):
        if i >= len(tokens):
            token_tick_labels.append("")
            continue
        tok = tokens[i].replace("\n", "\\n")
        token_tick_labels.append(tok.lstrip("_▁Ġ"))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=TICK_FONTSIZE, rotation=0, ha="center")
    ax.tick_params(axis="x", which="major", pad=1)

    token_ax = ax.secondary_xaxis(location=-0.1)
    token_ax.set_xticks(tick_positions, labels=token_tick_labels)
    token_ax.tick_params(axis="x", length=0, pad=0, labelsize=TICK_FONTSIZE)
    token_ax.spines["bottom"].set_linewidth(0)
    token_ax.set_xlabel("")
    for label in token_ax.get_xticklabels():
        label.set_rotation(-35)
        label.set_horizontalalignment("left")
        label.set_rotation_mode("anchor")

    row_by_layer = {layer: row for row, layer in enumerate(layer_indices_plot)}
    major_layers = [layer_indices_plot[0]] + [
        layer for layer in layer_indices_plot if layer % y_tick_step == 0
    ]

    # To make space for "Layer" label.
    major_layers = [layer for layer in major_layers if layer not in {35, 39}]

    major_layers = sorted(set(major_layers))
    major_yticks = [row_by_layer[layer] for layer in major_layers if layer in row_by_layer]
    major_ylabels = [str(layer) for layer in major_layers if layer in row_by_layer]
    ax.set_yticks(major_yticks)
    ax.set_yticklabels(major_ylabels, fontsize=TICK_FONTSIZE)
    ax.set_yticks(np.arange(num_layers), minor=True)
    ax.tick_params(axis="y", which="major", length=3, width=0.5)
    ax.tick_params(axis="y", which="minor", length=1.5, width=0.5, labelleft=False)

    ax.set_xlim(-0.5, (2 * num_tokens) - 0.5)
    ax.set_ylim(num_layers - 0.5, -0.5)

    _style_axes(ax)

    cbar_ap = fig.colorbar(im_ap, cax=cax_ap, orientation="vertical")
    cbar_ap.ax.tick_params(left=False, labelleft=False, right=False, labelright=False)
    cbar_ap.ax.set_frame_on(False)

    cbar_tpe = fig.colorbar(im_tpe, cax=cax_tpe, orientation="vertical")
    cbar_tpe.ax.yaxis.set_ticks_position("right")
    cbar_tpe.ax.tick_params(left=False, labelleft=False, labelsize=TICK_FONTSIZE, width=0.5, length=2, pad=1)
    cbar_tpe.ax.set_frame_on(False)

    cax_ap.tick_params(axis="x", bottom=False, labelbottom=False)
    cax_tpe.tick_params(axis="x", bottom=False, labelbottom=False)
    cax_ap.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
    for cax in (cax_ap, cax_tpe):
        for spine in cax.spines.values():
            spine.set_visible(False)

    cax_ap.text(
        0.55,
        0.99,
        "Standard",
        transform=cax_ap.transAxes,
        ha="center",
        va="top",
        rotation=90,
        color="white",
        fontsize=LABEL_FONTSIZE,
        zorder=10,
    )
    cax_tpe.text(
        0.55,
        0.99,
        "TPE-constructed",
        transform=cax_tpe.transAxes,
        ha="center",
        va="top",
        rotation=90,
        color="white",
        fontsize=LABEL_FONTSIZE,
        zorder=10,
    )

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interleave activation and TPE patching heatmaps into a Matplotlib figure."
    )
    parser.add_argument(
        "--activation-summary",
        type=Path,
        required=True,
        help="Path to summary.json from activation patching results.",
    )
    parser.add_argument(
        "--tpe-summary",
        type=Path,
        required=True,
        help="Path to summary.json from TPE activation patching results.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Output path without extension (writes .png and .pdf).",
    )
    parser.add_argument(
        "--y-tick-step",
        type=int,
        default=5,
        help="Show every Nth layer label.",
    )
    args = parser.parse_args()

    png_path, pdf_path = plot_interleaved_heatmap(
        activation_summary=args.activation_summary,
        tpe_summary=args.tpe_summary,
        output_stem=args.output_stem,
        y_tick_step=args.y_tick_step,
    )
    print(f"Saved plot: {png_path.resolve()}")
    print(f"Saved plot: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
