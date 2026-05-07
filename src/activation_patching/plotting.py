from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import plotly.express as px

from .utils import pretty_token


def write_heatmap_csv(path: Path, heatmap: np.ndarray, layer_labels: Sequence[str], token_labels: Sequence[str]) -> None:
    """Save a heatmap matrix to CSV with readable headers."""

    with path.open("w") as fh:
        header = ",".join(["layer"] + [label.replace(",", ";") for label in token_labels])
        fh.write(header + "\n")
        for idx, row in enumerate(heatmap):
            row_str = ",".join(f"{val:.6f}" for val in row)
            fh.write(f"{layer_labels[idx]},{row_str}\n")


def plot_activation_patching_results(
    data: np.ndarray,
    token_labels: Sequence[str],
    layer_labels: Sequence[str],
    *,
    token_positions: Optional[Sequence[int]] = None,
    plot_title: str = "Normalized logit restoration",
):
    """Render a token-layer heatmap with per-position labels."""

    display_tokens = []
    for idx, token in enumerate(token_labels):
        human_token = pretty_token(token)
        if token_positions is not None and idx < len(token_positions):
            pos = token_positions[idx]
            display_tokens.append(f"{human_token} ({pos})")
        else:
            display_tokens.append(f"{human_token} ({idx})")

    fig = px.imshow(
        data,
        x=display_tokens,
        y=layer_labels,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        labels={"x": "Token position", "y": "Layer", "color": "Restoration"},
        title=plot_title,
        aspect="auto",
    )
    fig.update_xaxes(type="category", tickangle=-45, tickfont=dict(size=12))
    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=80, r=80, t=80, b=120),
        coloraxis_colorbar=dict(title="Restoration"),
    )
    return fig


def plot_heatmap(
    path: Path,
    heatmap: np.ndarray,
    layer_labels: Sequence[str],
    token_labels: Sequence[str],
    token_positions: Optional[Sequence[int]] = None,
) -> None:
    """Plot activation patching results to an interactive HTML file."""

    fig = plot_activation_patching_results(
        heatmap,
        token_labels=token_labels,
        layer_labels=layer_labels,
        token_positions=token_positions,
        plot_title="Token-level activation patching",
    )
    fig.write_html(path)


__all__ = ["plot_activation_patching_results", "plot_heatmap", "write_heatmap_csv"]
