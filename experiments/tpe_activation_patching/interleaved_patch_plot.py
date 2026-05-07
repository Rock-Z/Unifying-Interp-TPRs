from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interleave activation and TPE patching heatmaps into a single plot."
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
        "--output",
        type=Path,
        required=True,
        help="Output HTML path for the interleaved plot.",
    )
    args = parser.parse_args()

    with args.activation_summary.open("r") as fh:
        activation_summary = json.load(fh)
    with args.tpe_summary.open("r") as fh:
        tpe_summary = json.load(fh)

    layer_labels = activation_summary.get("layer_labels", [])
    tpe_layer_labels = tpe_summary.get("layer_labels", [])
    if not layer_labels:
        raise ValueError("Activation summary is missing layer_labels.")
    if layer_labels != tpe_layer_labels:
        raise ValueError("Layer labels differ between activation and TPE summaries.")

    activation_positions = activation_summary.get("token_positions", [])
    tpe_positions = tpe_summary.get("token_positions", [])
    if not activation_positions:
        raise ValueError("Activation summary is missing token_positions.")
    if activation_positions != tpe_positions:
        raise ValueError("Token positions differ between activation and TPE summaries.")

    activation_tokens = activation_summary.get("token_labels", [])
    token_labels = []
    for idx, pos in enumerate(activation_positions):
        if idx < len(activation_tokens):
            token_labels.append(f"{activation_tokens[idx]} ({pos})")
        else:
            token_labels.append(f"pos {pos}")

    ap_heatmap = np.load(args.activation_summary.parent / "token_heatmap.npy")
    tpe_heatmap = np.load(args.tpe_summary.parent / "token_heatmap.npy")
    if ap_heatmap.shape != tpe_heatmap.shape:
        raise ValueError(
            f"Heatmap shapes differ: activation={ap_heatmap.shape} tpe={tpe_heatmap.shape}"
        )
    if ap_heatmap.shape[1] != len(token_labels):
        raise ValueError(
            f"Token label count ({len(token_labels)}) does not match heatmap width ({ap_heatmap.shape[1]})."
        )

    num_layers, num_tokens = ap_heatmap.shape
    x_labels = []
    ap_z = np.full((num_layers, num_tokens * 2), np.nan, dtype=np.float32)
    tpe_z = np.full((num_layers, num_tokens * 2), np.nan, dtype=np.float32)
    for idx, label in enumerate(token_labels):
        x_labels.append(f"{label} | activation")
        x_labels.append(f"{label} | tpe")
        ap_z[:, idx * 2] = ap_heatmap[:, idx]
        tpe_z[:, idx * 2 + 1] = tpe_heatmap[:, idx]

    zmin = 0.0
    zmax = float(np.nanmax([ap_heatmap.max(), tpe_heatmap.max()]))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=ap_z,
            x=x_labels,
            y=layer_labels,
            colorscale="Blues",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Activation", x=1.02),
            hoverongaps=False,
            showscale=True,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=tpe_z,
            x=x_labels,
            y=layer_labels,
            colorscale="Oranges",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="TPE", x=1.10),
            hoverongaps=False,
            showscale=True,
        )
    )
    fig.update_xaxes(type="category", tickangle=-45)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        width=1200,
        height=800,
        margin=dict(l=80, r=120, t=80, b=160),
        title="Activation vs TPE patching (interleaved)",
    )
    fig.write_html(args.output)


if __name__ == "__main__":
    main()
