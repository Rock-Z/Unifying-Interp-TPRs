"""
Plot side-by-side SAE feature summaries for three SAE types.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import textwrap
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from sae import (
    SparseAutoencoder,
    compute_feature_well_rankedness_per_feature,
    select_feature_labels_by_well_rankedness,
)
from sentences import load_sentences
from utils import load_dataset_with_embeddings

TITLE_FONTSIZE = 9
LABEL_FONTSIZE = 8
TICK_FONTSIZE = 6
TEXT_FONTSIZE = 6


@dataclass
class SAEPanelData:
    title: str
    feature_indices: np.ndarray
    activation_pct: np.ndarray
    well_rankedness: np.ndarray
    activations: np.ndarray
    sentences: list[str]
    semantic_keys: list
    semantic_labels: list[str]
    example_text: str = ""


def resolve_checkpoint_path(path: str) -> Path:
    base = Path(path).expanduser()
    if (base / "config.json").exists():
        return base
    best_model = base / "best_model"
    if (best_model / "config.json").exists():
        return best_model
    raise FileNotFoundError(f"Could not find config.json under {path} or {best_model}")


def load_sentence_dataset(dataset_path: str, embedding_model_name: str, embedding_cache_path: str):
    dataset, role_assigner = load_sentences(dataset_path, role_scheme="svo")
    dataset, _ = load_dataset_with_embeddings(
        dataset=dataset,
        dataset_path=dataset_path,
        embedding_model_name=embedding_model_name,
        embedding_cache_path=embedding_cache_path,
    )
    return dataset, role_assigner


def _sae_eval_collate(batch):
    embeddings = torch.tensor([ex["target_embeddings"] for ex in batch], dtype=torch.float)
    return {"target_embeddings": embeddings}


def compute_feature_activations(
    sae: SparseAutoencoder,
    dataset_split,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(dataset_split, batch_size=batch_size, collate_fn=_sae_eval_collate)
    activations = []
    sae.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["target_embeddings"].to(device)
            encoded = sae.get_sparse_features(embeddings)
            activations.append(encoded.cpu())
    return torch.cat(activations, dim=0)




def label_to_text(label, role_assigner, label_mode: str) -> str:
    if label is None:
        return "unknown"
    if label_mode == "filler":
        filler = role_assigner.noun_idx2filler.get(int(label), f"filler_{label}")
        return filler
    filler_id, role_id = label
    filler_id = int(filler_id)
    role_id = int(role_id)
    if role_id == role_assigner.role2idx.get("verb"):
        filler_idx = filler_id - len(role_assigner.noun_idx2filler)
        filler = role_assigner.verb_idx2filler.get(filler_idx, f"verb_{filler_idx}")
    else:
        filler = role_assigner.noun_idx2filler.get(filler_id, f"filler_{filler_id}")
    role = role_assigner.idx2role.get(role_id, f"role_{role_id}")
    return f"{filler} x {role}"


def label_to_key(label, role_assigner, label_mode: str) -> str:
    if label is None:
        return "unknown"
    if label_mode == "filler":
        return role_assigner.noun_idx2filler.get(int(label), f"filler_{label}")
    filler_id, role_id = label
    filler_id = int(filler_id)
    role_id = int(role_id)
    if role_id == role_assigner.role2idx.get("verb"):
        filler_idx = filler_id - len(role_assigner.noun_idx2filler)
        return role_assigner.verb_idx2filler.get(filler_idx, f"verb_{filler_idx}")
    return role_assigner.noun_idx2filler.get(filler_id, f"filler_{filler_id}")


def parse_feature_map_entry(entry, role_assigner):
    if isinstance(entry, dict):
        filler_id = entry.get("filler_id")
        role_id = entry.get("role_id")
        if filler_id is None or role_id is None:
            return None, "unknown"
        label = (int(filler_id), int(role_id))
        return label, label_to_text(label, role_assigner, "filler_role")
    if isinstance(entry, str):
        parts = entry.split(" x ")
        if len(parts) != 2:
            return None, entry
        filler, role = parts
        role_id = role_assigner.role2idx.get(role)
        if role_id is None:
            return None, entry
        if role == "verb":
            filler_id = role_assigner.verb_filler2idx.get(filler)
            if filler_id is None:
                return None, entry
            label = (filler_id + len(role_assigner.noun_idx2filler), role_id)
        else:
            filler_id = role_assigner.noun_filler2idx.get(filler)
            if filler_id is None:
                return None, entry
            label = (filler_id, role_id)
        return label, entry
    return None, "unknown"


def is_valid_label(label, role_assigner, label_mode: str) -> bool:
    if label is None:
        return False
    if label_mode == "filler":
        return True
    filler_id, role_id = label
    filler_id = int(filler_id)
    role_id = int(role_id)
    noun_vocab = len(role_assigner.noun_idx2filler)
    verb_role = role_assigner.role2idx.get("verb")
    if role_id == verb_role:
        return filler_id >= noun_vocab
    return filler_id < noun_vocab


def compute_feature_semantics(
    feature_labels: list,
    role_assigner,
    label_mode: str,
) -> tuple[list, list[str]]:
    labels = []
    keys = []
    for lbl in feature_labels:
        if not is_valid_label(lbl, role_assigner, label_mode):
            labels.append("unknown")
            keys.append(None)
            continue
        labels.append(label_to_text(lbl, role_assigner, label_mode))
        keys.append(label_to_key(lbl, role_assigner, label_mode))
    return keys, labels


def format_feature_block(
    feature_idx: int,
    semantic_label: str,
    activation_pct: np.ndarray,
    well_rankedness: np.ndarray,
    activations: np.ndarray,
    sentences: list[str],
    n_examples_per_feature: int,
    wrap_width: int,
) -> list[str]:
    lines = []
    pct = activation_pct[feature_idx]
    wr = well_rankedness[feature_idx]
    lines.append(f"Feature {feature_idx} | {semantic_label} | quality {wr:.2f}")
    acts = activations[:, feature_idx]
    top_indices = np.argsort(acts)[-n_examples_per_feature:][::-1]
    for rank, idx in enumerate(top_indices, start=1):
        sentence = sentences[int(idx)]
        activation = float(acts[int(idx)])
        entry = f"{rank:>2}. {activation:>7.4f}  {sentence}"
        lines.extend(textwrap.wrap(entry, width=wrap_width, subsequent_indent=" " * 6))
    return lines


def build_panel_data(
    title: str,
    checkpoint_path: str,
    dataset,
    device: torch.device,
    batch_size: int,
    label_mode: str,
    ignore_singleton_verb: bool,
    role_assigner,
) -> SAEPanelData:
    checkpoint = resolve_checkpoint_path(checkpoint_path)
    sae = SparseAutoencoder.from_pretrained(str(checkpoint)).to(device)
    activation_threshold = getattr(sae.config, "activation_threshold", None)
    activation_threshold = float(activation_threshold) if activation_threshold else 0.0

    train_activations = compute_feature_activations(
        sae, dataset["train"], device=device, batch_size=batch_size
    ).cpu()
    activations = compute_feature_activations(
        sae, dataset["test"], device=device, batch_size=batch_size
    ).cpu()
    activations_np = activations.numpy()
    activation_pct = (activations_np > activation_threshold).mean(axis=0) * 100.0

    label_mode = "filler_role"
    feature_labels = select_feature_labels_by_well_rankedness(
        dataset_split=dataset["train"],
        activations=train_activations,
        label_mode=label_mode,
        ignore_singleton_verb=ignore_singleton_verb,
    )
    well_rankedness = compute_feature_well_rankedness_per_feature(
        dataset_split=dataset["test"],
        activations=activations,
        feature_labels=feature_labels,
        label_mode=label_mode,
        ignore_singleton_verb=ignore_singleton_verb,
    )
    well_rankedness = well_rankedness.cpu().numpy()

    semantic_keys, semantic_labels = compute_feature_semantics(
        feature_labels=feature_labels,
        role_assigner=role_assigner,
        label_mode=label_mode,
    )

    sort_idx = np.argsort(-activation_pct)

    return SAEPanelData(
        title=title,
        feature_indices=sort_idx,
        activation_pct=activation_pct,
        well_rankedness=well_rankedness,
        activations=activations_np,
        sentences=dataset["test"]["sentence"],
        semantic_keys=semantic_keys,
        semantic_labels=semantic_labels,
    )


def plot_sae_comparison(
    panel_data: list[SAEPanelData],
    output_path: Path,
    num_features: int,
    example_labels: list[str],
    examples_per_feature: int,
    wrap_width: int,
):
    n_cols = len(panel_data)
    fig = plt.figure(figsize=(6.75, 4))
    grid = fig.add_gridspec(nrows=3, ncols=n_cols, height_ratios=[1.1, 1.0, 1.7], hspace=0.18, wspace=0.08)

    panel_blocks = []
    panel_example_indices = []
    for panel in panel_data:
        example_feature_indices = []
        for label in example_labels:
            matches = [i for i, lbl in enumerate(panel.semantic_labels) if lbl == label]
            if matches:
                best_match = max(matches, key=lambda idx: panel.activation_pct[idx])
                example_feature_indices.append(best_match)
            else:
                example_feature_indices.append(int(panel.feature_indices[0]))
        panel_example_indices.append(example_feature_indices)
        blocks = []
        for feature_idx in example_feature_indices:
            label_text = panel.semantic_labels[feature_idx]
            block_lines = format_feature_block(
                feature_idx=feature_idx,
                semantic_label=label_text,
                activation_pct=panel.activation_pct,
                well_rankedness=panel.well_rankedness,
                activations=panel.activations,
                sentences=panel.sentences,
                n_examples_per_feature=examples_per_feature,
                wrap_width=wrap_width,
            )
            blocks.append(block_lines)
        panel_blocks.append(blocks)

    row_max_lengths = []
    for row_idx in range(len(example_labels)):
        row_max_lengths.append(max(len(blocks[row_idx]) for blocks in panel_blocks))

    for col_idx, panel in enumerate(panel_data):
        feat_order = panel.feature_indices[:num_features]
        x = np.arange(len(feat_order))
        example_feature_indices = panel_example_indices[col_idx]

        ax_top = fig.add_subplot(grid[0, col_idx])
        ax_top.bar(
            x,
            panel.activation_pct[feat_order],
            color=plt.get_cmap("viridis")(panel.activation_pct[feat_order] / 100.0),
            width=1.0,
            linewidth=0,
        )
        ax_top.set_title(panel.title, fontsize=TITLE_FONTSIZE, pad=4)
        if col_idx == 0:
            ax_top.set_ylabel("% Active", fontsize=LABEL_FONTSIZE)
        else:
            ax_top.set_ylabel("")
        ax_top.set_ylim(0, 100)
        ax_top.set_xlim(-0.5, len(feat_order) - 0.5)
        ax_top.set_xticks([])
        ax_top.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=0.5, length=2)
        ax_top.grid(True, alpha=0.3, linewidth=0.3, axis="y")
        ax_top.set_axisbelow(True)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["bottom"].set_linewidth(0.5)
        ax_top.spines["left"].set_linewidth(0.5)

        ax_mid = fig.add_subplot(grid[1, col_idx], sharex=ax_top)
        ax_mid.bar(
            x,
            panel.well_rankedness[feat_order],
            color=plt.get_cmap("magma")(panel.well_rankedness[feat_order]),
            width=1.0,
            linewidth=0,
        )
        if col_idx == 0:
            ax_mid.set_ylabel("Quality", fontsize=LABEL_FONTSIZE)
        else:
            ax_mid.set_ylabel("")
        ax_mid.set_ylim(0, 1.0)
        ax_mid.set_xlim(-0.5, len(feat_order) - 0.5)
        ax_mid.set_xticks([])
        ax_mid.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=0.5, length=2)
        ax_mid.grid(True, alpha=0.3, linewidth=0.3, axis="y")
        ax_mid.set_axisbelow(True)
        ax_mid.spines["top"].set_visible(False)
        ax_mid.spines["right"].set_visible(False)
        ax_mid.spines["bottom"].set_linewidth(0.5)
        ax_mid.spines["left"].set_linewidth(0.5)

        ax_bottom = fig.add_subplot(grid[2, col_idx])
        ax_bottom.axis("off")
        blocks = panel_blocks[col_idx]
        padded_blocks = [
            block + [""] * (row_max_lengths[row_idx] - len(block))
            for row_idx, block in enumerate(blocks)
        ]
        lines = []
        for block in padded_blocks:
            lines.extend(block)
            lines.append("")
        panel.example_text = "\n".join(lines).strip()
        ax_bottom.text(
            0.0,
            1.0,
            panel.example_text,
            fontsize=5.0,
            va="top",
            ha="left",
            linespacing=1.15,
            family="monospace",
        )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, wspace=0.08, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[INFO] Saved plot to {output_path.with_suffix('.png')} and {output_path.with_suffix('.pdf')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAE feature summaries across models.")
    parser.add_argument(
        "--tpe-checkpoint",
        default="experiments/sae/results/modernbert",
        help="Checkpoint path for TPE-constructed SAE.",
    )
    parser.add_argument(
        "--supervised-checkpoint",
        default="experiments/sae/checkpoints/modernbert-embed-base/supervised",
        help="Checkpoint path for supervised SAE.",
    )
    parser.add_argument(
        "--topk-checkpoint",
        default="experiments/sae_sentences_topk/checkpoints/modernbert-embed-base/run-w4gfyb30",
        help="Checkpoint path for top-k SAE.",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/sentences_multiple",
        help="Path to sentences dataset directory.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="nomic-ai/modernbert-embed-base",
        help="Embedding model name for sentence embeddings.",
    )
    parser.add_argument(
        "--embedding-cache-path",
        default="data/sentences_multiple",
        help="Embedding cache base path (used to locate or write caches).",
    )
    parser.add_argument(
        "--output-path",
        default="experiments/sae/figures/sae_feature_comparison_modernbert",
        help="Output path (without extension) for plot files.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for SAE encoding.")
    parser.add_argument("--label-mode", choices=["filler", "filler_role"], default="filler_role")
    parser.add_argument("--no-ignore-singleton-verb", action="store_true", default=False)
    parser.add_argument("--example-feature-count", type=int, default=2)
    parser.add_argument("--examples-per-feature", type=int, default=5)
    parser.add_argument("--wrap-width", type=int, default=60)
    parser.add_argument("--num-features", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, role_assigner = load_sentence_dataset(
        dataset_path=args.dataset_path,
        embedding_model_name=args.embedding_model_name,
        embedding_cache_path=args.embedding_cache_path,
    )

    ignore_singleton_verb = not args.no_ignore_singleton_verb

    panels = [
        build_panel_data(
            title="TPE-constructed",
            checkpoint_path=args.tpe_checkpoint,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            label_mode=args.label_mode,
            ignore_singleton_verb=ignore_singleton_verb,
            role_assigner=role_assigner,
        ),
        build_panel_data(
            title="Supervised",
            checkpoint_path=args.supervised_checkpoint,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            label_mode=args.label_mode,
            ignore_singleton_verb=ignore_singleton_verb,
            role_assigner=role_assigner,
        ),
        build_panel_data(
            title="Top-K",
            checkpoint_path=args.topk_checkpoint,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            label_mode=args.label_mode,
            ignore_singleton_verb=ignore_singleton_verb,
            role_assigner=role_assigner,
        ),
    ]

    topk_panel = panels[2]
    valid_indices = [
        idx
        for idx, label in enumerate(topk_panel.semantic_labels)
        if label != "unknown"
        and topk_panel.activation_pct[idx] > 0.0
        and np.isfinite(topk_panel.well_rankedness[idx])
    ]
    rng = np.random.default_rng()
    if valid_indices:
        sample_size = min(args.example_feature_count, len(valid_indices))
        sampled_indices = rng.choice(valid_indices, size=sample_size, replace=False)
    else:
        sampled_indices = []
    example_labels = [topk_panel.semantic_labels[int(idx)] for idx in sampled_indices]

    plot_sae_comparison(
        panels,
        Path(args.output_path),
        num_features=args.num_features,
        example_labels=example_labels,
        examples_per_feature=args.examples_per_feature,
        wrap_width=args.wrap_width,
    )


if __name__ == "__main__":
    main()
