import argparse
import copy
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (  # noqa: E402
    get_cache_path,
    load_dataset_with_embeddings,
    set_random_seed,
)

from experiments.layerwise_decoder_tpe.train_layerwise_tpe import (  # noqa: E402
    _build_active_passive_dataset,
)


def _validate_layers(requested: Optional[Sequence[int]], num_layers: int) -> list[int]:
    if requested is None:
        return list(range(num_layers))
    layers = [int(x) for x in requested]
    for layer_id in layers:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(f"Invalid layer index {layer_id}; must be in [0, {num_layers}).")
    return layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache per-layer decoder-only embeddings.")
    parser.add_argument("--sentences-path", default="data/sentences", help="Path to IOI sentences.")
    parser.add_argument("--embedding-model-name", required=True, help="HF model id (decoder-only).")
    parser.add_argument(
        "--embedding-cache-path",
        default="data/sentences",
        help="Directory to store embedding cache files.",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=None,
        help="Comma-separated layer ids to cache; defaults to all layers.",
    )
    parser.add_argument("--role-scheme", default="svo", help="Role scheme for prompt builder.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Optional cap on examples per split.",
    )
    parser.add_argument(
        "--device", default=None, help="Device override (e.g., 'cuda', 'cpu'); defaults to auto."
    )
    return parser.parse_args()


def _parse_layer_indices(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main(args: argparse.Namespace) -> None:
    """Materialize decoder-only embeddings for each requested layer."""
    set_random_seed(args.random_seed)
    torch.set_grad_enabled(False)

    dataset, _ = _build_active_passive_dataset(
        sentences_path=args.sentences_path,
        embedding_model_name=args.embedding_model_name,
        role_scheme=args.role_scheme,
        seed=args.random_seed,
        max_examples_per_split=args.max_examples_per_split,
    )

    decoder_config = AutoConfig.from_pretrained(args.embedding_model_name, trust_remote_code=True)
    num_layers = int(
        getattr(decoder_config, "num_hidden_layers", getattr(decoder_config, "n_layer", 0))
    )
    if num_layers <= 0:
        raise ValueError("Could not determine decoder layer count from the model config.")

    layers = _validate_layers(_parse_layer_indices(args.layer_indices), num_layers)
    target_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


    layer_dataset = copy.deepcopy(dataset)
    _, embedding_dim = load_dataset_with_embeddings(
        dataset=layer_dataset,
        dataset_path=args.sentences_path,
        embedding_model_name=args.embedding_model_name,
        embedding_cache_path=args.embedding_cache_path,
        embedding_column_name="target_embeddings",
        encoder_model_type="decoder-only-full",
        device=target_device,
        decoder_layer_indices=layers,
    )
    # cache each layer separately
    for layer_id in layers:
        cache_path = get_cache_path(
            args.sentences_path, args.embedding_model_name, suffix=f"layer{int(layer_id)}"
        )
        print(
            f"[INFO] Cached embeddings for layer {layer_id} "
            f"({embedding_dim} dims) at {cache_path}"
        )


if __name__ == "__main__":
    parsed = _parse_args()
    main(parsed)
