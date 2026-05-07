import gin
import math
import argparse
import os
import json
import ast
import numpy as np
import random
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F  # noqa: F401 (may be unused for now)
from transformers.trainer_utils import set_seed as hf_set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Optional, Sequence, Literal, Union
from huggingface_hub import hf_hub_download

def load_gin_configs(gin_files: list, gin_bindings : list = []):
    """Load gin configuration files and in-line bindings.

    Based on the parsing approach in dopamine's runner. This function is the
    single entry point used by scripts to set up gin. It allows passing one or
    more `.gin` files plus CLI `--name=value` style overrides that are turned
    into bindings.

    Args:
        gin_files: List of paths to gin configuration files.
        gin_bindings: List of gin parameter bindings to override values in the
            provided config files (e.g., ["main.use_wandb = False"]).
    """

    gin.parse_config_files_and_bindings(
        gin_files, bindings=gin_bindings, skip_unknown=False
    )

def gin_config_to_readable_dictionary(gin_config: dict):
    """Convert a gin operative config to a flat dictionary for logging.

    Args:
        gin_config: The operative config (e.g., `gin.config._OPERATIVE_CONFIG`).

    Returns:
        dict: A flat mapping like {"main.use_wandb": False, ...} suitable for
        W&B or JSON serialization.
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data

def set_random_seed(random_seed: int) -> None:
    """Set seeds for `random`, `numpy`, `torch`, and HF transformers."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    hf_set_seed(random_seed)

def parse_args_for_gin():
    """Parse CLI args into gin files and `--name=value` bindings, then load.

    Usage pattern in entrypoints:
        - `parse_args_for_gin()` reads positional `.gin` files and any
          `--param=value` overrides, then calls `load_gin_configs(...)`.
    """
    parser = argparse.ArgumentParser(description='Parse gin configuration files and bindings.')
    parser.add_argument('gin_files', nargs='*', help='Paths to the gin configuration files.')

    args, unknown_args = parser.parse_known_args()

    gin_files = args.gin_files
    gin_bindings = []

    for arg in unknown_args:
        if arg.startswith('--'):
            key_value = arg[2:].split('=', 1)
            if len(key_value) == 2:
                key, raw_value = key_value
                stripped = raw_value.strip()
                # Preserve an explicit empty value as an empty string binding.
                if not stripped:
                    gin_value = repr("")
                # Keep quoted values intact to respect user-specified quoting.
                elif (stripped.startswith("'") and stripped.endswith("'")) or (
                    stripped.startswith('"') and stripped.endswith('"')
                ):
                    gin_value = stripped
                # Allow gin references and macros without forcing string quoting.
                elif stripped.startswith("@") or stripped.startswith("%"):
                    gin_value = stripped
                else:
                    # Let Python-literal values pass through; otherwise quote as string.
                    try:
                        ast.literal_eval(stripped)
                        gin_value = stripped
                    except (ValueError, SyntaxError):
                        gin_value = repr(stripped)
                gin_bindings.append(f"{key} = {gin_value}")

    load_gin_configs(gin_files, gin_bindings)

def _safe_model_name(embedding_model_name: str) -> str:
    return embedding_model_name.replace('/', '_').replace(':', '_')


def get_cache_path(dataset_path, embedding_model_name, suffix: Optional[str] = None):
    """Return the v2 index JSON path for a model's chunked cache.

    - Legacy (v1) files: `embeddings_{safe}.json`
    - Current (v2) index: `embeddings_{safe}.index.json`
    """
    safe_model_name = _safe_model_name(embedding_model_name)
    fname = f"embeddings_{safe_model_name}"
    if suffix is not None:
        fname += f"_{suffix}"
    fname += ".index.json"
    return os.path.join(dataset_path, fname)



def _chunk_dir_from_index(index_path: str) -> str:
    """Return the chunk directory for a given index path.

    Pattern: `embeddings_{safe}.index.json` -> `pt/{safe}/` inside the same
    dataset directory.
    """
    base = os.path.basename(index_path)
    safe = base
    if safe.startswith("embeddings_"):
        safe = safe[len("embeddings_"):]
    if safe.endswith(".index.json"):
        safe = safe[: -len(".index.json")]
    return os.path.join(os.path.dirname(index_path), "pt", safe)

def load_embeddings_from_cache(index_path: str, dataset):
    """Load embeddings from v2 chunked cache if available.

    Returns:
        (sent2emb, missing_sentences):
            - sent2emb: dict[str, np.ndarray] of any sentences found in cache.
            - missing_sentences: list[str] for which no cached vector exists.
    """
    sent2emb: dict[str, np.ndarray] = {}
    if index_path and os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        # minimal validation
        chunks = index.get("chunks", {})
        items = index.get("items", {})
        chunk_dir = _chunk_dir_from_index(index_path)
        loaded_chunks = {}
        for sentence, meta in items.items():
            cid = str(meta["chunk"]) if isinstance(meta.get("chunk"), int) else meta.get("chunk")
            row = int(meta["row"])
            if cid not in loaded_chunks:
                chunk_path = os.path.join(chunk_dir, f"chunk_{cid}.pt")
                if not os.path.exists(chunk_path):
                    continue  # tolerate missing; will be treated as missing sentence
                blob = torch.load(chunk_path, map_location="cpu", weights_only=True)
                loaded_chunks[cid] = blob["embeddings"]  # Tensor [n, D]
            emb = loaded_chunks[cid][row].numpy()
            sent2emb[sentence] = emb
    else:
        # No index found
        pass

    all_sentences = set()
    for split in dataset:
        all_sentences.update(dataset[split]["sentence"])
    missing_sentences = [s for s in all_sentences if s not in sent2emb]
    return sent2emb, missing_sentences

def save_embeddings_to_cache(index_path: str, updates: dict, embedding_dim: int, entries_per_chunk: int = 50000):
    """Append new sentence embeddings to the v2 chunked cache.

    Writes/extends `pt/chunk_*.pt` tensors and updates the JSON index atomically
    using a `.tmp` rename. Header fields include `embedding_dim` and
    `entries_per_chunk` for downstream dimension discovery.

    Args:
        index_path: Path to model-specific index JSON.
        updates: Mapping {sentence: np.ndarray[float32, D]} to add (skips any
            sentences already present in the index).
        embedding_dim: Final dimension D; validated against inputs.
        entries_per_chunk: Max rows per chunk file; starts new chunk as needed.
    """
    if not index_path:
        return
    os.makedirs(_chunk_dir_from_index(index_path), exist_ok=True)
    index = {
        "model": None,
        "dtype": "float32",
        "embedding_dim": embedding_dim,
        "entries_per_chunk": entries_per_chunk,
        "chunks": {},
        "items": {},
    }
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            try:
                index.update(json.load(f))
            except json.JSONDecodeError:
                pass
    # Ensure header fields are correct
    index["embedding_dim"] = embedding_dim
    index["entries_per_chunk"] = entries_per_chunk

    items = index.setdefault("items", {})
    chunks = index.setdefault("chunks", {})

    # Determine current chunk
    existing_chunk_ids = sorted([int(k) for k in chunks.keys()]) if chunks else []
    current_cid = existing_chunk_ids[-1] if existing_chunk_ids else 0
    chunk_dir = _chunk_dir_from_index(index_path)
    def load_chunk(cid: int):
        path = os.path.join(chunk_dir, f"chunk_{cid}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)["embeddings"]
        else:
            return torch.empty((0, embedding_dim), dtype=torch.float32)

    current = load_chunk(current_cid)
    current_count = int(chunks.get(str(current_cid), {}).get("count", current.shape[0]))
    entries_per_chunk = int(index.get("entries_per_chunk", entries_per_chunk))

    buffer_sentences = []
    buffer_embs = []
    for sent, emb in updates.items():
        if sent in items:
            continue
        e = np.asarray(emb, dtype=np.float32)
        if e.shape[-1] != embedding_dim:
            raise ValueError(f"Embedding dim mismatch for '{sent}': {e.shape[-1]} != {embedding_dim}")
        buffer_sentences.append(sent)
        buffer_embs.append(torch.from_numpy(e))

        # Flush if adding this would exceed capacity
        if current_count + len(buffer_embs) > entries_per_chunk:
            # fill current chunk up to capacity
            take = entries_per_chunk - current_count
            if take > 0:
                to_add = torch.stack(buffer_embs[:take], dim=0)
                new_current = torch.cat([current, to_add], dim=0) if current.numel() else to_add
                tmp_path = os.path.join(chunk_dir, f"chunk_{current_cid}.pt.tmp")
                torch.save({"embeddings": new_current}, tmp_path)
                os.replace(tmp_path, os.path.join(chunk_dir, f"chunk_{current_cid}.pt"))
                # update index for taken entries
                for i in range(take):
                    row = current_count + i
                    items[buffer_sentences[i]] = {"chunk": current_cid, "row": row}
                chunks[str(current_cid)] = {"path": f"pt/chunk_{current_cid}.pt", "count": entries_per_chunk}
                buffer_sentences = buffer_sentences[take:]
                buffer_embs = buffer_embs[take:]

            # start new chunk
            current_cid += 1
            current = torch.empty((0, embedding_dim), dtype=torch.float32)
            current_count = 0

    # Flush remaining buffer into current chunk
    if buffer_embs:
        if current.numel():
            new_current = torch.cat([current, torch.stack(buffer_embs, dim=0)], dim=0)
        else:
            new_current = torch.stack(buffer_embs, dim=0)
        tmp_path = os.path.join(chunk_dir, f"chunk_{current_cid}.pt.tmp")
        torch.save({"embeddings": new_current}, tmp_path)
        os.replace(tmp_path, os.path.join(chunk_dir, f"chunk_{current_cid}.pt"))
        start_row = current_count
        for i, sent in enumerate(buffer_sentences):
            items[sent] = {"chunk": current_cid, "row": start_row + i}
        chunks[str(current_cid)] = {"path": f"pt/chunk_{current_cid}.pt", "count": new_current.shape[0]}

    # Write index atomically
    tmp_index = index_path + ".tmp"
    with open(tmp_index, 'w') as f:
        json.dump(index, f)
    os.replace(tmp_index, index_path)

def get_st_dimension(repo_id: str) -> int:
    """Infer a SentenceTransformer embedding dimension without full model load.

    Tries pooling config first; falls back to base `config.json` `hidden_size`.
    Useful when everything is already cached and we only need the dimension.
    """
    try:
        # Most SentenceTransformers models
        cfg_path = hf_hub_download(repo_id,
                                   filename="1_Pooling/config.json",
                                   )  # < 1 kB
        with open(cfg_path) as f:
            return json.load(f)["word_embedding_dimension"]
    except Exception:
        # Fall-back: look at base transformer hidden_size
        base_cfg = hf_hub_download(repo_id, filename="config.json")
        with open(base_cfg) as f:
            return json.load(f).get("hidden_size")

def _load_decoder_only_full_with_layer_caches(
    dataset,
    dataset_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str],
    embedding_column_name: str,
    entries_per_chunk: int,
    device: Optional[str],
    decoder_layer_indices: Sequence[int],
    create_combined_column: bool = True,
):
    """Handle decoder-only-full embeddings with per-layer caches and one encode."""
    if not decoder_layer_indices:
        raise ValueError("decoder_layer_indices must be provided for decoder-only-full embeddings")
    decoder_layers = [int(x) for x in decoder_layer_indices]

    ordered_sentences: list[str] = []
    sentence_order: dict[str, int] = {}
    for split in dataset:
        for sentence in dataset[split]["sentence"]:
            if sentence not in sentence_order:
                sentence_order[sentence] = len(ordered_sentences)
                ordered_sentences.append(sentence)

    layer_cache_paths = {
        layer: get_cache_path(dataset_path, embedding_model_name, suffix=f"layer{layer}")
        for layer in decoder_layers
    }
    layer_sent2emb: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in decoder_layers}
    layer_dims: dict[int, int] = {}

    if embedding_cache_path:
        missing_sentences: set[str] = set()
        for layer in decoder_layers:
            cache_path = layer_cache_paths[layer]
            sent2emb_layer, missing_layer = load_embeddings_from_cache(cache_path, dataset)
            layer_sent2emb[layer] = sent2emb_layer
            if sent2emb_layer:
                layer_dims[layer] = len(next(iter(sent2emb_layer.values())))
            else:
                try:
                    with open(cache_path, "r") as f:
                        layer_dims[layer] = int(json.load(f).get("embedding_dim", 0))
                except Exception:
                    layer_dims[layer] = 0
            if missing_layer:
                missing_sentences.update(missing_layer)
    else:
        missing_sentences = set(ordered_sentences)

    sentences_to_encode = (
        sorted(missing_sentences, key=lambda s: sentence_order.get(s, float("inf")))
        if missing_sentences
        else []
    )
    if sentences_to_encode:
        new_embeddings, total_dim = encode_decoder_only_models(
            embedding_model_name,
            sentences_to_encode,
            decoder_layer_indices=decoder_layers,
            tokens="all",
            device=device,
        )
        per_layer_dim = int(total_dim) // len(decoder_layers) if decoder_layers else 0
        for idx, layer in enumerate(decoder_layers):
            start = idx * per_layer_dim
            end = start + per_layer_dim
            layer_block = new_embeddings[:, start:end] if per_layer_dim else np.empty((len(sentences_to_encode), 0), dtype=np.float32)
            updates = {
                sent: np.array(layer_block[row_idx], dtype=np.float32)
                for row_idx, sent in enumerate(sentences_to_encode)
            }
            layer_sent2emb[layer].update(updates)
            save_embeddings_to_cache(
                layer_cache_paths[layer],
                updates,
                per_layer_dim,
                entries_per_chunk,
            )
            layer_dims[layer] = per_layer_dim
    else:
        for layer in decoder_layers:
            if layer not in layer_dims:
                cache_path = layer_cache_paths[layer]
                try:
                    with open(cache_path, "r") as f:
                        layer_dims[layer] = int(json.load(f).get("embedding_dim", 0))
                except Exception:
                    layer_dims[layer] = 0

    missing_after = [
        sent
        for sent in ordered_sentences
        if any(sent not in layer_sent2emb[layer] for layer in decoder_layers)
    ]
    if missing_after:
        raise ValueError(f"Missing decoder-only-full embeddings for sentences: {missing_after[:5]}...")

    for split in dataset:
        ordered_sentences = dataset[split]["sentence"]
        # Vectorized combine to avoid Python loops/`tolist` overhead, then
        # convert to Arrow via FixedSizeListArray to keep 2D structure.
        per_layer_blocks = [
            np.stack([layer_sent2emb[layer][sent] for sent in ordered_sentences], dtype=np.float32)
            for layer in decoder_layers
        ]

        # Add per-layer columns to reduce per-row materialization cost.
        for layer, block in zip(decoder_layers, per_layer_blocks):
            layer_dim = block.shape[1] if block.size else 0
            if layer_dim == 0:
                column = [[] for _ in range(len(ordered_sentences))]
            else:
                flat = block.reshape(-1)
                flat_arrow = pa.array(flat, type=pa.float32())
                column = pa.FixedSizeListArray.from_arrays(flat_arrow, layer_dim)
            dataset[split] = dataset[split].add_column(
                f"{embedding_column_name}_layer{layer}",
                column,
                new_fingerprint=f"embeddings_layer{layer}",
            )

        if create_combined_column:
            combined_embeddings = np.concatenate(per_layer_blocks, axis=1) if per_layer_blocks else np.empty(
                (len(ordered_sentences), 0), dtype=np.float32
            )
            total_dim = combined_embeddings.shape[1]
            if total_dim == 0:
                column = [[] for _ in range(len(ordered_sentences))]
            else:
                flat = combined_embeddings.reshape(-1)
                flat_arrow = pa.array(flat, type=pa.float32())
                column = pa.FixedSizeListArray.from_arrays(flat_arrow, total_dim)
            dataset[split] = dataset[split].add_column(
                embedding_column_name,
                column,
                new_fingerprint="embeddings",
            )

    embedding_dim = sum(layer_dims.get(layer, 0) for layer in decoder_layers)
    return dataset, int(embedding_dim)

@gin.configurable
def load_dataset_with_embeddings(
    dataset,
    dataset_path: str,
    embedding_model_name: str,
    embedding_cache_path: Optional[str] = None,
    embedding_column_name: str = "target_embeddings",
    add_prefix: Optional[str] = None,
    entries_per_chunk: int = 50000,
    encoder_model_type: str = "sentence-transformers",
    device: Optional[str] = None,
    decoder_layer_indices: Optional[Sequence[int]] = None,
    create_combined_column: bool = True,
):
    """
    Compute or load sentence embeddings for each dataset split and attach them
    as a column, using the v2 chunked cache format.

    Supports multiple encoder backends selected via `encoder_model_type`:
    - "sentence-transformers" (default): uses `SentenceTransformer.encode`.
      Respects `add_prefix` if provided.
    - "decoder-only-punct": loads a causal LM and extracts the final hidden
      state at the token aligned to the single '.' in each sentence.
    - "decoder-only-full": loads a causal LM, collects hidden states from a
      specific transformer layer for every token (with padding to a fixed
      length), and flattens them into a single vector per sentence. Requires
      `decoder_layer_indices` to be provided.

    Caching behavior:
    - If `embedding_cache_path` is not None, embeddings are read from (and new
      ones written to) a model-specific index under `dataset_path` derived by
      `get_cache_path(dataset_path, embedding_model_name)`. Only missing
      sentences are encoded.
    - If `embedding_cache_path` is None, all sentences are encoded and the
      cache is written to the derived path.

    Args:
        dataset: DatasetDict-like with splits as keys and a "sentence" column.
        dataset_path: Directory of the dataset; also where caches are stored.
        embedding_model_name: HF model id. For sentence-transformers, a
            SentenceTransformer repo id; for decoder-only, a causal LM id
            (e.g., "Qwen/Qwen3-8B", "gpt2").
        embedding_cache_path: If not None, enables cache read/write. The actual
            cache path is derived; this parameter is treated as a flag.
        embedding_column_name: Column name to store embeddings.
        add_prefix: Optional prefix used only with sentence-transformers; ignored
            for decoder-only embeddings.
        entries_per_chunk: Max entries per `.pt` cache chunk.
        encoder_model_type: One of {"sentence-transformers", "decoder-only-punct",
            "decoder-only-full"}.
        device: Optional device override (e.g., "cuda", "cpu"); used for
            decoder-only embedding. If None, auto-detects.
        decoder_layer_indices: Required for "decoder-only-full"; list of decoder
            layers (0-based) to cache and concatenate.

    Returns:
        (dataset_with_embeddings, embedding_dim): the updated dataset and the
        embedding dimensionality (from the model or cache header).

    Raises:
        ValueError: Unknown `encoder_model_type`. For decoder-only, also raised
        if a sentence does not contain exactly one '.' or alignment fails.
    """
    model = None
    computed_emb_dim: Optional[int] = None

    # Decide add_prefix only for sentence-transformers; None for decoder-only
    if encoder_model_type != "sentence-transformers":
        add_prefix = None

    if encoder_model_type == "decoder-only-full":
        return _load_decoder_only_full_with_layer_caches(
            dataset,
            dataset_path,
            embedding_model_name,
            embedding_cache_path,
            embedding_column_name,
            entries_per_chunk,
            device,
            decoder_layer_indices,
            create_combined_column=create_combined_column,
        )

    layer_suffix = None

    if embedding_cache_path:
        embedding_cache_path = get_cache_path(dataset_path, embedding_model_name, suffix=layer_suffix)
        sent2emb, missing_sentences = load_embeddings_from_cache(
            embedding_cache_path, dataset)
        
        # Only load/compute if there are missing sentences to encode
        if missing_sentences:
            if encoder_model_type == "sentence-transformers":
                model = SentenceTransformer(embedding_model_name)
                model.eval()
                encode_sents = missing_sentences if not add_prefix else [add_prefix + s for s in missing_sentences]
                new_embeddings = model.encode(encode_sents, batch_size=1, show_progress_bar=True)
                computed_emb_dim = int(model.get_sentence_embedding_dimension())
            elif encoder_model_type == "decoder-only-punct":
                new_embeddings, computed_emb_dim = encode_decoder_only_models(
                    embedding_model_name,
                    missing_sentences,
                    decoder_layer_indices=None,
                    tokens="punct",
                    device=device,
                )
            else:
                raise ValueError(f"Unknown encoder_model_type: {encoder_model_type}")

            updates = {}
            for sent, emb in zip(missing_sentences, new_embeddings):
                e = np.array(emb, dtype=np.float32)
                sent2emb[sent] = e
                updates[sent] = e
            # Save only new items to chunked cache
            save_embeddings_to_cache(embedding_cache_path, updates, int(computed_emb_dim), entries_per_chunk)
        
        for split in dataset:
            embeddings = [sent2emb[s].tolist() for s in dataset[split]["sentence"]]
            dataset[split] = dataset[split].add_column(embedding_column_name, embeddings, new_fingerprint="embeddings")
    else:
        # No cache flag provided; encode all sentences and write cache
        all_sentences = set()
        for split in dataset:
            all_sentences.update(dataset[split]["sentence"])
        sent2emb = {}
        encode_sents = list(all_sentences)
        if encoder_model_type == "sentence-transformers":
            model = SentenceTransformer(embedding_model_name)
            model.eval()
            if add_prefix:
                encode_sents = [add_prefix + s for s in encode_sents]
            new_embeddings = model.encode(encode_sents, show_progress_bar=True)
            computed_emb_dim = int(model.get_sentence_embedding_dimension())
        elif encoder_model_type == "decoder-only-punct":
            new_embeddings, computed_emb_dim = encode_decoder_only_models(
                embedding_model_name,
                encode_sents,
                decoder_layer_indices=[-1],
                tokens="punct",
                device=device,
            )
        else:
            raise ValueError(f"Unknown encoder_model_type: {encoder_model_type}")

        updates = {}
        for sent, emb in zip(encode_sents, new_embeddings):
            e = np.array(emb, dtype=np.float32)
            sent2emb[sent] = e
            updates[sent] = e
        save_embeddings_to_cache(
            get_cache_path(dataset_path, embedding_model_name, suffix=layer_suffix),
            updates,
            int(computed_emb_dim),
            entries_per_chunk,
        )
        for split in dataset:
            embeddings = [sent2emb[s].tolist() for s in dataset[split]["sentence"]]
            dataset[split] = dataset[split].add_column(embedding_column_name, embeddings, new_fingerprint=None)

    # Get embedding dimension - avoid loading model if possible
    if computed_emb_dim is not None:
        embedding_dim = int(computed_emb_dim)
    elif model is not None and encoder_model_type == "sentence-transformers":
        embedding_dim = int(model.get_sentence_embedding_dimension())
    else:
        # All embeddings were cached; prefer reading dim from index header
        index_path = get_cache_path(dataset_path, embedding_model_name, suffix=layer_suffix)
        try:
            with open(index_path, 'r') as f:
                embedding_dim = int(json.load(f).get("embedding_dim"))
        except Exception:
            # Fallback to hub metadata if index missing or malformed
            embedding_dim = get_st_dimension(embedding_model_name)
    
    return dataset, embedding_dim

def ternary_search(objective_fn, left, right, precision, optimize: str = "min"):
    """Generic ternary search over a closed interval in log space.

    Args:
        objective_fn: Callable mapping x -> scalar score.
        left: Interval start (float).
        right: Interval end (float).
        precision: Stop when interval width < precision.
        optimize: 'min' or 'max' to choose the better direction.

    Returns:
        (x_best, score_best)
    """
    mode = str(optimize).lower()
    if mode not in {"max", "min"}:
        raise ValueError(f"optimize must be 'max' or 'min', found {optimize!r}")

    def _search(lft: float, rgt: float):
        if abs(rgt - lft) < float(precision):
            mid = 0.5 * (lft + rgt)
            return mid, float(objective_fn(mid))

        left_third = (2 * lft + rgt) / 3.0
        right_third = (lft + 2 * rgt) / 3.0
        score_left = float(objective_fn(left_third))
        score_right = float(objective_fn(right_third))

        if not math.isfinite(score_left) and not math.isfinite(score_right):
            return _search(left_third, right_third)
        if not math.isfinite(score_left):
            return _search(left_third, rgt)
        if not math.isfinite(score_right):
            return _search(lft, right_third)

        better = (score_left > score_right) if mode == "max" else (score_left < score_right)
        if better:
            return _search(lft, right_third)
        return _search(left_third, rgt)

    return _search(float(left), float(right))

def search_reg_param(
    objective_fn,
    optimize,
    log_bounds=(-12.0, 12.0),
    tolerance_ratio: float = 0.01,
):
    """Search for the best log-regularization value via ternary search.

    Args:
        objective_fn: Function mapping log10(lambda) -> score (float).
        optimize: 'max' or 'min'.
        log_bounds: Inclusive search interval (left, right) in log10 space.
        tolerance_ratio: Relative half-width for the reported bracket.

    Returns:
        (best_log, best_score, (lo, hi)) where best_log is log10(lambda).
    """
    log_lower, log_upper = log_bounds
    mode = str(optimize).lower()
    if mode not in {"max", "min"}:
        raise ValueError(f"optimize must be 'max' or 'min', found {optimize!r}")

    tol = max(float(tolerance_ratio), 0.0)
    precision = math.log10(1.0 + tol) if tol > 0 else 1e-6

    best_log, best_score = ternary_search(objective_fn, float(log_lower), float(log_upper), precision, optimize=mode)
    bracket = (
        max(log_lower, best_log - precision),
        min(log_upper, best_log + precision),
    )
    return best_log, best_score, bracket


def encode_decoder_only_models(
    model_name: str,
    sentences: list[str],
    decoder_layer_indices: Optional[Sequence[int]] = None,
    tokens: Literal["punct", "all"] = "punct",
    device: Optional[str] = None,
    use_safetensors: bool = True,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
) -> tuple[np.ndarray, int]:
    """Encode decoder-only models for punctuation tokens or full sequences at chosen layers."""
    layers = list(decoder_layer_indices) if decoder_layer_indices is not None else [-1]
    if len(layers) == 0:
        raise ValueError("decoder_layer_indices must be non-empty when provided")
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    use_auto = (device or "").lower() != "cpu" and torch.cuda.is_available()
    loader_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if use_auto else None,
        "use_safetensors": use_safetensors,
    }
    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        loader_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)
    if not use_auto:
        model.to(torch.device("cpu"))
    model.eval()
    target_device = model.get_input_embeddings().weight.device if use_auto else torch.device("cpu")

    outputs: list[np.ndarray] = []

    if tokens == "punct":
        dot_idx = []
        for s in sentences:
            if s.count(".") != 1:
                raise ValueError(f"decoder-only-punct requires exactly one '.' per sentence: {s!r}")
            dot_idx.append(s.index("."))

        bs = 32 if use_auto else 8
        for start in range(0, len(sentences), bs):
            chunk = sentences[start : start + bs]
            enc = tok(
                chunk,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(target_device) for k, v in enc.items()}
            positions: list[int] = []
            for idx, (o, dot) in enumerate(zip(offs, dot_idx[start : start + len(chunk)])):
                offsets = [(int(a), int(b)) for a, b in o]
                pos = next((t for t, (a, b) in enumerate(offsets) if a != b and a <= dot < b), None)
                if pos is None:
                    raise ValueError(f"Could not align '.' for: {chunk[idx]!r}")
                positions.append(pos)
            with torch.no_grad():
                hidden_states = model(**enc, output_hidden_states=True).hidden_states  # type: ignore[arg-type]
            layer_vecs = []
            for layer_idx in layers:
                resolved = -1 if layer_idx < 0 else layer_idx + 1
                layer_tensor = hidden_states[resolved]
                pos_tensor = torch.tensor(positions, device=layer_tensor.device, dtype=torch.long)
                layer_vecs.append(layer_tensor[torch.arange(layer_tensor.size(0), device=layer_tensor.device), pos_tensor, :])
            flat = torch.cat(layer_vecs, dim=1).to(torch.float32).cpu().numpy()
            outputs.append(flat)
        embedding_dim = outputs[0].shape[1] if outputs else 0
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, embedding_dim), dtype=np.float32), int(embedding_dim)

    if tokens != "all":
        raise ValueError(f"Unknown tokens mode '{tokens}', expected 'punct' or 'all'")

    lengths = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sentences]
    if lengths and len(set(lengths)) != 1:
        raise ValueError("decoder-only-full requires all prompts to tokenize to the same length")
    seq_len = lengths[0] if lengths else 0

    batch_size = 4
    layer_count: Optional[int] = None
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
            truncation=False,
        )
        enc = {k: v.to(target_device) for k, v in enc.items()}
        with torch.no_grad():
            hidden_states = model(**enc, output_hidden_states=True).hidden_states  # type: ignore[arg-type]
        if layer_count is None:
            layer_count = len(hidden_states) - 1
            max_layer = max(layers)
            if max_layer >= layer_count:
                raise ValueError(f"Requested layer {max_layer} but model has {layer_count} layers")
        layer_vecs = []
        for layer_idx in layers:
            resolved = -1 if layer_idx < 0 else layer_idx + 1
            target_layer = hidden_states[resolved]
            layer_vecs.append(target_layer.reshape(target_layer.shape[0], -1))
        flat = torch.cat(layer_vecs, dim=1).to(torch.float32).cpu().numpy()
        outputs.append(flat)

    embedding_dim = outputs[0].shape[1] if outputs else seq_len * len(layers)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, embedding_dim), dtype=np.float32), int(embedding_dim)

def calculate_variance_explained(model: Any, dataset: Any) -> dict:
    """Compute variance explained metrics between predictions and targets.

    Assumes the model returns `encoder_hidden_states` compatible with the
    dataset's `target_embeddings`. Metrics include total, residual, regression
    sums of squares, R^2, and explained variance ratio.

    Args:
        model: Module with `.eval()` and forward returning `encoder_hidden_states`.
        dataset: Iterable of examples providing `target_embeddings`.

    Returns:
        dict with keys: SS_Total, SS_Residual, SS_Regression, R_Squared,
        Explained_Variance_Ratio.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a dataloader for batch processing
    def _collate(batch: list[dict]) -> dict:
        def _stack(values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.stack(values, dim=0)
            return torch.tensor(np.stack(values, axis=0))

        return {
            key: [row["sentence"] for row in batch]
            if key == "sentence"
            else _stack([row[key] for row in batch])
            for key in batch[0].keys()
        }

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=_collate,
        shuffle=False,
    )
    
    all_predictions = []
    all_targets = []
    
    # Get predictions for all samples
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Variance Explained", unit="batch"):
            # Extract inputs and move to the same device as model
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items() if k != "target_embeddings" and k != "sentence"}
            
            # Forward pass
            outputs = model(**inputs)
            predictions = outputs.encoder_hidden_states
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch["target_embeddings"])
    
    # Concatenate all batches and flatten features to 2D for consistent math
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)

    # Calculate variance metrics
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_regression = np.sum((y_pred - y_mean) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return {
        "SS_Total": float(ss_total),
        "SS_Residual": float(ss_residual),
        "SS_Regression": float(ss_regression),
        "R_Squared": float(r_squared),
        "Explained_Variance_Ratio": float(ss_regression / ss_total) if ss_total != 0 else float('nan'),
    }
