from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from nnsight import LanguageModel


def torch_dtype_from_str(name: str) -> torch.dtype:
    """Convert a dtype alias into a torch dtype."""

    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return lookup[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {sorted(lookup)}") from exc


def resolve_device(device_str: str) -> torch.device:
    """Normalize device strings into torch.device objects."""

    if device_str.startswith("cuda"):
        return torch.device(device_str)
    if device_str in {"cpu", "mps"}:
        return torch.device(device_str)
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def maybe_empty_cache(device: torch.device) -> None:
    """Flush CUDA cache between tracing passes to reduce peak memory."""

    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_layers(model: LanguageModel) -> list:
    """Locate decoder layers on the wrapped model."""

    base = getattr(model, "model", None)
    if base is None:
        base = model

    if hasattr(base, "layers"):
        return list(base.layers)
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return list(base.model.layers)
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return list(base.transformer.h)
    raise ValueError("Unable to locate transformer layers on the language model")


def resolve_token_positions(
    token_strings: Sequence[str],
    explicit_positions: Optional[Sequence[int]],
    focus: Literal["all", "active_clause", "passive_clause"],
) -> list[int]:
    """Select token positions to patch based on focus or explicit list.

    focus options:
    - "all": every token position
    - "active_clause": tokens before the first period
    - "passive_clause": tokens after the first period
    """

    seq_len = len(token_strings)
    if explicit_positions:
        return [pos for pos in explicit_positions if 0 <= pos < seq_len]
    focus = (focus or "all").lower()
    if focus == "passive_clause":
        try:
            period_idx = token_strings.index(".")
            start = period_idx + 1
        except ValueError:
            start = seq_len // 2
        return list(range(start, seq_len))
    if focus == "active_clause":
        try:
            period_idx = token_strings.index(".")
            end = period_idx
        except ValueError:
            end = seq_len
        return list(range(end))
    return list(range(seq_len))


def pretty_token(token: str) -> str:
    """Render byte-pair tokens with a visible whitespace marker."""

    if token.startswith("Ġ"):
        return "▁" + token[1:]
    return token


def proxy_to_float(value) -> float:
    """Extract a float value from an nnsight proxy or tensor-like object."""

    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


__all__ = [
    "get_layers",
    "maybe_empty_cache",
    "pretty_token",
    "proxy_to_float",
    "resolve_device",
    "resolve_token_positions",
    "torch_dtype_from_str",
]
