"""Canonical helpers for flattening and restoring cache-shaped K/V tensors."""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "CANONICAL_CACHE_LAYOUT",
    "cache_layout_metadata",
    "flatten_cache_rows",
    "restore_cache_rows",
    "validate_cache_pair",
    "validate_cache_tensor",
]


CANONICAL_CACHE_LAYOUT: Final[str] = "lhsd_rows_v1"


def _ensure_cache_tensor(cache: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(cache, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if cache.ndim != 4:
        raise ValueError(f"{name} must have shape (layers, heads, seq_len, head_dim)")
    if cache.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(cache.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if int(cache.shape[-1]) < 2:
        raise ValueError(f"{name} head_dim must be >= 2")
    return cache


def validate_cache_tensor(cache: np.ndarray, *, name: str = "cache") -> tuple[int, int, int, int]:
    """Validate one cache tensor and return ``(layers, heads, seq_len, head_dim)``."""
    validated = _ensure_cache_tensor(cache, name=name)
    layers, heads, seq_len, head_dim = validated.shape
    return int(layers), int(heads), int(seq_len), int(head_dim)


def validate_cache_pair(keys: np.ndarray, values: np.ndarray) -> tuple[int, int, int, int]:
    """Validate key and value tensors and ensure they share the same cache shape."""
    key_shape = validate_cache_tensor(keys, name="keys")
    value_shape = validate_cache_tensor(values, name="values")
    if key_shape != value_shape:
        raise ValueError("keys and values must have identical cache shape")
    return key_shape


def flatten_cache_rows(cache: np.ndarray) -> np.ndarray:
    """Flatten ``(L, H, S, D)`` cache tensors into canonical ``(L*H*S, D)`` rows."""
    layers, heads, seq_len, head_dim = validate_cache_tensor(cache)
    return np.asarray(cache.reshape(layers * heads * seq_len, head_dim))


def restore_cache_rows(
    rows: np.ndarray,
    *,
    layers: int,
    heads: int,
    seq_len: int,
    head_dim: int,
) -> np.ndarray:
    """Restore canonical row-major cache data back to ``(L, H, S, D)`` form."""
    if not isinstance(rows, np.ndarray):
        raise TypeError("rows must be a numpy.ndarray")
    if rows.ndim != 2:
        raise ValueError("rows must have shape (n, d)")
    expected_shape = (int(layers) * int(heads) * int(seq_len), int(head_dim))
    if tuple(int(dim) for dim in rows.shape) != expected_shape:
        raise ValueError("rows shape does not match the requested cache layout")
    return np.asarray(rows.reshape(int(layers), int(heads), int(seq_len), int(head_dim)))


def cache_layout_metadata(*, layers: int, heads: int, seq_len: int, head_dim: int) -> dict[str, object]:
    """Return normalized metadata for the canonical cache layout."""
    for name, value in {
        "layers": layers,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    return {
        "layout": CANONICAL_CACHE_LAYOUT,
        "layers": int(layers),
        "heads": int(heads),
        "seq_len": int(seq_len),
        "head_dim": int(head_dim),
        "row_count": int(layers) * int(heads) * int(seq_len),
    }
