"""Hot-path helpers for the TurboQuant MSE preview codec."""

from __future__ import annotations

import numpy as np

from semafold.turboquant.codebook import TurboQuantScalarCodebook
from semafold.turboquant.rotation import apply_rotation, invert_rotation

__all__ = [
    "dequantize_rows",
    "normalize_rows",
    "quantize_rows",
    "restore_rows",
]


def _require_2d_rows(name: str, rows: np.ndarray) -> np.ndarray:
    array = np.asarray(rows)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if array.shape[1] < 2:
        raise ValueError(f"{name} must have dimension >= 2")
    return array


def normalize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized float32 vectors and float32 row norms."""

    array = _require_2d_rows("rows", rows)
    working = np.asarray(array, dtype=np.float64)
    norms64 = np.linalg.norm(working, axis=1)
    norms = np.asarray(norms64, dtype=np.float32)
    safe = np.where(norms64 > 0.0, norms64, 1.0)
    normalized = np.divide(
        working,
        safe[:, None],
        out=np.zeros_like(working, dtype=np.float64),
        where=safe[:, None] > 0.0,
    )
    return np.asarray(normalized, dtype=np.float32), norms


def quantize_rows(
    unit_rows: np.ndarray,
    *,
    rotation: np.ndarray,
    codebook: TurboQuantScalarCodebook,
) -> np.ndarray:
    """Rotate and quantize normalized rows with a TurboQuant scalar codebook."""

    unit_rows = np.asarray(_require_2d_rows("unit_rows", unit_rows), dtype=np.float32)
    rotated = apply_rotation(unit_rows, rotation)
    return codebook.quantize(np.clip(rotated, -1.0, 1.0))


def dequantize_rows(
    indices: np.ndarray,
    *,
    rotation: np.ndarray,
    codebook: TurboQuantScalarCodebook,
) -> np.ndarray:
    """Dequantize TurboQuant scalar indices back to normalized float32 rows."""

    array = _require_2d_rows("indices", np.asarray(indices))
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("indices must contain integers")
    rotated = codebook.dequantize(array)
    restored = invert_rotation(rotated, rotation)
    if not np.isfinite(restored).all():
        raise ValueError("dequantized TurboQuant rows contain non-finite values")
    return np.asarray(restored, dtype=np.float32)


def restore_rows(unit_rows: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Restore original row norms after TurboQuant dequantization."""

    unit_rows = np.asarray(_require_2d_rows("unit_rows", unit_rows), dtype=np.float32)
    norms = np.asarray(norms, dtype=np.float32)
    if norms.ndim != 1:
        raise ValueError("norms must be a 1D array")
    if unit_rows.shape[0] != norms.shape[0]:
        raise ValueError("norm count must match the row count")
    if not np.isfinite(norms).all() or np.any(norms < 0.0):
        raise ValueError("norms must be finite and non-negative")
    restored = unit_rows * norms[:, None]
    if not np.isfinite(restored).all():
        raise ValueError("restored TurboQuant rows contain non-finite values")
    return np.asarray(restored, dtype=np.float32)
