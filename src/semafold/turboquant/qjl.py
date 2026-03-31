"""QJL primitives used by TurboQuant residual-sign artifacts."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = [
    "qjl_decode_rows",
    "qjl_encode_rows",
    "seeded_gaussian_projection",
]


def _validate_dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an int")
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    return dimension


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    return seed


def _validate_projection(projection: np.ndarray) -> np.ndarray:
    if not isinstance(projection, np.ndarray):
        raise TypeError("projection must be a numpy.ndarray")
    if projection.ndim != 2:
        raise ValueError("projection must be a rank-2 array")
    if projection.shape[0] != projection.shape[1]:
        raise ValueError("projection must be square")
    if not np.issubdtype(projection.dtype, np.floating):
        raise TypeError("projection must contain floating-point values")
    if not np.isfinite(projection).all():
        raise ValueError("projection must contain only finite values")
    return np.asarray(projection, dtype=np.float32)


def _validate_residual_rows(rows: np.ndarray, *, dimension: int) -> np.ndarray:
    if not isinstance(rows, np.ndarray):
        raise TypeError("unit_residual_rows must be a numpy.ndarray")
    if rows.ndim != 2:
        raise ValueError("unit_residual_rows must be a rank-2 array")
    if rows.shape[1] != dimension:
        raise ValueError("unit_residual_rows dimension does not match projection")
    if not np.issubdtype(rows.dtype, np.floating):
        raise TypeError("unit_residual_rows must contain floating-point values")
    if not np.isfinite(rows).all():
        raise ValueError("unit_residual_rows must contain only finite values")
    return np.asarray(rows, dtype=np.float32)


def _validate_gamma(gamma: np.ndarray, *, row_count: int) -> np.ndarray:
    if not isinstance(gamma, np.ndarray):
        raise TypeError("gamma must be a numpy.ndarray")
    if gamma.ndim != 1:
        raise ValueError("gamma must be a rank-1 array")
    if gamma.shape[0] != row_count:
        raise ValueError("gamma length does not match sign row count")
    if not np.issubdtype(gamma.dtype, np.floating):
        raise TypeError("gamma must contain floating-point values")
    if not np.isfinite(gamma).all():
        raise ValueError("gamma must contain only finite values")
    if np.any(gamma < 0.0):
        raise ValueError("gamma must be >= 0")
    return np.asarray(gamma, dtype=np.float32)


@lru_cache(maxsize=32)
def seeded_gaussian_projection(dimension: int, seed: int) -> np.ndarray:
    """Return a deterministic dense Gaussian projection matrix.

    The matrix is cached and returned as a read-only ``float32`` array so
    encode/decode paths can share the same projection without re-sampling.
    """

    dimension = _validate_dimension(dimension)
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((dimension, dimension), dtype=np.float64).astype(np.float32)
    projection.setflags(write=False)
    return projection


def qjl_encode_rows(unit_residual_rows: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Encode residual directions into 1-bit QJL sign rows.

    Zero rows are encoded as all-zero sign rows by convention.
    Non-zero rows use a deterministic sign threshold of ``>= 0 -> 1``.
    """

    projection = _validate_projection(projection)
    rows = _validate_residual_rows(unit_residual_rows, dimension=int(projection.shape[0]))

    scores = np.asarray(rows @ projection.T, dtype=np.float32)
    sign_rows = np.asarray(scores >= 0.0, dtype=np.uint8)
    zero_mask = np.linalg.norm(rows.astype(np.float64), axis=1) == 0.0
    if np.any(zero_mask):
        sign_rows[zero_mask] = 0
    return sign_rows


def qjl_decode_rows(sign_rows: np.ndarray, gamma: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Decode QJL sign rows into residual estimates.

    The returned rows are in the same row-major vector space as the residuals.
    ``gamma`` carries the per-row residual norm that is factored out before the
    sign sketch is produced.
    """

    projection = _validate_projection(projection)
    if not isinstance(sign_rows, np.ndarray):
        raise TypeError("sign_rows must be a numpy.ndarray")
    if sign_rows.ndim != 2:
        raise ValueError("sign_rows must be a rank-2 array")
    if sign_rows.shape[1] != projection.shape[0]:
        raise ValueError("sign_rows dimension does not match projection")
    if not np.issubdtype(sign_rows.dtype, np.integer):
        raise TypeError("sign_rows must contain integers")
    if np.any((sign_rows != 0) & (sign_rows != 1)):
        raise ValueError("sign_rows must contain only 0/1 values")

    row_count = int(sign_rows.shape[0])
    gamma = _validate_gamma(gamma, row_count=row_count)
    sign_pm1 = np.where(sign_rows > 0, 1.0, -1.0).astype(np.float32)
    coefficient = np.float32(math.sqrt(math.pi / 2.0) / float(projection.shape[0]))
    decoded = np.asarray(sign_pm1 @ projection, dtype=np.float32) * coefficient
    decoded *= gamma.reshape(row_count, 1)
    zero_mask = gamma == 0.0
    if np.any(zero_mask):
        decoded[zero_mask] = 0.0
    return decoded
