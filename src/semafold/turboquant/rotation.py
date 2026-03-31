"""Deterministic rotation helpers for TurboQuant preview codecs."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "apply_rotation",
    "invert_rotation",
    "seeded_haar_rotation",
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


@lru_cache(maxsize=32)
def seeded_haar_rotation(dimension: int, seed: int) -> np.ndarray:
    """Return a deterministic orthogonal rotation matrix.

    The matrix is generated from a Gaussian QR factorization with a sign
    correction on the diagonal of ``R`` so the result is reproducible for a
    fixed ``(dimension, seed)`` pair.
    """

    dimension = _validate_dimension(dimension)
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension), dtype=np.float64)
    q, r = np.linalg.qr(gaussian, mode="reduced")
    diagonal_signs = np.sign(np.diag(r))
    diagonal_signs[diagonal_signs == 0.0] = 1.0
    q = q * diagonal_signs
    rotation = np.asarray(q, dtype=np.float32)
    rotation.setflags(write=False)
    return rotation


def apply_rotation(unit_rows: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate row-major vectors with an orthogonal matrix."""

    return np.asarray(unit_rows @ rotation.T, dtype=np.float32)


def invert_rotation(rotated_rows: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Invert a row-major rotation produced by :func:`apply_rotation`."""

    return np.asarray(rotated_rows @ rotation, dtype=np.float32)
