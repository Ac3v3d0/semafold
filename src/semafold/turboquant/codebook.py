"""Beta-marginal codebook utilities for TurboQuant preview codecs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "TurboQuantScalarCodebook",
    "numerical_codebook_distortion",
    "solve_beta_lloyd_max_codebook",
]


def _validate_dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an int")
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    return dimension


def _validate_bit_width(bit_width: int) -> int:
    if not isinstance(bit_width, int) or isinstance(bit_width, bool):
        raise TypeError("bit_width must be an int")
    if bit_width < 1 or bit_width > 4:
        raise ValueError("bit_width must be between 1 and 4 in the current TurboQuant preview")
    return bit_width


def _validate_grid_size(grid_size: int) -> int:
    if not isinstance(grid_size, int) or isinstance(grid_size, bool):
        raise TypeError("grid_size must be an int")
    if grid_size < 1025:
        raise ValueError("grid_size must be >= 1025")
    return grid_size


def _validate_iterations(max_iterations: int) -> int:
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an int")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    return max_iterations


def _validate_tolerance(tolerance: float) -> float:
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be float-compatible")
    result = float(tolerance)
    if result <= 0.0:
        raise ValueError("tolerance must be > 0")
    return result


def _readonly_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    array.setflags(write=False)
    return array


def _integration_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid_size = _validate_grid_size(grid_size)
    step = 2.0 / float(grid_size)
    points = np.linspace(
        -1.0 + (step / 2.0),
        1.0 - (step / 2.0),
        num=grid_size,
        dtype=np.float64,
    )
    weights = np.full((grid_size,), step, dtype=np.float64)
    return points, weights


def beta_coordinate_density(points: np.ndarray, dimension: int) -> np.ndarray:
    """Exact marginal density for one coordinate of a random sphere point."""

    dimension = _validate_dimension(dimension)
    points = np.asarray(points, dtype=np.float64)
    if np.any(points <= -1.0) or np.any(points >= 1.0):
        raise ValueError("points must lie strictly inside (-1, 1)")
    exponent = 0.5 * float(dimension - 3)
    log_norm = math.lgamma(dimension / 2.0) - (0.5 * math.log(math.pi)) - math.lgamma((dimension - 1) / 2.0)
    log_density = log_norm + exponent * np.log1p(-(points * points))
    return np.exp(log_density, dtype=np.float64)


def _weighted_mean(points: np.ndarray, masses: np.ndarray) -> float:
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return float(np.mean(points))
    return float(np.sum(points * masses) / total_mass)


def _initial_centers(points: np.ndarray, masses: np.ndarray, levels: int) -> np.ndarray:
    cumulative = np.cumsum(masses)
    cumulative /= cumulative[-1]
    quantiles = np.linspace(0.0, 1.0, num=levels + 1, dtype=np.float64)
    boundaries = np.interp(quantiles, cumulative, points, left=points[0], right=points[-1])
    centers = np.empty((levels,), dtype=np.float64)
    for index in range(levels):
        if index == levels - 1:
            mask = (points >= boundaries[index]) & (points <= boundaries[index + 1])
        else:
            mask = (points >= boundaries[index]) & (points < boundaries[index + 1])
        if np.any(mask):
            centers[index] = _weighted_mean(points[mask], masses[mask])
        else:
            centers[index] = 0.5 * (boundaries[index] + boundaries[index + 1])
    return centers


def _symmetrize_centers(centers: np.ndarray) -> np.ndarray:
    mirrored = 0.5 * (centers - centers[::-1])
    return np.sort(mirrored)


def _boundaries_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1D array with at least two entries")
    if not np.all(np.diff(centers) >= 0.0):
        raise ValueError("centers must be sorted in ascending order")
    boundaries = np.empty((centers.size + 1,), dtype=np.float64)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    return boundaries


@dataclass(frozen=True, slots=True)
class TurboQuantScalarCodebook:
    """Scalar codebook for the Beta marginal used by TurboQuant MSE."""

    dimension: int
    bits_per_scalar: int
    centers: np.ndarray
    boundaries: np.ndarray
    cell_masses: np.ndarray
    expected_coordinate_mse: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        _validate_dimension(self.dimension)
        _validate_bit_width(self.bits_per_scalar)
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool) or self.iterations < 0:
            raise ValueError("iterations must be a non-negative int")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        object.__setattr__(self, "centers", _readonly_float_array(self.centers))
        object.__setattr__(self, "boundaries", _readonly_float_array(self.boundaries))
        object.__setattr__(self, "cell_masses", _readonly_float_array(self.cell_masses))
        if self.centers.ndim != 1:
            raise ValueError("centers must be a 1D array")
        if self.boundaries.ndim != 1 or self.boundaries.size != self.centers.size + 1:
            raise ValueError("boundaries must be a 1D array with len(centers) + 1 entries")
        if self.cell_masses.ndim != 1 or self.cell_masses.shape != self.centers.shape:
            raise ValueError("cell_masses must match centers shape")
        if not np.all(np.diff(self.centers) >= 0.0):
            raise ValueError("centers must be sorted in ascending order")
        if not np.all(np.diff(self.boundaries) >= 0.0):
            raise ValueError("boundaries must be sorted in ascending order")
        if not np.isclose(float(self.boundaries[0]), -1.0, atol=1e-6):
            raise ValueError("first boundary must be -1")
        if not np.isclose(float(self.boundaries[-1]), 1.0, atol=1e-6):
            raise ValueError("last boundary must be 1")
        if not np.isfinite(self.expected_coordinate_mse) or float(self.expected_coordinate_mse) < 0.0:
            raise ValueError("expected_coordinate_mse must be finite and >= 0")
        total_mass = float(np.sum(self.cell_masses, dtype=np.float64))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("cell_masses must sum to a positive value")

    @property
    def levels(self) -> int:
        return int(self.centers.size)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.centers.shape

    @property
    def ndim(self) -> int:
        return self.centers.ndim

    @property
    def size(self) -> int:
        return int(self.centers.size)

    def __len__(self) -> int:
        return int(self.centers.size)

    def __array__(self, dtype: np.dtype[np.generic] | None = None) -> np.ndarray:
        return np.asarray(self.centers, dtype=dtype)

    def astype(self, dtype: np.dtype[np.generic]) -> np.ndarray:
        return np.asarray(self.centers, dtype=dtype)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        indices = np.searchsorted(self.boundaries[1:-1], np.clip(values, -1.0, 1.0), side="right")
        return np.asarray(indices, dtype=np.uint8)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        if not isinstance(indices, np.ndarray):
            raise TypeError("indices must be a numpy.ndarray")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("indices must contain integers")
        if np.any(indices < 0) or np.any(indices >= self.levels):
            raise ValueError("indices contain values outside the codebook range")
        return np.asarray(self.centers[indices], dtype=np.float32)

    @classmethod
    def from_centers(
        cls,
        dimension: int,
        bits_per_scalar: int,
        centers: np.ndarray,
        *,
        grid_size: int = 8193,
        iterations: int = 0,
        converged: bool = True,
    ) -> "TurboQuantScalarCodebook":
        dimension = _validate_dimension(dimension)
        bits_per_scalar = _validate_bit_width(bits_per_scalar)
        grid_size = _validate_grid_size(grid_size)
        centers = np.asarray(centers, dtype=np.float64)
        expected_levels = 1 << bits_per_scalar
        if centers.ndim != 1 or centers.size != expected_levels:
            raise ValueError("centers must contain exactly 2**bits_per_scalar entries")
        boundaries = _boundaries_from_centers(centers)
        points, weights = _integration_grid(grid_size)
        masses = beta_coordinate_density(points, dimension) * weights
        cell_masses = np.empty((expected_levels,), dtype=np.float64)
        quantized = np.empty_like(points)
        for index in range(expected_levels):
            left = boundaries[index]
            right = boundaries[index + 1]
            if index == expected_levels - 1:
                mask = (points >= left) & (points <= right)
            else:
                mask = (points >= left) & (points < right)
            cell_masses[index] = float(np.sum(masses[mask], dtype=np.float64))
            quantized[mask] = centers[index]
        total_mass = float(np.sum(cell_masses, dtype=np.float64))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise RuntimeError("failed to build a valid TurboQuant codebook from centers")
        cell_masses /= total_mass
        expected_coordinate_mse = float(np.sum(((points - quantized) ** 2) * masses, dtype=np.float64) / total_mass)
        return cls(
            dimension=dimension,
            bits_per_scalar=bits_per_scalar,
            centers=centers,
            boundaries=boundaries,
            cell_masses=cell_masses,
            expected_coordinate_mse=expected_coordinate_mse,
            iterations=iterations,
            converged=converged,
        )


@lru_cache(maxsize=64)
def solve_beta_lloyd_max_codebook(
    dimension: int,
    bit_width: int,
    *,
    grid_size: int = 8193,
    max_iterations: int = 128,
    tolerance: float = 1e-8,
) -> TurboQuantScalarCodebook:
    """Numerically solve a 1D Lloyd-Max codebook for the Beta marginal."""

    dimension = _validate_dimension(dimension)
    bit_width = _validate_bit_width(bit_width)
    grid_size = _validate_grid_size(grid_size)
    max_iterations = _validate_iterations(max_iterations)
    tolerance = _validate_tolerance(tolerance)

    points, weights = _integration_grid(grid_size)
    masses = beta_coordinate_density(points, dimension) * weights
    levels = 1 << bit_width
    centers = _initial_centers(points, masses, levels)
    centers = _symmetrize_centers(centers)

    iteration_count = 0
    converged = False
    for iteration_index in range(max_iterations):
        iteration_count = iteration_index + 1
        boundaries = _boundaries_from_centers(centers)
        updated = np.empty_like(centers)
        for index in range(levels):
            left = boundaries[index]
            right = boundaries[index + 1]
            if index == levels - 1:
                mask = (points >= left) & (points <= right)
            else:
                mask = (points >= left) & (points < right)
            if np.any(mask):
                updated[index] = _weighted_mean(points[mask], masses[mask])
            else:
                updated[index] = 0.5 * (left + right)
        updated = _symmetrize_centers(updated)
        if np.max(np.abs(updated - centers)) <= tolerance:
            centers = updated
            converged = True
            break
        centers = updated

    boundaries = _boundaries_from_centers(centers)
    cell_masses = np.empty((levels,), dtype=np.float64)
    quantized = np.empty_like(points)
    for index in range(levels):
        left = boundaries[index]
        right = boundaries[index + 1]
        if index == levels - 1:
            mask = (points >= left) & (points <= right)
        else:
            mask = (points >= left) & (points < right)
        cell_masses[index] = float(np.sum(masses[mask], dtype=np.float64))
        quantized[mask] = centers[index]
    total_mass = float(np.sum(cell_masses, dtype=np.float64))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("failed to integrate a valid TurboQuant codebook partition")
    cell_masses /= total_mass
    expected_coordinate_mse = float(np.sum(((points - quantized) ** 2) * masses, dtype=np.float64) / total_mass)

    return TurboQuantScalarCodebook(
        dimension=dimension,
        bits_per_scalar=bit_width,
        centers=centers,
        boundaries=boundaries,
        cell_masses=cell_masses,
        expected_coordinate_mse=expected_coordinate_mse,
        iterations=iteration_count,
        converged=converged,
    )


def numerical_codebook_distortion(
    dimension: int,
    codebook: TurboQuantScalarCodebook | np.ndarray,
    *,
    grid_size: int = 8193,
) -> float:
    """Approximate unit-norm TurboQuant MSE distortion for the current codebook."""

    dimension = _validate_dimension(dimension)
    if isinstance(codebook, TurboQuantScalarCodebook):
        if codebook.dimension != dimension:
            raise ValueError("codebook.dimension does not match the requested dimension")
        return float(dimension) * float(codebook.expected_coordinate_mse)

    centers = np.asarray(codebook, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("codebook must be a 1D array with at least two levels")
    if not np.all(np.diff(centers) >= 0.0):
        raise ValueError("codebook must be sorted in ascending order")

    points, weights = _integration_grid(grid_size)
    masses = beta_coordinate_density(points, dimension) * weights
    boundaries = 0.5 * (centers[:-1] + centers[1:])
    indices = np.searchsorted(boundaries, points, side="right")
    quantized = centers[indices]
    coordinate_mse = float(np.sum(((points - quantized) ** 2) * masses))
    return float(dimension) * coordinate_mse
