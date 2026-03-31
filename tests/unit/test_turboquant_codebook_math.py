from __future__ import annotations

import numpy as np

from semafold.turboquant.codebook import (
    TurboQuantScalarCodebook,
    numerical_codebook_distortion,
    solve_beta_lloyd_max_codebook,
)


def test_turboquant_codebook_is_sorted_symmetric_and_finite() -> None:
    codebook = solve_beta_lloyd_max_codebook(64, 3)
    assert isinstance(codebook, TurboQuantScalarCodebook)
    assert codebook.centers.shape == (8,)
    assert np.isfinite(codebook.centers).all()
    assert np.all(np.diff(codebook.centers) >= 0.0)
    assert np.allclose(codebook.centers, -codebook.centers[::-1], atol=1e-4)
    assert codebook.boundaries.shape == (9,)
    assert codebook.cell_masses.shape == (8,)
    assert float(np.sum(codebook.cell_masses)) > 0.99


def test_turboquant_codebook_distortion_improves_with_more_bits() -> None:
    low = solve_beta_lloyd_max_codebook(64, 1)
    high = solve_beta_lloyd_max_codebook(64, 4)
    low_distortion = numerical_codebook_distortion(64, low)
    high_distortion = numerical_codebook_distortion(64, high)
    assert high_distortion < low_distortion


def test_turboquant_codebook_quantize_and_dequantize_are_shape_preserving() -> None:
    codebook = solve_beta_lloyd_max_codebook(32, 2)
    values = np.linspace(-0.95, 0.95, num=15, dtype=np.float32).reshape(3, 5)
    indices = codebook.quantize(values)
    restored = codebook.dequantize(indices)
    assert indices.shape == values.shape
    assert restored.shape == values.shape
    assert np.isfinite(restored).all()
