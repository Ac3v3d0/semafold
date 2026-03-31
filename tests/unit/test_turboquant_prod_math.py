from __future__ import annotations

import math

import numpy as np

from semafold.turboquant.qjl import qjl_decode_rows, qjl_encode_rows, seeded_gaussian_projection


def _normalize_nonzero_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(rows.astype(np.float64), axis=1).astype(np.float32)
    normalized = np.zeros_like(rows, dtype=np.float32)
    nonzero = norms > 0.0
    if np.any(nonzero):
        normalized[nonzero] = rows[nonzero] / norms[nonzero, None]
    return normalized, norms


def test_turboquant_prod_residual_decode_matches_manual_formula() -> None:
    projection = seeded_gaussian_projection(8, 13)
    residual_rows = np.array(
        [
            [0.2, -0.1, 0.0, 0.3, -0.4, 0.1, 0.2, -0.2],
            [-0.2, 0.2, -0.1, 0.0, 0.1, -0.3, 0.4, 0.2],
        ],
        dtype=np.float32,
    )
    residual_hat, gamma = _normalize_nonzero_rows(residual_rows)
    sign_rows = qjl_encode_rows(residual_hat, projection)
    decoded = qjl_decode_rows(sign_rows, gamma, projection)

    sign_pm1 = np.where(sign_rows > 0, 1.0, -1.0).astype(np.float32)
    coefficient = np.float32(math.sqrt(math.pi / 2.0) / float(projection.shape[0]))
    expected = (sign_pm1 @ projection) * coefficient * gamma.reshape(-1, 1)

    assert np.allclose(decoded, expected, atol=1e-6)


def test_turboquant_prod_mixed_zero_and_nonzero_residual_rows_remain_finite() -> None:
    projection = seeded_gaussian_projection(8, 17)
    residual_rows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.1, 0.0, 0.2, -0.4, 0.1, 0.1, -0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    residual_hat, gamma = _normalize_nonzero_rows(residual_rows)
    sign_rows = qjl_encode_rows(residual_hat, projection)
    decoded = qjl_decode_rows(sign_rows, gamma, projection)

    assert np.isfinite(decoded).all()
    assert np.array_equal(decoded[0], np.zeros((8,), dtype=np.float32))
    assert np.array_equal(decoded[2], np.zeros((8,), dtype=np.float32))


def test_turboquant_prod_qjl_seed_determinism() -> None:
    residual_rows = np.array(
        [
            [0.25, -0.5, 0.0, 0.75, -0.25, 0.1, -0.2, 0.3],
            [-0.4, 0.2, 0.1, 0.0, 0.25, -0.1, 0.2, -0.3],
        ],
        dtype=np.float32,
    )
    residual_hat, gamma = _normalize_nonzero_rows(residual_rows)

    proj_a = seeded_gaussian_projection(8, 31)
    proj_b = seeded_gaussian_projection(8, 31)
    proj_c = seeded_gaussian_projection(8, 32)

    signs_a = qjl_encode_rows(residual_hat, proj_a)
    signs_b = qjl_encode_rows(residual_hat, proj_b)
    signs_c = qjl_encode_rows(residual_hat, proj_c)

    decoded_a = qjl_decode_rows(signs_a, gamma, proj_a)
    decoded_b = qjl_decode_rows(signs_b, gamma, proj_b)
    decoded_c = qjl_decode_rows(signs_c, gamma, proj_c)

    assert np.array_equal(signs_a, signs_b)
    assert np.allclose(decoded_a, decoded_b, atol=1e-6)
    assert not np.array_equal(signs_a, signs_c)
    assert not np.allclose(decoded_a, decoded_c, atol=1e-6)
