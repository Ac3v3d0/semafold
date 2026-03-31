from __future__ import annotations

import numpy as np
import pytest

from semafold.turboquant.packing import pack_scalar_indices, unpack_scalar_indices
from semafold.turboquant.qjl import qjl_decode_rows, qjl_encode_rows, seeded_gaussian_projection


def test_seeded_gaussian_projection_is_deterministic_and_read_only() -> None:
    first = seeded_gaussian_projection(16, 7)
    second = seeded_gaussian_projection(16, 7)
    third = seeded_gaussian_projection(16, 8)

    assert first.shape == (16, 16)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert first.flags.writeable is False


def test_qjl_sign_rows_roundtrip_through_bit_packing() -> None:
    projection = seeded_gaussian_projection(8, 11)
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sign_rows = qjl_encode_rows(rows, projection)
    packed = pack_scalar_indices(sign_rows.reshape(-1), 1)
    unpacked = unpack_scalar_indices(packed, count=sign_rows.size, bits_per_index=1).reshape(sign_rows.shape)
    assert np.array_equal(unpacked, sign_rows)


def test_qjl_zero_rows_decode_to_zero_contribution() -> None:
    projection = seeded_gaussian_projection(8, 19)
    rows = np.zeros((3, 8), dtype=np.float32)
    sign_rows = qjl_encode_rows(rows, projection)
    gamma = np.zeros((3,), dtype=np.float32)
    decoded = qjl_decode_rows(sign_rows, gamma, projection)
    assert np.array_equal(sign_rows, np.zeros_like(sign_rows))
    assert np.array_equal(decoded, np.zeros_like(decoded))


def test_qjl_decode_respects_gamma_not_original_row_norm() -> None:
    projection = seeded_gaussian_projection(8, 23)
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sign_rows = qjl_encode_rows(rows, projection)
    small = qjl_decode_rows(sign_rows, np.array([0.25, 0.5], dtype=np.float32), projection)
    large = qjl_decode_rows(sign_rows, np.array([0.5, 1.0], dtype=np.float32), projection)
    assert np.isfinite(small).all()
    assert np.isfinite(large).all()
    assert np.allclose(large, 2.0 * small, atol=1e-6)


def test_qjl_rejects_non_binary_sign_rows() -> None:
    projection = seeded_gaussian_projection(4, 5)
    sign_rows = np.array([[0, 1, 2, 0]], dtype=np.uint8)
    gamma = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="sign_rows must contain only 0/1 values"):
        qjl_decode_rows(sign_rows, gamma, projection)
