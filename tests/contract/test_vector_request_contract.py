from __future__ import annotations

import numpy as np
import pytest

from semafold import CompressionBudget
from semafold import VectorEncodeRequest


def test_vector_request_accepts_numpy_only() -> None:
    request = VectorEncodeRequest(
        data=np.array([1.0, 2.0], dtype=np.float32),
        objective="reconstruction",
        budget=CompressionBudget(target_bits_per_scalar=8.0),
    )
    assert request.data.shape == (2,)


def test_vector_request_rejects_non_numpy() -> None:
    with pytest.raises(TypeError):
        VectorEncodeRequest(data=[1.0, 2.0], objective="reconstruction")  # type: ignore[arg-type]


def test_vector_request_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError):
        VectorEncodeRequest(
            data=np.array([1.0, 2.0], dtype=np.float32),
            objective="reconstruction",
            seed=True,  # type: ignore[arg-type]
        )
