from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest


def test_vector_decode_returns_numpy_array() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(data=np.array([1.0, 2.0, 3.0], dtype=np.float32), objective="reconstruction")
    result = codec.decode(VectorDecodeRequest(encoding=codec.encode(request)))
    assert isinstance(result.data, np.ndarray)
    assert result.data.shape == request.data.shape


def test_target_layout_is_advisory_only_in_phase_1() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        objective="reconstruction",
    )
    result = codec.decode(
        VectorDecodeRequest(
            encoding=codec.encode(request),
            target_layout="column_major",
        )
    )
    assert np.array_equal(result.data, request.data)
    assert any("ignored in Phase 1" in note for note in result.materialization_notes)
