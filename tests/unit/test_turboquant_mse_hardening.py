from __future__ import annotations

import numpy as np
import pytest

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.errors import DecodeError
from semafold.vector.models import VectorEncoding
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec


def test_turboquant_mse_handles_zero_rows() -> None:
    codec = TurboQuantMSEVectorCodec()
    data = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5, -0.5],
        ],
        dtype=np.float32,
    )
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="reconstruction", seed=3))
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    assert decoded.data.shape == data.shape
    assert np.isfinite(decoded.data).all()


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_turboquant_mse_supports_additional_float_dtypes(dtype: np.dtype[np.generic]) -> None:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=2, default_rotation_seed=9)
    )
    data = np.linspace(-1.0, 1.0, num=48, dtype=np.float64).reshape(3, 16).astype(dtype)
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="reconstruction", metric="mse", seed=4))
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    assert decoded.data.dtype == data.dtype
    assert decoded.data.shape == data.shape


def test_turboquant_mse_decode_rejects_malformed_codebook_centers() -> None:
    codec = TurboQuantMSEVectorCodec()
    data = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(4, 8)
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="reconstruction", seed=3))
    reloaded = VectorEncoding.from_dict(encoding.to_dict())
    metadata_segment = next(segment for segment in reloaded.segments if segment.segment_kind == "metadata")
    assert isinstance(metadata_segment.payload, dict)
    metadata_segment.payload["codebook_centers"] = [0.0, 0.5]
    with pytest.raises(DecodeError):
        codec.decode(VectorDecodeRequest(encoding=reloaded))


def test_turboquant_mse_decode_rejects_sidecar_size_mismatch() -> None:
    codec = TurboQuantMSEVectorCodec()
    data = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(4, 8)
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="reconstruction", seed=3))
    reloaded = VectorEncoding.from_dict(encoding.to_dict())
    sidecar_segment = next(segment for segment in reloaded.segments if segment.segment_kind == "sidecar")
    assert isinstance(sidecar_segment.payload, bytes)
    sidecar_segment.payload = sidecar_segment.payload[:-4]
    with pytest.raises(DecodeError):
        codec.decode(VectorDecodeRequest(encoding=reloaded))
