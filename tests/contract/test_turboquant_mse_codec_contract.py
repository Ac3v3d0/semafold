from __future__ import annotations

import base64

import numpy as np
import pytest

from semafold import VectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold import VectorEncoding
from semafold.errors import CompatibilityError
from semafold.errors import DecodeError
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec


def _sample_matrix(dtype: type[np.float16] | type[np.float32] | type[np.float64]) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, -0.4],
            [0.4, 0.3, -0.2, -0.1],
        ],
        dtype=dtype,
    )


def _encoded_dict(
    *,
    dtype: type[np.float16] | type[np.float32] | type[np.float64] = np.float32,
) -> tuple[TurboQuantMSEVectorCodec, dict[str, object]]:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=2, default_rotation_seed=7)
    )
    request = VectorEncodeRequest(
        data=_sample_matrix(dtype),
        objective="reconstruction",
        metric="mse",
    )
    return codec, codec.encode(request).to_dict()


def _metadata_segment(payload: dict[str, object]) -> dict[str, object]:
    segments = payload["segments"]
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, dict)
        if segment["segment_kind"] == "metadata":
            return segment
    raise AssertionError("metadata segment missing from encoded payload")


def _metadata_payload_value(payload: dict[str, object]) -> dict[str, object]:
    segment = _metadata_segment(payload)
    payload_obj = segment["payload"]
    assert isinstance(payload_obj, dict)
    value = payload_obj["value"]
    assert isinstance(value, dict)
    return value


def _codebook_centers(payload: dict[str, object]) -> list[float]:
    value = _metadata_payload_value(payload)
    centers = value["codebook_centers"]
    assert isinstance(centers, list)
    result: list[float] = []
    for center in centers:
        assert isinstance(center, (int, float))
        result.append(float(center))
    return result


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_turboquant_mse_codec_emits_valid_preview_encoding_across_supported_float_dtypes(
    dtype: type[np.float16] | type[np.float32] | type[np.float64],
) -> None:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=2, default_rotation_seed=7)
    )
    request = VectorEncodeRequest(data=_sample_matrix(dtype), objective="reconstruction", metric="mse")

    assert isinstance(codec, VectorCodec)

    encoding = codec.encode(request)
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

    assert decoded.data.shape == request.data.shape
    assert decoded.data.dtype == request.data.dtype
    assert np.isfinite(decoded.data).all()
    assert np.array_equal(decoded.data[0], np.zeros((4,), dtype=request.data.dtype))
    assert encoding.codec_family == "turboquant"
    assert encoding.variant_id == "mse_beta_lloyd_qr_v2"
    assert {segment.segment_kind for segment in encoding.segments} == {"compressed", "sidecar", "metadata"}
    assert encoding.encoding_schema_version == "vector.encoding.v1"
    assert isinstance(encoding.config_fingerprint, str) and encoding.config_fingerprint
    assert encoding.footprint.baseline_bytes == int(request.data.nbytes)
    assert encoding.footprint.sidecar_bytes > 0
    assert any(guarantee.metric == "observed_mse" for guarantee in encoding.guarantees)


def test_turboquant_mse_rejects_unsupported_objective() -> None:
    codec = TurboQuantMSEVectorCodec()
    request = VectorEncodeRequest(
        data=_sample_matrix(np.float32),
        objective="inner_product_estimation",
        role="key_cache",
        seed=7,
    )

    with pytest.raises(CompatibilityError):
        codec.encode(request)


def test_turboquant_mse_codec_compresses_deterministic_dummy_vector_batches() -> None:
    rng = np.random.default_rng(17)
    data = rng.normal(size=(32, 64)).astype(np.float32)
    request = VectorEncodeRequest(
        data=data,
        objective="reconstruction",
        metric="mse",
        role="embedding",
        seed=19,
    )
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=7)
    )

    estimate = codec.estimate(request)
    encoding = codec.encode(request)
    assert estimate.estimated_total_bytes is not None
    assert estimate.estimated_compression_ratio is not None

    assert encoding.footprint.baseline_bytes == int(data.nbytes)
    assert encoding.footprint.total_bytes == estimate.estimated_total_bytes
    assert encoding.footprint.compression_ratio == pytest.approx(estimate.estimated_compression_ratio)
    assert encoding.footprint.compression_ratio > 1.0
    assert encoding.footprint.bytes_saved > 0


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (
            lambda payload: _metadata_segment(payload).__setitem__(
                "payload",
                {"kind": "bytes", "base64": base64.b64encode(b"not-json").decode("ascii")},
            ),
            "metadata payload must be a dict",
        ),
        (
            lambda payload: _metadata_payload_value(payload).__setitem__("format", "wrong.format"),
            "metadata format does not match codec variant",
        ),
        (
            lambda payload: _metadata_payload_value(payload).__setitem__(
                "codebook_centers",
                _codebook_centers(payload)[:-1],
            ),
            "codebook_centers size does not match bits_per_scalar",
        ),
        (
            lambda payload: _metadata_payload_value(payload).__setitem__("vector_count", 0),
            "vector_count and dimension must be positive",
        ),
    ],
)
def test_turboquant_mse_decode_rejects_malformed_metadata(
    mutate,
    expected_message: str,
) -> None:
    codec, payload = _encoded_dict()
    mutate(payload)
    encoding = VectorEncoding.from_dict(payload)

    with pytest.raises(DecodeError, match=expected_message):
        codec.decode(VectorDecodeRequest(encoding=encoding))
