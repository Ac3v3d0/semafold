from __future__ import annotations

import numpy as np
import pytest

from semafold import VectorDecodeRequest, VectorEncodeRequest, VectorEncoding
from semafold.errors import CompatibilityError, DecodeError
from semafold.turboquant import TurboQuantProdConfig, TurboQuantProdVectorCodec


def _sample_matrix(dtype: type[np.float16] | type[np.float32] | type[np.float64]) -> np.ndarray:
    return np.array(
        [
            [0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.4, -0.3],
            [-0.3, 0.2, -0.2, 0.1, 0.3, -0.4, 0.1, 0.2],
        ],
        dtype=dtype,
    )


def _encoded_dict(
    *,
    dtype: type[np.float16] | type[np.float32] | type[np.float64] = np.float32,
) -> tuple[TurboQuantProdVectorCodec, dict[str, object]]:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    request = VectorEncodeRequest(data=_sample_matrix(dtype), objective="inner_product_estimation", metric="dot_product_error")
    return codec, codec.encode(request).to_dict()


def _segment(payload: dict[str, object], segment_kind: str) -> dict[str, object]:
    segments = payload["segments"]
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, dict)
        if segment["segment_kind"] == segment_kind:
            return segment
    raise AssertionError(f"{segment_kind} segment missing")


def _metadata_value(payload: dict[str, object]) -> dict[str, object]:
    segment = _segment(payload, "metadata")
    payload_obj = segment["payload"]
    assert isinstance(payload_obj, dict)
    value = payload_obj["value"]
    assert isinstance(value, dict)
    return value


def test_turboquant_prod_codec_emits_valid_preview_encoding() -> None:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    request = VectorEncodeRequest(
        data=_sample_matrix(np.float32),
        objective="inner_product_estimation",
        metric="dot_product_error",
    )

    encoding = codec.encode(request)
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

    assert decoded.data.shape == request.data.shape
    assert decoded.data.dtype == request.data.dtype
    assert np.isfinite(decoded.data).all()
    assert encoding.codec_family == "turboquant"
    assert encoding.variant_id == "prod_qjl_residual_v1"
    assert encoding.encoding_schema_version == "vector.encoding.v1"
    assert {segment.segment_kind for segment in encoding.segments} == {
        "compressed",
        "sidecar",
        "residual_sketch",
        "residual_gamma",
        "metadata",
    }
    assert any(guarantee.metric == "estimator_unbiasedness" for guarantee in encoding.guarantees)
    assert {item.scope for item in encoding.evidence} == {
        "proxy_fidelity",
        "theory_proxy",
        "theorem_assumptions",
        "storage_accounting",
    }
    theory_proxy = next(item for item in encoding.evidence if item.scope == "theory_proxy")
    assert float(theory_proxy.metrics["mean_query_free_variance_factor"]) >= 0.0
    assert float(theory_proxy.metrics["max_query_free_variance_factor"]) >= 0.0
    theorem_assumptions = next(item for item in encoding.evidence if item.scope == "theorem_assumptions")
    assert theorem_assumptions.passed is True
    assert theorem_assumptions.metrics["qjl_residual_correction_enabled"] is True
    assert theorem_assumptions.metrics["base_bits_equals_total_minus_one"] is True


def test_turboquant_prod_accepts_only_inner_product_objective_and_dot_product_metric() -> None:
    codec = TurboQuantProdVectorCodec()
    bad_objective = VectorEncodeRequest(data=_sample_matrix(np.float32), objective="reconstruction")
    bad_metric = VectorEncodeRequest(
        data=_sample_matrix(np.float32),
        objective="inner_product_estimation",
        metric="mse",
    )

    with pytest.raises(CompatibilityError):
        codec.encode(bad_objective)
    with pytest.raises(CompatibilityError):
        codec.encode(bad_metric)


def test_turboquant_prod_codec_compresses_deterministic_dummy_vector_batches() -> None:
    rng = np.random.default_rng(23)
    data = rng.normal(size=(32, 64)).astype(np.float32)
    request = VectorEncodeRequest(
        data=data,
        objective="inner_product_estimation",
        metric="dot_product_error",
        role="embedding",
        seed=29,
    )
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
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
            lambda payload: _metadata_value(payload).__setitem__("mode", "mse"),
            "TurboQuant preview metadata mode must be 'prod'",
        ),
        (
            lambda payload: _metadata_value(payload).__setitem__("qjl_family", "wrong.family"),
            "unsupported qjl_family",
        ),
        (
            lambda payload: _metadata_value(payload).__setitem__("codebook_centers", [0.0, 0.5]),
            "codebook_centers size does not match base_bits_per_scalar",
        ),
        (
            lambda payload: _metadata_value(payload).__setitem__("base_bits_per_scalar", 4),
            "base_bits_per_scalar must equal total_bits_per_scalar - 1",
        ),
    ],
)
def test_turboquant_prod_decode_rejects_malformed_metadata(mutate, expected_message: str) -> None:
    codec, payload = _encoded_dict()
    mutate(payload)
    encoding = VectorEncoding.from_dict(payload)
    with pytest.raises(DecodeError, match=expected_message):
        codec.decode(VectorDecodeRequest(encoding=encoding))


def test_turboquant_prod_decode_rejects_segment_payload_size_mismatch() -> None:
    codec, payload = _encoded_dict()
    residual_gamma = _segment(payload, "residual_gamma")
    residual_payload = residual_gamma["payload"]
    assert isinstance(residual_payload, dict)
    base64_data = residual_payload["base64"]
    assert isinstance(base64_data, str)
    residual_payload["base64"] = base64_data[:-4]
    encoding = VectorEncoding.from_dict(payload)

    with pytest.raises(DecodeError, match="residual_gamma payload size does not match metadata"):
        codec.decode(VectorDecodeRequest(encoding=encoding))


def test_turboquant_prod_decode_is_artifact_driven() -> None:
    producer = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    consumer = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=5, default_rotation_seed=17, default_qjl_seed=23)
    )
    encoding = producer.encode(
        VectorEncodeRequest(data=_sample_matrix(np.float32), objective="inner_product_estimation", metric="dot_product_error")
    )
    decoded = consumer.decode(VectorDecodeRequest(encoding=encoding))

    assert decoded.data.shape == (2, 8)
    assert decoded.data.dtype == np.float32
    assert np.isfinite(decoded.data).all()
