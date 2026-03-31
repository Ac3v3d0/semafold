from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import pytest

import semafold
from semafold.errors import CompatibilityError
from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec
from semafold.turboquant.kv.layout import CANONICAL_CACHE_LAYOUT
from semafold.turboquant.kv.preview import TurboQuantKVCacheArtifact


EXPECTED_ROOT_ALL = {
    "__version__",
    "CompressionBudget",
    "CompressionEstimate",
    "CompressionFootprint",
    "CompressionGuarantee",
    "PassthroughVectorCodec",
    "ValidationEvidence",
    "VectorCodec",
    "VectorDecodeRequest",
    "VectorDecodeResult",
    "VectorEncodeRequest",
    "VectorEncoding",
    "VectorEncodingSegment",
}

EXPECTED_TURBOQUANT_ALL = [
    "TurboQuantMSEConfig",
    "TurboQuantMSEVectorCodec",
    "TurboQuantProdConfig",
    "TurboQuantProdVectorCodec",
]


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _sample_cache(*, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = _normalize_last_axis(rng.standard_normal((2, 3, 4, 16), dtype=np.float32))
    values = rng.standard_normal((2, 3, 4, 16), dtype=np.float32)
    return keys, values


def _shape_mismatch_inputs() -> tuple[np.ndarray, np.ndarray]:
    keys, values = _sample_cache()
    return keys, values[:, :, :-1, :]


def _int_key_inputs() -> tuple[np.ndarray, np.ndarray]:
    keys, values = _sample_cache()
    return keys.astype(np.int32), values


def _rank_mismatch_inputs() -> tuple[np.ndarray, np.ndarray]:
    keys, values = _sample_cache()
    return keys.reshape(2, 12, 16), values


def _non_finite_inputs() -> tuple[np.ndarray, np.ndarray]:
    keys, values = _sample_cache()
    keys = keys.copy()
    keys[0, 0, 0, 0] = np.nan
    return keys, values


def test_turboquant_kv_preview_codec_emits_preview_artifact_and_keeps_root_api_stable() -> None:
    turboquant_module = importlib.import_module("semafold.turboquant")
    keys, values = _sample_cache()
    codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=2,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    artifact = codec.compress(keys, values)

    assert isinstance(artifact, TurboQuantKVCacheArtifact)
    assert codec.variant_id == "kv_preview_v1"
    assert artifact.format == "kv_preview_v1"
    assert artifact.layout == CANONICAL_CACHE_LAYOUT
    assert (artifact.layers, artifact.heads, artifact.seq_len, artifact.head_dim) == keys.shape
    assert artifact.metadata["mode"] == "kv_cache_block_compression"
    assert artifact.metadata["key_total_bits_per_scalar"] == 3
    assert artifact.metadata["value_bits_per_scalar"] == 2
    assert artifact.metadata["key_variant_id"] == "prod_qjl_residual_v1"
    assert artifact.metadata["value_variant_id"] == "mse_beta_lloyd_qr_v2"
    assert artifact.key_encoding.codec_family == "turboquant"
    assert artifact.value_encoding.codec_family == "turboquant"
    assert artifact.key_encoding.variant_id == "prod_qjl_residual_v1"
    assert artifact.value_encoding.variant_id == "mse_beta_lloyd_qr_v2"
    assert artifact.key_encoding.metadata["objective"] == "inner_product_estimation"
    assert artifact.value_encoding.metadata["objective"] == "reconstruction"
    assert {segment.role for segment in artifact.key_encoding.segments} == {"key_cache"}
    assert {segment.role for segment in artifact.value_encoding.segments} == {"value_cache"}
    assert artifact.footprint.total_bytes == (
        artifact.key_encoding.footprint.total_bytes + artifact.value_encoding.footprint.total_bytes
    )
    assert set(semafold.__all__) == EXPECTED_ROOT_ALL
    assert turboquant_module.__all__ == EXPECTED_TURBOQUANT_ALL
    assert not hasattr(semafold, "TurboQuantKVConfig")
    assert not hasattr(semafold, "TurboQuantKVPreviewCodec")
    assert not hasattr(turboquant_module, "TurboQuantKVConfig")
    assert not hasattr(turboquant_module, "TurboQuantKVPreviewCodec")


def test_turboquant_kv_config_reuses_current_preview_bit_constraints() -> None:
    default = TurboQuantKVConfig()

    assert default.key_total_bits_per_scalar == 3
    assert default.value_bits_per_scalar == 3
    assert default.default_key_rotation_seed == 0
    assert default.default_key_qjl_seed == 0
    assert default.default_value_rotation_seed == 0
    assert default.normalization == "row_l2"

    with pytest.raises(ValueError, match="between 2 and 5"):
        TurboQuantKVConfig(key_total_bits_per_scalar=1)
    with pytest.raises(ValueError, match="between 1 and 4"):
        TurboQuantKVConfig(value_bits_per_scalar=0)
    with pytest.raises(ValueError, match="row_l2"):
        TurboQuantKVConfig(normalization=cast(Any, "none"))


@pytest.mark.parametrize(
    ("factory", "expected_exception", "expected_message"),
    [
        (_shape_mismatch_inputs, ValueError, "identical cache shape"),
        (_int_key_inputs, TypeError, "floating-point dtype"),
        (_rank_mismatch_inputs, ValueError, "shape \\(layers, heads, seq_len, head_dim\\)"),
        (_non_finite_inputs, CompatibilityError, "finite floating-point values"),
    ],
)
def test_turboquant_kv_preview_codec_rejects_invalid_cache_inputs(
    factory,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    codec = TurboQuantKVPreviewCodec(config=TurboQuantKVConfig())
    bad_keys, bad_values = factory()

    with pytest.raises(expected_exception, match=expected_message):
        codec.compress(bad_keys, bad_values)


def test_turboquant_kv_preview_codec_rejects_non_artifact_inputs_for_decompress_and_memory_stats() -> None:
    codec = TurboQuantKVPreviewCodec(config=TurboQuantKVConfig())

    with pytest.raises(TypeError, match="TurboQuantKVCacheArtifact"):
        codec.decompress(cast(Any, "not-an-artifact"))
    with pytest.raises(TypeError, match="TurboQuantKVCacheArtifact"):
        codec.memory_stats(cast(Any, "not-an-artifact"))
