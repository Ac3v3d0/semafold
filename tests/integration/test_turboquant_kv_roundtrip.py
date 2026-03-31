from __future__ import annotations

import json

import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec
from semafold.turboquant.kv.preview import TurboQuantKVCacheArtifact


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _sample_cache(*, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = _normalize_last_axis(rng.standard_normal((2, 2, 8, 16), dtype=np.float32))
    values = rng.standard_normal((2, 2, 8, 16), dtype=np.float32)
    return keys, values


def test_turboquant_kv_preview_compress_decompress_roundtrip_preserves_shape_dtype_and_finite_values() -> None:
    keys, values = _sample_cache()
    codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=3,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    artifact = codec.compress(keys, values)
    restored_keys, restored_values = codec.decompress(artifact)

    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    assert restored_keys.dtype == keys.dtype
    assert restored_values.dtype == values.dtype
    assert np.isfinite(restored_keys).all()
    assert np.isfinite(restored_values).all()
    assert artifact.key_encoding.variant_id == "prod_qjl_residual_v1"
    assert artifact.value_encoding.variant_id == "mse_beta_lloyd_qr_v2"
    assert float(np.mean(np.square(restored_values.astype(np.float64) - values.astype(np.float64)))) < 1.0


def test_turboquant_kv_preview_fresh_instance_decompresses_serialized_artifact() -> None:
    keys, values = _sample_cache(seed=23)

    producer = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=4,
            value_bits_per_scalar=2,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )
    consumer = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=2,
            value_bits_per_scalar=4,
            default_key_rotation_seed=19,
            default_key_qjl_seed=23,
            default_value_rotation_seed=29,
        )
    )

    artifact = producer.compress(keys, values)
    reloaded = TurboQuantKVCacheArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))

    expected_keys, expected_values = producer.decompress(artifact)
    actual_keys, actual_values = consumer.decompress(reloaded)

    assert np.array_equal(actual_keys, expected_keys)
    assert np.array_equal(actual_values, expected_values)
    assert actual_keys.shape == keys.shape
    assert actual_values.shape == values.shape
    assert actual_keys.dtype == keys.dtype
    assert actual_values.dtype == values.dtype
