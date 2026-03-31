from __future__ import annotations

import math

import numpy as np
import pytest

from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _sample_cache(*, seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = _normalize_last_axis(rng.standard_normal((2, 2, 6, 16), dtype=np.float32))
    values = rng.standard_normal((2, 2, 6, 16), dtype=np.float32)
    return keys, values


def test_turboquant_kv_preview_memory_stats_report_combined_bytes_and_fp16_bf16_ratios() -> None:
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
    stats = codec.memory_stats(artifact)

    expected_baseline_bytes = int(keys.nbytes + values.nbytes)
    expected_fp16_bytes = int((keys.size + values.size) * np.dtype(np.float16).itemsize)
    expected_bf16_bytes = int((keys.size + values.size) * 2)
    expected_combined_bytes = int(artifact.footprint.total_bytes)

    assert stats["baseline_bytes"] == expected_baseline_bytes
    assert stats["baseline_fp16_bytes"] == expected_fp16_bytes
    assert stats["baseline_bf16_bytes"] == expected_bf16_bytes
    assert stats["key_bytes"] == artifact.key_encoding.footprint.total_bytes
    assert stats["value_bytes"] == artifact.value_encoding.footprint.total_bytes
    assert stats["combined_bytes"] == expected_combined_bytes
    assert stats["combined_compression_ratio"] == pytest.approx(expected_baseline_bytes / expected_combined_bytes)
    assert stats["combined_compression_ratio_vs_fp16"] == pytest.approx(expected_fp16_bytes / expected_combined_bytes)
    assert stats["combined_compression_ratio_vs_bf16"] == pytest.approx(expected_bf16_bytes / expected_combined_bytes)

    for key in (
        "baseline_bytes",
        "baseline_fp16_bytes",
        "baseline_bf16_bytes",
        "key_bytes",
        "value_bytes",
        "combined_bytes",
        "combined_compression_ratio",
        "combined_compression_ratio_vs_fp16",
        "combined_compression_ratio_vs_bf16",
    ):
        value = stats[key]
        assert isinstance(value, (int, float))
        assert math.isfinite(float(value))
        assert float(value) >= 0.0


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 2, 16, 8), np.float16),
        ((2, 2, 8, 16), np.float32),
        ((2, 4, 12, 16), np.float32),
    ],
)
def test_turboquant_kv_memory_stats_multi_shape_smoke(
    shape: tuple[int, int, int, int],
    dtype: type[np.float16] | type[np.float32],
) -> None:
    layers, heads, seq_len, head_dim = shape
    rng = np.random.default_rng(sum(shape))
    keys = _normalize_last_axis(rng.standard_normal(shape, dtype=np.float32)).astype(dtype)
    values = rng.standard_normal(shape, dtype=np.float32).astype(dtype)

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
    stats = codec.memory_stats(artifact)

    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    assert restored_keys.dtype == keys.dtype
    assert restored_values.dtype == values.dtype
    assert np.isfinite(restored_keys).all()
    assert np.isfinite(restored_values).all()
    assert stats["combined_bytes"] == artifact.footprint.total_bytes
    assert stats["key_bytes"] + stats["value_bytes"] == stats["combined_bytes"]
    assert stats["baseline_bytes"] == int(keys.nbytes + values.nbytes)
    assert stats["baseline_fp16_bytes"] == stats["baseline_bf16_bytes"]

    if dtype is np.float32:
        assert stats["combined_compression_ratio_vs_fp16"] == pytest.approx(
            float(stats["combined_compression_ratio"]) / 2.0
        )
        assert stats["combined_compression_ratio_vs_bf16"] == pytest.approx(
            float(stats["combined_compression_ratio"]) / 2.0
        )
    else:
        assert stats["combined_compression_ratio_vs_fp16"] == pytest.approx(
            float(stats["combined_compression_ratio"])
        )
        assert stats["combined_compression_ratio_vs_bf16"] == pytest.approx(
            float(stats["combined_compression_ratio"])
        )
