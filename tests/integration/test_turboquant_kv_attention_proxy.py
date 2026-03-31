from __future__ import annotations

import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _softmax(array: np.ndarray, *, axis: int = -1) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _attention_output(queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(keys.shape[-1], dtype=np.float32))
    scores = np.einsum("bhqd,bhkd->bhqk", queries.astype(np.float64), keys.astype(np.float64)) / scale
    weights = _softmax(scores, axis=-1)
    return np.einsum("bhqk,bhkd->bhqd", weights, values.astype(np.float64))


def test_turboquant_kv_attention_proxy_preserves_behavior_on_synthetic_cache_blocks() -> None:
    rng = np.random.default_rng(123)
    queries = _normalize_last_axis(rng.standard_normal((2, 2, 5, 16), dtype=np.float32))
    keys = _normalize_last_axis(rng.standard_normal((2, 2, 7, 16), dtype=np.float32))
    values = rng.standard_normal((2, 2, 7, 16), dtype=np.float32)
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

    exact_output = _attention_output(queries, keys, values)
    approx_output = _attention_output(queries, restored_keys, restored_values)
    mse = float(np.mean(np.square(exact_output - approx_output)))
    cosine_similarity = float(
        np.sum(exact_output * approx_output)
        / ((np.linalg.norm(exact_output) + 1e-12) * (np.linalg.norm(approx_output) + 1e-12))
    )

    assert approx_output.shape == exact_output.shape
    assert np.isfinite(approx_output).all()
    assert artifact.key_encoding.variant_id == "prod_qjl_residual_v1"
    assert artifact.value_encoding.variant_id == "mse_beta_lloyd_qr_v2"
    assert mse < 0.02
    assert cosine_similarity > 0.95
