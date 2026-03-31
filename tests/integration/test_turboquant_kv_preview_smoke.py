from __future__ import annotations

import json

import numpy as np
import pytest

from semafold import VectorDecodeRequest, VectorEncodeRequest
from semafold.core.accounting import build_footprint
from semafold.turboquant import (
    TurboQuantMSEConfig,
    TurboQuantMSEVectorCodec,
    TurboQuantProdConfig,
    TurboQuantProdVectorCodec,
)
from semafold.turboquant.kv.layout import (
    CANONICAL_CACHE_LAYOUT,
    cache_layout_metadata,
    flatten_cache_rows,
    restore_cache_rows,
    validate_cache_pair,
)
from semafold.turboquant.kv.preview import TurboQuantKVCacheArtifact, build_kv_cache_artifact


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _sample_cache(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = _normalize_last_axis(rng.standard_normal((2, 3, 4, 16), dtype=np.float32))
    values = rng.standard_normal((2, 3, 4, 16), dtype=np.float32)
    return keys, values


def test_kv_layout_roundtrip_preserves_shape_and_order() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    assert key_rows.shape == (layers * heads * seq_len, head_dim)
    assert value_rows.shape == (layers * heads * seq_len, head_dim)
    assert np.array_equal(restore_cache_rows(key_rows, layers=layers, heads=heads, seq_len=seq_len, head_dim=head_dim), keys)
    assert np.array_equal(
        restore_cache_rows(value_rows, layers=layers, heads=heads, seq_len=seq_len, head_dim=head_dim),
        values,
    )


def test_kv_artifact_roundtrip_wraps_separate_key_and_value_encodings() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    key_codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )

    key_encoding = key_codec.encode(
        VectorEncodeRequest(
            data=key_rows,
            objective="inner_product_estimation",
            metric="dot_product_error",
            role="key_cache",
        )
    )
    value_encoding = value_codec.encode(
        VectorEncodeRequest(
            data=value_rows,
            objective="reconstruction",
            metric="mse",
            role="value_cache",
        )
    )

    artifact = build_kv_cache_artifact(
        layers=layers,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        key_encoding=key_encoding,
        value_encoding=value_encoding,
        metadata={"source": "integration_smoke"},
    )
    assert artifact.format == "kv_preview_v1"
    assert artifact.layout == CANONICAL_CACHE_LAYOUT
    assert artifact.footprint.total_bytes == key_encoding.footprint.total_bytes + value_encoding.footprint.total_bytes

    wire = artifact.to_dict()
    restored = TurboQuantKVCacheArtifact.from_dict(json.loads(json.dumps(wire)))

    assert restored.format == artifact.format
    assert restored.layout == artifact.layout
    assert restored.layers == artifact.layers
    assert restored.metadata["source"] == "integration_smoke"
    assert restored.key_encoding.variant_id == key_encoding.variant_id
    assert restored.value_encoding.variant_id == value_encoding.variant_id
    assert restored.footprint.total_bytes == artifact.footprint.total_bytes

    restored_key_rows = key_codec.decode(VectorDecodeRequest(encoding=restored.key_encoding)).data
    restored_value_rows = value_codec.decode(VectorDecodeRequest(encoding=restored.value_encoding)).data

    restored_keys = restore_cache_rows(
        restored_key_rows,
        layers=restored.layers,
        heads=restored.heads,
        seq_len=restored.seq_len,
        head_dim=restored.head_dim,
    )
    restored_values = restore_cache_rows(
        restored_value_rows,
        layers=restored.layers,
        heads=restored.heads,
        seq_len=restored.seq_len,
        head_dim=restored.head_dim,
    )

    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    assert float(np.mean(np.square(restored_values.astype(np.float64) - values.astype(np.float64)))) < 1.0


def test_kv_layout_rejects_shape_mismatch() -> None:
    keys, values = _sample_cache()
    with pytest.raises(ValueError, match="identical cache shape"):
        validate_cache_pair(keys, values[:, :, :-1, :])


def test_kv_artifact_rejects_role_mismatch() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    key_codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )

    bad_key_encoding = key_codec.encode(
        VectorEncodeRequest(
            data=key_rows,
            objective="inner_product_estimation",
            metric="dot_product_error",
            role="value_cache",
        )
    )
    value_encoding = value_codec.encode(
        VectorEncodeRequest(
            data=value_rows,
            objective="reconstruction",
            metric="mse",
            role="value_cache",
        )
    )

    with pytest.raises(ValueError, match="role='key_cache'"):
        build_kv_cache_artifact(
            layers=layers,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            key_encoding=bad_key_encoding,
            value_encoding=value_encoding,
        )


def test_kv_artifact_rejects_layout_shape_mismatch() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    key_codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )

    key_encoding = key_codec.encode(
        VectorEncodeRequest(
            data=key_rows,
            objective="inner_product_estimation",
            metric="dot_product_error",
            role="key_cache",
        )
    )
    value_encoding = value_codec.encode(
        VectorEncodeRequest(
            data=value_rows,
            objective="reconstruction",
            metric="mse",
            role="value_cache",
        )
    )

    with pytest.raises(ValueError, match="shape does not match KV layout"):
        build_kv_cache_artifact(
            layers=layers,
            heads=heads,
            seq_len=seq_len + 1,
            head_dim=head_dim,
            key_encoding=key_encoding,
            value_encoding=value_encoding,
        )


def test_kv_artifact_rejects_wrong_child_variant() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    wrong_key_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=7)
    )
    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )

    wrong_key_encoding = wrong_key_codec.encode(
        VectorEncodeRequest(
            data=key_rows,
            objective="reconstruction",
            metric="mse",
            role="key_cache",
        )
    )
    value_encoding = value_codec.encode(
        VectorEncodeRequest(
            data=value_rows,
            objective="reconstruction",
            metric="mse",
            role="value_cache",
        )
    )

    with pytest.raises(ValueError, match="key_encoding must use variant_id"):
        build_kv_cache_artifact(
            layers=layers,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            key_encoding=wrong_key_encoding,
            value_encoding=value_encoding,
        )


def test_kv_artifact_rejects_outer_footprint_mismatch() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    key_rows = flatten_cache_rows(keys)
    value_rows = flatten_cache_rows(values)

    key_codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )

    key_encoding = key_codec.encode(
        VectorEncodeRequest(
            data=key_rows,
            objective="inner_product_estimation",
            metric="dot_product_error",
            role="key_cache",
        )
    )
    value_encoding = value_codec.encode(
        VectorEncodeRequest(
            data=value_rows,
            objective="reconstruction",
            metric="mse",
            role="value_cache",
        )
    )

    with pytest.raises(ValueError, match="combined key/value encoding footprint"):
        TurboQuantKVCacheArtifact(
            format="kv_preview_v1",
            layout=CANONICAL_CACHE_LAYOUT,
            layers=layers,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            key_encoding=key_encoding,
            value_encoding=value_encoding,
            footprint=build_footprint(
                baseline_bytes=key_encoding.footprint.baseline_bytes + value_encoding.footprint.baseline_bytes,
                payload_bytes=key_encoding.footprint.payload_bytes + value_encoding.footprint.payload_bytes + 1,
                metadata_bytes=key_encoding.footprint.metadata_bytes + value_encoding.footprint.metadata_bytes,
                sidecar_bytes=key_encoding.footprint.sidecar_bytes + value_encoding.footprint.sidecar_bytes,
                protected_passthrough_bytes=(
                    key_encoding.footprint.protected_passthrough_bytes
                    + value_encoding.footprint.protected_passthrough_bytes
                ),
                decoder_state_bytes=(
                    key_encoding.footprint.decoder_state_bytes + value_encoding.footprint.decoder_state_bytes
                ),
            ),
        )


def test_cache_layout_metadata_matches_flattened_row_count() -> None:
    keys, values = _sample_cache()
    layers, heads, seq_len, head_dim = validate_cache_pair(keys, values)
    metadata = cache_layout_metadata(layers=layers, heads=heads, seq_len=seq_len, head_dim=head_dim)
    assert metadata == {
        "layout": CANONICAL_CACHE_LAYOUT,
        "layers": layers,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "row_count": layers * heads * seq_len,
    }
