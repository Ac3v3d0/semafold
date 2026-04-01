"""Cache-shaped K/V compression helpers built on top of TurboQuant codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Mapping

import numpy as np

from semafold.core.accounting import CompressionFootprint, build_footprint
from semafold.errors import CompatibilityError, DecodeError
from semafold.turboquant.codec_mse import TurboQuantMSEConfig, TurboQuantMSEVectorCodec
from semafold.turboquant.codec_prod import TurboQuantProdConfig, TurboQuantProdVectorCodec
from semafold.turboquant.kv.layout import (
    CANONICAL_CACHE_LAYOUT,
    flatten_cache_rows,
    restore_cache_rows,
    validate_cache_pair,
)
from semafold.vector.models import EncodeObjective, EncodeMetric, VectorDecodeRequest, VectorEncodeRequest, VectorEncoding

__all__ = [
    "TurboQuantKVConfig",
    "TurboQuantKVCacheArtifact",
    "TurboQuantKVPreviewCodec",
    "build_kv_cache_artifact",
]


_ARTIFACT_FORMAT: Final[str] = "kv_preview_v1"
_KEY_OBJECTIVE: Final[EncodeObjective] = EncodeObjective.INNER_PRODUCT_ESTIMATION
_KEY_METRIC: Final[EncodeMetric] = EncodeMetric.DOT_PRODUCT_ERROR
_KEY_ROLE: Final[str] = "key_cache"
_VALUE_OBJECTIVE: Final[EncodeObjective] = EncodeObjective.RECONSTRUCTION
_VALUE_METRIC: Final[EncodeMetric] = EncodeMetric.MSE
_VALUE_ROLE: Final[str] = "value_cache"
_KEY_VARIANT_ID: Final[str] = TurboQuantProdVectorCodec.variant_id
_VALUE_VARIANT_ID: Final[str] = TurboQuantMSEVectorCodec.variant_id


def _required_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _required_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _metadata_dict(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("metadata must be a mapping[str, object]")
    return dict(value)


def _expected_role(name: str, encoding: VectorEncoding, expected_role: str) -> None:
    roles = {segment.role for segment in encoding.segments}
    if roles != {expected_role}:
        raise ValueError(f"{name} segments must all use role={expected_role!r}")


def _encoding_scope_shape(name: str, encoding: VectorEncoding) -> tuple[int, int]:
    shapes: set[tuple[int, int]] = set()
    for segment in encoding.segments:
        kind = segment.scope.get("kind")
        if kind != "full":
            continue
        row_start = segment.scope.get("row_start")
        row_stop = segment.scope.get("row_stop")
        col_start = segment.scope.get("col_start")
        col_stop = segment.scope.get("col_stop")
        if not isinstance(row_start, int) or not isinstance(row_stop, int):
            raise ValueError(f"{name} full scopes must contain integer row bounds")
        if not isinstance(col_start, int) or not isinstance(col_stop, int):
            raise ValueError(f"{name} full scopes must contain integer column bounds")
        if row_start != 0 or col_start != 0:
            raise ValueError(f"{name} full scopes must start at zero")
        if row_stop <= 0 or col_stop <= 0:
            raise ValueError(f"{name} full scopes must have positive extents")
        shapes.add((row_stop, col_stop))
    if len(shapes) != 1:
        raise ValueError(f"{name} must expose exactly one full-scope shape")
    return next(iter(shapes))


def _validate_encoding_contracts(
    *,
    layers: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    key_encoding: VectorEncoding,
    value_encoding: VectorEncoding,
) -> None:
    expected_row_count = layers * heads * seq_len
    if key_encoding.codec_family != "turboquant":
        raise ValueError("key_encoding must use codec_family='turboquant'")
    if value_encoding.codec_family != "turboquant":
        raise ValueError("value_encoding must use codec_family='turboquant'")
    if key_encoding.variant_id != _KEY_VARIANT_ID:
        raise ValueError(f"key_encoding must use variant_id={_KEY_VARIANT_ID!r}")
    if value_encoding.variant_id != _VALUE_VARIANT_ID:
        raise ValueError(f"value_encoding must use variant_id={_VALUE_VARIANT_ID!r}")
    if key_encoding.metadata.get("objective") != "inner_product_estimation":
        raise ValueError("key_encoding must preserve inner_product_estimation objective metadata")
    if value_encoding.metadata.get("objective") != "reconstruction":
        raise ValueError("value_encoding must preserve reconstruction objective metadata")
    _expected_role("key_encoding", key_encoding, _KEY_ROLE)
    _expected_role("value_encoding", value_encoding, _VALUE_ROLE)
    if _encoding_scope_shape("key_encoding", key_encoding) != (expected_row_count, head_dim):
        raise ValueError("key_encoding shape does not match KV layout")
    if _encoding_scope_shape("value_encoding", value_encoding) != (expected_row_count, head_dim):
        raise ValueError("value_encoding shape does not match KV layout")


def _combined_footprint(*, key_encoding: VectorEncoding, value_encoding: VectorEncoding) -> CompressionFootprint:
    return build_footprint(
        baseline_bytes=key_encoding.footprint.baseline_bytes + value_encoding.footprint.baseline_bytes,
        payload_bytes=key_encoding.footprint.payload_bytes + value_encoding.footprint.payload_bytes,
        metadata_bytes=key_encoding.footprint.metadata_bytes + value_encoding.footprint.metadata_bytes,
        sidecar_bytes=key_encoding.footprint.sidecar_bytes + value_encoding.footprint.sidecar_bytes,
        protected_passthrough_bytes=(
            key_encoding.footprint.protected_passthrough_bytes + value_encoding.footprint.protected_passthrough_bytes
        ),
        decoder_state_bytes=key_encoding.footprint.decoder_state_bytes + value_encoding.footprint.decoder_state_bytes,
    )


def _validate_exact_footprint(
    *,
    footprint: CompressionFootprint,
    key_encoding: VectorEncoding,
    value_encoding: VectorEncoding,
) -> None:
    expected = _combined_footprint(key_encoding=key_encoding, value_encoding=value_encoding)
    if footprint.to_dict() != expected.to_dict():
        raise ValueError("footprint must equal the combined key/value encoding footprint")


def _validate_key_total_bits_per_scalar(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("key_total_bits_per_scalar must be an int")
    if value < 2 or value > 5:
        raise ValueError("key_total_bits_per_scalar must be between 2 and 5")
    return value


def _validate_value_bits_per_scalar(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value_bits_per_scalar must be an int")
    if value < 1 or value > 4:
        raise ValueError("value_bits_per_scalar must be between 1 and 4")
    return value


def _validate_seed(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    return value


def _hypothetical_baseline_bytes(
    *,
    layers: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    bytes_per_scalar: int,
) -> int:
    scalar_count_per_tensor = int(layers) * int(heads) * int(seq_len) * int(head_dim)
    return 2 * scalar_count_per_tensor * int(bytes_per_scalar)


@dataclass(frozen=True, slots=True)
class TurboQuantKVConfig:
    """Configuration for cache-shaped TurboQuant K/V block compression.

    Keys use the inner-product oriented TurboQuant Prod path while values use
    the reconstruction-oriented TurboQuant MSE path.
    """

    key_total_bits_per_scalar: int = 3
    value_bits_per_scalar: int = 3
    default_key_rotation_seed: int = 0
    default_key_qjl_seed: int = 0
    default_value_rotation_seed: int = 0
    normalization: Literal["row_l2"] = "row_l2"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key_total_bits_per_scalar",
            _validate_key_total_bits_per_scalar(self.key_total_bits_per_scalar),
        )
        object.__setattr__(self, "value_bits_per_scalar", _validate_value_bits_per_scalar(self.value_bits_per_scalar))
        object.__setattr__(
            self,
            "default_key_rotation_seed",
            _validate_seed("default_key_rotation_seed", self.default_key_rotation_seed),
        )
        object.__setattr__(
            self,
            "default_key_qjl_seed",
            _validate_seed("default_key_qjl_seed", self.default_key_qjl_seed),
        )
        object.__setattr__(
            self,
            "default_value_rotation_seed",
            _validate_seed("default_value_rotation_seed", self.default_value_rotation_seed),
        )
        if self.normalization != "row_l2":
            raise ValueError("normalization must be 'row_l2' in the current KV TurboQuant surface")


@dataclass(slots=True)
class TurboQuantKVCacheArtifact:
    """Outer artifact that binds key and value encodings to one cache layout."""

    format: str
    layout: str
    layers: int
    heads: int
    seq_len: int
    head_dim: int
    key_encoding: VectorEncoding
    value_encoding: VectorEncoding
    footprint: CompressionFootprint
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.format = _required_str("format", self.format)
        if self.format != _ARTIFACT_FORMAT:
            raise ValueError(f"format must be {_ARTIFACT_FORMAT!r}")
        self.layout = _required_str("layout", self.layout)
        if self.layout != CANONICAL_CACHE_LAYOUT:
            raise ValueError(f"layout must be {CANONICAL_CACHE_LAYOUT!r}")
        self.layers = _required_int("layers", self.layers)
        self.heads = _required_int("heads", self.heads)
        self.seq_len = _required_int("seq_len", self.seq_len)
        self.head_dim = _required_int("head_dim", self.head_dim)
        if self.head_dim < 2:
            raise ValueError("head_dim must be >= 2")
        if not isinstance(self.key_encoding, VectorEncoding):
            raise TypeError("key_encoding must be a VectorEncoding")
        if not isinstance(self.value_encoding, VectorEncoding):
            raise TypeError("value_encoding must be a VectorEncoding")
        if not isinstance(self.footprint, CompressionFootprint):
            raise TypeError("footprint must be a CompressionFootprint")
        _validate_encoding_contracts(
            layers=self.layers,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            key_encoding=self.key_encoding,
            value_encoding=self.value_encoding,
        )
        _validate_exact_footprint(
            footprint=self.footprint,
            key_encoding=self.key_encoding,
            value_encoding=self.value_encoding,
        )
        self.metadata = _metadata_dict(self.metadata)

    def to_dict(self) -> dict[str, object]:
        """Serialize the KV artifact into a JSON-friendly mapping."""
        return {
            "format": self.format,
            "layout": self.layout,
            "layers": self.layers,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "head_dim": self.head_dim,
            "key_encoding": self.key_encoding.to_dict(),
            "value_encoding": self.value_encoding.to_dict(),
            "footprint": self.footprint.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TurboQuantKVCacheArtifact":
        """Hydrate a KV artifact from a validated JSON-like mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("value must be a mapping")
        key_encoding = value.get("key_encoding")
        value_encoding = value.get("value_encoding")
        footprint = value.get("footprint")
        if not isinstance(key_encoding, Mapping):
            raise TypeError("key_encoding must be a mapping")
        if not isinstance(value_encoding, Mapping):
            raise TypeError("value_encoding must be a mapping")
        if not isinstance(footprint, Mapping):
            raise TypeError("footprint must be a mapping")
        return cls(
            format=_required_str("format", value["format"]),
            layout=_required_str("layout", value["layout"]),
            layers=_required_int("layers", value["layers"]),
            heads=_required_int("heads", value["heads"]),
            seq_len=_required_int("seq_len", value["seq_len"]),
            head_dim=_required_int("head_dim", value["head_dim"]),
            key_encoding=VectorEncoding.from_dict(dict(key_encoding)),
            value_encoding=VectorEncoding.from_dict(dict(value_encoding)),
            footprint=CompressionFootprint.from_dict(dict(footprint)),
            metadata=_metadata_dict(value.get("metadata")),
        )


def build_kv_cache_artifact(
    *,
    layers: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    key_encoding: VectorEncoding,
    value_encoding: VectorEncoding,
    format: str = _ARTIFACT_FORMAT,
    layout: str = CANONICAL_CACHE_LAYOUT,
    metadata: Mapping[str, object] | None = None,
) -> TurboQuantKVCacheArtifact:
    """Build a KV artifact from already-encoded key and value envelopes."""
    return TurboQuantKVCacheArtifact(
        format=format,
        layout=layout,
        layers=layers,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        key_encoding=key_encoding,
        value_encoding=value_encoding,
        footprint=_combined_footprint(key_encoding=key_encoding, value_encoding=value_encoding),
        metadata=_metadata_dict(metadata),
    )


class TurboQuantKVPreviewCodec:
    """Cache-shaped wrapper that compresses keys and values with dedicated codecs."""

    variant_id = _ARTIFACT_FORMAT

    def __init__(self, *, config: TurboQuantKVConfig | None = None) -> None:
        if config is not None and not isinstance(config, TurboQuantKVConfig):
            raise TypeError("config must be a TurboQuantKVConfig or None")
        self.config = config if config is not None else TurboQuantKVConfig()
        self._key_encode_codec = TurboQuantProdVectorCodec(
            config=TurboQuantProdConfig(
                total_bits_per_scalar=self.config.key_total_bits_per_scalar,
                default_rotation_seed=self.config.default_key_rotation_seed,
                default_qjl_seed=self.config.default_key_qjl_seed,
                normalization=self.config.normalization,
            )
        )
        self._value_encode_codec = TurboQuantMSEVectorCodec(
            config=TurboQuantMSEConfig(
                default_bits_per_scalar=self.config.value_bits_per_scalar,
                default_rotation_seed=self.config.default_value_rotation_seed,
                normalization=self.config.normalization,
            )
        )
        self._key_decode_codec = TurboQuantProdVectorCodec()
        self._value_decode_codec = TurboQuantMSEVectorCodec()

    def compress(self, keys: np.ndarray, values: np.ndarray) -> TurboQuantKVCacheArtifact:
        """Compress cache-shaped key/value tensors into one Semafold KV artifact."""
        layers, heads, seq_len, head_dim = self._validate_inputs(keys, values)
        key_rows = flatten_cache_rows(keys)
        value_rows = flatten_cache_rows(values)
        key_encoding = self._key_encode_codec.encode(
            VectorEncodeRequest(
                data=key_rows,
                objective=_KEY_OBJECTIVE,
                metric=_KEY_METRIC,
                role=_KEY_ROLE,
            )
        )
        value_encoding = self._value_encode_codec.encode(
            VectorEncodeRequest(
                data=value_rows,
                objective=_VALUE_OBJECTIVE,
                metric=_VALUE_METRIC,
                role=_VALUE_ROLE,
            )
        )
        return build_kv_cache_artifact(
            layers=layers,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            key_encoding=key_encoding,
            value_encoding=value_encoding,
            metadata={
                "mode": "kv_cache_block_compression",
                "key_total_bits_per_scalar": self.config.key_total_bits_per_scalar,
                "value_bits_per_scalar": self.config.value_bits_per_scalar,
                "key_variant_id": key_encoding.variant_id,
                "value_variant_id": value_encoding.variant_id,
            },
        )

    def decompress(self, artifact: TurboQuantKVCacheArtifact) -> tuple[np.ndarray, np.ndarray]:
        """Decompress one KV artifact back into ``(keys, values)`` tensors."""
        validated = self._validate_artifact(artifact)
        key_rows = self._key_decode_codec.decode(VectorDecodeRequest(encoding=validated.key_encoding)).data
        value_rows = self._value_decode_codec.decode(VectorDecodeRequest(encoding=validated.value_encoding)).data
        try:
            keys = restore_cache_rows(
                key_rows,
                layers=validated.layers,
                heads=validated.heads,
                seq_len=validated.seq_len,
                head_dim=validated.head_dim,
            )
            values = restore_cache_rows(
                value_rows,
                layers=validated.layers,
                heads=validated.heads,
                seq_len=validated.seq_len,
                head_dim=validated.head_dim,
            )
        except (TypeError, ValueError) as exc:
            raise DecodeError("decoded rows do not match artifact KV layout") from exc
        return keys, values

    def memory_stats(self, artifact: TurboQuantKVCacheArtifact) -> dict[str, float | int]:
        """Return combined KV size statistics, including fp16/bf16 baselines."""
        validated = self._validate_artifact(artifact)
        combined_bytes = validated.footprint.total_bytes
        baseline_bytes = validated.footprint.baseline_bytes
        baseline_fp16_bytes = _hypothetical_baseline_bytes(
            layers=validated.layers,
            heads=validated.heads,
            seq_len=validated.seq_len,
            head_dim=validated.head_dim,
            bytes_per_scalar=2,
        )
        baseline_bf16_bytes = baseline_fp16_bytes
        return {
            "baseline_bytes": baseline_bytes,
            "baseline_fp16_bytes": baseline_fp16_bytes,
            "baseline_bf16_bytes": baseline_bf16_bytes,
            "key_bytes": validated.key_encoding.footprint.total_bytes,
            "value_bytes": validated.value_encoding.footprint.total_bytes,
            "combined_bytes": combined_bytes,
            "combined_compression_ratio": (
                float(baseline_bytes) / float(combined_bytes) if combined_bytes > 0 else 0.0
            ),
            "combined_compression_ratio_vs_fp16": (
                float(baseline_fp16_bytes) / float(combined_bytes) if combined_bytes > 0 else 0.0
            ),
            "combined_compression_ratio_vs_bf16": (
                float(baseline_bf16_bytes) / float(combined_bytes) if combined_bytes > 0 else 0.0
            ),
        }

    @staticmethod
    def _validate_inputs(keys: np.ndarray, values: np.ndarray) -> tuple[int, int, int, int]:
        shape = validate_cache_pair(keys, values)
        if not np.isfinite(keys).all():
            raise CompatibilityError("keys must contain only finite floating-point values")
        if not np.isfinite(values).all():
            raise CompatibilityError("values must contain only finite floating-point values")
        return shape

    @staticmethod
    def _validate_artifact(artifact: TurboQuantKVCacheArtifact) -> TurboQuantKVCacheArtifact:
        if not isinstance(artifact, TurboQuantKVCacheArtifact):
            raise TypeError("artifact must be a TurboQuantKVCacheArtifact")
        return artifact
