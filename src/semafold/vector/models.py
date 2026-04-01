"""Stable vector envelope models shared by Semafold codecs."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, TypeVar

import numpy as np

from semafold.core import CompressionBudget, CompressionFootprint, CompressionGuarantee, ValidationEvidence

__all__ = [
    "EncodeMetric",
    "EncodeObjective",
    "EncodingSegmentKind",
    "VectorDecodeRequest",
    "VectorDecodeResult",
    "VectorEncodeRequest",
    "VectorEncoding",
    "VectorEncodingSegment",
    "fingerprint_config",
    "normalize_to_2d",
]


class EncodeObjective(str, Enum):
    """High-level goal of an encoding request."""
    RECONSTRUCTION = "reconstruction"
    STORAGE_ONLY = "storage_only"
    INNER_PRODUCT_ESTIMATION = "inner_product_estimation"


class EncodeMetric(str, Enum):
    """Target metric for distortion limits or budget evaluations."""
    MSE = "mse"
    L2 = "l2"
    DOT_PRODUCT_ERROR = "dot_product_error"


class EncodingSegmentKind(str, Enum):
    """Standard classification of physical segments within an encoding."""
    COMPRESSED = "compressed"
    SIDECAR = "sidecar"
    METADATA = "metadata"
    RESIDUAL_SKETCH = "residual_sketch"
    RESIDUAL_GAMMA = "residual_gamma"
    PASSTHROUGH = "passthrough"


TEnum = TypeVar("TEnum", bound=Enum)


def _coerce_enum(name: str, value: object, enum_cls: type[TEnum]) -> TEnum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            valid = [e.value for e in enum_cls]
            raise ValueError(f"{name} must be one of {valid!r}, got {value!r}") from None
    raise TypeError(f"{name} must be a {enum_cls.__name__} or str, got {type(value).__name__!r}")


def _coerce_optional_enum(name: str, value: object | None, enum_cls: type[TEnum]) -> TEnum | None:
    if value is None:
        return None
    return _coerce_enum(name, value, enum_cls)


def _coerce_optional_str(name: str, value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _coerce_required_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _coerce_object_mapping(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping[str, object]")
    return dict(value)


def _coerce_optional_object_mapping(name: str, value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    return _coerce_object_mapping(name, value)


def _coerce_optional_footprint_mapping(
    name: str,
    value: object | None,
) -> dict[str, int | float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping[str, int | float] or None")
    copied = dict(value)
    for item in copied.values():
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"{name} values must be int or float")
    return copied


def _coerce_object_mapping_list(name: str, value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list[dict[str, object]]")
    result: list[dict[str, object]] = []
    for item in value:
        result.append(_coerce_object_mapping(name, item))
    return result


def _copy_object_dict(name: str, value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping[str, object]")
    return dict(value)


def _copy_optional_footprint(
    value: Mapping[str, int | float] | None,
) -> dict[str, int | float] | None:
    if value is None:
        return None
    copied = dict(value)
    for key, item in copied.items():
        if not isinstance(key, str):
            raise TypeError("footprint keys must be strings")
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError("footprint values must be int or float")
    return copied


def _ensure_numpy_array(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.ndim not in (1, 2):
        raise ValueError("data must have shape (d,) or (n, d)")
    if data.size == 0:
        raise ValueError("data must not be empty")
    return data


def normalize_to_2d(array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Normalize rank-1 or rank-2 vector input to a 2D row-major view."""
    array = _ensure_numpy_array(array)
    original_shape = tuple(int(dim) for dim in array.shape)
    original_rank = int(array.ndim)
    if array.ndim == 1:
        return array.reshape(1, -1), original_shape, original_rank
    return array, original_shape, original_rank


def restore_from_2d(array: np.ndarray, original_shape: tuple[int, ...], _original_rank: int) -> np.ndarray:
    return np.asarray(array.reshape(original_shape))


def array_layout(array: np.ndarray) -> str:
    return "c_contiguous" if array.flags.c_contiguous else "non_contiguous"


def fingerprint_config(config: Mapping[str, object]) -> str:
    """Return a stable SHA-256 fingerprint for a codec configuration payload."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _encode_payload(payload: bytes | dict[str, object]) -> dict[str, object]:
    if isinstance(payload, bytes):
        return {"kind": "bytes", "base64": base64.b64encode(payload).decode("ascii")}
    return {"kind": "json", "value": dict(payload)}


def _decode_payload(value: dict[str, object]) -> bytes | dict[str, object]:
    kind = value.get("kind")
    if kind == "bytes":
        encoded = value.get("base64")
        if not isinstance(encoded, str):
            raise TypeError("base64 payload must be a string")
        return base64.b64decode(encoded.encode("ascii"))
    if kind == "json":
        raw = value.get("value", {})
        return _coerce_object_mapping("json payload", raw)
    raise ValueError("unknown payload wrapper kind")


@dataclass(slots=True)
class VectorEncodeRequest:
    """Stable input envelope passed into vector codecs.

    The request keeps algorithm selection out of the stable surface while still
    carrying the information a codec needs to choose the right compression path:
    objective, optional metric hints, an optional budget, and caller metadata.
    """

    data: np.ndarray
    objective: EncodeObjective
    role: str | None = None
    metric: EncodeMetric | None = None
    budget: CompressionBudget | None = None
    component_id: str | None = None
    profile_id: str | None = None
    seed: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = _ensure_numpy_array(self.data)
        self.objective = _coerce_enum("objective", self.objective, EncodeObjective)
        self.role = _coerce_optional_str("role", self.role)
        self.metric = _coerce_optional_enum("metric", self.metric, EncodeMetric)
        if self.budget is not None and not isinstance(self.budget, CompressionBudget):
            raise TypeError("budget must be a CompressionBudget or None")
        self.component_id = _coerce_optional_str("component_id", self.component_id)
        self.profile_id = _coerce_optional_str("profile_id", self.profile_id)
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an int or None")
        self.metadata = _copy_object_dict("metadata", self.metadata)


@dataclass(slots=True)
class VectorEncodingSegment:
    """One physical segment inside a vector encoding envelope.

    Segments let Semafold describe mixed artifacts such as compressed payloads,
    sidecars, passthrough regions, and metadata without inventing a separate
    envelope type for each codec family.
    """

    segment_kind: EncodingSegmentKind
    role: str | None
    scope: dict[str, object]
    payload: bytes | dict[str, object]
    payload_format: str
    footprint: Mapping[str, int | float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.segment_kind = _coerce_enum("segment_kind", self.segment_kind, EncodingSegmentKind)
        self.role = _coerce_optional_str("role", self.role)
        self.scope = _coerce_object_mapping("scope", self.scope)
        if not isinstance(self.payload, (bytes, Mapping)):
            raise TypeError("payload must be bytes or dict[str, object]")
        if isinstance(self.payload, Mapping):
            self.payload = _coerce_object_mapping("payload", self.payload)
        if not isinstance(self.payload_format, str) or not self.payload_format:
            raise TypeError("payload_format must be a non-empty string")
        self.footprint = _copy_optional_footprint(self.footprint)
        self.metadata = _copy_object_dict("metadata", self.metadata)

    def to_dict(self) -> dict[str, object]:
        """Serialize the segment into a JSON-friendly mapping."""
        return {
            "segment_kind": self.segment_kind.value,
            "role": self.role,
            "scope": dict(self.scope),
            "payload": _encode_payload(self.payload),
            "payload_format": self.payload_format,
            "footprint": dict(self.footprint) if self.footprint is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "VectorEncodingSegment":
        """Hydrate a segment from a validated JSON-like mapping."""
        payload_wrapper = value.get("payload")
        if not isinstance(payload_wrapper, Mapping):
            raise TypeError("payload wrapper must be an object")
        return cls(
            segment_kind=EncodingSegmentKind(value["segment_kind"]),
            role=_coerce_optional_str("role", value.get("role")),
            scope=_coerce_object_mapping("scope", value["scope"]),
            payload=_decode_payload(dict(payload_wrapper)),
            payload_format=_coerce_required_str("payload_format", value["payload_format"]),
            footprint=_coerce_optional_footprint_mapping("footprint", value.get("footprint")),
            metadata=_coerce_optional_object_mapping("metadata", value.get("metadata")),
        )


@dataclass(slots=True)
class VectorEncoding:
    """Stable artifact envelope returned by vector codecs.

    A ``VectorEncoding`` combines identity, measured accounting, guarantees,
    evidence, and a list of physical segments. It is the canonical wire shape
    for Semafold vector artifacts.
    """

    codec_family: str
    variant_id: str
    implementation_version: str
    encoding_schema_version: str
    config_fingerprint: str
    segments: list[VectorEncodingSegment]
    footprint: CompressionFootprint
    guarantees: list[CompressionGuarantee]
    evidence: list[ValidationEvidence]
    profile_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "codec_family",
            "variant_id",
            "implementation_version",
            "encoding_schema_version",
            "config_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{field_name} must be a non-empty string")
        if self.profile_id is not None and not isinstance(self.profile_id, str):
            raise TypeError("profile_id must be a string or None")
        if not isinstance(self.segments, list) or not self.segments:
            raise TypeError("segments must be a non-empty list")
        if not all(isinstance(segment, VectorEncodingSegment) for segment in self.segments):
            raise TypeError("segments must contain VectorEncodingSegment items")
        if not isinstance(self.footprint, CompressionFootprint):
            raise TypeError("footprint must be a CompressionFootprint")
        if not isinstance(self.guarantees, list) or not self.guarantees:
            raise TypeError("guarantees must be a non-empty list")
        if not all(isinstance(item, CompressionGuarantee) for item in self.guarantees):
            raise TypeError("guarantees must contain CompressionGuarantee items")
        if not isinstance(self.evidence, list) or not self.evidence:
            raise TypeError("evidence must be a non-empty list")
        if not all(isinstance(item, ValidationEvidence) for item in self.evidence):
            raise TypeError("evidence must contain ValidationEvidence items")
        self.metadata = _copy_object_dict("metadata", self.metadata)

    def to_dict(self) -> dict[str, object]:
        """Serialize the encoding into a JSON-friendly mapping."""
        return {
            "codec_family": self.codec_family,
            "variant_id": self.variant_id,
            "profile_id": self.profile_id,
            "implementation_version": self.implementation_version,
            "encoding_schema_version": self.encoding_schema_version,
            "config_fingerprint": self.config_fingerprint,
            "segments": [segment.to_dict() for segment in self.segments],
            "footprint": self.footprint.to_dict(),
            "guarantees": [guarantee.to_dict() for guarantee in self.guarantees],
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "VectorEncoding":
        """Hydrate an encoding from a validated JSON-like mapping."""
        segment_items = _coerce_object_mapping_list("segments", value["segments"])
        guarantee_items = _coerce_object_mapping_list("guarantees", value["guarantees"])
        evidence_items = _coerce_object_mapping_list("evidence", value["evidence"])
        return cls(
            codec_family=_coerce_required_str("codec_family", value["codec_family"]),
            variant_id=_coerce_required_str("variant_id", value["variant_id"]),
            profile_id=_coerce_optional_str("profile_id", value.get("profile_id")),
            implementation_version=_coerce_required_str("implementation_version", value["implementation_version"]),
            encoding_schema_version=_coerce_required_str("encoding_schema_version", value["encoding_schema_version"]),
            config_fingerprint=_coerce_required_str("config_fingerprint", value["config_fingerprint"]),
            segments=[VectorEncodingSegment.from_dict(item) for item in segment_items],
            footprint=CompressionFootprint.from_dict(
                _coerce_object_mapping("footprint", value["footprint"]),
            ),
            guarantees=[CompressionGuarantee.from_dict(item) for item in guarantee_items],
            evidence=[ValidationEvidence.from_dict(item) for item in evidence_items],
            metadata=_coerce_optional_object_mapping("metadata", value.get("metadata")),
        )


@dataclass(slots=True)
class VectorDecodeRequest:
    """Stable decode request used to materialize an encoded vector artifact."""

    encoding: VectorEncoding
    target_layout: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, VectorEncoding):
            raise TypeError("encoding must be a VectorEncoding")
        self.target_layout = _coerce_optional_str("target_layout", self.target_layout)
        self.metadata = _copy_object_dict("metadata", self.metadata)


@dataclass(slots=True)
class VectorDecodeResult:
    """Materialized NumPy data plus optional decode-time notes."""

    data: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    materialization_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = _ensure_numpy_array(self.data)
        self.metadata = _copy_object_dict("metadata", self.metadata)
        if not isinstance(self.materialization_notes, list) or not all(
            isinstance(item, str) for item in self.materialization_notes
        ):
            raise TypeError("materialization_notes must be a list[str]")
        self.materialization_notes = list(self.materialization_notes)
