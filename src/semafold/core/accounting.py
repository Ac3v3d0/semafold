"""Measured and estimated byte accounting for Semafold artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "CompressionEstimate",
    "CompressionFootprint",
    "aggregate_footprints",
    "build_footprint",
    "json_byte_size",
    "segment_footprint",
]


def _coerce_optional_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _coerce_required_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _coerce_signed_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    return value


def _coerce_optional_float(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float-compatible value or None")
    coerced = float(value)
    if coerced < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return coerced


def _coerce_required_float(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float-compatible value")
    coerced = float(value)
    if coerced < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return coerced


def _coerce_component_bytes(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int-compatible numeric value")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{name} must be a whole number of bytes")
    coerced = int(value)
    if coerced < 0:
        raise ValueError(f"{name} must be >= 0")
    return coerced


def _require_estimate_components(
    components: list[int | None],
) -> tuple[int, int, int, int, int] | None:
    first, second, third, fourth, fifth = components
    if (
        first is None
        or second is None
        or third is None
        or fourth is None
        or fifth is None
    ):
        return None
    return (first, second, third, fourth, fifth)


def _validate_optional_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_required_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_signed_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")


def _validate_optional_float(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float-compatible value or None")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


def json_byte_size(value: Mapping[str, object] | list[object] | tuple[object, ...] | str | int | float | bool | None) -> int:
    """Return the canonical UTF-8 JSON size for a JSON-like value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return len(encoded)


def _ratio(baseline_bytes: int, total_bytes: int) -> float:
    return float(baseline_bytes) / float(total_bytes) if total_bytes > 0 else 0.0


@dataclass(slots=True)
class CompressionEstimate:
    """Pre-run size estimate emitted before materializing an artifact.

    Estimates are used for planning and UX. They intentionally mirror
    ``CompressionFootprint`` so callers can compare predicted and measured costs
    without re-mapping field names.
    """

    baseline_bytes: int | None = None
    estimated_payload_bytes: int | None = None
    estimated_metadata_bytes: int | None = None
    estimated_sidecar_bytes: int | None = None
    estimated_protected_passthrough_bytes: int | None = None
    estimated_decoder_state_bytes: int | None = None
    estimated_total_bytes: int | None = None
    estimated_compression_ratio: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_int("baseline_bytes", self.baseline_bytes)
        _validate_optional_int("estimated_payload_bytes", self.estimated_payload_bytes)
        _validate_optional_int("estimated_metadata_bytes", self.estimated_metadata_bytes)
        _validate_optional_int("estimated_sidecar_bytes", self.estimated_sidecar_bytes)
        _validate_optional_int(
            "estimated_protected_passthrough_bytes",
            self.estimated_protected_passthrough_bytes,
        )
        _validate_optional_int("estimated_decoder_state_bytes", self.estimated_decoder_state_bytes)
        _validate_optional_int("estimated_total_bytes", self.estimated_total_bytes)
        _validate_optional_float("estimated_compression_ratio", self.estimated_compression_ratio)
        components = [
            self.estimated_payload_bytes,
            self.estimated_metadata_bytes,
            self.estimated_sidecar_bytes,
            self.estimated_protected_passthrough_bytes,
            self.estimated_decoder_state_bytes,
        ]
        typed_components = _require_estimate_components(components)
        if self.estimated_total_bytes is not None and typed_components is not None:
            expected_total = sum(typed_components)
            if self.estimated_total_bytes != expected_total:
                raise ValueError("estimated_total_bytes does not match component sum")
        if (
            self.baseline_bytes is not None
            and self.estimated_total_bytes is not None
            and self.estimated_compression_ratio is not None
        ):
            expected_ratio = _ratio(self.baseline_bytes, self.estimated_total_bytes)
            if abs(float(self.estimated_compression_ratio) - expected_ratio) > 1e-12:
                raise ValueError("estimated_compression_ratio does not match baseline_bytes / estimated_total_bytes")

    def to_dict(self) -> dict[str, object]:
        """Serialize the estimate into a JSON-friendly mapping."""
        return {
            "baseline_bytes": self.baseline_bytes,
            "estimated_payload_bytes": self.estimated_payload_bytes,
            "estimated_metadata_bytes": self.estimated_metadata_bytes,
            "estimated_sidecar_bytes": self.estimated_sidecar_bytes,
            "estimated_protected_passthrough_bytes": self.estimated_protected_passthrough_bytes,
            "estimated_decoder_state_bytes": self.estimated_decoder_state_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "estimated_compression_ratio": self.estimated_compression_ratio,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompressionEstimate":
        """Hydrate an estimate from a validated JSON-like mapping."""
        return cls(
            baseline_bytes=_coerce_optional_int("baseline_bytes", value.get("baseline_bytes")),
            estimated_payload_bytes=_coerce_optional_int(
                "estimated_payload_bytes",
                value.get("estimated_payload_bytes"),
            ),
            estimated_metadata_bytes=_coerce_optional_int(
                "estimated_metadata_bytes",
                value.get("estimated_metadata_bytes"),
            ),
            estimated_sidecar_bytes=_coerce_optional_int(
                "estimated_sidecar_bytes",
                value.get("estimated_sidecar_bytes"),
            ),
            estimated_protected_passthrough_bytes=_coerce_optional_int(
                "estimated_protected_passthrough_bytes",
                value.get("estimated_protected_passthrough_bytes"),
            ),
            estimated_decoder_state_bytes=_coerce_optional_int(
                "estimated_decoder_state_bytes",
                value.get("estimated_decoder_state_bytes"),
            ),
            estimated_total_bytes=_coerce_optional_int(
                "estimated_total_bytes",
                value.get("estimated_total_bytes"),
            ),
            estimated_compression_ratio=_coerce_optional_float(
                "estimated_compression_ratio",
                value.get("estimated_compression_ratio"),
            ),
        )


@dataclass(slots=True)
class CompressionFootprint:
    """Measured byte footprint for a concrete encoded artifact.

    This record is the canonical source of truth for how many bytes were
    actually emitted after payload, metadata, sidecars, and passthrough regions
    are accounted for.
    """

    baseline_bytes: int
    payload_bytes: int
    metadata_bytes: int
    sidecar_bytes: int
    protected_passthrough_bytes: int
    decoder_state_bytes: int
    total_bytes: int
    bytes_saved: int
    compression_ratio: float

    def __post_init__(self) -> None:
        _validate_required_int("baseline_bytes", self.baseline_bytes)
        _validate_required_int("payload_bytes", self.payload_bytes)
        _validate_required_int("metadata_bytes", self.metadata_bytes)
        _validate_required_int("sidecar_bytes", self.sidecar_bytes)
        _validate_required_int("protected_passthrough_bytes", self.protected_passthrough_bytes)
        _validate_required_int("decoder_state_bytes", self.decoder_state_bytes)
        _validate_required_int("total_bytes", self.total_bytes)
        _validate_signed_int("bytes_saved", self.bytes_saved)
        if not isinstance(self.compression_ratio, (int, float)) or isinstance(self.compression_ratio, bool):
            raise TypeError("compression_ratio must be a float-compatible value")
        expected_total = (
            self.payload_bytes
            + self.metadata_bytes
            + self.sidecar_bytes
            + self.protected_passthrough_bytes
            + self.decoder_state_bytes
        )
        if self.total_bytes != expected_total:
            raise ValueError("total_bytes does not match component sum")
        expected_saved = self.baseline_bytes - self.total_bytes
        if self.bytes_saved != expected_saved:
            raise ValueError("bytes_saved must equal baseline_bytes - total_bytes")
        expected_ratio = _ratio(self.baseline_bytes, self.total_bytes)
        if abs(float(self.compression_ratio) - expected_ratio) > 1e-12:
            raise ValueError("compression_ratio does not match baseline_bytes / total_bytes")

    def to_dict(self) -> dict[str, object]:
        """Serialize the measured footprint into a JSON-friendly mapping."""
        return {
            "baseline_bytes": self.baseline_bytes,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "sidecar_bytes": self.sidecar_bytes,
            "protected_passthrough_bytes": self.protected_passthrough_bytes,
            "decoder_state_bytes": self.decoder_state_bytes,
            "total_bytes": self.total_bytes,
            "bytes_saved": self.bytes_saved,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompressionFootprint":
        """Hydrate a measured footprint from a validated JSON-like mapping."""
        return cls(
            baseline_bytes=_coerce_required_int("baseline_bytes", value["baseline_bytes"]),
            payload_bytes=_coerce_required_int("payload_bytes", value["payload_bytes"]),
            metadata_bytes=_coerce_required_int("metadata_bytes", value["metadata_bytes"]),
            sidecar_bytes=_coerce_required_int("sidecar_bytes", value["sidecar_bytes"]),
            protected_passthrough_bytes=_coerce_required_int(
                "protected_passthrough_bytes",
                value["protected_passthrough_bytes"],
            ),
            decoder_state_bytes=_coerce_required_int(
                "decoder_state_bytes",
                value["decoder_state_bytes"],
            ),
            total_bytes=_coerce_required_int("total_bytes", value["total_bytes"]),
            bytes_saved=_coerce_signed_int("bytes_saved", value["bytes_saved"]),
            compression_ratio=_coerce_required_float(
                "compression_ratio",
                value["compression_ratio"],
            ),
        )


def build_footprint(
    *,
    baseline_bytes: int,
    payload_bytes: int = 0,
    metadata_bytes: int = 0,
    sidecar_bytes: int = 0,
    protected_passthrough_bytes: int = 0,
    decoder_state_bytes: int = 0,
) -> CompressionFootprint:
    """Build a measured footprint from explicit byte components."""
    total_bytes = (
        payload_bytes
        + metadata_bytes
        + sidecar_bytes
        + protected_passthrough_bytes
        + decoder_state_bytes
    )
    return CompressionFootprint(
        baseline_bytes=baseline_bytes,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        sidecar_bytes=sidecar_bytes,
        protected_passthrough_bytes=protected_passthrough_bytes,
        decoder_state_bytes=decoder_state_bytes,
        total_bytes=total_bytes,
        bytes_saved=baseline_bytes - total_bytes,
        compression_ratio=_ratio(baseline_bytes, total_bytes),
    )


def segment_footprint(
    *,
    payload_bytes: int = 0,
    metadata_bytes: int = 0,
    sidecar_bytes: int = 0,
    protected_passthrough_bytes: int = 0,
    decoder_state_bytes: int = 0,
) -> dict[str, int]:
    """Create a segment-local byte summary with a normalized field layout."""
    payload_bytes = _coerce_component_bytes("payload_bytes", payload_bytes)
    metadata_bytes = _coerce_component_bytes("metadata_bytes", metadata_bytes)
    sidecar_bytes = _coerce_component_bytes("sidecar_bytes", sidecar_bytes)
    protected_passthrough_bytes = _coerce_component_bytes(
        "protected_passthrough_bytes",
        protected_passthrough_bytes,
    )
    decoder_state_bytes = _coerce_component_bytes("decoder_state_bytes", decoder_state_bytes)
    total_bytes = (
        payload_bytes
        + metadata_bytes
        + sidecar_bytes
        + protected_passthrough_bytes
        + decoder_state_bytes
    )
    return {
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "sidecar_bytes": sidecar_bytes,
        "protected_passthrough_bytes": protected_passthrough_bytes,
        "decoder_state_bytes": decoder_state_bytes,
        "total_bytes": total_bytes,
    }


def aggregate_footprints(
    *,
    baseline_bytes: int,
    segment_footprints: list[Mapping[str, int | float] | None],
) -> CompressionFootprint:
    """Aggregate segment-local byte summaries into one measured footprint."""
    payload_bytes = 0
    metadata_bytes = 0
    sidecar_bytes = 0
    protected_passthrough_bytes = 0
    decoder_state_bytes = 0
    for raw in segment_footprints:
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise TypeError("segment_footprints must contain mappings or None")
        payload_value = _coerce_component_bytes("payload_bytes", raw.get("payload_bytes", 0))
        metadata_value = _coerce_component_bytes("metadata_bytes", raw.get("metadata_bytes", 0))
        sidecar_value = _coerce_component_bytes("sidecar_bytes", raw.get("sidecar_bytes", 0))
        protected_value = _coerce_component_bytes(
            "protected_passthrough_bytes",
            raw.get("protected_passthrough_bytes", 0),
        )
        decoder_state_value = _coerce_component_bytes(
            "decoder_state_bytes",
            raw.get("decoder_state_bytes", 0),
        )
        payload_bytes += payload_value
        metadata_bytes += metadata_value
        sidecar_bytes += sidecar_value
        protected_passthrough_bytes += protected_value
        decoder_state_bytes += decoder_state_value
        expected_total = (
            payload_value
            + metadata_value
            + sidecar_value
            + protected_value
            + decoder_state_value
        )
        if "total_bytes" in raw and _coerce_component_bytes("total_bytes", raw["total_bytes"]) != expected_total:
            raise ValueError("segment footprint total_bytes does not match component sum")
    return build_footprint(
        baseline_bytes=baseline_bytes,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        sidecar_bytes=sidecar_bytes,
        protected_passthrough_bytes=protected_passthrough_bytes,
        decoder_state_bytes=decoder_state_bytes,
    )
