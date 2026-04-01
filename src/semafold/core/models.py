"""Stable core records shared across Semafold codecs and envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, TypeVar
__all__ = ["CompressionBudget", "CompressionGuarantee", "EncodingBoundType", "WorkloadSuitability"]


class EncodingBoundType(str, Enum):
    OBSERVED = "observed"
    PAPER_REFERENCE = "paper_reference"
    THEOREM_REFERENCE = "theorem_reference"
    EXACT = "exact"


class WorkloadSuitability(str, Enum):
    EMBEDDING_STORAGE = "embedding_storage"
    RECONSTRUCTION_ONLY = "reconstruction_only"
    VECTOR_DATABASE = "vector_database"
    QUERY_TIME_INNER_PRODUCT = "query_time_inner_product"


_TEnum = TypeVar("_TEnum", bound=Enum)


def _coerce_enum(name: str, value: object, enum_cls: type[_TEnum]) -> _TEnum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            valid = [e.value for e in enum_cls]
            raise ValueError(f"{name} must be one of {valid!r}, got {value!r}") from None
    raise TypeError(f"{name} must be a {enum_cls.__name__} or str, got {type(value).__name__!r}")


def _coerce_optional_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
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


def _coerce_bool(name: str, value: object | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _coerce_metadata(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("metadata must be a mapping[str, object]")
    return dict(value)


def _coerce_string_list(name: str, value: object | None) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{name} must be a list[str] or None")
    return list(value)


def _coerce_guarantee_value(value: object | None) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, int, str)):
        return value
    raise TypeError("value must be a float, int, str, bool, or None")


def _validate_optional_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_optional_float(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float-compatible value or None")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _validate_optional_str(name: str, value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")


@dataclass(slots=True)
class CompressionBudget:
    """Caller-supplied constraints that guide encode-time planning.

    Budgets are intentionally generic so the same record can describe vector,
    KV-cache, or future domains without leaking codec-specific terminology into
    the stable surface.
    """

    target_bytes: int | None = None
    target_ratio: float | None = None
    target_bits_per_scalar: float | None = None
    max_vectors: int | None = None
    max_tokens: int | None = None
    allow_passthrough: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_optional_int("target_bytes", self.target_bytes)
        _validate_optional_float("target_ratio", self.target_ratio)
        _validate_optional_float("target_bits_per_scalar", self.target_bits_per_scalar)
        _validate_optional_int("max_vectors", self.max_vectors)
        _validate_optional_int("max_tokens", self.max_tokens)
        if not isinstance(self.allow_passthrough, bool):
            raise TypeError("allow_passthrough must be a bool")
        self.metadata = _coerce_metadata(self.metadata)

    def to_dict(self) -> dict[str, object]:
        """Serialize the budget into a JSON-friendly mapping."""
        return {
            "target_bytes": self.target_bytes,
            "target_ratio": self.target_ratio,
            "target_bits_per_scalar": self.target_bits_per_scalar,
            "max_vectors": self.max_vectors,
            "max_tokens": self.max_tokens,
            "allow_passthrough": self.allow_passthrough,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompressionBudget":
        """Hydrate a budget from a validated JSON-like mapping."""
        return cls(
            target_bytes=_coerce_optional_int("target_bytes", value.get("target_bytes")),
            target_ratio=_coerce_optional_float("target_ratio", value.get("target_ratio")),
            target_bits_per_scalar=_coerce_optional_float(
                "target_bits_per_scalar",
                value.get("target_bits_per_scalar"),
            ),
            max_vectors=_coerce_optional_int("max_vectors", value.get("max_vectors")),
            max_tokens=_coerce_optional_int("max_tokens", value.get("max_tokens")),
            allow_passthrough=_coerce_bool(
                "allow_passthrough",
                value.get("allow_passthrough"),
                default=False,
            ),
            metadata=_coerce_metadata(value.get("metadata")),
        )


@dataclass(slots=True)
class CompressionGuarantee:
    """Declared fidelity, safety, or accounting property for one artifact.

    Guarantees describe what a codec claims about the emitted artifact, such as
    exact accounting, observed reconstruction error, or theorem-backed inner
    product behavior.
    """

    objective: str
    metric: str
    bound_type: EncodingBoundType
    value: float | int | str | bool | None = None
    units: str | None = None
    scope: str | None = None
    workload_suitability: list[WorkloadSuitability] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        self.objective = _coerce_required_str("objective", self.objective)
        self.metric = _coerce_required_str("metric", self.metric)
        self.bound_type = _coerce_enum("bound_type", self.bound_type, EncodingBoundType)
        self.value = _coerce_guarantee_value(self.value)
        _validate_optional_str("units", self.units)
        _validate_optional_str("scope", self.scope)
        _validate_optional_str("notes", self.notes)
        if self.workload_suitability is not None:
            if not isinstance(self.workload_suitability, list):
                raise TypeError("workload_suitability must be a list[WorkloadSuitability] or None")
            self.workload_suitability = [
                _coerce_enum("workload_suitability item", item, WorkloadSuitability)
                for item in self.workload_suitability
            ]

    def to_dict(self) -> dict[str, object]:
        """Serialize the guarantee into a JSON-friendly mapping."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "bound_type": self.bound_type,
            "value": self.value,
            "units": self.units,
            "scope": self.scope,
            "workload_suitability": (
                None if self.workload_suitability is None else list(self.workload_suitability)
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CompressionGuarantee":
        """Hydrate a guarantee from a validated JSON-like mapping."""
        ws = value.get("workload_suitability")
        if ws is not None:
            if not isinstance(ws, list):
                raise TypeError("workload_suitability must be a list or None")
            workload_suitability: list[WorkloadSuitability] | None = [WorkloadSuitability(w) for w in ws]
        else:
            workload_suitability = None
        return cls(
            objective=_coerce_required_str("objective", value["objective"]),
            metric=_coerce_required_str("metric", value["metric"]),
            bound_type=EncodingBoundType(_coerce_required_str("bound_type", value["bound_type"])),
            value=_coerce_guarantee_value(value.get("value")),
            units=_coerce_optional_str("units", value.get("units")),
            scope=_coerce_optional_str("scope", value.get("scope")),
            workload_suitability=workload_suitability,
            notes=_coerce_optional_str("notes", value.get("notes")),
        )
