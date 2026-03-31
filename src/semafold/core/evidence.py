"""Validation evidence records attached to encoded Semafold artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["ValidationEvidence"]

def _coerce_scope(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("scope must be a non-empty string")
    return value


def _coerce_environment(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("environment must be a mapping[str, object]")
    return dict(value)


def _coerce_metrics(value: object | None) -> dict[str, float | int | str | bool]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("metrics must be a mapping[str, scalar]")
    copied = dict(value)
    for metric_name, metric_value in copied.items():
        if not isinstance(metric_value, (float, int, str, bool)):
            raise TypeError(f"metrics[{metric_name!r}] must be a float, int, str, or bool")
    return copied


def _coerce_optional_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("passed must be a bool or None")
    return value


def _coerce_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("summary must be a string or None")
    return value


def _coerce_artifact_refs(value: object | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("artifact_refs must be a list[str]")
    return list(value)


@dataclass(slots=True)
class ValidationEvidence:
    """Observed validation evidence produced while building an artifact.

    Evidence is intentionally lightweight: a scope name, environment hints,
    scalar metrics, and an optional pass/fail summary that can be serialized
    alongside an encoding.
    """

    scope: str
    environment: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    passed: bool | None = None
    summary: str | None = None
    artifact_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scope = _coerce_scope(self.scope)
        self.environment = _coerce_environment(self.environment)
        self.metrics = _coerce_metrics(self.metrics)
        self.passed = _coerce_optional_bool(self.passed)
        self.summary = _coerce_optional_str(self.summary)
        self.artifact_refs = _coerce_artifact_refs(self.artifact_refs)

    def to_dict(self) -> dict[str, object]:
        """Serialize the evidence into a JSON-friendly mapping."""
        return {
            "scope": self.scope,
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "summary": self.summary,
            "artifact_refs": list(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ValidationEvidence":
        """Hydrate evidence from a validated JSON-like mapping."""
        return cls(
            scope=_coerce_scope(value["scope"]),
            environment=_coerce_environment(value.get("environment")),
            metrics=_coerce_metrics(value.get("metrics")),
            passed=_coerce_optional_bool(value.get("passed")),
            summary=_coerce_optional_str(value.get("summary")),
            artifact_refs=_coerce_artifact_refs(value.get("artifact_refs")),
        )
