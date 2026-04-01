from __future__ import annotations

import pytest

from semafold import CompressionBudget
from semafold import CompressionGuarantee
from semafold import EncodingBoundType, WorkloadSuitability
from semafold import ValidationEvidence
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.vector.models import EncodeObjective, EncodeMetric


def test_budget_copies_metadata() -> None:
    metadata: dict[str, object] = {"a": 1}
    budget = CompressionBudget(metadata=metadata)
    metadata["a"] = 2
    assert budget.metadata["a"] == 1


def test_budget_from_dict_copies_mapping_metadata() -> None:
    metadata: dict[str, object] = {"a": 1}
    budget = CompressionBudget.from_dict({"metadata": metadata})
    metadata["a"] = 2
    assert budget.metadata["a"] == 1


def test_budget_rejects_invalid_allow_passthrough() -> None:
    with pytest.raises(TypeError):
        CompressionBudget(allow_passthrough="yes")  # type: ignore[arg-type]


def test_budget_rejects_negative_target_bytes() -> None:
    with pytest.raises(ValueError):
        CompressionBudget(target_bytes=-1)


def test_budget_rejects_non_string_metadata_keys() -> None:
    with pytest.raises(TypeError):
        CompressionBudget(metadata={1: "x"})  # type: ignore[dict-item]


def test_guarantee_requires_strings() -> None:
    with pytest.raises(TypeError):
        CompressionGuarantee(objective="", metric=EncodeMetric.MSE, bound_type=EncodingBoundType.OBSERVED)


def test_guarantee_rejects_invalid_value_type() -> None:
    with pytest.raises(TypeError):
        CompressionGuarantee(
            objective=EncodeObjective.RECONSTRUCTION,
            metric=EncodeMetric.MSE,
            bound_type=EncodingBoundType.OBSERVED,
            value={"not": "scalar"},  # type: ignore[arg-type]
        )


def test_guarantee_copies_workload_suitability() -> None:
    workload_suitability = [WorkloadSuitability.EMBEDDING_STORAGE]
    guarantee = CompressionGuarantee(
        objective=EncodeObjective.RECONSTRUCTION,
        metric=EncodeMetric.MSE,
        bound_type=EncodingBoundType.OBSERVED,
        workload_suitability=workload_suitability,
    )
    workload_suitability.append(WorkloadSuitability.VECTOR_DATABASE)
    assert guarantee.workload_suitability == [WorkloadSuitability.EMBEDDING_STORAGE]


def test_guarantee_from_dict_rejects_non_string_required_fields() -> None:
    with pytest.raises(TypeError):
        CompressionGuarantee.from_dict(
            {
                "objective": 1,
                "metric": "mse",
                "bound_type": "observed",
            },
        )


def test_validation_evidence_requires_string_refs() -> None:
    with pytest.raises(TypeError):
        ValidationEvidence(scope="compatibility", artifact_refs=[1])  # type: ignore[list-item]


def test_validation_evidence_rejects_invalid_environment_keys() -> None:
    with pytest.raises(TypeError):
        ValidationEvidence(
            scope="compatibility",
            environment={1: "bad"},  # type: ignore[dict-item]
        )


def test_validation_evidence_rejects_non_scalar_metrics() -> None:
    with pytest.raises(TypeError):
        ValidationEvidence(
            scope="compatibility",
            metrics={"mse": {"bad": 1}},  # type: ignore[dict-item]
        )


def test_validation_evidence_copies_collections() -> None:
    environment: dict[str, object] = {"codec_family": "passthrough"}
    metrics: dict[str, float | int | str | bool] = {"supported": True}
    artifact_refs = ["artifact://one"]
    evidence = ValidationEvidence(
        scope="compatibility",
        environment=environment,
        metrics=metrics,
        artifact_refs=artifact_refs,
    )
    environment["codec_family"] = "scalar_reference"
    metrics["supported"] = False
    artifact_refs.append("artifact://two")
    assert evidence.environment == {"codec_family": "passthrough"}
    assert evidence.metrics == {"supported": True}
    assert evidence.artifact_refs == ["artifact://one"]


def test_validation_evidence_from_dict_rejects_non_mapping_metrics() -> None:
    with pytest.raises(TypeError):
        ValidationEvidence.from_dict(
            {
                "scope": "compatibility",
                "metrics": ["not", "a", "mapping"],
            },
        )


def test_validation_evidence_from_dict_rejects_non_string_scope() -> None:
    with pytest.raises(TypeError):
        ValidationEvidence.from_dict({"scope": 123})
