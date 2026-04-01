from __future__ import annotations

import numpy as np

from semafold import CompressionGuarantee
from semafold import PassthroughVectorCodec
from semafold import ValidationEvidence
from semafold import VectorEncodeRequest
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.vector.models import EncodeObjective


def test_guarantee_and_evidence_serialization_roundtrip() -> None:
    guarantee = CompressionGuarantee(
        objective=EncodeObjective.RECONSTRUCTION,
        metric="observed_mse",
        bound_type=EncodingBoundType.OBSERVED,
        value=0.01,
        workload_suitability=[WorkloadSuitability.RECONSTRUCTION_ONLY],
    )
    evidence = ValidationEvidence(
        scope="proxy_fidelity",
        environment={"codec_family": "scalar_reference"},
        metrics={"mse": 0.01},
        passed=True,
    )
    assert CompressionGuarantee.from_dict(guarantee.to_dict()).to_dict() == guarantee.to_dict()
    assert ValidationEvidence.from_dict(evidence.to_dict()).to_dict() == evidence.to_dict()


def test_codec_emitted_guarantees_and_evidence_remain_consumer_readable() -> None:
    passthrough = PassthroughVectorCodec().encode(
        VectorEncodeRequest(
            data=np.array([1.0, 2.0], dtype=np.float32),
            objective=EncodeObjective.RECONSTRUCTION,
            role="embedding",
        )
    )
    passthrough_guarantees = [item.to_dict() for item in passthrough.guarantees]
    passthrough_evidence = [item.to_dict() for item in passthrough.evidence]

    assert passthrough_guarantees[0]["objective"] == "reconstruction"
    assert passthrough_guarantees[0]["workload_suitability"] == [WorkloadSuitability.EMBEDDING_STORAGE, WorkloadSuitability.RECONSTRUCTION_ONLY]
    assert {item["scope"] for item in passthrough_evidence} == {"compatibility", "storage_accounting"}
    assert [
        CompressionGuarantee.from_dict(item).to_dict() for item in passthrough_guarantees
    ] == passthrough_guarantees
    assert [ValidationEvidence.from_dict(item).to_dict() for item in passthrough_evidence] == passthrough_evidence

    scalar = ScalarReferenceVectorCodec().encode(
        VectorEncodeRequest(
            data=np.array([[1.0, 1.5, 2.0]], dtype=np.float32),
            objective=EncodeObjective.RECONSTRUCTION,
            role="embedding",
        )
    )
    scalar_guarantees = [item.to_dict() for item in scalar.guarantees]
    scalar_evidence = [item.to_dict() for item in scalar.evidence]

    assert scalar_guarantees[0]["objective"] == "reconstruction"
    assert scalar_guarantees[0]["metric"] == "observed_mse"
    assert scalar_guarantees[0]["workload_suitability"] == [WorkloadSuitability.EMBEDDING_STORAGE, WorkloadSuitability.RECONSTRUCTION_ONLY]
    assert {item["scope"] for item in scalar_evidence} == {"proxy_fidelity", "storage_accounting"}
    proxy_fidelity = next(item for item in scalar_evidence if item["scope"] == "proxy_fidelity")
    metrics = proxy_fidelity["metrics"]
    assert isinstance(metrics, dict)
    assert float(metrics["mse"]) > 0.0
    assert [
        CompressionGuarantee.from_dict(item).to_dict() for item in scalar_guarantees
    ] == scalar_guarantees
    assert [ValidationEvidence.from_dict(item).to_dict() for item in scalar_evidence] == scalar_evidence
