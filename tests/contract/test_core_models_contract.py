from __future__ import annotations

from semafold import CompressionBudget
from semafold import CompressionGuarantee
from semafold import EncodingBoundType
from semafold import ValidationEvidence
from semafold.core.accounting import CompressionEstimate
from semafold.core.accounting import CompressionFootprint
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.vector.models import EncodeObjective


def test_core_models_roundtrip_dicts() -> None:
    budget = CompressionBudget(target_bytes=32, metadata={"x": "y"})
    estimate = CompressionEstimate(baseline_bytes=64, estimated_total_bytes=32)
    footprint = CompressionFootprint(
        baseline_bytes=64,
        payload_bytes=16,
        metadata_bytes=4,
        sidecar_bytes=4,
        protected_passthrough_bytes=0,
        decoder_state_bytes=0,
        total_bytes=24,
        bytes_saved=40,
        compression_ratio=64 / 24,
    )
    guarantee = CompressionGuarantee(
        objective=EncodeObjective.RECONSTRUCTION,
        metric="observed_mse",
        bound_type=EncodingBoundType.OBSERVED,
        value=0.1,
    )
    evidence = ValidationEvidence(scope="compatibility", metrics={"passed": True})
    assert CompressionBudget.from_dict(budget.to_dict()).to_dict() == budget.to_dict()
    assert CompressionEstimate.from_dict(estimate.to_dict()).to_dict() == estimate.to_dict()
    assert CompressionFootprint.from_dict(footprint.to_dict()).to_dict() == footprint.to_dict()
    assert CompressionGuarantee.from_dict(guarantee.to_dict()).to_dict() == guarantee.to_dict()
    assert ValidationEvidence.from_dict(evidence.to_dict()).to_dict() == evidence.to_dict()
