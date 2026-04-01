"""Stable root exports for Phase 1."""

from semafold._version import __version__
from semafold.core import (
    CompressionBudget,
    CompressionEstimate,
    CompressionFootprint,
    CompressionGuarantee,
    ValidationEvidence,
)
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.vector import (
    PassthroughVectorCodec,
    VectorCodec,
    VectorDecodeRequest,
    VectorDecodeResult,
    VectorEncodeRequest,
    VectorEncoding,
    VectorEncodingSegment,
)
from semafold.vector.models import EncodeMetric, EncodeObjective, EncodingSegmentKind

__all__ = [
    "__version__",
    "CompressionBudget",
    "CompressionEstimate",
    "CompressionFootprint",
    "CompressionGuarantee",
    "EncodeMetric",
    "EncodeObjective",
    "EncodingBoundType",
    "EncodingSegmentKind",
    "PassthroughVectorCodec",
    "ValidationEvidence",
    "VectorCodec",
    "VectorDecodeRequest",
    "VectorDecodeResult",
    "VectorEncodeRequest",
    "VectorEncoding",
    "VectorEncodingSegment",
    "WorkloadSuitability",
]
