"""Stable root exports for Phase 1."""

from semafold._version import __version__
from semafold.core import (
    CompressionBudget,
    CompressionEstimate,
    CompressionFootprint,
    CompressionGuarantee,
    ValidationEvidence,
)
from semafold.vector import (
    PassthroughVectorCodec,
    VectorCodec,
    VectorDecodeRequest,
    VectorDecodeResult,
    VectorEncodeRequest,
    VectorEncoding,
    VectorEncodingSegment,
)

__all__ = [
    "__version__",
    "CompressionBudget",
    "CompressionEstimate",
    "CompressionFootprint",
    "CompressionGuarantee",
    "PassthroughVectorCodec",
    "ValidationEvidence",
    "VectorCodec",
    "VectorDecodeRequest",
    "VectorDecodeResult",
    "VectorEncodeRequest",
    "VectorEncoding",
    "VectorEncodingSegment",
]
