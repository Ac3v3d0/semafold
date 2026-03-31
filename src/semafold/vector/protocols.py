"""Stable protocol definitions for Semafold vector codecs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semafold.core import CompressionEstimate
from semafold.vector.models import VectorDecodeRequest, VectorDecodeResult, VectorEncodeRequest, VectorEncoding

__all__ = ["VectorCodec"]


@runtime_checkable
class VectorCodec(Protocol):
    """Stable protocol implemented by vector codecs exposed by Semafold."""

    def estimate(self, request: VectorEncodeRequest) -> CompressionEstimate:
        """Return a pre-run size estimate for ``request`` without emitting an artifact."""
        ...

    def encode(self, request: VectorEncodeRequest) -> VectorEncoding:
        """Encode ``request`` and return a measured artifact envelope."""
        ...

    def decode(self, request: VectorDecodeRequest) -> VectorDecodeResult:
        """Materialize a ``VectorEncoding`` back into NumPy data."""
        ...
