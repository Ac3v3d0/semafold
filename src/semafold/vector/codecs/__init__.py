"""Concrete codecs available inside the vector package."""

from semafold.vector.codecs.passthrough import PassthroughVectorCodec
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec

__all__ = ["PassthroughVectorCodec", "ScalarReferenceVectorCodec"]
