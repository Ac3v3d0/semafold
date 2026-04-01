from __future__ import annotations

import numpy as np

from semafold import EncodeObjective
from semafold import PassthroughVectorCodec
from semafold import VectorEncodeRequest


def test_vector_encoding_envelope_invariants() -> None:
    codec = PassthroughVectorCodec()
    encoding = codec.encode(
        VectorEncodeRequest(
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            objective=EncodeObjective.RECONSTRUCTION,
        )
    )
    assert encoding.segments
    assert encoding.guarantees
    assert encoding.evidence
    assert encoding.footprint.total_bytes > 0
