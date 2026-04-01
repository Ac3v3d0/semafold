from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorEncodeRequest
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec
from semafold.vector.models import EncodeObjective


def test_passthrough_footprint_golden() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective=EncodeObjective.RECONSTRUCTION)
    encoding = codec.encode(request)
    assert encoding.footprint.to_dict() == {
        "baseline_bytes": 8,
        "payload_bytes": 8,
        "metadata_bytes": 95,
        "sidecar_bytes": 0,
        "protected_passthrough_bytes": 0,
        "decoder_state_bytes": 0,
        "total_bytes": 103,
        "bytes_saved": -95,
        "compression_ratio": 0.07766990291262135,
    }
    assert [segment.footprint for segment in encoding.segments] == [
        {
            "payload_bytes": 8,
            "metadata_bytes": 0,
            "sidecar_bytes": 0,
            "protected_passthrough_bytes": 0,
            "decoder_state_bytes": 0,
            "total_bytes": 8,
        },
        {
            "payload_bytes": 0,
            "metadata_bytes": 95,
            "sidecar_bytes": 0,
            "protected_passthrough_bytes": 0,
            "decoder_state_bytes": 0,
            "total_bytes": 95,
        },
    ]


def test_scalar_reference_footprint_golden() -> None:
    codec = ScalarReferenceVectorCodec()
    request = VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective=EncodeObjective.RECONSTRUCTION)
    encoding = codec.encode(request)
    assert encoding.footprint.to_dict() == {
        "baseline_bytes": 8,
        "payload_bytes": 2,
        "metadata_bytes": 103,
        "sidecar_bytes": 8,
        "protected_passthrough_bytes": 0,
        "decoder_state_bytes": 0,
        "total_bytes": 113,
        "bytes_saved": -105,
        "compression_ratio": 0.07079646017699115,
    }
    assert [segment.footprint for segment in encoding.segments] == [
        {
            "payload_bytes": 2,
            "metadata_bytes": 0,
            "sidecar_bytes": 0,
            "protected_passthrough_bytes": 0,
            "decoder_state_bytes": 0,
            "total_bytes": 2,
        },
        {
            "payload_bytes": 0,
            "metadata_bytes": 0,
            "sidecar_bytes": 8,
            "protected_passthrough_bytes": 0,
            "decoder_state_bytes": 0,
            "total_bytes": 8,
        },
        {
            "payload_bytes": 0,
            "metadata_bytes": 103,
            "sidecar_bytes": 0,
            "protected_passthrough_bytes": 0,
            "decoder_state_bytes": 0,
            "total_bytes": 103,
        },
    ]
