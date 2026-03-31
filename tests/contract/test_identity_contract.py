from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorEncodeRequest


def test_identity_fields_are_deterministic() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective="reconstruction")
    left = codec.encode(request)
    right = codec.encode(request)
    assert left.codec_family == "passthrough"
    assert left.variant_id == "raw_bytes_v1"
    assert left.encoding_schema_version == right.encoding_schema_version
    assert left.config_fingerprint == right.config_fingerprint
