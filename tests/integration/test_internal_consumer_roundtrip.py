from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold import VectorEncoding


def _persist_encoding_roundtrip(encoding: VectorEncoding) -> VectorEncoding:
    """Simulate a consumer persisting and reloading a Semafold envelope."""

    return VectorEncoding.from_dict(encoding.to_dict())


def test_internal_consumer_roundtrip_uses_stable_root_exports_only() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        objective="reconstruction",
        role="embedding",
        component_id="consumer.embedding_batch",
        profile_id="phase1.public_smoke",
    )
    encoding = _persist_encoding_roundtrip(codec.encode(request))
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

    assert np.array_equal(decoded.data, request.data)
    assert encoding.profile_id == "phase1.public_smoke"
    assert encoding.encoding_schema_version == "vector.encoding.v1"
    assert [segment.segment_kind for segment in encoding.segments] == ["passthrough", "metadata"]
    assert {item.scope for item in encoding.evidence} == {"compatibility", "storage_accounting"}
