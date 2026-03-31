from __future__ import annotations

import json
from typing import TypeAlias

import numpy as np
import pytest

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec
from semafold.vector.models import VectorEncoding
from semafold.vector.models import VectorEncodingSegment

CodecUnderTest: TypeAlias = PassthroughVectorCodec | ScalarReferenceVectorCodec


@pytest.mark.parametrize(
    ("codec", "encode_request", "expected_segment_kinds", "exact_roundtrip"),
    [
        (
            PassthroughVectorCodec(),
            VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective="reconstruction"),
            {"passthrough", "metadata"},
            True,
        ),
        (
            ScalarReferenceVectorCodec(),
            VectorEncodeRequest(
                data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                objective="reconstruction",
            ),
            {"compressed", "sidecar", "metadata"},
            False,
        ),
    ],
)
def test_consumer_can_transport_inspect_and_decode_vector_encoding(
    codec: CodecUnderTest,
    encode_request: VectorEncodeRequest,
    expected_segment_kinds: set[str],
    exact_roundtrip: bool,
) -> None:
    encoding = codec.encode(encode_request)
    wire = json.loads(json.dumps(encoding.to_dict()))

    assert wire["codec_family"] == encoding.codec_family
    assert wire["variant_id"] == encoding.variant_id
    assert wire["encoding_schema_version"] == encoding.encoding_schema_version
    assert wire["footprint"]["total_bytes"] == encoding.footprint.total_bytes

    wire_segments = {segment["segment_kind"]: segment for segment in wire["segments"]}
    assert set(wire_segments) == expected_segment_kinds
    assert wire_segments["metadata"]["payload"]["kind"] == "json"
    assert wire_segments["metadata"]["scope"]["kind"] == "encoding_metadata"
    for segment_kind, segment in wire_segments.items():
        assert isinstance(segment["payload_format"], str)
        assert isinstance(segment["scope"], dict)
        if segment_kind == "metadata":
            continue
        assert segment["payload"]["kind"] == "bytes"
        assert isinstance(segment["payload"]["base64"], str)

    restored = VectorEncoding.from_dict(wire)
    assert restored.to_dict() == encoding.to_dict()
    assert restored.metadata["objective"] == encode_request.objective

    restored_segments = {segment.segment_kind: segment for segment in restored.segments}
    assert set(restored_segments) == expected_segment_kinds
    assert isinstance(restored_segments["metadata"].payload, dict)
    for segment_kind, segment in restored_segments.items():
        if segment_kind == "metadata":
            continue
        assert isinstance(segment.payload, bytes)

    decoded = codec.decode(VectorDecodeRequest(encoding=restored))
    assert decoded.data.shape == encode_request.data.shape
    if exact_roundtrip:
        assert np.array_equal(decoded.data, encode_request.data)
    else:
        assert decoded.data.dtype == encode_request.data.dtype


def test_vector_encoding_from_dict_rejects_non_string_identity_fields() -> None:
    codec = PassthroughVectorCodec()
    encoding = codec.encode(VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective="reconstruction"))
    wire = encoding.to_dict()
    wire["codec_family"] = 123

    with pytest.raises(TypeError):
        VectorEncoding.from_dict(wire)


def test_vector_segment_from_dict_rejects_non_string_required_fields() -> None:
    segment = VectorEncodingSegment(
        segment_kind="metadata",
        role=None,
        scope={"kind": "encoding_metadata"},
        payload={"shape": [2]},
        payload_format="json",
    )
    wire = segment.to_dict()
    wire["payload_format"] = 123

    with pytest.raises(TypeError):
        VectorEncodingSegment.from_dict(wire)
