from __future__ import annotations

import numpy as np
import pytest

from semafold import EncodeObjective, EncodingSegmentKind
from semafold.errors import DecodeError
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest


def test_scalar_reference_codec_emits_lossy_valid_encoding() -> None:
    codec = ScalarReferenceVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        objective=EncodeObjective.RECONSTRUCTION,
    )
    encoding = codec.encode(request)
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    assert decoded.data.shape == request.data.shape
    assert encoding.variant_id == "uniform_affine_u8_v1"
    assert encoding.footprint.sidecar_bytes > 0
    assert any(e.scope == "proxy_fidelity" for e in encoding.evidence)


def test_scalar_reference_decode_rejects_non_finite_sidecar() -> None:
    codec = ScalarReferenceVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        objective=EncodeObjective.RECONSTRUCTION,
    )
    encoding = codec.encode(request)
    sidecar_segment = next(segment for segment in encoding.segments if segment.segment_kind == EncodingSegmentKind.SIDECAR)
    sidecar_segment.payload = np.array(
        [[np.nan, np.inf], [np.nan, np.inf]],
        dtype=np.float32,
    ).tobytes()
    with pytest.raises(DecodeError):
        codec.decode(VectorDecodeRequest(encoding=encoding))
