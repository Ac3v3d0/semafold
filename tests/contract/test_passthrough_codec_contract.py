from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest


def test_passthrough_codec_is_exact() -> None:
    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[1, 2], [3, 4]], dtype=np.int32),
        objective="reconstruction",
    )
    encoding = codec.encode(request)
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    assert np.array_equal(decoded.data, request.data)
    assert encoding.footprint.payload_bytes == request.data.nbytes
