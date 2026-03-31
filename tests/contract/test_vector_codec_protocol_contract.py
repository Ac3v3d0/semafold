from __future__ import annotations

import inspect

from semafold import PassthroughVectorCodec
from semafold import VectorCodec


def test_passthrough_matches_protocol_shape() -> None:
    codec = PassthroughVectorCodec()
    assert isinstance(codec, VectorCodec)
    for method_name in ("estimate", "encode", "decode"):
        assert callable(getattr(codec, method_name))
        signature = inspect.signature(getattr(codec, method_name))
        assert "request" in signature.parameters
