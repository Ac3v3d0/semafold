"""Stable-root smoke example for Semafold's exact passthrough path."""

from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold import VectorEncoding
from semafold.vector.models import EncodeObjective


def _format_bytes(value: int) -> str:
    return f"{value:,} B"


def _format_summary(*, codec_family: str, variant_id: str, baseline_bytes: int, artifact_bytes: int) -> str:
    delta = artifact_bytes - baseline_bytes
    return "\n".join(
        [
            "Semafold wire roundtrip",
            f"codec: {codec_family}/{variant_id}",
            f"baseline bytes: {_format_bytes(baseline_bytes)}",
            f"artifact bytes: {_format_bytes(artifact_bytes)}",
            f"bytes delta: {delta:+,d} B",
        ]
    )


def main() -> None:
    """Run a lossless wire-roundtrip smoke test against the stable root surface."""

    codec = PassthroughVectorCodec()
    request = VectorEncodeRequest(
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        objective=EncodeObjective.RECONSTRUCTION,
        role="embedding",
        profile_id="examples.wire_roundtrip",
    )
    encoded = codec.encode(request)
    encoding = VectorEncoding.from_dict(encoded.to_dict())
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    if not np.array_equal(decoded.data, request.data):
        raise SystemExit("round-trip mismatch")
    print(
        _format_summary(
            codec_family=encoding.codec_family,
            variant_id=encoding.variant_id,
            baseline_bytes=int(request.data.nbytes),
            artifact_bytes=int(encoding.footprint.total_bytes),
        )
    )
    print("lossless roundtrip: yes")
    print(f"segment kinds: {[segment.segment_kind for segment in encoding.segments]}")


if __name__ == "__main__":
    main()
