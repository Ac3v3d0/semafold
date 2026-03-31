from __future__ import annotations

import semafold


def test_public_api_exports_snapshot() -> None:
    assert set(semafold.__all__) == {
        "__version__",
        "CompressionBudget",
        "CompressionEstimate",
        "CompressionFootprint",
        "CompressionGuarantee",
        "PassthroughVectorCodec",
        "ValidationEvidence",
        "VectorCodec",
        "VectorDecodeRequest",
        "VectorDecodeResult",
        "VectorEncodeRequest",
        "VectorEncoding",
        "VectorEncodingSegment",
    }
