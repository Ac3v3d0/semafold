from __future__ import annotations

import importlib

import semafold


EXPECTED_PUBLIC_ALL = {
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


def test_turboquant_deep_import_does_not_change_public_root_exports() -> None:
    assert set(semafold.__all__) == EXPECTED_PUBLIC_ALL
    assert not hasattr(semafold, "TurboQuantMSEVectorCodec")
    assert not hasattr(semafold, "TurboQuantProdVectorCodec")
    assert "turboquant" not in semafold.__all__

    module = importlib.import_module("semafold.turboquant")
    assert module.__all__ == [
        "TurboQuantMSEConfig",
        "TurboQuantMSEVectorCodec",
        "TurboQuantProdConfig",
        "TurboQuantProdVectorCodec",
    ]
    assert hasattr(module, "TurboQuantMSEConfig")
    assert hasattr(module, "TurboQuantMSEVectorCodec")
    assert hasattr(module, "TurboQuantProdConfig")
    assert hasattr(module, "TurboQuantProdVectorCodec")

    assert set(semafold.__all__) == EXPECTED_PUBLIC_ALL
    assert not hasattr(semafold, "TurboQuantMSEVectorCodec")
    assert not hasattr(semafold, "TurboQuantProdVectorCodec")
    assert "turboquant" not in semafold.__all__
