from __future__ import annotations

import importlib

import semafold


EXPECTED_ROOT_ALL = {
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

EXPECTED_TURBOQUANT_ALL = [
    "TurboQuantMSEConfig",
    "TurboQuantMSEVectorCodec",
    "TurboQuantProdConfig",
    "TurboQuantProdVectorCodec",
]


def test_kv_preview_deep_import_keeps_root_and_turboquant_boundaries_stable() -> None:
    turboquant_module = importlib.import_module("semafold.turboquant")

    assert set(semafold.__all__) == EXPECTED_ROOT_ALL
    assert turboquant_module.__all__ == EXPECTED_TURBOQUANT_ALL
    assert not hasattr(semafold, "TurboQuantKVCacheArtifact")
    assert not hasattr(semafold, "TurboQuantKVConfig")
    assert not hasattr(semafold, "TurboQuantKVPreviewCodec")
    assert not hasattr(turboquant_module, "TurboQuantKVCacheArtifact")
    assert not hasattr(turboquant_module, "TurboQuantKVConfig")
    assert not hasattr(turboquant_module, "TurboQuantKVPreviewCodec")

    kv_module = importlib.import_module("semafold.turboquant.kv")

    assert kv_module.__all__ == ["TurboQuantKVConfig", "TurboQuantKVPreviewCodec"]

    assert set(semafold.__all__) == EXPECTED_ROOT_ALL
    assert turboquant_module.__all__ == EXPECTED_TURBOQUANT_ALL
    assert not hasattr(semafold, "TurboQuantKVCacheArtifact")
    assert not hasattr(semafold, "TurboQuantKVConfig")
    assert not hasattr(semafold, "TurboQuantKVPreviewCodec")
    assert not hasattr(turboquant_module, "TurboQuantKVCacheArtifact")
    assert not hasattr(turboquant_module, "TurboQuantKVConfig")
    assert not hasattr(turboquant_module, "TurboQuantKVPreviewCodec")
