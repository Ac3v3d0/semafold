from __future__ import annotations

import semafold


def test_beta_or_future_symbols_not_root_exported() -> None:
    assert "turboquant" not in semafold.__all__
    for symbol in (
        "TurboQuantMSEVectorCodec",
        "TurboQuantProdVectorCodec",
        "ScalarReferenceVectorCodec",
        "TextCompressor",
        "VectorQueryable",
        "AsyncVectorCodec",
    ):
        assert not hasattr(semafold, symbol)
