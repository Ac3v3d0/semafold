from __future__ import annotations

import time

import numpy as np

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantProdConfig
from semafold.turboquant import TurboQuantProdVectorCodec


def test_turboquant_prod_encode_decode_perf_smoke() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(64, 256)).astype(np.float32)
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(
            total_bits_per_scalar=3,
            default_rotation_seed=7,
            default_qjl_seed=11,
        )
    )
    request = VectorEncodeRequest(
        data=data,
        objective="inner_product_estimation",
        metric="dot_product_error",
    )

    # Warm caches so the smoke threshold tracks the steady-state hot path.
    codec.encode(request)

    encode_start = time.perf_counter()
    encoding = codec.encode(request)
    encode_seconds = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    decode_seconds = time.perf_counter() - decode_start

    assert np.isfinite(decoded.data).all()
    assert encode_seconds < 1.0
    assert decode_seconds < 0.5
