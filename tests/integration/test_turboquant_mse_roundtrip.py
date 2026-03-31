from __future__ import annotations

import numpy as np
import pytest

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.vector.models import VectorEncoding
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec


@pytest.mark.parametrize(
    "data",
    [
        np.linspace(-1.0, 1.0, num=32, dtype=np.float32),
        np.linspace(-1.0, 1.0, num=3 * 16, dtype=np.float32).reshape(3, 16),
    ],
)
def test_turboquant_mse_consumer_wire_roundtrip_preserves_shape_profile_and_finite_decode(data: np.ndarray) -> None:
    request = VectorEncodeRequest(
        data=data,
        objective="reconstruction",
        metric="mse",
        role="embedding",
        seed=11,
        profile_id="turboquant.mse.integration",
    )
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=0)
    )
    encoding = codec.encode(request)
    reloaded = VectorEncoding.from_dict(encoding.to_dict())
    decoded = codec.decode(VectorDecodeRequest(encoding=reloaded))

    assert decoded.data.shape == data.shape
    assert decoded.data.dtype == data.dtype
    assert np.isfinite(decoded.data).all()
    assert reloaded.codec_family == encoding.codec_family
    assert reloaded.variant_id == encoding.variant_id
    assert reloaded.profile_id == "turboquant.mse.integration"
    mse = float(np.mean((data.astype(np.float64) - decoded.data.astype(np.float64)) ** 2))
    assert mse >= 0.0


def test_turboquant_mse_roundtrip_serialization_is_seed_deterministic() -> None:
    data = np.array(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            [-0.3, 0.2, 0.1, -0.5, 0.7, 0.4],
        ],
        dtype=np.float32,
    )
    request = VectorEncodeRequest(data=data, objective="reconstruction", metric="mse", seed=11)
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=0)
    )
    first = codec.encode(request)
    second = codec.encode(request)

    assert first.config_fingerprint == second.config_fingerprint
    assert first.to_dict() == second.to_dict()


def test_turboquant_more_bits_improves_observed_mse_on_fixed_input() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(6, 32)).astype(np.float32)
    request = VectorEncodeRequest(data=data, objective="reconstruction", metric="mse", seed=5)

    low = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=1)
    ).encode(request)
    high = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=4)
    ).encode(request)

    low_mse = next(guarantee.value for guarantee in low.guarantees if guarantee.metric == "observed_mse")
    high_mse = next(guarantee.value for guarantee in high.guarantees if guarantee.metric == "observed_mse")
    assert isinstance(low_mse, float)
    assert isinstance(high_mse, float)
    assert high_mse < low_mse
