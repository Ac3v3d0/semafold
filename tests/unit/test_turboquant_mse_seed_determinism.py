from __future__ import annotations

import numpy as np

from semafold import EncodeObjective
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec


def _normalized_vector() -> np.ndarray:
    data = np.linspace(-2.0, 2.0, num=32, dtype=np.float32)
    norm = float(np.linalg.norm(data))
    return data if norm == 0.0 else data / norm


def test_turboquant_mse_seeded_encoding_is_deterministic() -> None:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )
    request = VectorEncodeRequest(
        data=_normalized_vector(),
        objective=EncodeObjective.RECONSTRUCTION,
        role="embedding",
        seed=17,
    )

    left = codec.encode(request)
    right = codec.encode(request)

    assert left.config_fingerprint == right.config_fingerprint
    assert left.to_dict() == right.to_dict()


def test_turboquant_mse_seeded_estimate_is_deterministic() -> None:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=17)
    )
    request = VectorEncodeRequest(
        data=_normalized_vector(),
        objective=EncodeObjective.STORAGE_ONLY,
        role="embedding",
        seed=17,
    )

    left = codec.estimate(request)
    right = codec.estimate(request)

    assert left.to_dict() == right.to_dict()
