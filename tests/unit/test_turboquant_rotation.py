from __future__ import annotations

import numpy as np

from semafold.turboquant.rotation import seeded_haar_rotation


def test_seeded_haar_rotation_is_deterministic_and_orthogonal() -> None:
    first = seeded_haar_rotation(8, 123)
    second = seeded_haar_rotation(8, 123)
    third = seeded_haar_rotation(8, 124)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)

    gram = first.T @ first
    assert np.allclose(gram, np.eye(8, dtype=np.float32), atol=1e-4)
