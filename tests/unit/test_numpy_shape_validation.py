from __future__ import annotations

import numpy as np
import pytest

from semafold import EncodeObjective
from semafold import VectorEncodeRequest


def test_rank_1_and_rank_2_are_accepted() -> None:
    VectorEncodeRequest(data=np.array([1.0, 2.0], dtype=np.float32), objective=EncodeObjective.RECONSTRUCTION)
    VectorEncodeRequest(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        objective=EncodeObjective.RECONSTRUCTION,
    )


def test_rank_3_is_rejected() -> None:
    with pytest.raises(ValueError):
        VectorEncodeRequest(
            data=np.zeros((1, 2, 3), dtype=np.float32),
            objective=EncodeObjective.RECONSTRUCTION,
        )
