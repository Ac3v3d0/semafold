from __future__ import annotations

import numpy as np

from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec


def test_constant_row_reconstructs_exactly() -> None:
    codec = ScalarReferenceVectorCodec()
    row = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    quantized, mins, maxs = codec._quantize_rows(row)
    restored = codec._reconstruct(quantized, mins, maxs)
    assert np.allclose(restored, row)
