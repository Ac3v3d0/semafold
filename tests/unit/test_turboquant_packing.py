from __future__ import annotations

import numpy as np

from semafold.turboquant.packing import pack_scalar_indices, packed_byte_count, unpack_scalar_indices


def test_turboquant_bit_packing_roundtrip_for_small_widths() -> None:
    rng = np.random.default_rng(42)
    for bits in (1, 2, 3, 4):
        values = rng.integers(0, 1 << bits, size=73, dtype=np.uint8)
        packed = pack_scalar_indices(values, bits)
        unpacked = unpack_scalar_indices(packed, count=73, bits_per_index=bits)
        assert unpacked.dtype == np.uint8
        assert np.array_equal(unpacked, values)
        assert len(packed) == packed_byte_count(73, bits)
