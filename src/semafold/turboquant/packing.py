"""Bit-packing helpers shared by TurboQuant payload and sketch segments."""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "pack_scalar_indices",
    "packed_byte_count",
    "unpack_scalar_indices",
]


def _validate_bits_per_index(bits_per_index: int) -> int:
    if not isinstance(bits_per_index, int) or isinstance(bits_per_index, bool):
        raise TypeError("bits_per_index must be an int")
    if bits_per_index < 1 or bits_per_index > 4:
        raise ValueError("bits_per_index must be between 1 and 4 in the current TurboQuant preview")
    return bits_per_index


def packed_byte_count(count: int, bits_per_index: int) -> int:
    """Return the exact byte count needed to pack ``count`` indices."""
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an int")
    if count < 0:
        raise ValueError("count must be >= 0")
    bits_per_index = _validate_bits_per_index(bits_per_index)
    return int(math.ceil((count * bits_per_index) / 8.0))


def pack_scalar_indices(indices: np.ndarray, bits_per_index: int) -> bytes:
    """Pack flat scalar indices into a dense little-endian bitstream."""

    bits_per_index = _validate_bits_per_index(bits_per_index)
    if not isinstance(indices, np.ndarray):
        raise TypeError("indices must be a numpy.ndarray")
    if indices.ndim != 1:
        raise ValueError("indices must be a flat array")
    if indices.size == 0:
        return b""
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must contain integers")
    limit = 1 << bits_per_index
    if np.any(indices < 0) or np.any(indices >= limit):
        raise ValueError("indices contain values outside the supported bit range")

    flat = np.asarray(indices, dtype=np.uint8)
    bit_offsets = np.arange(bits_per_index, dtype=np.uint8)
    bit_matrix = ((flat[:, None] >> bit_offsets) & np.uint8(1)).astype(np.uint8, copy=False)
    bitstream = np.ascontiguousarray(bit_matrix.reshape(-1))
    packed = np.packbits(bitstream, bitorder="little")
    expected_size = packed_byte_count(int(flat.size), bits_per_index)
    return packed[:expected_size].tobytes()


def unpack_scalar_indices(payload: bytes, count: int, bits_per_index: int) -> np.ndarray:
    """Unpack flat scalar indices from a dense little-endian bitstream."""

    bits_per_index = _validate_bits_per_index(bits_per_index)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an int")
    if count < 0:
        raise ValueError("count must be >= 0")
    expected_size = packed_byte_count(count, bits_per_index)
    if len(payload) != expected_size:
        raise ValueError("payload length does not match count and bits_per_index")
    if count == 0:
        return np.empty((0,), dtype=np.uint8)

    raw = np.frombuffer(payload, dtype=np.uint8)
    bitstream = np.unpackbits(raw, bitorder="little", count=count * bits_per_index)
    bit_matrix = np.ascontiguousarray(bitstream.reshape(count, bits_per_index), dtype=np.uint8)
    weights = (1 << np.arange(bits_per_index, dtype=np.uint8)).astype(np.uint8, copy=False)
    decoded = np.sum(bit_matrix * weights.reshape(1, bits_per_index), axis=1, dtype=np.uint16)
    return np.asarray(decoded, dtype=np.uint8)
