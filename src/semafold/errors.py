"""Public error hierarchy for Semafold."""

from __future__ import annotations

__all__ = [
    "CompressionError",
    "CompatibilityError",
    "DecodeError",
    "ValidationError",
]


class CompressionError(Exception):
    """Base exception for Semafold compression failures."""


class CompatibilityError(CompressionError):
    """Raised for unsupported inputs or incompatible configurations."""


class DecodeError(CompressionError):
    """Raised when an encoding cannot be decoded safely."""


class ValidationError(CompressionError):
    """Raised when invariants or validation evidence are invalid."""
