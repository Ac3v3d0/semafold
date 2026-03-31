from __future__ import annotations

from pathlib import Path


def test_reserved_modules_absent() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "semafold"
    forbidden = [
        root / "text",
        root / "runtime",
        root / "diagnostics",
        root / "experimental",
        root / "core" / "serde.py",
        root / "core" / "hashing.py",
        root / "core" / "types.py",
        root / "vector" / "segments.py",
        root / "vector" / "testing.py",
    ]
    assert all(not path.exists() for path in forbidden)
