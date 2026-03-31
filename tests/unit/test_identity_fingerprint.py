from __future__ import annotations

from semafold.vector.models import fingerprint_config


def test_fingerprint_is_key_order_invariant() -> None:
    left = fingerprint_config({"a": 1, "b": 2})
    right = fingerprint_config({"b": 2, "a": 1})
    assert left == right
