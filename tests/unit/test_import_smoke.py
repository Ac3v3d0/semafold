from __future__ import annotations

import semafold


def test_import_smoke() -> None:
    assert semafold.__version__ == "0.1.0"
