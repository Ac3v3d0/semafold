from __future__ import annotations

import shutil
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"


def _cleanup_generated_artifacts() -> None:
    shutil.rmtree(PACKAGE_ROOT / ".pytest_cache", ignore_errors=True)
    for pycache_dir in PACKAGE_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache_dir, ignore_errors=True)
    for compiled in PACKAGE_ROOT.rglob("*.py[co]"):
        compiled.unlink(missing_ok=True)


sys.dont_write_bytecode = True
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_sessionstart(session) -> None:
    _cleanup_generated_artifacts()


def pytest_sessionfinish(session, exitstatus: int) -> None:
    _cleanup_generated_artifacts()
