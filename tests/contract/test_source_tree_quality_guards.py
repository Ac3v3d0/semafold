from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "semafold"
EXPECTED_SOURCE_FILES = {
    "src/semafold/__init__.py",
    "src/semafold/_version.py",
    "src/semafold/errors.py",
    "src/semafold/py.typed",
    "src/semafold/core/__init__.py",
    "src/semafold/core/accounting.py",
    "src/semafold/core/evidence.py",
    "src/semafold/core/models.py",
    "src/semafold/vector/__init__.py",
    "src/semafold/vector/codecs/__init__.py",
    "src/semafold/vector/codecs/passthrough.py",
    "src/semafold/vector/codecs/scalar_reference.py",
    "src/semafold/vector/models.py",
    "src/semafold/vector/protocols.py",
    "src/semafold/turboquant/__init__.py",
    "src/semafold/turboquant/codebook.py",
    "src/semafold/turboquant/codec_mse.py",
    "src/semafold/turboquant/codec_prod.py",
    "src/semafold/turboquant/backends/__init__.py",
    "src/semafold/turboquant/backends/_mlx.py",
    "src/semafold/turboquant/backends/_numpy.py",
    "src/semafold/turboquant/backends/_protocol.py",
    "src/semafold/turboquant/backends/_registry.py",
    "src/semafold/turboquant/backends/_torch.py",
    "src/semafold/turboquant/kv/__init__.py",
    "src/semafold/turboquant/kv/layout.py",
    "src/semafold/turboquant/kv/preview.py",
    "src/semafold/turboquant/packing.py",
    "src/semafold/turboquant/qjl.py",
    "src/semafold/turboquant/quantizer.py",
    "src/semafold/turboquant/rotation.py",
}


def test_source_tree_matches_current_inventory() -> None:
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert actual == EXPECTED_SOURCE_FILES


def test_generated_artifacts_are_not_tracked_in_git() -> None:
    git_exec = shutil.which("git")
    if not git_exec:
        pytest.skip("git executable not found on the system")

    completed = subprocess.run(
        [git_exec, "ls-files", "."],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    forbidden = {
        path
        for path in tracked
        if "__pycache__/" in path or path.startswith(".pytest_cache/") or path.endswith((".pyc", ".pyo"))
    }
    assert forbidden == set()


def test_package_tree_has_no_generated_bytecode_artifacts() -> None:
    # Check only the git-tracked surface: bytecode must not be committed.
    # Runtime __pycache__ produced by the test suite itself is expected and ignored.
    # The companion test `test_generated_artifacts_are_not_tracked_in_git` enforces
    # that none of these artefacts are actually committed to the repository.
    def _is_ignored(p: Path) -> bool:
        return any(part.startswith(".venv") or part == "venv" or part == "__pycache__" for part in p.parts)

    bytecode_files = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py[co]")
        if not _is_ignored(path)
    )
    assert bytecode_files == [], f"committed bytecode files found: {bytecode_files}"


def test_source_tree_has_no_type_ignore_markers() -> None:
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "type: ignore" in text or "pyright: ignore" in text:
            offenders.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert offenders == []
