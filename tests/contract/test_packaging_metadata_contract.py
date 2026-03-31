from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_matches_phase1_packaging_contract() -> None:
    text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    expected_lines = {
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        'name = "semafold"',
        'requires-python = ">=3.10"',
        'dependencies = ["numpy>=1.26"]',
        'dev = ["build>=1.2", "pyright>=1.1", "pytest>=9"]',
        'exclude = [".pytest_cache/**", "**/__pycache__/**", "**/*.py[cod]"]',
        'packages = ["src/semafold"]',
        'pythonVersion = "3.10"',
        'include = ["src", "tests", "examples", "benchmarks"]',
        'typeCheckingMode = "basic"',
    }

    missing = sorted(line for line in expected_lines if line not in text)
    assert missing == []


def test_packaging_docs_match_distribution_and_import_names() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "- distribution: `semafold`" in readme
    assert "- import: `semafold`" in readme
    assert "pip install -e \".[dev]\"" in readme
