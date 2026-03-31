from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _run_example(script_name: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-B", f"examples/{script_name}"],
        cwd=PACKAGE_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(PACKAGE_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "example failed\n"
            f"script: {script_name}\n"
            f"returncode: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def test_wire_roundtrip_example_smoke() -> None:
    output = _run_example("wire_roundtrip.py")
    assert "Semafold wire roundtrip" in output
    assert "lossless roundtrip: yes" in output


def test_turboquant_embedding_example_smoke() -> None:
    output = _run_example("turboquant_embedding.py")
    assert "Semafold TurboQuant embedding example" in output
    assert "compression ratio:" in output
    assert "mse:" in output


def test_turboquant_kv_block_example_smoke() -> None:
    output = _run_example("turboquant_kv_block.py")
    assert "Semafold TurboQuant KV block example" in output
    assert "combined compression ratio:" in output
    assert "smaller vs float32:" in output
