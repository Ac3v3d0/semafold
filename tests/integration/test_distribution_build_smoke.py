from __future__ import annotations

import subprocess
import sys
import venv
import zipfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(venv_root: Path) -> Path:
    if sys.platform == "win32":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    import os
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "command failed\n"
            f"cwd: {cwd}\n"
            f"command: {command}\n"
            f"returncode: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def test_wheel_build_smoke(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    completed = _run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(dist_dir),
        ],
        cwd=PACKAGE_ROOT,
    )

    assert "Successfully built semafold" in completed.stdout + completed.stderr
    wheels = sorted(dist_dir.glob("semafold-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert "semafold/__init__.py" in names
    assert "semafold/py.typed" in names
    assert "semafold/turboquant/kv/__init__.py" in names
    assert not any("__pycache__/" in name or name.endswith((".pyc", ".pyo")) for name in names)
    assert not any(".pytest_cache/" in name for name in names)


def test_editable_install_smoke(tmp_path: Path) -> None:
    venv_root = tmp_path / "venv"
    # Keep this venv isolated so the smoke test validates the editable install
    # rather than accidentally importing an ambient system package.
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(venv_root)
    python = _venv_python(venv_root)

    _run_checked([str(python), "-m", "pip", "install", "-e", "."], cwd=PACKAGE_ROOT)

    _run_checked(
        [
            str(python),
            "-c",
            "\n".join(
                [
                    "import semafold",
                    "assert 'PassthroughVectorCodec' in semafold.__all__",
                    "assert 'ScalarReferenceVectorCodec' not in semafold.__all__",
                ]
            ),
        ],
        cwd=tmp_path,
    )
