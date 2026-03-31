from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

import pytest


def _resolve_pyright(test_root: Path) -> str | None:
    candidate = shutil.which("pyright")
    if candidate is not None:
        return candidate
    for parent in (test_root, *test_root.parents):
        local = parent / ".venv" / "bin" / "pyright"
        if local.exists():
            return str(local)
    return None


def test_public_imports_typecheck(tmp_path: Path) -> None:
    test_root = Path(__file__).resolve().parents[2]
    pyright = _resolve_pyright(test_root)
    if pyright is None:
        pytest.skip("pyright is not installed")
    snippet = tmp_path / "public_imports.py"
    snippet.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "from semafold import (",
                "    CompressionBudget,",
                "    CompressionEstimate,",
                "    CompressionFootprint,",
                "    CompressionGuarantee,",
                "    PassthroughVectorCodec,",
                "    ValidationEvidence,",
                "    VectorCodec,",
                "    VectorDecodeRequest,",
                "    VectorDecodeResult,",
                "    VectorEncodeRequest,",
                "    VectorEncoding,",
                "    VectorEncodingSegment,",
                ")",
                "",
                "codec: VectorCodec = PassthroughVectorCodec()",
                "budget = CompressionBudget(target_bytes=16, allow_passthrough=True)",
                "request = VectorEncodeRequest(",
                '    data=np.array([[1.0, 2.0]], dtype=np.float32),',
                '    objective=\"reconstruction\",',
                "    budget=budget,",
                ")",
                "estimate = codec.estimate(request)",
                "encoding = codec.encode(request)",
                "decoded = codec.decode(VectorDecodeRequest(encoding=encoding))",
                "guarantee = CompressionGuarantee(",
                '    objective=\"reconstruction\",',
                '    metric=\"exact_roundtrip\",',
                '    bound_type=\"exact\",',
                "    value=True,",
                ")",
                "evidence = ValidationEvidence(",
                '    scope=\"typing_smoke\",',
                "    environment={},",
                "    metrics={\"ok\": True},",
                "    artifact_refs=[],",
                ")",
                "segment: VectorEncodingSegment = encoding.segments[0]",
                "footprint: CompressionFootprint = encoding.footprint",
                "typed_encoding: VectorEncoding = encoding",
                "typed_decode: VectorDecodeResult = decoded",
                "typed_estimate: CompressionEstimate = estimate",
                "assert segment.segment_kind",
                "assert footprint.total_bytes >= 0",
                "assert typed_encoding.codec_family == 'passthrough'",
                "assert typed_decode.data.shape == request.data.shape",
                "assert typed_estimate.estimated_total_bytes is not None",
                "assert guarantee.value is True",
                "assert evidence.metrics['ok'] is True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [pyright, "--project", str(test_root / "pyproject.toml"), str(snippet)],
        cwd=test_root,
        env={**os.environ, "PYTHONPATH": str(test_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
