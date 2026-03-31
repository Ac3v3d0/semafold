from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_turboquant_synthetic_kv_runner_emits_deterministic_smoke_metrics(tmp_path: Path) -> None:
    module = _load_module(
        PACKAGE_ROOT / "benchmarks" / "turboquant_synthetic_kv_benchmark.py",
        "turboquant_synthetic_kv_benchmark",
    )

    output_path = tmp_path / "synthetic_kv.json"
    result = module.run_synthetic_kv_benchmark(
        layers=2,
        heads=2,
        seq_len=64,
        head_dim=64,
        query_count=16,
        output_path=output_path,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == result
    assert result["kind"] == "turboquant_synthetic_kv_benchmark"

    parameters = result["parameters"]
    expected_rows = parameters["layers"] * parameters["heads"] * parameters["seq_len"]
    assert result["keys"]["row_count"] == expected_rows
    assert result["values"]["row_count"] == expected_rows
    assert result["keys"]["dimension"] == parameters["head_dim"]
    assert result["values"]["dimension"] == parameters["head_dim"]

    assert result["combined"]["compression_ratio"] > 1.0
    assert result["keys"]["compression_ratio"] > 1.0
    assert result["values"]["compression_ratio"] > 1.0
    assert result["keys"]["mean_abs_error"] < 1.0
    assert result["values"]["observed_mse"] < 1.0

    for section in ("keys", "values"):
        for metric_name in ("payload_bytes", "sidecar_bytes", "total_bytes", "compression_ratio", "encode_seconds", "decode_seconds"):
            metric_value = result[section][metric_name]
            assert math.isfinite(metric_value)
            assert metric_value >= 0.0

    rerun = module.run_synthetic_kv_benchmark(
        layers=2,
        heads=2,
        seq_len=64,
        head_dim=64,
        query_count=16,
    )
    for section in ("keys", "values", "combined"):
        left = dict(result[section])
        right = dict(rerun[section])
        left.pop("encode_seconds", None)
        left.pop("decode_seconds", None)
        right.pop("encode_seconds", None)
        right.pop("decode_seconds", None)
        assert left == right
