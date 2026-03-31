from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_timings(records: list[dict[str, object]]) -> list[dict[str, object]]:
    stripped: list[dict[str, object]] = []
    for record in records:
        copied = dict(record)
        copied.pop("encode_seconds", None)
        copied.pop("decode_seconds", None)
        stripped.append(copied)
    return stripped


def test_turboquant_paper_validation_runner_emits_deterministic_json_and_trends(tmp_path: Path) -> None:
    module = _load_module(
        PACKAGE_ROOT / "benchmarks" / "turboquant_paper_validation.py",
        "turboquant_paper_validation_benchmark",
    )

    output_path = tmp_path / "paper_validation.json"
    result = module.run_paper_validation(
        dimensions=(32, 64),
        vector_count=64,
        query_count=24,
        output_path=output_path,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == result
    assert result["kind"] == "turboquant_paper_validation"

    mse_records = result["mse"]
    prod_records = result["prod"]
    assert len(mse_records) == 8
    assert len(prod_records) == 6

    for dimension in (32, 64):
        by_bits = {
            record["bits_per_scalar"]: record
            for record in mse_records
            if record["dimension"] == dimension
        }
        assert by_bits[1]["observed_mse"] > by_bits[4]["observed_mse"]
        assert by_bits[1]["compression_ratio"] > by_bits[4]["compression_ratio"]

        prod_by_bits = {
            record["total_bits_per_scalar"]: record
            for record in prod_records
            if record["dimension"] == dimension
        }
        assert prod_by_bits[2]["mean_abs_error"] > prod_by_bits[4]["mean_abs_error"]
        assert prod_by_bits[2]["empirical_variance"] > prod_by_bits[4]["empirical_variance"]
        assert prod_by_bits[2]["theory_proxy"] > prod_by_bits[4]["theory_proxy"]

    rerun = module.run_paper_validation(
        dimensions=(32, 64),
        vector_count=64,
        query_count=24,
    )
    assert _strip_timings(rerun["mse"]) == _strip_timings(result["mse"])
    assert _strip_timings(rerun["prod"]) == _strip_timings(result["prod"])
