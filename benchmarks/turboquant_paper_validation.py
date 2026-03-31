from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from semafold import __version__
from semafold import VectorDecodeRequest, VectorEncodeRequest
from semafold.turboquant import (
    TurboQuantMSEConfig,
    TurboQuantMSEVectorCodec,
    TurboQuantProdConfig,
    TurboQuantProdVectorCodec,
)


def _unit_rows(*, vector_count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((vector_count, dimension), dtype=np.float32)
    norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(rows / norms, dtype=np.float32)


def _find_metric(encoding_metrics: Sequence[dict[str, object]] | Sequence[object], scope: str, metric: str) -> float:
    for evidence in encoding_metrics:
        current_scope = getattr(evidence, "scope", None)
        current_metrics = getattr(evidence, "metrics", None)
        if current_scope == scope and isinstance(current_metrics, dict) and metric in current_metrics:
            value = current_metrics[metric]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    raise KeyError(f"missing metric {metric!r} in scope {scope!r}")


def _mse_record(*, rows: np.ndarray, bits_per_scalar: int, rotation_seed: int) -> dict[str, object]:
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(
            default_bits_per_scalar=bits_per_scalar,
            default_rotation_seed=rotation_seed,
        )
    )
    request = VectorEncodeRequest(data=rows, objective="reconstruction", metric="mse")

    encode_start = time.perf_counter()
    encoding = codec.encode(request)
    encode_seconds = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding)).data
    decode_seconds = time.perf_counter() - decode_start

    observed_mse = float(np.mean(np.square(decoded.astype(np.float64) - rows.astype(np.float64))))
    return {
        "dimension": int(rows.shape[1]),
        "vector_count": int(rows.shape[0]),
        "bits_per_scalar": int(bits_per_scalar),
        "observed_mse": observed_mse,
        "compression_ratio": float(encoding.footprint.compression_ratio),
        "payload_bytes": int(encoding.footprint.payload_bytes),
        "sidecar_bytes": int(encoding.footprint.sidecar_bytes),
        "total_bytes": int(encoding.footprint.total_bytes),
        "encode_seconds": float(encode_seconds),
        "decode_seconds": float(decode_seconds),
        "guarantee_value": float(encoding.guarantees[0].value) if isinstance(encoding.guarantees[0].value, (int, float)) else None,
    }


def _prod_record(
    *,
    rows: np.ndarray,
    queries: np.ndarray,
    total_bits_per_scalar: int,
    rotation_seed: int,
    qjl_seed: int,
) -> dict[str, object]:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(
            total_bits_per_scalar=total_bits_per_scalar,
            default_rotation_seed=rotation_seed,
            default_qjl_seed=qjl_seed,
        )
    )
    request = VectorEncodeRequest(
        data=rows,
        objective="inner_product_estimation",
        metric="dot_product_error",
    )

    encode_start = time.perf_counter()
    encoding = codec.encode(request)
    encode_seconds = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding)).data
    decode_seconds = time.perf_counter() - decode_start

    exact_scores = queries.astype(np.float64) @ rows.astype(np.float64).T
    approx_scores = queries.astype(np.float64) @ decoded.astype(np.float64).T
    errors = approx_scores - exact_scores

    return {
        "dimension": int(rows.shape[1]),
        "vector_count": int(rows.shape[0]),
        "query_count": int(queries.shape[0]),
        "total_bits_per_scalar": int(total_bits_per_scalar),
        "base_bits_per_scalar": int(total_bits_per_scalar - 1),
        "mean_error": float(np.mean(errors)),
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "empirical_variance": float(np.var(errors)),
        "theory_proxy": _find_metric(encoding.evidence, "theory_proxy", "mean_query_free_variance_factor"),
        "compression_ratio": float(encoding.footprint.compression_ratio),
        "payload_bytes": int(encoding.footprint.payload_bytes),
        "sidecar_bytes": int(encoding.footprint.sidecar_bytes),
        "total_bytes": int(encoding.footprint.total_bytes),
        "encode_seconds": float(encode_seconds),
        "decode_seconds": float(decode_seconds),
    }


def run_paper_validation(
    *,
    dimensions: Sequence[int] = (32, 64, 128),
    vector_count: int = 128,
    query_count: int = 32,
    mse_bits: Sequence[int] = (1, 2, 3, 4),
    prod_total_bits: Sequence[int] = (2, 3, 4),
    data_seed: int = 123,
    query_seed: int = 456,
    rotation_seed: int = 7,
    qjl_seed: int = 11,
    output_path: Path | None = None,
) -> dict[str, object]:
    mse_records: list[dict[str, object]] = []
    prod_records: list[dict[str, object]] = []

    for dimension in dimensions:
        rows = _unit_rows(vector_count=vector_count, dimension=int(dimension), seed=data_seed + int(dimension))
        queries = _unit_rows(vector_count=query_count, dimension=int(dimension), seed=query_seed + int(dimension))
        for bits_per_scalar in mse_bits:
            mse_records.append(
                _mse_record(rows=rows, bits_per_scalar=int(bits_per_scalar), rotation_seed=rotation_seed)
            )
        for total_bits_per_scalar in prod_total_bits:
            prod_records.append(
                _prod_record(
                    rows=rows,
                    queries=queries,
                    total_bits_per_scalar=int(total_bits_per_scalar),
                    rotation_seed=rotation_seed,
                    qjl_seed=qjl_seed,
                )
            )

    result: dict[str, object] = {
        "kind": "turboquant_paper_validation",
        "implementation_version": __version__,
        "parameters": {
            "dimensions": [int(dimension) for dimension in dimensions],
            "vector_count": int(vector_count),
            "query_count": int(query_count),
            "mse_bits": [int(bits) for bits in mse_bits],
            "prod_total_bits": [int(bits) for bits in prod_total_bits],
            "data_seed": int(data_seed),
            "query_seed": int(query_seed),
            "rotation_seed": int(rotation_seed),
            "qjl_seed": int(qjl_seed),
        },
        "mse": mse_records,
        "prod": prod_records,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic TurboQuant paper-shaped validation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    result = run_paper_validation(output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
