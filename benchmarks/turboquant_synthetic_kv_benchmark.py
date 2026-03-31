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


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def generate_synthetic_kv(
    *,
    layers: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
    value_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((layers, heads, seq_len, head_dim), dtype=np.float32)
    values = rng.standard_normal((layers, heads, seq_len, head_dim), dtype=np.float32) * np.float32(value_scale)
    return _normalize_last_axis(keys), np.asarray(values, dtype=np.float32)


def flatten_cache_rows(cache: np.ndarray) -> np.ndarray:
    if cache.ndim != 4:
        raise ValueError("cache must have shape (layers, heads, seq_len, head_dim)")
    layers, heads, seq_len, head_dim = (int(dim) for dim in cache.shape)
    return np.asarray(cache.reshape(layers * heads * seq_len, head_dim), dtype=np.float32)


def _unit_queries(*, query_count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((query_count, dimension), dtype=np.float32)
    return _normalize_last_axis(rows)


def _find_metric(encoding_metrics: Sequence[object], scope: str, metric: str) -> float:
    for evidence in encoding_metrics:
        current_scope = getattr(evidence, "scope", None)
        current_metrics = getattr(evidence, "metrics", None)
        if current_scope == scope and isinstance(current_metrics, dict) and metric in current_metrics:
            value = current_metrics[metric]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    raise KeyError(f"missing metric {metric!r} in scope {scope!r}")


def run_synthetic_kv_benchmark(
    *,
    layers: int = 2,
    heads: int = 4,
    seq_len: int = 128,
    head_dim: int = 64,
    key_total_bits: int = 3,
    value_bits: int = 3,
    query_count: int = 32,
    seed: int = 123,
    query_seed: int = 456,
    key_rotation_seed: int = 7,
    qjl_seed: int = 11,
    value_rotation_seed: int = 17,
    value_scale: float = 1.0,
    output_path: Path | None = None,
) -> dict[str, object]:
    key_cache, value_cache = generate_synthetic_kv(
        layers=layers,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        seed=seed,
        value_scale=value_scale,
    )
    key_rows = flatten_cache_rows(key_cache)
    value_rows = flatten_cache_rows(value_cache)
    queries = _unit_queries(query_count=query_count, dimension=head_dim, seed=query_seed)

    key_codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(
            total_bits_per_scalar=key_total_bits,
            default_rotation_seed=key_rotation_seed,
            default_qjl_seed=qjl_seed,
        )
    )
    key_request = VectorEncodeRequest(
        data=key_rows,
        objective="inner_product_estimation",
        metric="dot_product_error",
        role="key_cache",
    )
    key_encode_start = time.perf_counter()
    key_encoding = key_codec.encode(key_request)
    key_encode_seconds = time.perf_counter() - key_encode_start
    key_decode_start = time.perf_counter()
    key_decoded = key_codec.decode(VectorDecodeRequest(encoding=key_encoding)).data
    key_decode_seconds = time.perf_counter() - key_decode_start

    exact_scores = queries.astype(np.float64) @ key_rows.astype(np.float64).T
    approx_scores = queries.astype(np.float64) @ key_decoded.astype(np.float64).T
    key_errors = approx_scores - exact_scores

    value_codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(
            default_bits_per_scalar=value_bits,
            default_rotation_seed=value_rotation_seed,
        )
    )
    value_request = VectorEncodeRequest(
        data=value_rows,
        objective="reconstruction",
        metric="mse",
        role="value_cache",
    )
    value_encode_start = time.perf_counter()
    value_encoding = value_codec.encode(value_request)
    value_encode_seconds = time.perf_counter() - value_encode_start
    value_decode_start = time.perf_counter()
    value_decoded = value_codec.decode(VectorDecodeRequest(encoding=value_encoding)).data
    value_decode_seconds = time.perf_counter() - value_decode_start

    value_mse = float(np.mean(np.square(value_decoded.astype(np.float64) - value_rows.astype(np.float64))))

    combined_baseline = int(key_rows.nbytes + value_rows.nbytes)
    combined_total = int(key_encoding.footprint.total_bytes + value_encoding.footprint.total_bytes)
    combined_ratio = float(combined_baseline) / float(combined_total) if combined_total > 0 else 0.0

    result: dict[str, object] = {
        "kind": "turboquant_synthetic_kv_benchmark",
        "implementation_version": __version__,
        "parameters": {
            "layers": int(layers),
            "heads": int(heads),
            "seq_len": int(seq_len),
            "head_dim": int(head_dim),
            "key_total_bits": int(key_total_bits),
            "value_bits": int(value_bits),
            "query_count": int(query_count),
            "seed": int(seed),
            "query_seed": int(query_seed),
            "key_rotation_seed": int(key_rotation_seed),
            "qjl_seed": int(qjl_seed),
            "value_rotation_seed": int(value_rotation_seed),
            "value_scale": float(value_scale),
        },
        "keys": {
            "row_count": int(key_rows.shape[0]),
            "dimension": int(key_rows.shape[1]),
            "mean_error": float(np.mean(key_errors)),
            "mean_abs_error": float(np.mean(np.abs(key_errors))),
            "empirical_variance": float(np.var(key_errors)),
            "theory_proxy": _find_metric(key_encoding.evidence, "theory_proxy", "mean_query_free_variance_factor"),
            "payload_bytes": int(key_encoding.footprint.payload_bytes),
            "sidecar_bytes": int(key_encoding.footprint.sidecar_bytes),
            "total_bytes": int(key_encoding.footprint.total_bytes),
            "compression_ratio": float(key_encoding.footprint.compression_ratio),
            "encode_seconds": float(key_encode_seconds),
            "decode_seconds": float(key_decode_seconds),
        },
        "values": {
            "row_count": int(value_rows.shape[0]),
            "dimension": int(value_rows.shape[1]),
            "observed_mse": value_mse,
            "payload_bytes": int(value_encoding.footprint.payload_bytes),
            "sidecar_bytes": int(value_encoding.footprint.sidecar_bytes),
            "total_bytes": int(value_encoding.footprint.total_bytes),
            "compression_ratio": float(value_encoding.footprint.compression_ratio),
            "encode_seconds": float(value_encode_seconds),
            "decode_seconds": float(value_decode_seconds),
        },
        "combined": {
            "baseline_bytes": combined_baseline,
            "total_bytes": combined_total,
            "compression_ratio": combined_ratio,
        },
    }
    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic TurboQuant KV-style benchmarks.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    result = run_synthetic_kv_benchmark(output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
