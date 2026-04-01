from __future__ import annotations

import math

import numpy as np
import pytest

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec
from semafold.turboquant import TurboQuantProdConfig
from semafold.turboquant import TurboQuantProdVectorCodec
from semafold.vector.models import EncodeObjective, EncodeMetric

_MONTE_CARLO_SAMPLES = 256
_DIMENSION = 32


def _unit_vector(seed: int, dimension: int = _DIMENSION) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=(dimension,)).astype(np.float32)
    vector /= np.linalg.norm(vector.astype(np.float64))
    return vector


def _prod_inner_product_errors(
    *,
    x: np.ndarray,
    y: np.ndarray,
    total_bits_per_scalar: int,
    sample_count: int = _MONTE_CARLO_SAMPLES,
) -> tuple[np.ndarray, float]:
    errors: list[float] = []
    variance_bound: float | None = None
    true_inner_product = float(np.dot(y.astype(np.float64), x.astype(np.float64)))
    query_norm_squared = float(np.dot(y.astype(np.float64), y.astype(np.float64)))

    for qjl_seed in range(sample_count):
        codec = TurboQuantProdVectorCodec(
            config=TurboQuantProdConfig(
                total_bits_per_scalar=total_bits_per_scalar,
                default_rotation_seed=7,
                default_qjl_seed=qjl_seed,
            )
        )
        encoding = codec.encode(
            VectorEncodeRequest(
                data=x,
                objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
                metric=EncodeMetric.DOT_PRODUCT_ERROR,
            )
        )
        if variance_bound is None:
            theory_proxy = next(item for item in encoding.evidence if item.scope == "theory_proxy")
            variance_bound = float(theory_proxy.metrics["mean_query_free_variance_factor"]) * query_norm_squared
        decoded = codec.decode(VectorDecodeRequest(encoding=encoding)).data.astype(np.float64)
        errors.append(float(np.dot(y.astype(np.float64), decoded) - true_inner_product))

    assert variance_bound is not None
    return np.asarray(errors, dtype=np.float64), variance_bound


def _mse_inner_product_errors(
    *,
    x: np.ndarray,
    y: np.ndarray,
    bits_per_scalar: int,
    sample_count: int = _MONTE_CARLO_SAMPLES,
) -> np.ndarray:
    errors: list[float] = []
    true_inner_product = float(np.dot(y.astype(np.float64), x.astype(np.float64)))

    for rotation_seed in range(sample_count):
        codec = TurboQuantMSEVectorCodec(
            config=TurboQuantMSEConfig(default_bits_per_scalar=bits_per_scalar, default_rotation_seed=0)
        )
        encoding = codec.encode(
            VectorEncodeRequest(
                data=x,
                objective=EncodeObjective.RECONSTRUCTION,
                metric=EncodeMetric.MSE,
                seed=rotation_seed,
            )
        )
        decoded = codec.decode(VectorDecodeRequest(encoding=encoding)).data.astype(np.float64)
        errors.append(float(np.dot(y.astype(np.float64), decoded) - true_inner_product))

    return np.asarray(errors, dtype=np.float64)


@pytest.mark.parametrize(
    ("total_bits_per_scalar", "query_seed"),
    [
        (2, 123),
        (2, 456),
        (3, 456),
        (4, 456),
    ],
)
def test_turboquant_prod_monte_carlo_inner_product_estimator_is_empirically_unbiased(
    total_bits_per_scalar: int,
    query_seed: int,
) -> None:
    x = _unit_vector(123)
    y = x.copy() if query_seed == 123 else _unit_vector(query_seed)

    errors, _ = _prod_inner_product_errors(
        x=x,
        y=y,
        total_bits_per_scalar=total_bits_per_scalar,
    )
    sample_mean = float(np.mean(errors))
    sample_std = float(np.std(errors, ddof=1))
    tolerance = max(0.012, 4.0 * sample_std / math.sqrt(float(errors.size)))

    assert abs(sample_mean) <= tolerance


@pytest.mark.parametrize("total_bits_per_scalar", [2, 3, 4])
def test_turboquant_prod_empirical_variance_respects_theory_proxy(total_bits_per_scalar: int) -> None:
    x = _unit_vector(123)
    y = _unit_vector(456)

    errors, variance_bound = _prod_inner_product_errors(
        x=x,
        y=y,
        total_bits_per_scalar=total_bits_per_scalar,
    )
    empirical_variance = float(np.var(errors))

    assert variance_bound >= 0.0
    assert empirical_variance <= (variance_bound * 1.4) + 1e-9


def test_turboquant_prod_reduces_inner_product_bias_relative_to_mse_codec() -> None:
    x = _unit_vector(123)
    y = x.copy()

    prod_errors, _ = _prod_inner_product_errors(
        x=x,
        y=y,
        total_bits_per_scalar=2,
    )
    mse_errors = _mse_inner_product_errors(
        x=x,
        y=y,
        bits_per_scalar=2,
    )

    assert abs(float(np.mean(prod_errors))) < (0.2 * abs(float(np.mean(mse_errors))))


def test_turboquant_prod_theory_proxy_improves_with_more_bits_on_fixed_input() -> None:
    x = _unit_vector(123)
    y = _unit_vector(456)

    low_errors, low_proxy = _prod_inner_product_errors(x=x, y=y, total_bits_per_scalar=2, sample_count=64)
    high_errors, high_proxy = _prod_inner_product_errors(x=x, y=y, total_bits_per_scalar=4, sample_count=64)

    assert np.var(high_errors) < np.var(low_errors)
    assert high_proxy < low_proxy
