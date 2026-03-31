from __future__ import annotations

import numpy as np

from semafold import VectorDecodeRequest, VectorEncodeRequest, VectorEncoding
from semafold.turboquant import TurboQuantProdConfig, TurboQuantProdVectorCodec


def test_turboquant_prod_roundtrip_preserves_shape_dtype_and_finite_decode_for_rank2() -> None:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=5, default_qjl_seed=9)
    )
    data = np.linspace(-1.0, 1.0, num=48, dtype=np.float32).reshape(6, 8)
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="inner_product_estimation", metric="dot_product_error"))
    decoded = codec.decode(VectorDecodeRequest(encoding=VectorEncoding.from_dict(encoding.to_dict())))

    assert decoded.data.shape == data.shape
    assert decoded.data.dtype == data.dtype
    assert np.isfinite(decoded.data).all()


def test_turboquant_prod_roundtrip_preserves_shape_dtype_and_finite_decode_for_rank1() -> None:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=4, default_rotation_seed=5, default_qjl_seed=9)
    )
    data = np.linspace(-0.75, 0.75, num=8, dtype=np.float64)
    encoding = codec.encode(VectorEncodeRequest(data=data, objective="inner_product_estimation"))
    decoded = codec.decode(VectorDecodeRequest(encoding=VectorEncoding.from_dict(encoding.to_dict())))

    assert decoded.data.shape == data.shape
    assert decoded.data.dtype == data.dtype
    assert np.isfinite(decoded.data).all()


def test_turboquant_prod_profile_id_survives_roundtrip() -> None:
    codec = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=1, default_qjl_seed=2)
    )
    data = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(2, 8)
    request = VectorEncodeRequest(
        data=data,
        objective="inner_product_estimation",
        profile_id="tq-prod-preview",
    )
    encoding = codec.encode(request)
    reloaded = VectorEncoding.from_dict(encoding.to_dict())

    assert reloaded.profile_id == "tq-prod-preview"
    assert reloaded.variant_id == "prod_qjl_residual_v1"


def test_turboquant_prod_same_seed_same_artifact_and_different_seed_changes_artifact() -> None:
    data = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(4, 8)
    first = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    second = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=7, default_qjl_seed=11)
    )
    third = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=3, default_rotation_seed=8, default_qjl_seed=12)
    )

    first_encoding = first.encode(VectorEncodeRequest(data=data, objective="inner_product_estimation", metric="dot_product_error"))
    second_encoding = second.encode(VectorEncodeRequest(data=data, objective="inner_product_estimation", metric="dot_product_error"))
    third_encoding = third.encode(VectorEncodeRequest(data=data, objective="inner_product_estimation", metric="dot_product_error"))

    assert first_encoding.to_dict() == second_encoding.to_dict()
    assert first_encoding.to_dict() != third_encoding.to_dict()
