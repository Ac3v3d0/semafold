"""TurboQuant inner-product codec with QJL residual correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, TypedDict

import numpy as np

from semafold._version import __version__
from semafold.core import CompressionEstimate, CompressionGuarantee, ValidationEvidence
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.core.accounting import aggregate_footprints, json_byte_size, segment_footprint
from semafold.errors import CompatibilityError, DecodeError
from semafold.turboquant.codebook import TurboQuantScalarCodebook, solve_beta_lloyd_max_codebook
from semafold.turboquant.packing import pack_scalar_indices, packed_byte_count, unpack_scalar_indices
from semafold.turboquant.qjl import qjl_decode_rows, qjl_encode_rows, seeded_gaussian_projection
from semafold.turboquant.quantizer import dequantize_rows, normalize_rows, quantize_rows, restore_rows
from semafold.turboquant.rotation import seeded_haar_rotation
from semafold.vector.models import (
    EncodeMetric,
    EncodeObjective,
    EncodingSegmentKind,
    VectorDecodeRequest,
    VectorDecodeResult,
    VectorEncodeRequest,
    VectorEncoding,
    VectorEncodingSegment,
    fingerprint_config,
    normalize_to_2d,
    restore_from_2d,
)

__all__ = ["TurboQuantProdConfig", "TurboQuantProdVectorCodec"]


class _DecodedProdMetadata(TypedDict):
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...]
    rank: int
    vector_count: int
    dimension: int
    total_bits_per_scalar: int
    base_bits_per_scalar: int
    base_rotation_seed: int
    qjl_seed: int
    codebook_centers: list[float]
    solver_iterations: int
    solver_converged: bool


def _decode_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DecodeError(f"{name} must be an int")
    return value


def _decode_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DecodeError("shape must be a list[int]")
    shape: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise DecodeError("shape must contain only ints")
        shape.append(item)
    return tuple(shape)


def _decode_required_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise DecodeError(f"{name} must be a non-empty string")
    return value


def _decode_float_list(name: str, value: object) -> list[float]:
    if not isinstance(value, list):
        raise DecodeError(f"{name} must be a list[float]")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise DecodeError(f"{name} must contain only float-compatible values")
        result.append(float(item))
    return result


def _validate_supported_dtype(dtype: np.dtype[np.generic]) -> None:
    if dtype not in {np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)}:
        raise CompatibilityError("TurboQuantProdVectorCodec requires float16, float32, or float64 input")


def _validate_total_bits_per_scalar(total_bits_per_scalar: int) -> int:
    if not isinstance(total_bits_per_scalar, int) or isinstance(total_bits_per_scalar, bool):
        raise TypeError("total_bits_per_scalar must be an int")
    if total_bits_per_scalar < 2 or total_bits_per_scalar > 5:
        raise ValueError("total_bits_per_scalar must be between 2 and 5 in the current TurboQuant preview")
    return total_bits_per_scalar


def _validate_grid_size(grid_size: int) -> int:
    if not isinstance(grid_size, int) or isinstance(grid_size, bool):
        raise TypeError("grid_size must be an int")
    if grid_size < 1025:
        raise ValueError("grid_size must be >= 1025")
    return grid_size


def _validate_iterations(max_iterations: int) -> int:
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an int")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    return max_iterations


def _validate_tolerance(tolerance: float) -> float:
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be float-compatible")
    result = float(tolerance)
    if result <= 0.0:
        raise ValueError("tolerance must be > 0")
    return result


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    return seed


def _full_scope(array_2d: np.ndarray) -> dict[str, object]:
    return {
        "kind": "full",
        "row_start": 0,
        "row_stop": int(array_2d.shape[0]),
        "col_start": 0,
        "col_stop": int(array_2d.shape[1]),
    }


def _normalize_residual_rows(residual_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.linalg.norm(residual_rows.astype(np.float64), axis=1).astype(np.float32)
    normalized = np.zeros_like(residual_rows, dtype=np.float32)
    nonzero = gamma > 0.0
    if np.any(nonzero):
        normalized[nonzero] = residual_rows[nonzero] / gamma[nonzero, None]
    return normalized, gamma


@dataclass(frozen=True, slots=True)
class TurboQuantProdConfig:
    """Encode-time defaults for the TurboQuant Prod codec."""

    total_bits_per_scalar: int = 3
    default_rotation_seed: int = 0
    default_qjl_seed: int = 0
    normalization: Literal["row_l2"] = "row_l2"
    grid_size: int = 8193
    max_iterations: int = 128
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_bits_per_scalar", _validate_total_bits_per_scalar(self.total_bits_per_scalar))
        object.__setattr__(self, "default_rotation_seed", _validate_seed(self.default_rotation_seed))
        object.__setattr__(self, "default_qjl_seed", _validate_seed(self.default_qjl_seed))
        if self.normalization != "row_l2":
            raise ValueError("normalization must be 'row_l2' in the current TurboQuant Prod preview")
        object.__setattr__(self, "grid_size", _validate_grid_size(self.grid_size))
        object.__setattr__(self, "max_iterations", _validate_iterations(self.max_iterations))
        object.__setattr__(self, "tolerance", _validate_tolerance(self.tolerance))


class TurboQuantProdVectorCodec:
    """TurboQuant codec for inner-product estimation workloads.

    The codec uses a lower-bit TurboQuant MSE base representation plus a QJL
    sign sketch over the residual direction, following the paper's two-stage
    formulation.
    """

    codec_family = "turboquant"
    variant_id = "prod_qjl_residual_v1"
    encoding_schema_version = "vector.encoding.v1"
    _supported_objectives = {EncodeObjective.INNER_PRODUCT_ESTIMATION}
    _supported_metrics = {None, EncodeMetric.DOT_PRODUCT_ERROR}

    def __init__(self, *, config: TurboQuantProdConfig | None = None) -> None:
        if config is not None and not isinstance(config, TurboQuantProdConfig):
            raise TypeError("config must be a TurboQuantProdConfig or None")
        self.config = config if config is not None else TurboQuantProdConfig()

    def estimate(self, request: VectorEncodeRequest) -> CompressionEstimate:
        """Estimate artifact size for a TurboQuant Prod encoding request."""
        array_2d, _, _ = self._validate_request(request)
        vector_count = int(array_2d.shape[0])
        dimension = int(array_2d.shape[1])
        total_bits = self.config.total_bits_per_scalar
        base_bits = total_bits - 1
        rotation_seed = self._resolve_seed(request.seed, self.config.default_rotation_seed)
        qjl_seed = self.config.default_qjl_seed
        codebook = self._solve_codebook(dimension, base_bits)
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=tuple(int(dim) for dim in request.data.shape),
            original_rank=int(request.data.ndim),
            total_bits_per_scalar=total_bits,
            base_bits_per_scalar=base_bits,
            base_rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
            codebook=codebook,
        )
        payload_bytes = packed_byte_count(vector_count * dimension, base_bits)
        residual_sketch_bytes = packed_byte_count(vector_count * dimension, 1)
        sidecar_bytes = vector_count * np.dtype(np.float32).itemsize
        metadata_bytes = json_byte_size(metadata_payload)
        total_bytes = payload_bytes + residual_sketch_bytes + (2 * sidecar_bytes) + metadata_bytes
        baseline_bytes = int(request.data.nbytes)
        return CompressionEstimate(
            baseline_bytes=baseline_bytes,
            estimated_payload_bytes=payload_bytes + residual_sketch_bytes,
            estimated_metadata_bytes=metadata_bytes,
            estimated_sidecar_bytes=2 * sidecar_bytes,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=total_bytes,
            estimated_compression_ratio=(float(baseline_bytes) / float(total_bytes) if total_bytes > 0 else 0.0),
        )

    def encode(self, request: VectorEncodeRequest) -> VectorEncoding:
        """Encode vectors into a TurboQuant Prod artifact with residual QJL sidecars."""
        array_2d, original_shape, original_rank = self._validate_request(request)
        dimension = int(array_2d.shape[1])
        total_bits = self.config.total_bits_per_scalar
        base_bits = total_bits - 1
        base_rotation_seed = self._resolve_seed(request.seed, self.config.default_rotation_seed)
        qjl_seed = self.config.default_qjl_seed

        codebook = self._solve_codebook(dimension, base_bits)
        rotation = seeded_haar_rotation(dimension, base_rotation_seed)
        projection = seeded_gaussian_projection(dimension, qjl_seed)

        unit_rows, row_norms = normalize_rows(array_2d)
        base_indices = quantize_rows(unit_rows, rotation=rotation, codebook=codebook)
        base_unit = dequantize_rows(base_indices, rotation=rotation, codebook=codebook)
        residual_rows = np.asarray(unit_rows - base_unit, dtype=np.float32)
        residual_hat, residual_gamma = _normalize_residual_rows(residual_rows)
        sign_rows = qjl_encode_rows(residual_hat, projection)

        compressed_payload = pack_scalar_indices(base_indices.reshape(-1), base_bits)
        sidecar_payload = np.ascontiguousarray(row_norms).tobytes()
        residual_sketch_payload = pack_scalar_indices(sign_rows.reshape(-1), 1)
        residual_gamma_payload = np.ascontiguousarray(residual_gamma).tobytes()
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=original_shape,
            original_rank=original_rank,
            total_bits_per_scalar=total_bits,
            base_bits_per_scalar=base_bits,
            base_rotation_seed=base_rotation_seed,
            qjl_seed=qjl_seed,
            codebook=codebook,
        )

        compressed_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.COMPRESSED,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=compressed_payload,
            payload_format=f"bitpack.lsb.u{base_bits}",
            footprint=segment_footprint(payload_bytes=len(compressed_payload)),
            metadata={},
        )
        sidecar_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.SIDECAR,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=sidecar_payload,
            payload_format="numpy.float32.row_norm",
            footprint=segment_footprint(sidecar_bytes=len(sidecar_payload)),
            metadata={},
        )
        residual_sketch_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.RESIDUAL_SKETCH,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=residual_sketch_payload,
            payload_format="bitpack.sign.u1",
            footprint=segment_footprint(payload_bytes=len(residual_sketch_payload)),
            metadata={},
        )
        residual_gamma_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.RESIDUAL_GAMMA,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=residual_gamma_payload,
            payload_format="numpy.float32.row_residual_norm",
            footprint=segment_footprint(sidecar_bytes=len(residual_gamma_payload)),
            metadata={},
        )
        metadata_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.METADATA,
            role=request.role,
            scope={"kind": "encoding_metadata"},
            payload=metadata_payload,
            payload_format="json",
            footprint=segment_footprint(metadata_bytes=json_byte_size(metadata_payload)),
            metadata={},
        )
        footprint = aggregate_footprints(
            baseline_bytes=int(request.data.nbytes),
            segment_footprints=[
                compressed_segment.footprint,
                sidecar_segment.footprint,
                residual_sketch_segment.footprint,
                residual_gamma_segment.footprint,
                metadata_segment.footprint,
            ],
        )

        base_restored = restore_from_2d(restore_rows(base_unit, row_norms).astype(request.data.dtype, copy=False), original_shape, original_rank)
        base_observed_mse = float(np.mean((request.data.astype(np.float64) - base_restored.astype(np.float64)) ** 2))
        mean_residual_norm_squared = float(np.mean((residual_gamma.astype(np.float64)) ** 2))
        zero_residual_fraction = float(np.mean(residual_gamma == 0.0))
        query_free_variance_factors = (
            (math.pi / (2.0 * float(dimension))) * (residual_gamma.astype(np.float64) ** 2)
        )
        mean_query_free_variance_factor = float(np.mean(query_free_variance_factors))
        max_query_free_variance_factor = float(np.max(query_free_variance_factors))

        guarantees = [
            CompressionGuarantee(
                objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
                metric="estimator_unbiasedness",
                bound_type=EncodingBoundType.THEOREM_REFERENCE,
                value=True,
                units=None,
                scope="unit_norm_row",
                workload_suitability=[
                    WorkloadSuitability.VECTOR_DATABASE,
                    WorkloadSuitability.QUERY_TIME_INNER_PRODUCT,
                ],
                notes="Preview Theorem 2 reference under unit-norm rows with QJL residual correction; not a general-purpose runtime guarantee.",
            ),
        ]
        evidence = [
            ValidationEvidence(
                scope="proxy_fidelity",
                environment={
                    "codec_family": self.codec_family,
                    "variant_id": self.variant_id,
                    "dtype": str(request.data.dtype),
                    "base_rotation_seed": base_rotation_seed,
                    "qjl_seed": qjl_seed,
                    "total_bits_per_scalar": total_bits,
                    "base_bits_per_scalar": base_bits,
                },
                metrics={
                    "base_observed_mse": base_observed_mse,
                    "mean_residual_norm_squared": mean_residual_norm_squared,
                    "zero_residual_fraction": zero_residual_fraction,
                    "solver_iterations": codebook.iterations,
                    "solver_converged": codebook.converged,
                },
                passed=True,
                summary="TurboQuant Prod preview artifact-local fidelity summary.",
                artifact_refs=[],
            ),
            ValidationEvidence(
                scope="theory_proxy",
                environment={
                    "codec_family": self.codec_family,
                    "variant_id": self.variant_id,
                    "dimension": dimension,
                    "qjl_family": "seeded_gaussian_sign_v1",
                },
                metrics={
                    "mean_query_free_variance_factor": mean_query_free_variance_factor,
                    "max_query_free_variance_factor": max_query_free_variance_factor,
                    "mean_residual_gamma_squared": mean_residual_norm_squared,
                },
                passed=True,
                summary=(
                    "Query-free QJL variance factor for inner-product queries; multiply by ||y||^2 "
                    "to obtain the artifact-local Lemma 4/Theorem 2 proxy."
                ),
                artifact_refs=[],
            ),
            ValidationEvidence(
                scope="theorem_assumptions",
                environment={
                    "codec_family": self.codec_family,
                    "variant_id": self.variant_id,
                    "dimension": dimension,
                    "total_bits_per_scalar": total_bits,
                    "base_bits_per_scalar": base_bits,
                    "rotation_family": "seeded_haar_qr",
                    "qjl_family": "seeded_gaussian_sign_v1",
                    "normalization": self.config.normalization,
                },
                metrics={
                    "unit_norm_row_envelope": True,
                    "residual_defined_in_normalized_space": True,
                    "base_bits_equals_total_minus_one": True,
                    "qjl_residual_correction_enabled": True,
                    "zero_residual_rows_canonical": True,
                    "row_norm_sidecar_nonnegative": bool(np.all(row_norms >= 0.0)),
                    "residual_gamma_sidecar_nonnegative": bool(np.all(residual_gamma >= 0.0)),
                    "codebook_centers_size_valid": len(codebook.centers) == (1 << base_bits),
                    "solver_converged": codebook.converged,
                },
                passed=True,
                summary=(
                    "Theorem-reference assumptions carried by the current TurboQuant Prod preview artifact."
                ),
                artifact_refs=[],
            ),
            ValidationEvidence(
                scope="storage_accounting",
                environment={"codec_family": self.codec_family, "variant_id": self.variant_id},
                metrics={
                    "baseline_bytes": footprint.baseline_bytes,
                    "payload_bytes": footprint.payload_bytes,
                    "metadata_bytes": footprint.metadata_bytes,
                    "sidecar_bytes": footprint.sidecar_bytes,
                    "total_bytes": footprint.total_bytes,
                    "compression_ratio": footprint.compression_ratio,
                    "residual_sketch_bytes": len(residual_sketch_payload),
                    "residual_gamma_bytes": len(residual_gamma_payload),
                },
                passed=True,
                summary="TurboQuant Prod preview accounting computed exactly.",
                artifact_refs=[],
            ),
        ]
        config = self._config_payload(
            total_bits_per_scalar=total_bits,
            base_rotation_seed=base_rotation_seed,
            qjl_seed=qjl_seed,
        )
        return VectorEncoding(
            codec_family=self.codec_family,
            variant_id=self.variant_id,
            profile_id=request.profile_id,
            implementation_version=__version__,
            encoding_schema_version=self.encoding_schema_version,
            config_fingerprint=fingerprint_config(config),
            segments=[
                compressed_segment,
                sidecar_segment,
                residual_sketch_segment,
                residual_gamma_segment,
                metadata_segment,
            ],
            footprint=footprint,
            guarantees=guarantees,
            evidence=evidence,
            metadata={"objective": request.objective, "mode": "prod"},
        )

    def decode(self, request: VectorDecodeRequest) -> VectorDecodeResult:
        """Decode a TurboQuant Prod artifact back into the original tensor shape."""
        encoding = request.encoding
        if encoding.codec_family != self.codec_family:
            raise CompatibilityError("encoding does not belong to the TurboQuant codec family")
        if encoding.variant_id != self.variant_id:
            raise CompatibilityError("encoding does not belong to a supported TurboQuant Prod preview variant")
        if encoding.encoding_schema_version != self.encoding_schema_version:
            raise CompatibilityError("unsupported encoding schema version")

        compressed_segment = self._require_single_segment(encoding, EncodingSegmentKind.COMPRESSED)
        sidecar_segment = self._require_single_segment(encoding, EncodingSegmentKind.SIDECAR)
        residual_sketch_segment = self._require_single_segment(encoding, EncodingSegmentKind.RESIDUAL_SKETCH)
        residual_gamma_segment = self._require_single_segment(encoding, EncodingSegmentKind.RESIDUAL_GAMMA)
        metadata_segment = self._require_single_segment(encoding, EncodingSegmentKind.METADATA)

        if not isinstance(compressed_segment.payload, bytes):
            raise DecodeError("compressed payload must be bytes")
        if not isinstance(sidecar_segment.payload, bytes):
            raise DecodeError("sidecar payload must be bytes")
        if not isinstance(residual_sketch_segment.payload, bytes):
            raise DecodeError("residual_sketch payload must be bytes")
        if not isinstance(residual_gamma_segment.payload, bytes):
            raise DecodeError("residual_gamma payload must be bytes")
        if not isinstance(metadata_segment.payload, dict):
            raise DecodeError("metadata payload must be a dict")

        if metadata_segment.payload_format != "json":
            raise DecodeError("unexpected metadata payload_format")
        if sidecar_segment.payload_format != "numpy.float32.row_norm":
            raise DecodeError("unexpected sidecar payload_format")
        if residual_gamma_segment.payload_format != "numpy.float32.row_residual_norm":
            raise DecodeError("unexpected residual_gamma payload_format")

        metadata = self._decode_metadata(metadata_segment.payload, encoding.variant_id)
        expected_compressed_format = f"bitpack.lsb.u{metadata['base_bits_per_scalar']}"
        if compressed_segment.payload_format != expected_compressed_format:
            raise DecodeError("compressed payload_format does not match metadata base_bits_per_scalar")
        if residual_sketch_segment.payload_format != "bitpack.sign.u1":
            raise DecodeError("unexpected residual_sketch payload_format")

        vector_count = metadata["vector_count"]
        dimension = metadata["dimension"]
        base_bits = metadata["base_bits_per_scalar"]
        expected_compressed_bytes = packed_byte_count(vector_count * dimension, base_bits)
        if len(compressed_segment.payload) != expected_compressed_bytes:
            raise DecodeError("compressed payload size does not match metadata")
        expected_sketch_bytes = packed_byte_count(vector_count * dimension, 1)
        if len(residual_sketch_segment.payload) != expected_sketch_bytes:
            raise DecodeError("residual_sketch payload size does not match metadata")
        expected_sidecar_bytes = vector_count * np.dtype(np.float32).itemsize
        if len(sidecar_segment.payload) != expected_sidecar_bytes:
            raise DecodeError("sidecar payload size does not match metadata")
        if len(residual_gamma_segment.payload) != expected_sidecar_bytes:
            raise DecodeError("residual_gamma payload size does not match metadata")

        try:
            row_norms = np.frombuffer(sidecar_segment.payload, dtype=np.float32).copy()
            residual_gamma = np.frombuffer(residual_gamma_segment.payload, dtype=np.float32).copy()
        except ValueError as exc:  # pragma: no cover - defensive
            raise DecodeError("sidecar payloads cannot be interpreted as float32 arrays") from exc
        if row_norms.shape != (vector_count,):
            raise DecodeError("sidecar row norms do not match metadata")
        if residual_gamma.shape != (vector_count,):
            raise DecodeError("residual_gamma values do not match metadata")
        if not np.isfinite(row_norms).all() or np.any(row_norms < 0.0):
            raise DecodeError("sidecar payload contains invalid row norms")
        if not np.isfinite(residual_gamma).all() or np.any(residual_gamma < 0.0):
            raise DecodeError("residual_gamma payload contains invalid values")

        base_indices = unpack_scalar_indices(
            compressed_segment.payload,
            count=vector_count * dimension,
            bits_per_index=base_bits,
        ).reshape(vector_count, dimension)
        sign_rows = unpack_scalar_indices(
            residual_sketch_segment.payload,
            count=vector_count * dimension,
            bits_per_index=1,
        ).reshape(vector_count, dimension)
        rotation = seeded_haar_rotation(dimension, metadata["base_rotation_seed"])
        projection = seeded_gaussian_projection(dimension, metadata["qjl_seed"])
        codebook = TurboQuantScalarCodebook.from_centers(
            dimension=dimension,
            bits_per_scalar=base_bits,
            centers=np.asarray(metadata["codebook_centers"], dtype=np.float64),
            iterations=metadata["solver_iterations"],
            converged=metadata["solver_converged"],
        )
        base_unit = dequantize_rows(base_indices, rotation=rotation, codebook=codebook)
        residual_tilde = qjl_decode_rows(sign_rows, residual_gamma, projection)
        restored_unit = np.asarray(base_unit + residual_tilde, dtype=np.float32)
        restored_rows = restore_rows(restored_unit, row_norms)
        dtype = np.dtype(metadata["dtype"])
        restored = restore_from_2d(restored_rows.astype(dtype, copy=False), metadata["shape"], metadata["rank"])
        notes: list[str] = []
        if request.target_layout is not None:
            notes.append(f"target_layout={request.target_layout!r} ignored in the current TurboQuant preview.")
        return VectorDecodeResult(data=restored, metadata={}, materialization_notes=notes)

    def _validate_request(self, request: VectorEncodeRequest) -> tuple[np.ndarray, tuple[int, ...], int]:
        if request.objective not in self._supported_objectives:
            raise CompatibilityError("TurboQuantProdVectorCodec only supports 'inner_product_estimation'")
        if request.metric not in self._supported_metrics:
            raise CompatibilityError("TurboQuantProdVectorCodec only supports metric=None or 'dot_product_error'")
        _validate_supported_dtype(request.data.dtype)
        if not np.isfinite(request.data).all():
            raise CompatibilityError("TurboQuantProdVectorCodec requires finite floating-point values")
        array_2d, original_shape, original_rank = normalize_to_2d(request.data)
        if int(array_2d.shape[1]) < 2:
            raise CompatibilityError("TurboQuantProdVectorCodec requires dimension >= 2")
        return array_2d, original_shape, original_rank

    def _solve_codebook(self, dimension: int, base_bits_per_scalar: int) -> TurboQuantScalarCodebook:
        return solve_beta_lloyd_max_codebook(
            dimension,
            base_bits_per_scalar,
            grid_size=self.config.grid_size,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
        )

    @staticmethod
    def _resolve_seed(request_seed: int | None, default_seed: int) -> int:
        return default_seed if request_seed is None else _validate_seed(request_seed)

    def _config_payload(
        self,
        *,
        total_bits_per_scalar: int,
        base_rotation_seed: int,
        qjl_seed: int,
    ) -> dict[str, object]:
        return {
            "codec_family": self.codec_family,
            "variant_id": self.variant_id,
            "encoding_schema_version": self.encoding_schema_version,
            "mode": "prod",
            "total_bits_per_scalar": total_bits_per_scalar,
            "base_bits_per_scalar": total_bits_per_scalar - 1,
            "rotation_family": "seeded_haar_qr",
            "base_rotation_seed": base_rotation_seed,
            "qjl_family": "seeded_gaussian_sign_v1",
            "qjl_seed": qjl_seed,
            "normalization": self.config.normalization,
            "codebook_family": "beta_lloyd_max",
            "grid_size": self.config.grid_size,
            "max_iterations": self.config.max_iterations,
            "tolerance": self.config.tolerance,
        }

    def _metadata_payload(
        self,
        *,
        data: np.ndarray,
        array_2d: np.ndarray,
        original_shape: tuple[int, ...],
        original_rank: int,
        total_bits_per_scalar: int,
        base_bits_per_scalar: int,
        base_rotation_seed: int,
        qjl_seed: int,
        codebook: TurboQuantScalarCodebook,
    ) -> dict[str, object]:
        metadata = {
            "format": self.variant_id,
            "mode": "prod",
            "shape": list(original_shape),
            "rank": original_rank,
            "dtype": str(data.dtype),
            "vector_count": int(array_2d.shape[0]),
            "dimension": int(array_2d.shape[1]),
            "total_bits_per_scalar": total_bits_per_scalar,
            "base_bits_per_scalar": base_bits_per_scalar,
            "base_rotation_seed": base_rotation_seed,
            "rotation_family": "seeded_haar_qr",
            "normalization": self.config.normalization,
            "codebook_family": "beta_lloyd_max",
            "codebook_centers": [float(item) for item in np.asarray(codebook.centers, dtype=np.float64)],
            "qjl_seed": qjl_seed,
            "qjl_family": "seeded_gaussian_sign_v1",
        }
        metadata["solver_iterations"] = codebook.iterations
        metadata["solver_converged"] = codebook.converged
        return metadata

    def _decode_metadata(self, metadata: Mapping[str, object], envelope_variant_id: str) -> _DecodedProdMetadata:
        format_value = _decode_required_str("format", metadata.get("format"))
        if format_value != self.variant_id:
            raise DecodeError("metadata format does not match codec variant")
        if envelope_variant_id != self.variant_id:
            raise DecodeError("encoding variant is not a supported TurboQuant Prod preview variant")
        if format_value != envelope_variant_id:
            raise DecodeError("metadata format does not match codec variant")

        mode = metadata.get("mode")
        if mode != "prod":
            raise DecodeError("TurboQuant preview metadata mode must be 'prod'")

        dtype = np.dtype(_decode_required_str("dtype", metadata.get("dtype")))
        _validate_supported_dtype(dtype)
        shape = _decode_shape(metadata.get("shape"))
        rank = _decode_int("rank", metadata.get("rank"))
        vector_count = _decode_int("vector_count", metadata.get("vector_count"))
        dimension = _decode_int("dimension", metadata.get("dimension"))
        total_bits = _decode_int("total_bits_per_scalar", metadata.get("total_bits_per_scalar"))
        total_bits = _validate_total_bits_per_scalar(total_bits)
        base_bits = _decode_int("base_bits_per_scalar", metadata.get("base_bits_per_scalar"))
        if base_bits != total_bits - 1:
            raise DecodeError("base_bits_per_scalar must equal total_bits_per_scalar - 1")
        if base_bits < 1 or base_bits > 4:
            raise DecodeError("base_bits_per_scalar must be between 1 and 4")
        base_rotation_seed = _decode_int("base_rotation_seed", metadata.get("base_rotation_seed"))
        qjl_seed = _decode_int("qjl_seed", metadata.get("qjl_seed"))
        rotation_family = _decode_required_str("rotation_family", metadata.get("rotation_family"))
        normalization = _decode_required_str("normalization", metadata.get("normalization"))
        codebook_family = _decode_required_str("codebook_family", metadata.get("codebook_family"))
        qjl_family = _decode_required_str("qjl_family", metadata.get("qjl_family"))
        centers = _decode_float_list("codebook_centers", metadata.get("codebook_centers"))
        solver_iterations = metadata.get("solver_iterations", 0)
        if not isinstance(solver_iterations, int) or isinstance(solver_iterations, bool) or solver_iterations < 0:
            raise DecodeError("solver_iterations must be a non-negative int when present")
        solver_converged = metadata.get("solver_converged", True)
        if not isinstance(solver_converged, bool):
            raise DecodeError("solver_converged must be a bool when present")

        if rotation_family != "seeded_haar_qr":
            raise DecodeError("unsupported rotation_family")
        if normalization != "row_l2":
            raise DecodeError("unsupported normalization")
        if codebook_family != "beta_lloyd_max":
            raise DecodeError("unsupported codebook_family")
        if qjl_family != "seeded_gaussian_sign_v1":
            raise DecodeError("unsupported qjl_family")
        if len(centers) != (1 << base_bits):
            raise DecodeError("codebook_centers size does not match base_bits_per_scalar")

        self._validate_shape_metadata(shape=shape, rank=rank, vector_count=vector_count, dimension=dimension)
        return {
            "dtype": dtype,
            "shape": shape,
            "rank": rank,
            "vector_count": vector_count,
            "dimension": dimension,
            "total_bits_per_scalar": total_bits,
            "base_bits_per_scalar": base_bits,
            "base_rotation_seed": base_rotation_seed,
            "qjl_seed": qjl_seed,
            "codebook_centers": centers,
            "solver_iterations": solver_iterations,
            "solver_converged": solver_converged,
        }

    @staticmethod
    def _validate_shape_metadata(
        *,
        shape: tuple[int, ...],
        rank: int,
        vector_count: int,
        dimension: int,
    ) -> None:
        if rank not in (1, 2):
            raise DecodeError("rank must be 1 or 2")
        if len(shape) != rank:
            raise DecodeError("shape length must match rank")
        if vector_count <= 0 or dimension <= 0:
            raise DecodeError("vector_count and dimension must be positive")
        if rank == 1:
            if vector_count != 1 or shape != (dimension,):
                raise DecodeError("rank-1 TurboQuant metadata is inconsistent")
            return
        if shape != (vector_count, dimension):
            raise DecodeError("rank-2 TurboQuant metadata is inconsistent")

    @staticmethod
    def _require_single_segment(encoding: VectorEncoding, segment_kind: EncodingSegmentKind) -> VectorEncodingSegment:
        matches = [segment for segment in encoding.segments if segment.segment_kind == segment_kind]
        if len(matches) != 1:
            raise DecodeError(f"expected exactly one {segment_kind!r} segment")
        return matches[0]
