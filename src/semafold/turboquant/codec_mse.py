"""TurboQuant MSE codec implemented against the Semafold vector envelope."""

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
from semafold.turboquant.codebook import (
    TurboQuantScalarCodebook,
    numerical_codebook_distortion,
    solve_beta_lloyd_max_codebook,
)
from semafold.turboquant.packing import pack_scalar_indices, packed_byte_count, unpack_scalar_indices
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

__all__ = ["TurboQuantMSEConfig", "TurboQuantMSEVectorCodec"]


class _DecodedMetadata(TypedDict):
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...]
    rank: int
    vector_count: int
    dimension: int
    bits_per_scalar: int
    rotation_seed: int
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
        raise CompatibilityError("TurboQuantMSEVectorCodec requires float16, float32, or float64 input")


def _validate_bits_per_scalar(bits_per_scalar: int) -> int:
    if not isinstance(bits_per_scalar, int) or isinstance(bits_per_scalar, bool):
        raise TypeError("bits_per_scalar must be an int")
    if bits_per_scalar < 1 or bits_per_scalar > 4:
        raise ValueError("bits_per_scalar must be between 1 and 4")
    return bits_per_scalar


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


@dataclass(frozen=True, slots=True)
class TurboQuantMSEConfig:
    """Encode-time defaults for the TurboQuant MSE codec."""

    default_bits_per_scalar: int = 3
    default_rotation_seed: int = 0
    normalization: Literal["row_l2"] = "row_l2"
    grid_size: int = 8193
    max_iterations: int = 128
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_bits_per_scalar", _validate_bits_per_scalar(self.default_bits_per_scalar))
        object.__setattr__(self, "default_rotation_seed", _validate_seed(self.default_rotation_seed))
        if self.normalization != "row_l2":
            raise ValueError("normalization must be 'row_l2' in the current TurboQuant MSE preview")
        object.__setattr__(self, "grid_size", _validate_grid_size(self.grid_size))
        object.__setattr__(self, "max_iterations", _validate_iterations(self.max_iterations))
        object.__setattr__(self, "tolerance", _validate_tolerance(self.tolerance))


class TurboQuantMSEVectorCodec:
    """TurboQuant codec optimized for reconstruction-oriented workloads.

    This codec applies row normalization, deterministic orthogonal rotation,
    Beta-distribution Lloyd-Max scalar quantization, and packed index storage.
    """

    codec_family = "turboquant"
    variant_id = "mse_beta_lloyd_qr_v2"
    encoding_schema_version = "vector.encoding.v1"
    _supported_objectives = {EncodeObjective.RECONSTRUCTION, EncodeObjective.STORAGE_ONLY}
    _supported_metrics = {None, EncodeMetric.MSE, EncodeMetric.L2}
    def __init__(self, *, config: TurboQuantMSEConfig | None = None) -> None:
        if config is not None and not isinstance(config, TurboQuantMSEConfig):
            raise TypeError("config must be a TurboQuantMSEConfig or None")
        self.config = config if config is not None else TurboQuantMSEConfig()

    def estimate(self, request: VectorEncodeRequest) -> CompressionEstimate:
        """Estimate artifact size for a TurboQuant MSE encoding request."""
        array_2d, _, _ = self._validate_request(request)
        bits_per_scalar = self.config.default_bits_per_scalar
        rotation_seed = self._resolve_rotation_seed(request.seed)
        codebook = self._solve_codebook(int(array_2d.shape[1]), bits_per_scalar)
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=tuple(int(dim) for dim in request.data.shape),
            original_rank=int(request.data.ndim),
            bits_per_scalar=bits_per_scalar,
            rotation_seed=rotation_seed,
            codebook=codebook,
        )
        payload_bytes = packed_byte_count(int(array_2d.size), bits_per_scalar)
        sidecar_bytes = int(array_2d.shape[0] * np.dtype(np.float32).itemsize)
        metadata_bytes = json_byte_size(metadata_payload)
        total_bytes = payload_bytes + sidecar_bytes + metadata_bytes
        baseline_bytes = int(request.data.nbytes)
        return CompressionEstimate(
            baseline_bytes=baseline_bytes,
            estimated_payload_bytes=payload_bytes,
            estimated_metadata_bytes=metadata_bytes,
            estimated_sidecar_bytes=sidecar_bytes,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=total_bytes,
            estimated_compression_ratio=(float(baseline_bytes) / float(total_bytes) if total_bytes > 0 else 0.0),
        )

    def encode(self, request: VectorEncodeRequest) -> VectorEncoding:
        """Encode vectors into a TurboQuant MSE artifact with measured accounting."""
        array_2d, original_shape, original_rank = self._validate_request(request)
        bits_per_scalar = self.config.default_bits_per_scalar
        rotation_seed = self._resolve_rotation_seed(request.seed)
        codebook = self._solve_codebook(int(array_2d.shape[1]), bits_per_scalar)
        rotation = seeded_haar_rotation(int(array_2d.shape[1]), rotation_seed)

        unit_rows, norms = normalize_rows(array_2d)
        indices = quantize_rows(unit_rows, rotation=rotation, codebook=codebook)
        packed = pack_scalar_indices(indices.reshape(-1), bits_per_scalar)
        sidecar = np.ascontiguousarray(norms).tobytes()

        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=original_shape,
            original_rank=original_rank,
            bits_per_scalar=bits_per_scalar,
            rotation_seed=rotation_seed,
            codebook=codebook,
        )
        compressed_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.COMPRESSED,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=packed,
            payload_format=f"bitpack.lsb.u{bits_per_scalar}",
            footprint=segment_footprint(payload_bytes=len(packed)),
            metadata={},
        )
        sidecar_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.SIDECAR,
            role=request.role,
            scope=_full_scope(array_2d),
            payload=sidecar,
            payload_format="numpy.float32.row_norm",
            footprint=segment_footprint(sidecar_bytes=len(sidecar)),
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
                metadata_segment.footprint,
            ],
        )

        decoded_unit = dequantize_rows(indices, rotation=rotation, codebook=codebook)
        decoded_rows = restore_rows(decoded_unit, norms)
        restored = restore_from_2d(decoded_rows.astype(request.data.dtype, copy=False), original_shape, original_rank)

        observed_mse = float(np.mean((request.data.astype(np.float64) - restored.astype(np.float64)) ** 2))
        row_l2_squared = float(
            np.mean(
                np.sum(
                    (array_2d.astype(np.float64) - decoded_rows.astype(np.float64)) ** 2,
                    axis=1,
                ),
            )
        )
        max_abs_error = float(np.max(np.abs(request.data.astype(np.float64) - restored.astype(np.float64))))
        numerical_reference = numerical_codebook_distortion(
            int(array_2d.shape[1]),
            codebook,
            grid_size=self.config.grid_size,
        )
        paper_reference = (math.sqrt(3.0 * math.pi) / 2.0) * (4.0 ** (-bits_per_scalar))

        guarantees = [
            CompressionGuarantee(
                objective=request.objective,
                metric="observed_mse",
                bound_type=EncodingBoundType.OBSERVED,
                value=observed_mse,
                units="mean_squared_error",
                scope="full_tensor",
                workload_suitability=[
                    WorkloadSuitability.EMBEDDING_STORAGE,
                    WorkloadSuitability.RECONSTRUCTION_ONLY,
                    WorkloadSuitability.VECTOR_DATABASE,
                ],
                notes="Observed reconstruction MSE for this TurboQuant preview artifact.",
            ),
            CompressionGuarantee(
                objective=EncodeObjective.RECONSTRUCTION,
                metric="unit_norm_row_l2_squared_distortion",
                bound_type=EncodingBoundType.PAPER_REFERENCE,
                value=paper_reference,
                units="expected_l2_squared",
                scope="unit_norm_row",
                workload_suitability=[
                    WorkloadSuitability.EMBEDDING_STORAGE,
                    WorkloadSuitability.VECTOR_DATABASE,
                ],
                notes="Theorem 1 reference constant for unit-norm rows: (sqrt(3*pi)/2) * 4^-b.",
            ),
        ]
        evidence = [
            ValidationEvidence(
                scope="proxy_fidelity",
                environment={
                    "codec_family": self.codec_family,
                    "variant_id": self.variant_id,
                    "dtype": str(request.data.dtype),
                    "rotation_seed": rotation_seed,
                    "bits_per_scalar": bits_per_scalar,
                },
                metrics={
                    "observed_mse": observed_mse,
                    "mean_row_l2_squared_error": row_l2_squared,
                    "max_abs_error": max_abs_error,
                    "expected_coordinate_mse": codebook.expected_coordinate_mse,
                    "expected_unit_norm_row_l2_squared_distortion": numerical_reference,
                    "solver_iterations": codebook.iterations,
                    "solver_converged": codebook.converged,
                },
                passed=True,
                summary="TurboQuant MSE preview fidelity measured on the current payload.",
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
                },
                passed=True,
                summary="TurboQuant MSE preview accounting computed exactly.",
                artifact_refs=[],
            ),
        ]
        config = self._config_payload(bits_per_scalar=bits_per_scalar, rotation_seed=rotation_seed)
        return VectorEncoding(
            codec_family=self.codec_family,
            variant_id=self.variant_id,
            profile_id=request.profile_id,
            implementation_version=__version__,
            encoding_schema_version=self.encoding_schema_version,
            config_fingerprint=fingerprint_config(config),
            segments=[compressed_segment, sidecar_segment, metadata_segment],
            footprint=footprint,
            guarantees=guarantees,
            evidence=evidence,
            metadata={"objective": request.objective, "mode": "mse"},
        )

    def decode(self, request: VectorDecodeRequest) -> VectorDecodeResult:
        """Decode a TurboQuant MSE artifact back into its original tensor shape."""
        encoding = request.encoding
        if encoding.codec_family != self.codec_family:
            raise CompatibilityError("encoding does not belong to the TurboQuant codec family")
        if encoding.variant_id != self.variant_id:
            raise CompatibilityError("encoding does not belong to a supported TurboQuant MSE preview variant")
        if encoding.encoding_schema_version != self.encoding_schema_version:
            raise CompatibilityError("unsupported encoding schema version")

        compressed_segment = self._require_single_segment(encoding, EncodingSegmentKind.COMPRESSED)
        sidecar_segment = self._require_single_segment(encoding, EncodingSegmentKind.SIDECAR)
        metadata_segment = self._require_single_segment(encoding, EncodingSegmentKind.METADATA)

        if not isinstance(compressed_segment.payload, bytes):
            raise DecodeError("compressed payload must be bytes")
        if not isinstance(sidecar_segment.payload, bytes):
            raise DecodeError("sidecar payload must be bytes")
        if not isinstance(metadata_segment.payload, dict):
            raise DecodeError("metadata payload must be a dict")
        if metadata_segment.payload_format != "json":
            raise DecodeError("unexpected metadata payload_format")
        if sidecar_segment.payload_format != "numpy.float32.row_norm":
            raise DecodeError("unexpected sidecar payload_format")

        metadata = self._decode_metadata(metadata_segment.payload, encoding.variant_id)
        expected_payload_format = f"bitpack.lsb.u{metadata['bits_per_scalar']}"
        if compressed_segment.payload_format != expected_payload_format:
            raise DecodeError("compressed payload_format does not match metadata bits_per_scalar")

        vector_count = metadata["vector_count"]
        dimension = metadata["dimension"]
        bits_per_scalar = metadata["bits_per_scalar"]
        expected_packed_bytes = packed_byte_count(vector_count * dimension, bits_per_scalar)
        if len(compressed_segment.payload) != expected_packed_bytes:
            raise DecodeError("compressed payload size does not match metadata")

        expected_sidecar_bytes = vector_count * np.dtype(np.float32).itemsize
        if len(sidecar_segment.payload) != expected_sidecar_bytes:
            raise DecodeError("sidecar payload size does not match metadata")

        try:
            norms = np.frombuffer(sidecar_segment.payload, dtype=np.float32).copy()
        except ValueError as exc:  # pragma: no cover - defensive
            raise DecodeError("sidecar payload cannot be interpreted as float32 row norms") from exc
        if norms.shape != (vector_count,):
            raise DecodeError("sidecar row norms do not match metadata")
        if not np.isfinite(norms).all() or np.any(norms < 0.0):
            raise DecodeError("sidecar payload contains invalid row norms")

        indices = unpack_scalar_indices(
            compressed_segment.payload,
            count=vector_count * dimension,
            bits_per_index=bits_per_scalar,
        ).reshape(vector_count, dimension)
        rotation = seeded_haar_rotation(dimension, metadata["rotation_seed"])
        codebook = TurboQuantScalarCodebook.from_centers(
            dimension=dimension,
            bits_per_scalar=bits_per_scalar,
            centers=np.asarray(metadata["codebook_centers"], dtype=np.float64),
            iterations=metadata["solver_iterations"],
            converged=metadata["solver_converged"],
        )
        restored_unit = dequantize_rows(indices, rotation=rotation, codebook=codebook)
        restored_rows = restore_rows(restored_unit, norms)
        dtype = np.dtype(metadata["dtype"])
        restored = restore_from_2d(
            restored_rows.astype(dtype, copy=False),
            metadata["shape"],
            metadata["rank"],
        )
        notes: list[str] = []
        if request.target_layout is not None:
            notes.append(f"target_layout={request.target_layout!r} ignored in the current TurboQuant preview.")
        return VectorDecodeResult(data=restored, metadata={}, materialization_notes=notes)

    def _validate_request(self, request: VectorEncodeRequest) -> tuple[np.ndarray, tuple[int, ...], int]:
        if request.objective not in self._supported_objectives:
            raise CompatibilityError(
                "TurboQuantMSEVectorCodec only supports 'reconstruction' and 'storage_only' objectives"
            )
        if request.metric not in self._supported_metrics:
            raise CompatibilityError("TurboQuantMSEVectorCodec only supports metric=None, 'mse', or 'l2'")
        _validate_supported_dtype(request.data.dtype)
        if not np.isfinite(request.data).all():
            raise CompatibilityError("TurboQuantMSEVectorCodec requires finite floating-point values")
        array_2d, original_shape, original_rank = normalize_to_2d(request.data)
        if int(array_2d.shape[1]) < 2:
            raise CompatibilityError("TurboQuantMSEVectorCodec requires dimension >= 2")
        return array_2d, original_shape, original_rank

    def _resolve_rotation_seed(self, request_seed: int | None) -> int:
        return self.config.default_rotation_seed if request_seed is None else _validate_seed(request_seed)

    def _solve_codebook(self, dimension: int, bits_per_scalar: int) -> TurboQuantScalarCodebook:
        return solve_beta_lloyd_max_codebook(
            dimension,
            bits_per_scalar,
            grid_size=self.config.grid_size,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
        )

    def _config_payload(self, *, bits_per_scalar: int, rotation_seed: int) -> dict[str, object]:
        return {
            "codec_family": self.codec_family,
            "variant_id": self.variant_id,
            "encoding_schema_version": self.encoding_schema_version,
            "mode": "mse",
            "bits_per_scalar": bits_per_scalar,
            "rotation_family": "seeded_haar_qr",
            "rotation_seed": rotation_seed,
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
        bits_per_scalar: int,
        rotation_seed: int,
        codebook: TurboQuantScalarCodebook,
    ) -> dict[str, object]:
        metadata = {
            "format": self.variant_id,
            "mode": "mse",
            "shape": list(original_shape),
            "rank": original_rank,
            "dtype": str(data.dtype),
            "vector_count": int(array_2d.shape[0]),
            "dimension": int(array_2d.shape[1]),
            "bits_per_scalar": bits_per_scalar,
            "rotation_seed": rotation_seed,
            "rotation_family": "seeded_haar_qr",
            "normalization": self.config.normalization,
            "codebook_family": "beta_lloyd_max",
            "codebook_centers": [float(item) for item in np.asarray(codebook.centers, dtype=np.float64)],
        }
        metadata["solver_iterations"] = codebook.iterations
        metadata["solver_converged"] = codebook.converged
        return metadata

    def _decode_metadata(self, metadata: Mapping[str, object], envelope_variant_id: str) -> _DecodedMetadata:
        format_value = _decode_required_str("format", metadata.get("format"))
        if format_value != self.variant_id:
            raise DecodeError("metadata format does not match codec variant")
        if envelope_variant_id != self.variant_id:
            raise DecodeError("encoding variant is not a supported TurboQuant MSE preview variant")
        if format_value != envelope_variant_id:
            raise DecodeError("metadata format does not match codec variant")

        mode = metadata.get("mode")
        if mode != "mse":
            raise DecodeError("TurboQuant preview metadata mode must be 'mse'")

        dtype = np.dtype(_decode_required_str("dtype", metadata.get("dtype")))
        _validate_supported_dtype(dtype)
        shape = _decode_shape(metadata.get("shape"))
        rank = _decode_int("rank", metadata.get("rank"))
        vector_count = _decode_int("vector_count", metadata.get("vector_count"))
        dimension = _decode_int("dimension", metadata.get("dimension"))
        bits_per_scalar = _decode_int(
            "bits_per_scalar",
            metadata.get("bits_per_scalar"),
        )
        bits_per_scalar = _validate_bits_per_scalar(bits_per_scalar)
        rotation_seed = _decode_int("rotation_seed", metadata.get("rotation_seed"))
        rotation_family = _decode_required_str(
            "rotation_family",
            metadata.get("rotation_family", "seeded_haar_qr"),
        )
        normalization = _decode_required_str(
            "normalization",
            metadata.get("normalization", "row_l2"),
        )
        codebook_family = _decode_required_str(
            "codebook_family",
            metadata.get("codebook_family", "beta_lloyd_max"),
        )
        centers = _decode_float_list(
            "codebook_centers",
            metadata.get("codebook_centers"),
        )
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
        if len(centers) != (1 << bits_per_scalar):
            raise DecodeError("codebook_centers size does not match bits_per_scalar")

        self._validate_shape_metadata(
            shape=shape,
            rank=rank,
            vector_count=vector_count,
            dimension=dimension,
        )
        return {
            "dtype": dtype,
            "shape": shape,
            "rank": rank,
            "vector_count": vector_count,
            "dimension": dimension,
            "bits_per_scalar": bits_per_scalar,
            "rotation_seed": rotation_seed,
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
