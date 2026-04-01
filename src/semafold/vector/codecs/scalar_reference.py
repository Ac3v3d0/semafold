"""Deterministic scalar baseline used for correctness-first lossy comparisons."""

from __future__ import annotations

import numpy as np

from semafold._version import __version__
from semafold.core import (
    CompressionEstimate,
    CompressionGuarantee,
    ValidationEvidence,
)
from semafold.core.models import EncodingBoundType, WorkloadSuitability
from semafold.core.accounting import aggregate_footprints, json_byte_size, segment_footprint
from semafold.errors import CompatibilityError, DecodeError
from semafold.vector.models import (
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

__all__ = ["ScalarReferenceVectorCodec"]


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


class ScalarReferenceVectorCodec:
    """Row-wise affine scalar codec that serves as a simple lossy baseline."""

    codec_family = "scalar_reference"
    variant_id = "uniform_affine_u8_v1"
    encoding_schema_version = "vector.encoding.v1"
    _supported_objectives = {EncodeObjective.RECONSTRUCTION, EncodeObjective.STORAGE_ONLY}
    _supported_dtypes = {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }

    def estimate(self, request: VectorEncodeRequest) -> CompressionEstimate:
        """Estimate payload, sidecar, and metadata bytes for the scalar baseline."""
        self._validate_request(request)
        array_2d, original_shape, original_rank = normalize_to_2d(request.data)
        compressed_bytes = int(array_2d.size)
        sidecar_bytes = int(array_2d.shape[0] * 2 * np.dtype(np.float32).itemsize)
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=original_shape,
            original_rank=original_rank,
        )
        metadata_bytes = json_byte_size(metadata_payload)
        total_bytes = compressed_bytes + sidecar_bytes + metadata_bytes
        baseline_bytes = int(request.data.nbytes)
        return CompressionEstimate(
            baseline_bytes=baseline_bytes,
            estimated_payload_bytes=compressed_bytes,
            estimated_metadata_bytes=metadata_bytes,
            estimated_sidecar_bytes=sidecar_bytes,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=total_bytes,
            estimated_compression_ratio=(float(baseline_bytes) / float(total_bytes) if total_bytes > 0 else 0.0),
        )

    def encode(self, request: VectorEncodeRequest) -> VectorEncoding:
        """Encode floating-point vectors into an 8-bit row-wise affine artifact."""
        self._validate_request(request)
        original_2d, original_shape, original_rank = normalize_to_2d(request.data)
        working = original_2d.astype(np.float32, copy=False)
        quantized, min_values, max_values = self._quantize_rows(working)
        compressed_bytes = np.ascontiguousarray(quantized).tobytes()
        minmax = np.stack([min_values, max_values], axis=1).astype(np.float32, copy=False)
        sidecar_bytes = np.ascontiguousarray(minmax).tobytes()
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=working,
            original_shape=original_shape,
            original_rank=original_rank,
        )

        compressed_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.COMPRESSED,
            role=request.role,
            scope=self._full_scope(working),
            payload=compressed_bytes,
            payload_format="numpy.uint8.row_major",
            footprint=segment_footprint(payload_bytes=len(compressed_bytes)),
            metadata={},
        )
        sidecar_segment = VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.SIDECAR,
            role=request.role,
            scope=self._full_scope(working),
            payload=sidecar_bytes,
            payload_format="numpy.float32.row_min_max",
            footprint=segment_footprint(sidecar_bytes=len(sidecar_bytes)),
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

        reconstructed = self._reconstruct(quantized, min_values, max_values).astype(request.data.dtype, copy=False)
        restored = restore_from_2d(reconstructed, original_shape, original_rank)
        mse = float(np.mean((request.data.astype(np.float64) - restored.astype(np.float64)) ** 2))
        max_abs_error = float(np.max(np.abs(request.data.astype(np.float64) - restored.astype(np.float64))))
        cosine_similarity = self._cosine_similarity(request.data.astype(np.float64), restored.astype(np.float64))

        guarantees = [
            CompressionGuarantee(
                objective=request.objective,
                metric="observed_mse",
                bound_type=EncodingBoundType.OBSERVED,
                value=mse,
                units="squared_error",
                scope="full_tensor",
                workload_suitability=[
                    WorkloadSuitability.EMBEDDING_STORAGE,
                    WorkloadSuitability.RECONSTRUCTION_ONLY,
                ],
                notes="Observed reconstruction metric for the current payload.",
            ),
            CompressionGuarantee(
                objective=EncodeObjective.STORAGE_ONLY,
                metric="accounting_exactness",
                bound_type=EncodingBoundType.EXACT,
                value=True,
                scope="envelope",
                notes="Measured footprint is emitted exactly.",
            ),
        ]
        evidence = [
            ValidationEvidence(
                scope="proxy_fidelity",
                environment={
                    "codec_family": self.codec_family,
                    "variant_id": self.variant_id,
                    "dtype": str(request.data.dtype),
                },
                metrics={
                    "mse": mse,
                    "max_abs_error": max_abs_error,
                    "cosine_similarity": cosine_similarity,
                },
                passed=True,
                summary="Scalar reference fidelity measured on the current payload.",
                artifact_refs=[],
            ),
            ValidationEvidence(
                scope="storage_accounting",
                environment={"codec_family": self.codec_family},
                metrics={
                    "baseline_bytes": footprint.baseline_bytes,
                    "payload_bytes": footprint.payload_bytes,
                    "metadata_bytes": footprint.metadata_bytes,
                    "sidecar_bytes": footprint.sidecar_bytes,
                    "total_bytes": footprint.total_bytes,
                    "compression_ratio": footprint.compression_ratio,
                },
                passed=True,
                summary="Scalar reference accounting computed exactly.",
                artifact_refs=[],
            ),
        ]
        config = {
            "codec_family": self.codec_family,
            "variant_id": self.variant_id,
            "encoding_schema_version": self.encoding_schema_version,
            "quantization": "rowwise_uniform_affine_u8",
        }
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
            metadata={"objective": request.objective},
        )

    def decode(self, request: VectorDecodeRequest) -> VectorDecodeResult:
        """Decode a scalar-reference artifact back into the original tensor shape."""
        encoding = request.encoding
        if encoding.codec_family != self.codec_family or encoding.variant_id != self.variant_id:
            raise CompatibilityError("encoding does not belong to ScalarReferenceVectorCodec")
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
        if compressed_segment.payload_format != "numpy.uint8.row_major":
            raise DecodeError("unexpected compressed payload_format")
        if sidecar_segment.payload_format != "numpy.float32.row_min_max":
            raise DecodeError("unexpected sidecar payload_format")
        if metadata_segment.payload_format != "json":
            raise DecodeError("unexpected metadata payload_format")
        metadata = metadata_segment.payload
        try:
            vector_count = _decode_int("vector_count", metadata["vector_count"])
            dimension = _decode_int("dimension", metadata["dimension"])
            dtype = np.dtype(str(metadata["dtype"]))
            original_shape = _decode_shape(metadata["shape"])
            original_rank = _decode_int("rank", metadata["rank"])
        except Exception as exc:  # pragma: no cover - defensive
            raise DecodeError("invalid scalar reference metadata payload") from exc
        if metadata.get("format") != self.variant_id:
            raise DecodeError("metadata format does not match codec variant")
        if dtype not in self._supported_dtypes:
            raise DecodeError("scalar reference dtype must be float16, float32, or float64")
        self._validate_shape_metadata(
            shape=original_shape,
            rank=original_rank,
            vector_count=vector_count,
            dimension=dimension,
        )
        expected_compressed_bytes = vector_count * dimension
        if len(compressed_segment.payload) != expected_compressed_bytes:
            raise DecodeError("compressed payload size does not match metadata")
        expected_sidecar_bytes = vector_count * 2 * np.dtype(np.float32).itemsize
        if len(sidecar_segment.payload) != expected_sidecar_bytes:
            raise DecodeError("sidecar payload size does not match metadata")
        try:
            quantized = (
                np.frombuffer(compressed_segment.payload, dtype=np.uint8)
                .copy()
                .reshape(vector_count, dimension)
            )
            minmax = (
                np.frombuffer(sidecar_segment.payload, dtype=np.float32)
                .copy()
                .reshape(vector_count, 2)
            )
        except ValueError as exc:
            raise DecodeError("scalar reference payload cannot be reshaped from metadata") from exc
        if not np.isfinite(minmax).all():
            raise DecodeError("sidecar payload contains non-finite values")
        if np.any(minmax[:, 1] < minmax[:, 0]):
            raise DecodeError("sidecar payload min/max ordering is invalid")
        reconstructed = self._reconstruct(quantized, minmax[:, 0], minmax[:, 1]).astype(dtype, copy=False)
        restored = restore_from_2d(reconstructed, original_shape, original_rank)
        notes: list[str] = []
        if request.target_layout is not None:
            notes.append(f"target_layout={request.target_layout!r} ignored in Phase 1.")
        return VectorDecodeResult(data=restored, metadata={}, materialization_notes=notes)

    def _validate_request(self, request: VectorEncodeRequest) -> None:
        if request.seed is not None:
            raise CompatibilityError("ScalarReferenceVectorCodec does not support seed in Phase 1")
        if request.objective not in self._supported_objectives:
            raise CompatibilityError(f"unsupported objective: {request.objective}")
        if request.data.dtype not in self._supported_dtypes:
            raise CompatibilityError("ScalarReferenceVectorCodec requires float16, float32, or float64")
        if not np.isfinite(request.data).all():
            raise CompatibilityError("ScalarReferenceVectorCodec requires finite floating-point values")

    def _metadata_payload(
        self,
        data: np.ndarray,
        array_2d: np.ndarray,
        original_shape: tuple[int, ...],
        original_rank: int,
    ) -> dict[str, object]:
        return {
            "shape": list(original_shape),
            "rank": original_rank,
            "dtype": str(data.dtype),
            "vector_count": int(array_2d.shape[0]),
            "dimension": int(array_2d.shape[1]),
            "format": self.variant_id,
        }

    @staticmethod
    def _full_scope(array_2d: np.ndarray) -> dict[str, object]:
        return {
            "kind": "full",
            "row_start": 0,
            "row_stop": int(array_2d.shape[0]),
            "col_start": 0,
            "col_stop": int(array_2d.shape[1]),
        }

    @staticmethod
    def _quantize_rows(array_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_values = array_2d.min(axis=1).astype(np.float32)
        max_values = array_2d.max(axis=1).astype(np.float32)
        ranges = max_values - min_values
        normalized = np.zeros_like(array_2d, dtype=np.float32)
        np.divide(
            array_2d - min_values[:, None],
            ranges[:, None],
            out=normalized,
            where=ranges[:, None] != 0.0,
        )
        quantized = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
        return quantized, min_values, max_values

    @staticmethod
    def _reconstruct(quantized: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
        ranges = (max_values - min_values).astype(np.float32)
        restored = min_values[:, None] + (quantized.astype(np.float32) / 255.0) * ranges[:, None]
        zero_range_rows = ranges == 0.0
        if np.any(zero_range_rows):
            restored[zero_range_rows] = min_values[zero_range_rows, None]
        return restored

    @staticmethod
    def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs_flat = lhs.reshape(-1)
        rhs_flat = rhs.reshape(-1)
        lhs_norm = float(np.linalg.norm(lhs_flat))
        rhs_norm = float(np.linalg.norm(rhs_flat))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return 1.0 if lhs_norm == rhs_norm else 0.0
        return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))

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
                raise DecodeError("rank-1 scalar reference metadata is inconsistent")
            return
        if shape != (vector_count, dimension):
            raise DecodeError("rank-2 scalar reference metadata is inconsistent")

    @staticmethod
    def _require_single_segment(
        encoding: VectorEncoding,
        segment_kind: EncodingSegmentKind,
    ) -> VectorEncodingSegment:
        matches = [
            segment
            for segment in encoding.segments
            if segment.segment_kind == segment_kind
        ]
        if not matches:
            raise DecodeError(f"missing required segment: {segment_kind}")
        if len(matches) > 1:
            raise DecodeError(f"multiple {segment_kind} segments are not supported in Phase 1")
        return matches[0]
