"""Exact passthrough codec used as the lossless Semafold baseline."""

from __future__ import annotations

import numpy as np

from semafold._version import __version__
from semafold.core import (
    CompressionEstimate,
    CompressionGuarantee,
    ValidationEvidence,
)
from semafold.core.accounting import aggregate_footprints, json_byte_size, segment_footprint
from semafold.errors import CompatibilityError, DecodeError
from semafold.vector.models import (
    VectorDecodeRequest,
    VectorDecodeResult,
    VectorEncodeRequest,
    VectorEncoding,
    VectorEncodingSegment,
    array_layout,
    fingerprint_config,
    normalize_to_2d,
)

__all__ = ["PassthroughVectorCodec"]


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


class PassthroughVectorCodec:
    """Lossless baseline codec that stores the original numeric payload verbatim."""

    codec_family = "passthrough"
    variant_id = "raw_bytes_v1"
    encoding_schema_version = "vector.encoding.v1"

    def estimate(self, request: VectorEncodeRequest) -> CompressionEstimate:
        """Estimate the exact byte cost of storing ``request.data`` verbatim."""
        self._validate_request(request)
        array_2d, original_shape, original_rank = normalize_to_2d(request.data)
        payload_bytes = len(np.ascontiguousarray(request.data).tobytes())
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=original_shape,
            original_rank=original_rank,
        )
        metadata_bytes = json_byte_size(metadata_payload)
        total_bytes = payload_bytes + metadata_bytes
        baseline_bytes = int(request.data.nbytes)
        return CompressionEstimate(
            baseline_bytes=baseline_bytes,
            estimated_payload_bytes=payload_bytes,
            estimated_metadata_bytes=metadata_bytes,
            estimated_sidecar_bytes=0,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=total_bytes,
            estimated_compression_ratio=(float(baseline_bytes) / float(total_bytes) if total_bytes > 0 else 0.0),
        )

    def encode(self, request: VectorEncodeRequest) -> VectorEncoding:
        """Emit a passthrough artifact containing raw NumPy bytes and metadata."""
        self._validate_request(request)
        array_2d, original_shape, original_rank = normalize_to_2d(request.data)
        raw_bytes = np.ascontiguousarray(request.data).tobytes()
        payload_segment = VectorEncodingSegment(
            segment_kind="passthrough",
            role=request.role,
            scope=self._full_scope(array_2d),
            payload=raw_bytes,
            payload_format="numpy.raw",
            footprint=segment_footprint(payload_bytes=len(raw_bytes)),
            metadata={},
        )
        metadata_payload = self._metadata_payload(
            data=request.data,
            array_2d=array_2d,
            original_shape=original_shape,
            original_rank=original_rank,
        )
        metadata_segment = VectorEncodingSegment(
            segment_kind="metadata",
            role=request.role,
            scope={"kind": "encoding_metadata"},
            payload=metadata_payload,
            payload_format="json",
            footprint=segment_footprint(metadata_bytes=json_byte_size(metadata_payload)),
            metadata={},
        )
        footprint = aggregate_footprints(
            baseline_bytes=int(request.data.nbytes),
            segment_footprints=[payload_segment.footprint, metadata_segment.footprint],
        )
        guarantees = [
            CompressionGuarantee(
                objective=request.objective,
                metric="exact_roundtrip",
                bound_type="exact",
                value=True,
                scope="full_tensor",
                workload_suitability=["embedding_storage", "reconstruction_only"],
                notes="Passthrough preserves every scalar exactly.",
            ),
            CompressionGuarantee(
                objective="storage_only",
                metric="accounting_exactness",
                bound_type="exact",
                value=True,
                scope="envelope",
                notes="Measured footprint is emitted exactly.",
            ),
        ]
        evidence = [
            ValidationEvidence(
                scope="compatibility",
                environment={"codec_family": self.codec_family, "variant_id": self.variant_id},
                metrics={
                    "rank": int(request.data.ndim),
                    "vector_count": int(array_2d.shape[0]),
                    "dimension": int(array_2d.shape[1]),
                },
                passed=True,
                summary="Passthrough request accepted.",
                artifact_refs=[],
            ),
            ValidationEvidence(
                scope="storage_accounting",
                environment={"codec_family": self.codec_family},
                metrics={
                    "baseline_bytes": footprint.baseline_bytes,
                    "payload_bytes": footprint.payload_bytes,
                    "metadata_bytes": footprint.metadata_bytes,
                    "total_bytes": footprint.total_bytes,
                    "compression_ratio": footprint.compression_ratio,
                },
                passed=True,
                summary="Passthrough accounting computed exactly.",
                artifact_refs=[],
            ),
        ]
        config = {
            "codec_family": self.codec_family,
            "variant_id": self.variant_id,
            "encoding_schema_version": self.encoding_schema_version,
        }
        return VectorEncoding(
            codec_family=self.codec_family,
            variant_id=self.variant_id,
            profile_id=request.profile_id,
            implementation_version=__version__,
            encoding_schema_version=self.encoding_schema_version,
            config_fingerprint=fingerprint_config(config),
            segments=[payload_segment, metadata_segment],
            footprint=footprint,
            guarantees=guarantees,
            evidence=evidence,
            metadata={"objective": request.objective},
        )

    def decode(self, request: VectorDecodeRequest) -> VectorDecodeResult:
        """Reconstruct the original numeric tensor from a passthrough artifact."""
        encoding = request.encoding
        if encoding.codec_family != self.codec_family or encoding.variant_id != self.variant_id:
            raise CompatibilityError("encoding does not belong to PassthroughVectorCodec")
        if encoding.encoding_schema_version != self.encoding_schema_version:
            raise CompatibilityError("unsupported encoding schema version")
        payload_segment = self._require_single_segment(encoding, "passthrough")
        metadata_segment = self._require_single_segment(encoding, "metadata")
        if not isinstance(payload_segment.payload, bytes):
            raise DecodeError("passthrough payload must be bytes")
        if not isinstance(metadata_segment.payload, dict):
            raise DecodeError("metadata payload must be a dict")
        if payload_segment.payload_format != "numpy.raw":
            raise DecodeError("unexpected passthrough payload_format")
        if metadata_segment.payload_format != "json":
            raise DecodeError("unexpected metadata payload_format")
        metadata = metadata_segment.payload
        try:
            dtype = np.dtype(str(metadata["dtype"]))
            shape = _decode_shape(metadata["shape"])
            rank = _decode_int("rank", metadata["rank"])
            vector_count = _decode_int("vector_count", metadata["vector_count"])
            dimension = _decode_int("dimension", metadata["dimension"])
        except Exception as exc:  # pragma: no cover - defensive
            raise DecodeError("invalid passthrough metadata payload") from exc
        if not np.issubdtype(dtype, np.number):
            raise DecodeError("passthrough dtype must be numeric")
        self._validate_shape_metadata(
            shape=shape,
            rank=rank,
            vector_count=vector_count,
            dimension=dimension,
        )
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected_bytes != len(payload_segment.payload):
            raise DecodeError("passthrough payload size does not match metadata")
        try:
            array = np.frombuffer(payload_segment.payload, dtype=dtype).copy().reshape(shape)
        except ValueError as exc:
            raise DecodeError("passthrough payload cannot be reshaped from metadata") from exc
        notes: list[str] = []
        if request.target_layout is not None:
            notes.append(f"target_layout={request.target_layout!r} ignored in Phase 1.")
        return VectorDecodeResult(data=array, metadata={}, materialization_notes=notes)

    def _validate_request(self, request: VectorEncodeRequest) -> None:
        if request.seed is not None:
            raise CompatibilityError("PassthroughVectorCodec does not support seed in Phase 1")
        if not np.issubdtype(request.data.dtype, np.number):
            raise CompatibilityError("PassthroughVectorCodec requires a numeric NumPy dtype")

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
            "layout": array_layout(data),
            "vector_count": int(array_2d.shape[0]),
            "dimension": int(array_2d.shape[1]),
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
                raise DecodeError("rank-1 passthrough metadata is inconsistent")
            return
        if shape != (vector_count, dimension):
            raise DecodeError("rank-2 passthrough metadata is inconsistent")

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
    def _require_single_segment(
        encoding: VectorEncoding,
        segment_kind: str,
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
