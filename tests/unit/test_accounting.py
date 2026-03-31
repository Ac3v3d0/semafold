from __future__ import annotations

import pytest

from semafold.core.accounting import CompressionEstimate
from semafold.core.accounting import CompressionFootprint
from semafold.core.accounting import aggregate_footprints
from semafold.core.accounting import build_footprint
from semafold.core.accounting import segment_footprint


def test_build_footprint_invariants() -> None:
    footprint = build_footprint(
        baseline_bytes=100,
        payload_bytes=40,
        metadata_bytes=10,
        sidecar_bytes=5,
        protected_passthrough_bytes=0,
        decoder_state_bytes=0,
    )
    assert footprint.total_bytes == 55
    assert footprint.bytes_saved == 45


def test_aggregate_footprints() -> None:
    footprint = aggregate_footprints(
        baseline_bytes=64,
        segment_footprints=[
            segment_footprint(payload_bytes=16),
            segment_footprint(metadata_bytes=4),
            segment_footprint(sidecar_bytes=8),
        ],
    )
    assert footprint.total_bytes == 28
    assert footprint.payload_bytes == 16
    assert footprint.metadata_bytes == 4
    assert footprint.sidecar_bytes == 8


def test_build_footprint_allows_expansion() -> None:
    footprint = build_footprint(
        baseline_bytes=8,
        payload_bytes=12,
        metadata_bytes=4,
    )
    assert footprint.total_bytes == 16
    assert footprint.bytes_saved == -8


def test_estimate_rejects_mismatched_total() -> None:
    with pytest.raises(ValueError):
        CompressionEstimate(
            baseline_bytes=100,
            estimated_payload_bytes=40,
            estimated_metadata_bytes=5,
            estimated_sidecar_bytes=5,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=60,
        )


def test_estimate_rejects_mismatched_ratio() -> None:
    with pytest.raises(ValueError):
        CompressionEstimate(
            baseline_bytes=100,
            estimated_payload_bytes=40,
            estimated_metadata_bytes=5,
            estimated_sidecar_bytes=5,
            estimated_protected_passthrough_bytes=0,
            estimated_decoder_state_bytes=0,
            estimated_total_bytes=50,
            estimated_compression_ratio=3.0,
        )


def test_footprint_rejects_negative_baseline() -> None:
    with pytest.raises(ValueError):
        CompressionFootprint(
            baseline_bytes=-1,
            payload_bytes=0,
            metadata_bytes=0,
            sidecar_bytes=0,
            protected_passthrough_bytes=0,
            decoder_state_bytes=0,
            total_bytes=0,
            bytes_saved=-1,
            compression_ratio=0.0,
        )


def test_segment_footprint_rejects_fractional_bytes() -> None:
    with pytest.raises(TypeError):
        segment_footprint(payload_bytes=1.5)  # type: ignore[arg-type]


def test_aggregate_footprints_rejects_mismatched_segment_total() -> None:
    with pytest.raises(ValueError):
        aggregate_footprints(
            baseline_bytes=32,
            segment_footprints=[
                {
                    "payload_bytes": 10,
                    "metadata_bytes": 2,
                    "sidecar_bytes": 0,
                    "protected_passthrough_bytes": 0,
                    "decoder_state_bytes": 0,
                    "total_bytes": 99,
                },
            ],
        )


def test_aggregate_footprints_rejects_non_mapping_segments() -> None:
    with pytest.raises(TypeError):
        aggregate_footprints(
            baseline_bytes=32,
            segment_footprints=[123],  # type: ignore[list-item]
        )
