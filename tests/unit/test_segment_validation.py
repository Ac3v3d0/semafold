from __future__ import annotations

import pytest

from semafold import EncodingSegmentKind
from semafold import VectorEncodingSegment


def test_segment_rejects_invalid_kind_string() -> None:
    with pytest.raises(ValueError):
        VectorEncodingSegment(
            segment_kind="not_a_valid_kind",  # type: ignore[arg-type]
            role=None,
            scope={},
            payload=b"abc",
            payload_format="raw",
        )


def test_segment_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError):
        VectorEncodingSegment(
            segment_kind=EncodingSegmentKind.METADATA,
            role=None,
            scope={},
            payload=123,  # type: ignore[arg-type]
            payload_format="json",
        )
