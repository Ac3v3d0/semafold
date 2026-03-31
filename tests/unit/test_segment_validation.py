from __future__ import annotations

import pytest

from semafold import VectorEncodingSegment


def test_segment_requires_non_empty_kind() -> None:
    with pytest.raises(TypeError):
        VectorEncodingSegment(
            segment_kind="",
            role=None,
            scope={},
            payload=b"abc",
            payload_format="raw",
        )


def test_segment_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError):
        VectorEncodingSegment(
            segment_kind="metadata",
            role=None,
            scope={},
            payload=123,  # type: ignore[arg-type]
            payload_format="json",
        )
