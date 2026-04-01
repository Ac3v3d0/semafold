from __future__ import annotations

import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorEncoding
from semafold import VectorEncodeRequest
from semafold.vector.codecs.scalar_reference import ScalarReferenceVectorCodec
from semafold.vector.models import EncodeObjective


def test_schema_version_is_consistent() -> None:
    payload = np.array([1.0, 2.0], dtype=np.float32)
    passthrough = PassthroughVectorCodec().encode(VectorEncodeRequest(data=payload, objective=EncodeObjective.RECONSTRUCTION))
    scalar = ScalarReferenceVectorCodec().encode(VectorEncodeRequest(data=payload, objective=EncodeObjective.RECONSTRUCTION))
    assert passthrough.encoding_schema_version == "vector.encoding.v1"
    assert scalar.encoding_schema_version == "vector.encoding.v1"


def test_schema_and_variant_identity_survive_wire_roundtrip() -> None:
    payload = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    passthrough = PassthroughVectorCodec().encode(
        VectorEncodeRequest(data=payload, objective=EncodeObjective.RECONSTRUCTION, profile_id="consumer.passthrough")
    )
    scalar = ScalarReferenceVectorCodec().encode(
        VectorEncodeRequest(data=payload, objective=EncodeObjective.RECONSTRUCTION, profile_id="consumer.scalar")
    )

    passthrough_wire = VectorEncoding.from_dict(passthrough.to_dict())
    scalar_wire = VectorEncoding.from_dict(scalar.to_dict())

    assert (passthrough_wire.encoding_schema_version, passthrough_wire.variant_id, passthrough_wire.profile_id) == (
        "vector.encoding.v1",
        "raw_bytes_v1",
        "consumer.passthrough",
    )
    assert (scalar_wire.encoding_schema_version, scalar_wire.variant_id, scalar_wire.profile_id) == (
        "vector.encoding.v1",
        "uniform_affine_u8_v1",
        "consumer.scalar",
    )
