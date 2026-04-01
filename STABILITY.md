# Stability Policy

## Stable Now

Stable imports are limited to documented root exports from `semafold`.

Stable surface:

- `__version__`
- `CompressionBudget`
- `CompressionEstimate`
- `CompressionFootprint`
- `CompressionGuarantee`
- `ValidationEvidence`
- `EncodingBoundType`
- `WorkloadSuitability`
- `VectorEncodeRequest`
- `VectorEncodingSegment`
- `VectorEncoding`
- `VectorDecodeRequest`
- `VectorDecodeResult`
- `VectorCodec`
- `PassthroughVectorCodec`
- `EncodeObjective`
- `EncodeMetric`
- `EncodingSegmentKind`

## Intentionally Not Stable

- deep imports
- `semafold.turboquant` and anything below it
- `semafold.turboquant.kv` and any KV preview config, codec, helper, or artifact below it
- `ScalarReferenceVectorCodec`
- examples and tests
- internal package layout under `src/semafold/...`
- any future `text`, `runtime`, or `diagnostics` namespace
- any selector, profile, or manifest surface

## Working Rule

If an import path is not listed under `Stable Now` and exported from the `semafold` root, it may move or disappear without notice during incubation.

The current package is Phase 1 software, not a broad platform release. Docs, examples, and tests should reflect that narrower reality rather than earlier planning language.

The README reference example is allowed to use a non-stable deep import for internal closeout smoke coverage. That does not make the deep import stable.

TurboQuant preview work, including `semafold.turboquant.kv`, is deep-import-only and may change or disappear without notice during incubation. `TurboQuantKVConfig`, `TurboQuantKVPreviewCodec`, and any `compress` / `decompress` / `memory_stats` behavior there remain preview-only and must not be promoted into the stable root surface without an explicit API freeze decision.
