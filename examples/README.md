# Examples

Runnable Semafold examples live here. They are intentionally small, deterministic,
and useful as real smoke checks for the public package surface.

## Example Set

- `wire_roundtrip.py`: stable root-envelope smoke test using the lossless passthrough codec
- `turboquant_embedding.py`: synthetic embedding compression example using `TurboQuantMSEVectorCodec`
- `turboquant_kv_block.py`: cache-shaped K/V compression example using `TurboQuantKVPreviewCodec`

## How To Run

From the project root:

```bash
PYTHONPATH=src python3 examples/wire_roundtrip.py
PYTHONPATH=src python3 examples/turboquant_embedding.py
PYTHONPATH=src python3 examples/turboquant_kv_block.py
```

If the package is installed editable:

```bash
python3 examples/wire_roundtrip.py
python3 examples/turboquant_embedding.py
python3 examples/turboquant_kv_block.py
```

## What Each Example Teaches

- `wire_roundtrip.py` shows the stable root API, JSON wire conversion, and exact reconstruction
- `turboquant_embedding.py` shows vector compression ratio, reconstruction error, and deterministic TurboQuant MSE behavior
- `turboquant_kv_block.py` shows cache-shaped K/V compression, restoration of shapes, and combined memory accounting

TurboQuant examples use the current deep-import surface under `semafold.turboquant`
and `semafold.turboquant.kv`; those imports are intentionally outside the stable
root API.
