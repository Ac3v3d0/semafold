# TurboQuant Benchmarks

This directory contains deterministic, NumPy-only TurboQuant benchmark runners.

Current contents:

- synthetic paper-shaped validation
- synthetic KV-style row benchmarks
- benchmark summary:
  [turboquant_benchmark_report.md](turboquant_benchmark_report.md)

These runners are validation tooling only. They are not the package KV preview API and do not define `TurboQuantKVPreviewCodec.compress`, `.decompress`, or `.memory_stats`.

Non-goals:

- real-model downloads
- runtime adapters
- hardware probing
- machine-specific artifact storage

Typical usage from the package root:

```bash
PYTHONPATH=src python benchmarks/turboquant_paper_validation.py --output /tmp/turboquant-paper.json
PYTHONPATH=src python benchmarks/turboquant_synthetic_kv_benchmark.py --output /tmp/turboquant-kv.json
```

The benchmark report in this directory summarizes measured outputs from those runners and presents the current compression results in Markdown form.

These runners are intended for local validation and CI-adjacent smoke usage. They must remain deterministic by seed and must not require network access.
