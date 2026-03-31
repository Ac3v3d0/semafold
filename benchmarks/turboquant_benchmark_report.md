# Semafold TurboQuant Benchmark Report

Date: 2026-03-31  
Package: `semafold` `0.1.0`  

## Current Scope

This report summarizes the current Semafold TurboQuant benchmark results.

Included:
- vector / embedding-style compression
- cache-shaped K/V tensor compression
- measured artifact sizes, not nominal bit budgets
- deterministic synthetic benchmark inputs
- measured payload, sidecar, metadata, and total artifact bytes

This report is about **measured vector and KV tensor compression behavior** for the current Semafold implementation.

## Primary Reference

- TurboQuant paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)

## Executive Summary

- For embedding-style `float32` vectors, Semafold TurboQuant currently reduces storage by roughly **87% to 94%**, depending on bit-width.
- A practical middle point is **3-bit** compression, which yielded about **90.5% smaller** artifacts for reconstruction-oriented vector storage and about **90.4% smaller** artifacts for inner-product-oriented vector storage.
- For cache-shaped K/V tensors, a representative synthetic benchmark produced **9.47x compression vs dense float32** and **4.74x compression vs dense fp16/bf16**, while preserving finite outputs and passing synthetic attention-proxy checks.
- Overhead is real on very small caches. For tiny K/V blocks, TurboQuant can still beat `float32` while being slightly worse than a plain `fp16` dense baseline.

## What “Normal Systems” Means Here

In this report, “normal systems” means dense storage of the exact same logical data:

- dense `float32`
- dense `fp16`
- dense `bf16`

This is the fairest baseline for the current Semafold surface because today we measure:
- exact artifact bytes
- sidecars and metadata included
- the same tensors and shapes

## Measured Setup

### Embedding / Vector Compression

- Shape: `128 x 1536`
- Source dtype: `float32`
- Dense `float32` baseline: `786,432 B`
- Dense `fp16` / `bf16` baseline: `393,216 B`
- Data: deterministic Gaussian rows
- For `TurboQuantProd`, rows were unit-normalized and evaluated against deterministic unit-normalized queries

### KV Tensor Compression

- Shape: `(layers=4, heads=8, seq_len=256, head_dim=128)`
- Source dtype: `float32`
- Dense `float32` baseline: `8,388,608 B`
- Dense `fp16` / `bf16` baseline: `4,194,304 B`
- Data: deterministic synthetic K and V tensors
- Key path: `TurboQuantProd`
- Value path: `TurboQuantMSE`
- Benchmark sources:
  - [turboquant_paper_validation.py](turboquant_paper_validation.py)
  - [turboquant_synthetic_kv_benchmark.py](turboquant_synthetic_kv_benchmark.py)

## Results: Embedding / Vector Compression

### Reconstruction-Oriented Storage (`TurboQuantMSE`)

| Setting | Dense `float32` | Dense `fp16/bf16` | Semafold Artifact | Saved vs `float32` | Smaller vs `float32` | Smaller vs `fp16` | Ratio vs `float32` | Ratio vs `fp16` | Quality Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2-bit MSE` | `786,432 B` | `393,216 B` | `50,077 B` | `736,355 B` | `93.63%` | `87.26%` | `15.70x` | `7.85x` | observed MSE `0.1178` |
| `3-bit MSE` | `786,432 B` | `393,216 B` | `74,738 B` | `711,694 B` | `90.50%` | `80.99%` | `10.52x` | `5.26x` | observed MSE `0.0348` |
| `4-bit MSE` | `786,432 B` | `393,216 B` | `99,484 B` | `686,948 B` | `87.35%` | `74.70%` | `7.91x` | `3.95x` | observed MSE `0.0098` |

### Inner-Product-Oriented Storage (`TurboQuantProd`)

| Setting | Dense `float32` | Dense `fp16/bf16` | Semafold Artifact | Saved vs `float32` | Smaller vs `float32` | Smaller vs `fp16` | Ratio vs `float32` | Ratio vs `fp16` | Quality Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2 total bits` | `786,432 B` | `393,216 B` | `50,638 B` | `735,794 B` | `93.56%` | `87.12%` | `15.53x` | `7.77x` | mean abs dot error `0.0156` |
| `3 total bits` | `786,432 B` | `393,216 B` | `75,255 B` | `711,177 B` | `90.43%` | `80.86%` | `10.45x` | `5.23x` | mean abs dot error `0.0089` |
| `4 total bits` | `786,432 B` | `393,216 B` | `99,916 B` | `686,516 B` | `87.30%` | `74.59%` | `7.87x` | `3.94x` | mean abs dot error `0.0048` |

### Interpretation

- If you want a practical default for embedding-style storage, **3-bit** is the most natural headline.
- At `3-bit`, Semafold is roughly:
  - **10.5x smaller than dense float32**
  - **5.2x smaller than dense fp16/bf16**
- Higher bit-widths trade compression ratio for better fidelity, exactly as expected.

## Results: KV Tensor Compression

### Representative Cache-Shaped Tensor Result

| Workload | Shape | Dense `float32` | Dense `fp16/bf16` | Semafold Combined Artifact | Smaller vs `float32` | Smaller vs `fp16` | Ratio vs `float32` | Ratio vs `fp16/bf16` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| KV block compression | `(4, 8, 256, 128)` | `8,388,608 B` | `4,194,304 B` | `885,734 B` | `89.44%` | `78.88%` | `9.47x` | `4.74x` |

### K / V Split

| Component | Artifact Bytes |
|---|---:|
| Keys (`TurboQuantProd`) | `459,255 B` |
| Values (`TurboQuantMSE`) | `426,479 B` |
| Combined | `885,734 B` |

### Interpretation

- For cache-shaped K/V tensors, the current Semafold implementation is already a **real compression layer**, not a paper-only skeleton.
- In this synthetic benchmark, it reduced the combined K/V artifact to about **10.6% of dense float32 size**.
- Even compared to dense `fp16/bf16`, it still reduced storage by about **78.9%**.

## Small-Block Caveat

TurboQuant has real metadata and sidecar overhead. That overhead amortizes well on larger tensors, but not on very small ones.

Measured tiny-block example:

| Workload | Shape | Dense `float32` | Dense `fp16` | Semafold Combined Artifact | Smaller vs `float32` | Smaller vs `fp16` | Ratio vs `float32` | Ratio vs `fp16` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny KV block | `(2, 2, 6, 16)` | `3,072 B` | `1,536 B` | `1,550 B` | `49.54%` | `-0.91%` | `1.98x` | `0.99x` |

This is important because it shows the honest tradeoff:
- TurboQuant already beats dense `float32`
- but on tiny cache blocks, it can be slightly worse than just storing dense `fp16`

So the system is most compelling when:
- vectors are moderately large
- cache blocks are non-trivial in size
- sidecar overhead is amortized

## What This Means In Practice

### Good Use Cases Right Now

- embedding storage
- long-term vector memory in AI orchestrators
- custom vector retrieval pipelines
- cache-shaped K/V tensor compression in custom inference stacks

## Reproducibility

The current benchmark and validation entry points live here:

- [turboquant_paper_validation.py](turboquant_paper_validation.py)
- [turboquant_synthetic_kv_benchmark.py](turboquant_synthetic_kv_benchmark.py)

Representative commands:

```bash
PYTHONPATH=src python benchmarks/turboquant_paper_validation.py

PYTHONPATH=src python benchmarks/turboquant_synthetic_kv_benchmark.py
```

## Final Takeaway

Current milestone:

- Semafold TurboQuant already provides measured value for vector / embedding compression.
- It also provides measured value for cache-shaped KV tensor compression.
- Runtime adapters and backend-specific serving integrations are a separate next step.

Headline summary:

> Semafold TurboQuant already delivers roughly **8x–16x compression vs dense float32** on representative synthetic vector workloads, and about **9.5x compression vs dense float32** on a representative cache-shaped KV tensor workload, with strong even-against-fp16 savings when tensors are not too small.
