# B3 · Transformer Attention Mechanism

**[Full project report → REPORT.md](REPORT.md)** — bottleneck analysis, kernel
design, timing methodology, metrics, charts, and insights.

Scaled dot-product attention implemented from scratch with numba CUDA and
optimised in three steps. Everything except attention (embedding, projections,
layer norms, FFN) stays in PyTorch on the CPU — only the profiled bottleneck
is replaced.

| Version | File | Key idea |
|---|---|---|
| `CpuPipeline` | `src/cpu_baseline.py` | PyTorch reference + timing baseline |
| `GpuV1` | `src/gpu_v1.py` | naive three-kernel attention, everything through global memory |
| `GpuV2` | `src/gpu_v2.py` | tiled QKᵀ in shared memory + single-pass online softmax |
| `GpuV3` | `src/gpu_v3.py` | FlashAttention-style fused kernel, never materialises the N×N matrix |

## Layout

- `src/abstract.py` — shared transformer block; subclasses implement `attention`
- `src/bench.py` — profiling and timing helpers
- `notebooks/attention.ipynb` — report: bottleneck profile, correctness checks, benchmarks, charts
- `tests/test_correctness.py` — every version vs `torch` SDPA (GPU tests auto-skip without CUDA)

## Run

```sh
uv sync
uv run pytest
```

The notebook needs a CUDA GPU (e.g. Colab T4) for the V1–V3 benchmarks; the
CPU profile chart runs anywhere.
