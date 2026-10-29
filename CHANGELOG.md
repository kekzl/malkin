# Changelog

## Unreleased

- Q6_K wide (prefill) kernel: signed weights, activations + 128, the per-row shift preloaded into the integer accumulators, both 16-value halves scaled once per Q8 block.
- Vectorized SiLU (`kern::silu_mul`), threaded residual adds.
- `bench_layers`: decode (n = 1) bandwidth over distinct packed weights larger than L3; Q8_0, Q4_K, Q6_K reach 59-61 GB/s on the 9800X3D.

| pp512 tok/s, 9800X3D, 8 threads, best of 3 interleaved, host load 7-9 | 0.1.0 | now | llama.cpp |
|---|---:|---:|---:|
| Qwen3-14B Q6_K | 145.60 | 167.40 | 116.31 |
| Qwen3-4B Q4_K_M | 493.00 | 513.50 | 241.07 |
| Qwen3-4B Q8_0 | 476.80 | 506.70 | 116.83 |

## 0.1.0 (2026-10-29)

First release. Qwen3 dense GGUF models on AVX-512 (Zen 4 / Zen 5, Sapphire Rapids+) and Arm Neoverse-N1+ CPUs.

### Server

| | |
|---|---|
| API | OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`; Anthropic `/v1/messages`; `/health`; SSE streaming |
| Reasoning | `<think>` to `reasoning_content` / Anthropic `thinking` blocks; `enable_thinking=false` |
| Cache | KV reuse of the longest common prefix with the previous request |
| CLI | `serve`, `run`, `bench`, `ppl`; `--draft` model for speculative decoding |

### Engine

| | |
|---|---|
| Weights | F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K from GGUF v3 (mmap) |
| Prefill | AVX-512 VNNI panel GEMM (16 rows per panel, compact and wide panels, dynamic work items) |
| Attention | GQA heads share K/V row loads, BF16 KV cache |
| Speculation | greedy tree (draft top-1 chain + top-2 sibling per depth), byte-identical to plain greedy; sampling with exact rejection |

### Measured (Ryzen 7 9800X3D, 8 threads, vs llama.cpp build 8642541b2)

| Case | malkin | llama.cpp | |
|---|---:|---:|---:|
| Prefill 512, Qwen3-4B Q8_0 | 502.8 tok/s | 121.86 tok/s | 4.13x |
| Prefill 512, Qwen3-4B Q4_K_M | 514.8 tok/s | 251.24 tok/s | 2.05x |
| Prefill 512, Qwen3-14B Q6_K | 143.3 tok/s | 121.37 tok/s | 1.18x |
| Decode, Qwen3-8B Q8_0 + 0.6B draft vs llama.cpp default | 15.95 tok/s | 6.70 tok/s | 2.38x |
| Decode, Qwen3-8B Q8_0 + 0.6B draft vs llama.cpp + same draft | 15.95 tok/s | 10.32 tok/s | 1.55x |
| Perplexity, Qwen3-4B Q8_0, `-c 512` | 18.0197 | 18.0256 | |

### Distribution

| | |
|---|---|
| Image | `ghcr.io/kekzl/malkin:0.1.0`, `linux/amd64` + `linux/arm64`, SBOM, provenance, cosign signature |
| Binaries | `malkin-0.1.0-linux-{x86_64,aarch64}.tar.gz`, libc++ and libomp bundled, glibc 2.38+, cosign bundles |
| Toolchain | C++26, LLVM 23, CMake 4.4, Ninja 1.13 |

### Limits

Qwen3 dense only, one sequence at a time, decode without a draft model level with llama.cpp (memory-bound), Neoverse kernels verified under qemu only.
