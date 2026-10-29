# Changelog

## Unreleased

- Q6_K wide (prefill) kernel: signed weights, activations + 128, the per-row shift preloaded into the integer accumulators, both 16-value halves scaled once per Q8 block.
- Vectorized SiLU (`kern::silu_mul`), threaded residual adds.
- Continuous batching: `serve --parallel N` (default 4) decodes concurrent requests in one forward pass per step, one KV sequence per slot with prefix reuse; 4B Q8_0 with 8 requests 100.07 tok/s vs llama-server 58.01. `tools/bench_parallel.sh`, `tools/check_parallel.sh`.
- Speculative decoding for concurrent requests: draft proposals and tree verification batched across slots; Qwen3-8B Q8_0 + 0.6B draft, 4 requests 50.17 tok/s vs llama-server 23.42, 8 requests 73.06 vs 37.66.
- NEON panel GEMM (Q8_0, Q4_K, Q6_K; `sdot` by lane, 16 rows per panel), vectorized NEON attention, softmax and SiLU. Neoverse-N1 VPS, 4 threads: 0.6B Q8_0 pp512 31.0 -> 158.3 tok/s, tg128 14.48 -> 21.93; 1.7B Q4_K_M pp512 16.7 -> 49.9 (llama.cpp 55.37).
- `matmul_group`: q/k/v and gate/up in one parallel region (9800X3D 4B Q8_0 tg128 13.14 -> 13.36).
- Auto context size measures free RAM after the packed weights are resident, counts the draft KV, and honors cgroup v2 limits (`docker --memory`, systemd `MemoryMax`). Before, Qwen3-1.7B Q4_K_M + 0.6B draft under `docker --memory 6g` picked ctx 32768 and was OOM-killed; now ctx 2464.
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
