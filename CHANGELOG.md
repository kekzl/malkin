# Changelog

## Unreleased

- `/v1/embeddings` for Qwen3-Embedding GGUFs: final-norm state of the appended `<|endoftext|>` token, L2-normalized; `input` string or array, `dimensions` (prefix, normalized again), `encoding_format` float or base64. Qwen3-Embedding-0.6B Q8_0, 4 inputs, cosine to llama.cpp (`--embedding --pooling last`, same file): 0.9980, 0.9779, 0.9955, 0.9968; token counts equal (35).
- `/v1/rerank` for Qwen3-Reranker GGUFs: `relevance_score` = P("yes") over {"yes", "no"} after the judge prompt; `top_n`, `instruction`, `return_documents`. Documents of one request share the system + instruction + query prefix, whose KV is reused. Qwen3-Reranker-0.6B Q8_0, 4 documents: same order as llama.cpp (`--reranking`), scores 0.996 / 0.0001 / 0.0001 / 0.9991 vs 0.998 / 0.0001 / 0.0001 / 0.9991.
- `Engine::probe`: evaluates a token sequence between generations and returns the last token's hidden state and chosen logits. `tools/check_embed.sh` compares both endpoints with llama.cpp; `tests/api_smoke.sh` has 17 checks.
- Retrieval over 50 notes and 30 queries (secondBrain `brain eval`, 9800X3D): recall@5 equal to llama.cpp in all modes (vector 30/30, FTS + vector fused 28/30, reranked 30/30); reranked mode (query embedding + up to 20 documents scored) 440 ms vs 3355 ms per query (llama.cpp run at load 3.18, malkin at 0.79: not an idle-host comparison).

## 0.2.0 (2026-11-17)

- Q6_K wide (prefill) kernel: signed weights, activations + 128, the per-row shift preloaded into the integer accumulators, both 16-value halves scaled once per Q8 block.
- Vectorized SiLU (`kern::silu_mul`), threaded residual adds.
- Continuous batching: `serve --parallel N` (default 4) decodes concurrent requests in one forward pass per step, one KV sequence per slot with prefix reuse; 4B Q8_0 with 8 requests 100.07 tok/s vs llama-server 58.01. `tools/bench_parallel.sh`, `tools/check_parallel.sh`.
- Speculative decoding for concurrent requests: draft proposals and tree verification batched across slots; Qwen3-8B Q8_0 + 0.6B draft, 4 requests 50.17 tok/s vs llama-server 23.42, 8 requests 73.06 vs 37.66.
- NEON prompt attention in blocks (`attend_block`: 4 rows of token x head per K/V row conversion): N1 0.6B Q8_0 pp512 156.0 -> 214.8 tok/s, attention share 53.2 % -> 37.7 %.
- NEON panel GEMM (Q8_0, Q4_K, Q6_K; `sdot` by lane, 16 rows per panel), vectorized NEON attention, softmax and SiLU. Neoverse-N1 VPS, 4 threads: 0.6B Q8_0 pp512 31.0 -> 158.3 tok/s, tg128 14.48 -> 21.93; 1.7B Q4_K_M pp512 16.7 -> 49.9 (llama.cpp 55.37).
- `matmul_group`: q/k/v and gate/up in one parallel region (9800X3D 4B Q8_0 tg128 13.14 -> 13.36).
- Auto context size measures free RAM after the packed weights are resident, counts the draft KV, and honors cgroup v2 limits (`docker --memory`, systemd `MemoryMax`). Before, Qwen3-1.7B Q4_K_M + 0.6B draft under `docker --memory 6g` picked ctx 32768 and was OOM-killed; now ctx 2464.
- Small hosts: wide prompt panels are packed only where they leave room for the KV cache plus 512 MiB (before: always, Qwen3-8B Q4_K_M under `docker --memory 6g` was OOM-killed at load). All panels of a model share one buffer (8B Q4_K_M: 5398 -> 5204 MiB anonymous without wide panels, 13752 -> 13270 MiB with), packed weights are unmapped from the GGUF (mapped file pages 2931 -> 344 MiB peak) and the load-time prefetch touches only what is still read from the file.
- KV cache in lazily faulted 4 KiB pages (`LazyBytes`) and token embeddings without load-time prefetch: both get resident only as far as they are used. Qwen3-8B Q4_K_M, `--parallel 2 --ctx 2048`, after the load: 5204 -> 4628 MiB anonymous, 351 -> 13 MiB mapped file; after a 69-token request 4638 MiB. `MALKIN_KV_HUGE=1` puts the KV in huge pages.
- KV cache in int8 with one float scale per head row instead of BF16: half the bytes (Qwen3-8B: 1056 instead of 2048 bytes per K or V row, 2 x 2048 ctx 0.29 instead of 0.56 GiB). 8B Q4_K_M: PPL 17.0506 (BF16 17.0812, llama.cpp 17.1490), pp1536 148.0 / 150.8 vs 149.8 / 151.2 tok/s, tg64 11.22 / 11.06 vs 10.80 / 10.74, anonymous peak after 1600 tokens 4855 vs 5069 MiB.
- `bench_layers`: decode (n = 1) bandwidth over distinct packed weights larger than L3; Q8_0, Q4_K, Q6_K reach 59-61 GB/s on the 9800X3D.

| pp512 tok/s, 9800X3D, 8 threads, best of 3 interleaved, host load 7-9 | 0.1.0 | 0.2.0 | llama.cpp |
|---|---:|---:|---:|
| Qwen3-14B Q6_K | 145.60 | 167.40 | 116.31 |
| Qwen3-4B Q4_K_M | 493.00 | 513.50 | 241.07 |
| Qwen3-4B Q8_0 | 476.80 | 506.70 | 116.83 |

Not measured on hardware: the int8 KV cache and the small-host changes on the EPYC 9634 and Neoverse-N1 VPS (9800X3D and qemu only).

| | |
|---|---|
| Image | `ghcr.io/kekzl/malkin:0.2.0`, `linux/amd64` + `linux/arm64`, SBOM, provenance, cosign signature |
| Binaries | `malkin-0.2.0-linux-{x86_64,aarch64}.tar.gz`, libc++ and libomp bundled, glibc 2.38+, cosign bundles |

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
