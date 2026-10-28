# malkin

**LLM inference server for two CPU families. Nothing else.**

| ISA | CPUs | Kernels |
|---|---|---|
| x86-64: AVX-512 + VNNI + BF16 | AMD Zen 4 / Zen 5 (Ryzen 7000/9000, EPYC 9004/9005), Intel Sapphire Rapids and newer | `src/kernels_avx512.cpp` |
| aarch64: Armv8.2-A + dotprod + fp16 | Arm Neoverse-N1 and newer (Ampere Altra, Graviton2+) | `src/kernels_neon.cpp` |

No generic fallback: on any other CPU malkin exits with the missing feature list. One container, OpenAI and Anthropic API, GGUF models, C++23, MIT.

## Quick start

```bash
docker build -t malkin:local .
docker run --rm -p 127.0.0.1:8080:8080 -v /path/to/models:/models:ro \
  malkin:local --model /models/Qwen3-4B-Instruct-2507-Q8_0.gguf

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

arm64 image from an amd64 host, cross-compiled, no emulation: `docker buildx build --platform linux/arm64 -t malkin:arm64 .`

## What it serves

| Surface | Paths |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/models` (SSE streaming, `stream_options.include_usage`) |
| Anthropic | `/v1/messages` (SSE streaming, `thinking` blocks) |
| Ops | `/health` |

Hybrid reasoning models: `<think>` output goes to `reasoning_content` (OpenAI) or a `thinking` block (Anthropic); `chat_template_kwargs.enable_thinking=false` turns it off. One sequence at a time; the KV of the longest common token prefix with the previous request is reused.

## Models

| | Supported |
|---|---|
| Architecture | `qwen3` (dense) |
| Weights | F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K |
| Tokenizer | byte-level BPE (`gpt2`), pre-tokenizers `qwen2`, `llama-bpe` |
| Chat format | ChatML (`<|im_start|>`) |
| KV cache | BF16 |

## Measured

Ryzen 7 9800X3D (Zen 5, 8 cores, WSL2), 2026-10-28, `malkin bench -pp 512 -tg 128` vs `llama-bench -p 512 -n 128 -t 8` (ghcr.io/ggml-org/llama.cpp:full), tokens/s:

| Model | malkin pp512 | llama.cpp pp512 | malkin tg128 | llama.cpp tg128 |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q8_0 | 34.0 | 120.26 | 12.05 | 13.15 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 34.9 | 247.28 | 18.15 | 19.67 |
| Qwen3-14B Q6_K | 14.8 | 125.69 | 4.62 | 4.89 |

Decode reaches 92 to 95 % of llama.cpp; prefill is 3.5x to 8.5x behind (no GEMM micro-kernel yet, see Roadmap).

Correctness, perplexity on the same text and windows (`malkin ppl` vs `llama-perplexity`, `-c 512`):

| Model | Tokens | malkin | llama.cpp | Δ |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Q8_0 | 2040 | 18.0120 | 18.0256 | -0.08 % |
| Qwen3-4B-Instruct-2507 Q4_K_M | 2040 | 18.2484 | 18.1540 | +0.52 % |
| Qwen3-4B-Instruct-2507 Q5_K_M | 2040 | 18.0426 | 18.0570 | -0.08 % |
| Qwen3-14B Q6_K | 1020 | 13.4946 | 13.4977 | -0.02 % |

llama.cpp's error bar on these runs: ±1.41 (7.8 %). arm64 under `qemu-aarch64 -cpu neoverse-n1` vs x86 on Qwen3-0.6B Q8_0, 252 tokens: 39.0757 vs 38.9896.

## Threads

| Phase | Default | Why |
|---|---|---|
| decode | physical cores (`--threads`) | bandwidth-bound; 4B Q8_0 on the 9800X3D: 13.2 tok/s on 8 threads, 7.3 on 16 |
| prefill | all logical CPUs (`--threads-batch`) | compute-bound; pp128 42.5 -> 52.8 tok/s with SMT |

All compute runs on one engine thread with one OpenMP pool. Weights are first-touched in parallel at load, so on NUMA hosts (EPYC NPS2/NPS4) pages spread across nodes. Matmul tiles shrink to keep at least 4 tiles per thread (84-core EPYC 9634: 1024-row matrices get 3-row tiles).

## CLI

```
malkin [serve] --model PATH [--host 0.0.0.0] [--port 8080] [--ctx N] [--threads N] [--threads-batch N]
malkin run     --model PATH -p TEXT [-n 512] [--temp 0] [--no-think]
malkin bench   --model PATH [-pp 512] [-tg 128]
malkin ppl     --model PATH -f FILE [--window 512]
```

`--ctx` defaults to min(model context, 32768), capped so the BF16 KV takes at most half of the RAM left after the weights.

## Build and test

```bash
docker build --target test .                          # x86 kernel tests (skip without AVX-512 BF16)
docker build --target test --platform linux/arm64 .   # arm64 kernel tests through qemu, Neoverse-N1 CPU model
tests/api_smoke.sh http://127.0.0.1:8080              # 14 API checks against a running server
```

## Roadmap

1. Prefill: GEMM micro-kernels (multi-token weight reuse) for AVX-512 and NEON.
2. Architectures: `qwen3moe`, Gemma, gpt-oss (MXFP4).
3. Tool calling, Jinja chat templates.
4. Continuous batching for concurrent requests.
5. Neoverse-V1/V2 paths (SVE, i8mm, BF16), AMX on Intel.
