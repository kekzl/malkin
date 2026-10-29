<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo.svg">
    <img src="docs/logo-light.svg" alt="malkin" width="560">
  </picture>
</p>

<p align="center"><b>Two CPU families. Nothing else.</b><br>
AVX-512 (Zen 4 / Zen 5, EPYC 9004/9005, Sapphire Rapids+) · Arm Neoverse-N1+ · one <code>docker run</code> · OpenAI and Anthropic API</p>

<p align="center">
  <a href="https://github.com/kekzl/malkin/actions/workflows/ci.yml"><img src="https://github.com/kekzl/malkin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/kekzl/malkin/actions/workflows/codeql.yml"><img src="https://github.com/kekzl/malkin/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <img src="https://img.shields.io/badge/C++-26-00599C?style=flat&logo=cplusplus" alt="C++26">
  <img src="https://img.shields.io/badge/LLVM-23-262D3A?style=flat&logo=llvm" alt="LLVM 23">
  <img src="https://img.shields.io/badge/ISA-AVX--512%20VNNI%20%7C%20NEON%20dotprod-9b5cff?style=flat" alt="ISA">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat" alt="MIT">
</p>

**malkin is an LLM server for CPUs, built for two instruction sets only:** x86-64 with AVX-512 VNNI + BF16, and Armv8.2-A with dotprod (Neoverse-N1 and newer). Weights are repacked at load into panels for the CPU's dot-product instruction; prompt reading runs as a register-blocked VNNI GEMM, generation can verify draft tokens losslessly. On any other CPU it exits with the list of missing features.

### Measured on one Ryzen 7 9800X3D

| Qwen3, tokens/s, 8 threads | malkin prompt | llama.cpp prompt | | malkin answer | llama.cpp answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B Q8_0 | **506.7** | 116.83 | **4.34x** | 13.85 | 13.79 |
| 4B Q4_K_M | **513.5** | 241.07 | **2.13x** | 23.04 | 22.83 |
| 14B Q6_K | **167.4** | 116.31 | 1.44x | **5.32** | 4.69 |

<!-- [PROV: prompt columns: date=2026-10-29 hw=Ryzen7-9800X3D WSL2 branch=q6k-wide-epilogue cmd=malkin bench --ctx 4096 -pp 512 -tg 1 vs llama-bench -p 512 -n 0 -t 8 -r 1 (ghcr.io/ggml-org/llama.cpp:full build 8642541b2), interleaved with the main build, best of 3, host load 7-9 from other tenants. Answer columns: date=2026-10-29 commit=fee0379 cmd=`MALKIN_BUILD=build-q6 tools/bench_vs_llama.sh MODEL 16 128 3`, best of 3, host load 0.1 at the start (4B Q8_0), 12 at the end (14B)] -->

Prompt = 512-token prefill, answer = 128-token generation, same GGUF file in both engines. Answer speed without a draft model is memory-bound (both engines read every weight once per token; malkin streams 59-61 GB/s, `bench_layers`) and level with or ahead of llama.cpp; with a draft model (below) malkin generates the same tokens 1.85x to 2.38x faster.

**Same quality.** Perplexity on the same text and windows (`malkin ppl` vs `llama-perplexity -c 512`, 2040 tokens): 4B Q8_0 18.0245 vs 18.0256, 4B Q4_K_M 18.2496 vs 18.1540, 14B Q6_K 14.0375 vs 14.0468 (llama.cpp's error bar: ±1.41 and ±1.02).

**Answers with a draft model, lossless.** `--draft qwen3-0.6b-q4_k_m.gguf`: the draft proposes a chain of tokens plus its second choice at every step, the target checks the whole tree in one pass (same weight read as one token). Greedy output is byte-identical to plain decoding; sampled output follows the target's distribution exactly (rejection step, `tests/test_sampler.cpp`).

| Qwen3 target, Qwen3-0.6B Q4_K_M draft, tokens/s | llama.cpp | malkin | malkin + draft | |
| --- | ---: | ---: | ---: | ---: |
| 8B Q8_0 | 6.70 | 7.27 | **15.95** | **2.38x** |
| 4B-Instruct-2507 Q8_0 | 12.30 | 12.60 | **22.74** | **1.85x** |
| 14B Q6_K | 4.61 | 4.39 | **9.59** | **2.08x** |

<!-- [PROV: date=2026-10-29 hw=Ryzen7-9800X3D WSL2 commit=c4776b8 cmd=`tools/bench_decode.sh TARGET qwen3-0.6b-q4_k_m.gguf 200`: malkin run, 4 prompts (explanation, Python code, list, summary), 200 greedy tokens each, total tokens / total time; llama.cpp = llama-bench -p 0 -n 128 -t 8 -r 2 on the same GGUF; host load 5-10 from other tenants] -->

Four prompt types (explanation, code, list, summary), 200 greedy tokens each. llama.cpp column: `llama-bench` generation on its defaults (no draft). The 0.6B draft fits the original Qwen3 release (8B) better than the newer 4B-Instruct-2507 post-training (one-prompt runs: 55 % vs 47 % of drafts accepted). Draft from the Hugging Face weights: `llama.cpp:full --convert --outtype q8_0 Qwen3-0.6B`, then `--quantize ... Q4_K_M`.

**Against llama.cpp with its own draft** (`llama-speculative-simple --spec-type draft-simple`, same 0.6B Q4_K_M draft, greedy, same 4 prompts, best of draft length 4 / 8 / 16): Qwen3-8B Q8_0 10.32 tok/s, malkin + draft 15.95 tok/s, **1.55x** (`tools/bench_spec_llama.sh`).

**Concurrent requests, continuous batching.** `serve` decodes up to `--parallel` requests (default 4) in one forward pass per step: every weight is read once for all of them. Greedy answers are byte-identical to one request at a time (`tools/check_parallel.sh`).

| Qwen3, aggregate tokens/s, N requests of 128 tokens at once | slots | llama-server | malkin | |
| --- | ---: | ---: | ---: | ---: |
| 4B Q8_0, N = 1 | 4 | 12.83 | 14.01 | 1.09x |
| 4B Q8_0, N = 4 | 4 | 40.96 | **53.61** | **1.31x** |
| 4B Q8_0, N = 8 | 8 | 58.01 | **100.07** | **1.73x** |
| 14B Q6_K, N = 4 | 4 | 16.98 | **19.88** | **1.17x** |

<!-- [PROV: date=2026-10-29 hw=Ryzen7-9800X3D WSL2 branch=parallel-slots cmd=`tools/bench_parallel.sh URL N 128` against `malkin serve --ctx 4096 --parallel S` and `llama-server -t 8 -np S -c 4096*S` (ghcr.io/ggml-org/llama.cpp:full), interleaved, best of 2 (14B: 1 round), host load 5] -->

### Where malkin loses, and what it is not

- **Two ISAs.** No AVX2-only x86, no Apple, no SVE paths yet (Neoverse-V1/V2 run the N1 kernels).
- **Q6_K prompt reading** is 1.44x llama.cpp, not 2x: its scale per 16 weights needs a float multiply per 4 `vpdpbusd` on the same ports (one core: 0.43 TMAC/s, 0.51 in an ablation build without scales).
- **Answers without a draft model** are level with llama.cpp: memory bandwidth, not compute, sets the pace. llama.cpp can run a draft model too; the tables compare both engines on their defaults.
- **One architecture:** Qwen3 dense (`qwen3`). No MoE, no vision, no tool calls yet.
- **Speculation is single-sequence:** with `--draft`, a request decodes speculatively only while it is the only one in flight; concurrent requests are batched without the draft.
- **Neoverse numbers are emulated** (`qemu -cpu neoverse-n1`): correct, not benchmarked on hardware.

## Quick start

```bash
docker run --rm -p 127.0.0.1:8080:8080 -v /path/to/models:/models:ro \
  ghcr.io/kekzl/malkin:latest --model /models/Qwen3-8B-Q8_0.gguf --draft /models/qwen3-0.6b-q4_k_m.gguf

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

| Install | How |
|---|---|
| Image | `ghcr.io/kekzl/malkin:<version>` or `:latest`, one manifest for `linux/amd64` and `linux/arm64` |
| Binary | `malkin-<version>-linux-{x86_64,aarch64}.tar.gz` from [Releases](https://github.com/kekzl/malkin/releases), unpack, run `bin/malkin` (libc++ and libomp bundled, needs glibc 2.38+) |
| Source | `docker build -t malkin:local .`; arm64 from an amd64 host, cross-compiled: `docker buildx build --platform linux/arm64 -t malkin:arm64 .` |

Release images and tarballs are signed keyless with cosign:

```bash
cosign verify ghcr.io/kekzl/malkin:latest --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp '^https://github\.com/kekzl/malkin/\.github/workflows/release\.yml@refs/tags/'
cosign verify-blob --bundle malkin-0.1.0-linux-x86_64.tar.gz.sigstore.json \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp '^https://github\.com/kekzl/malkin/\.github/workflows/release\.yml@refs/tags/' \
  malkin-0.1.0-linux-x86_64.tar.gz
```

## What it serves

| Surface | Paths |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/models` (SSE streaming, `stream_options.include_usage`) |
| Anthropic | `/v1/messages` (SSE streaming, `thinking` blocks) |
| Ops | `/health` |

Hybrid reasoning models: `<think>` output goes to `reasoning_content` (OpenAI) or a `thinking` block (Anthropic); `chat_template_kwargs.enable_thinking=false` turns it off. The KV of the longest common token prefix with the previous request is reused. Responses carry `timings` (prefill and generation tokens/s, `draft_n`, `draft_n_accepted`).

## Models

| | Supported |
|---|---|
| Architecture | `qwen3` (dense) |
| Weights | F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K |
| Tokenizer | byte-level BPE (`gpt2`), pre-tokenizers `qwen2`, `llama-bpe` |
| Chat format | ChatML (`<|im_start|>`) |
| KV cache | BF16 |

## How it works

| Piece | What it does |
|---|---|
| Panels | 16 weight rows interleaved per 64-byte vector, one `vpdpbusd` lane per row; compact panels (about GGUF size) for generation, wide panels (one byte per weight) for K-quant prompt reading |
| GEMM | 2 panels x up to 6 tokens per call, activations as Q8 blocks of 32 with per-half sums for the K-quant minimum terms; work handed out dynamically per (panel pair, token chunk) |
| Attention | the query heads of one KV head (GQA) share every K/V row load; vectorized exp |
| Speculation | greedy: draft top-1 chain + top-2 sibling per depth, verified as a tree (each node attends to the prefix and its ancestors, RoPE at its depth), accepted path compacted in the KV cache; sampling: chain with the exact rejection step |
| Threads | one engine thread, one OpenMP pool; physical cores for both phases (prefill 348.1 vs 188.7 tok/s on 16 SMT threads, 9800X3D) |
| NUMA | weights first-touched in parallel at load, so pages spread across EPYC NPS2/NPS4 nodes |

## CLI

```
malkin [serve] --model PATH [--host 0.0.0.0] [--port 8080] [--ctx N] [--threads N] [--threads-batch N]
               [--draft PATH] [--n-draft N]
malkin run     --model PATH -p TEXT [-n 512] [--temp 0] [--no-think] [--draft PATH]
malkin bench   --model PATH [-pp 512] [-tg 128]
malkin ppl     --model PATH -f FILE [--window 512]
```

| Environment | Effect |
|---|---|
| `MALKIN_NO_WIDE=1` | no wide prompt panels (saves 1.09 bytes per Q4_K/Q5_K weight, 1.20 per Q6_K weight of RAM) |
| `MALKIN_NO_PACK=1` | row-by-row kernels only (A/B reference) |
| `MALKIN_PROFILE=1` | wall time per forward stage at exit |

## Build and test

Toolchain: C++26, LLVM 23 (clang, lld, libc++, libomp), CMake 4.4, Ninja 1.13, all in Docker.

```bash
docker build --target test .                          # x86 kernel + sampler tests (kernels skip without AVX-512 BF16)
docker build --target test --platform linux/arm64 .   # same through qemu, Neoverse-N1 CPU model
tests/api_smoke.sh http://127.0.0.1:8080              # 14 API checks against a running server
tools/bench_vs_llama.sh /path/model.gguf 512 128 2    # malkin vs llama.cpp, back to back
```

## Roadmap

1. Neoverse panel GEMM (NEON `sdot` by lane); Q6_K with 256-value activation blocks (one token-scale multiply per superblock).
2. Long context: adaptive-precision attention after [APA](https://github.com/kekzl/apa) (INT8 cold KV tiles, exact hot tiles).
3. Architectures: `qwen3moe`, Gemma, gpt-oss (MXFP4); tool calling, Jinja chat templates.
4. Concurrent requests (continuous batching).

MIT. Sibling project for NVIDIA Blackwell GPUs: [imp](https://github.com/kekzl/imp).
