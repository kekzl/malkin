# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

malkin: GGUF LLM server (C++26, LLVM 23: clang + libc++ + libomp + lld, CMake 4.4) for exactly two CPU ISAs, AVX-512 (VNNI + BF16; Ryzen 9800X3D, EPYC 9634) and Armv8.2-A dotprod (Neoverse-N1+). No generic fallback, no other ISA.

## Commands

Build and run only in Docker; the host has no compiler. Pass `--user $(id -u):$(id -g)` or build dirs end up root-owned.

| Task | Command |
|---|---|
| Dev image | `docker build --target deps -t malkin-dev:local .` |
| Build x86 | `docker run --rm --user $(id -u):$(id -g) -v $PWD:/src -w /src malkin-dev:local bash -c 'cmake -B build -G Ninja && cmake --build build'` |
| Kernel test | `./build/test_quant` in the dev container (only ctest test: `ctest --test-dir build -R quant`; exit 77 = skipped, CPU lacks the ISA) |
| Cross image | `docker buildx build --platform linux/arm64 --target deps --load -t malkin-cross:local .` (amd64 clang + arm64 libc++/libomp) |
| Build arm64 | `cmake -B build-arm64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake && cmake --build build-arm64` in `malkin-cross:local` |
| Run arm64 | `qemu-aarch64 -cpu neoverse-n1 build-arm64/test_quant` (host has no binfmt; ctest uses the emulator via the toolchain file) |
| CI-equivalent | `docker build --target test .` and `docker buildx build --platform linux/arm64 --target test .` |
| Runtime image | `docker build -t malkin:local .` (arm64: `docker buildx build --platform linux/arm64 --load ...`) |
| API test | start `malkin:local --model /models/<file>.gguf` with `-p 127.0.0.1:18080:8080`, then `tests/api_smoke.sh http://127.0.0.1:18080` (14 checks; host port 8080 belongs to imp) |
| Correctness | `malkin ppl --model M -f /models/ppl_4k.txt --window 512` vs `docker run --rm -v /home/kekz/models:/models:ro ghcr.io/ggml-org/llama.cpp:full --perplexity -m M -f /models/ppl_4k.txt -c 512` |
| Speed | `tools/bench_vs_llama.sh HOST_GGUF 512 128 2` (prefill + decode, back to back); `tools/bench_decode.sh TARGET DRAFT` (4 prompts, plain vs draft vs llama.cpp tg128); `bench_gemm [n] [threads] [rows cols] [type]` (kernel TMAC/s; single-thread runs are the stable A/B on a shared host) |
| Lint | `tools/lint.sh` in the deps image: clang-format, clang-tidy (bugprone, performance, warnings are errors), shellcheck; CI adds actionlint |
| Release | bump `project(VERSION)` in `CMakeLists.txt`, add a `CHANGELOG.md` section, then `gh release create vX.Y.Z --notes-file <section>`; `.github/workflows/release.yml` pushes `ghcr.io/kekzl/malkin:X.Y.Z`, `X.Y`, `latest` (amd64 + arm64, SBOM, provenance, cosign) and attaches signed tarballs (`--target bundle`) |

Models live in `/home/kekz/models` (mount read-only at `/models`). Reference model: `Qwen3-4B-Instruct-2507-Q8_0.gguf`, PPL 18.0120 (llama.cpp 18.0256).

## Architecture

Request path: `server.cpp` (httplib handler) -> `Engine::generate` (mutex, then `on_worker`) -> prefix match against `cache_` -> `Model::forward` in chunks of `kMaxBatch` = 256 -> `matmul` -> `kern::*`.

- **One compute thread.** All model work runs on the engine worker thread so there is one OpenMP pool. OpenMP thread counts are per calling thread: a pool per HTTP thread ran decode at 7.2 instead of 13.2 tok/s. Decode and prefill both default to physical cores (SMT prefill: 188.7 vs 348.1 tok/s); `omp_set_num_threads` is set per phase in `engine.cpp`.
- **ISA split.** `kernels.h` is the only ISA boundary; CMake picks `kernels_avx512.cpp` or `kernels_neon.cpp` from `CMAKE_SYSTEM_PROCESSOR`. Everything else is portable C++ (no intrinsics in `model.cpp`, `quant.cpp`). A new weight type needs: scalar `dequantize_row` (the reference), a dot kernel in both ISA files, dispatch in `vec_dot`, a case in `tests/test_quant.cpp`.
- **Activations.** `BlockQ8` = 32 int8 + float scale + int16 sums of each 16-value half. The sums carry the K-quant min terms (Q4_K/Q5_K `dmin*m`) and the compact Q6_K `-32` bias, so kernels never sign-extend weights. The Q6_K wide kernel instead takes signed weights and `qs ^ 0x80` activations (`gemm_unsigned_acts`, converted in `matmul`). llama.cpp uses Q8_K (256-value scale) for K-quants, so PPL vs llama.cpp differs by up to ~0.5 % there; Q8_0 and Q6_K match within 0.1 %.
- **Kernel port budget (Zen 5).** `vpdpbusd` shares ports with float mul/FMA, not with add/convert: each extra mul or FMA per accumulator per 16 weights cost 12 % (Q6_K ablation), so epilogues fold offsets into integer accumulator init and scale once per Q8 block.
- **Weights** stay in the mmap'd GGUF (`Weight` is a view); load-time `prefetch` first-touches pages in parallel for NUMA spread. KV cache is BF16, `[layer][pos][kv_head*head_dim]`.
- **Tokenization.** `ChatFormat::tokenize` encodes template parts with control tokens enabled and message content with them disabled, so user text cannot inject `<|im_end|>`. Only ChatML (Qwen) and only arch `qwen3` are supported; `read_config` rejects others.
- **Streaming.** `Engine::generate` withholds incomplete UTF-8 and stop-string prefixes before calling `on_text`; `ReasoningSplitter` (server) splits `<think>` into `reasoning_content` / Anthropic `thinking` blocks and withholds partial tags.
- **Image.** Build stages run on `$BUILDPLATFORM`; arm64 is cross-compiled (amd64 host only). The runtime stage must stay `COPY`-only (no `RUN`), so it builds without emulation; `docker/collect-libs.sh` copies libc++, libc++abi and libomp for the target arch, found via `LD_LIBRARY_PATH`. LLVM dev packages for amd64 and arm64 are not co-installable, so `docker/deps.sh` installs the target arch only.

## Rules

- Kernel or numeric change: both ISAs, `test_quant` green on both, PPL vs `llama-perplexity` on the same file and `-c` in the PR.
- Perf claim: `malkin bench` vs `llama-bench` with the same threads on an idle host (check `/proc/loadavg`; other tenants like bookforge share this machine).
- x86 build tuning must not favor one target (9800X3D Zen 5 vs EPYC 9634 Zen 4) without a measurement on both.
