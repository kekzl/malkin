# malkin

CPU LLM server: AVX-512 (VNNI, BF16) and Neoverse-N1+ (NEON dotprod) only. No generic fallback.

## Layout

| Path | Role |
|---|---|
| `src/kernels.h` | ISA interface; one `.cpp` per ISA, picked by CMake from `CMAKE_SYSTEM_PROCESSOR` |
| `src/quant.cpp` | GGUF block dequant (reference), dispatch, matmul tiling |
| `src/model.cpp` | qwen3 forward, BF16 KV |
| `src/engine.cpp` | worker thread, prefix cache, sampling loop, perplexity |
| `src/server.cpp` | OpenAI + Anthropic HTTP |

## Rules

- Build and test only in Docker: `docker compose --profile dev run --rm dev bash`, then `cmake -B build -G Ninja && cmake --build build`
- arm64: `docker build --target cross-arm64 -t malkin-cross:local .`, toolchain `cmake/aarch64-linux-gnu.cmake`, run with `qemu-aarch64 -cpu neoverse-n1 -L /usr/aarch64-linux-gnu`
- New kernel: both ISAs, plus a case in `tests/test_quant.cpp` against `dequantize_row`
- Numeric change: perplexity vs `llama-perplexity` (`ghcr.io/ggml-org/llama.cpp:full`) on the same file and `-c`, both numbers in the PR
- Perf claim: `malkin bench` vs `llama-bench`, same threads, numbers from the log
- Compute only on the engine worker thread (one OpenMP pool)
