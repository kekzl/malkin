// GEMM throughput per weight type: rows x cols matrix (default 9728 x 2560, Qwen3-4B ffn), n tokens, best of 30.
// usage: bench_gemm [n=256] [threads] [rows cols] [type, e.g. q6_K]
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "quant.h"

using namespace malkin;

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 256;
  if (argc > 2) omp_set_num_threads(std::atoi(argv[2]));
  const int64_t rows = argc > 4 ? std::atoll(argv[3]) : 9728, cols = argc > 4 ? std::atoll(argv[4]) : 2560;
  const char* only = argc > 5 ? argv[5] : nullptr;
  std::mt19937 rng(1);
  std::vector<float> x(static_cast<size_t>(n) * cols);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
  std::vector<BlockQ8> xq;
  quantize_acts(x.data(), n, cols, xq);
  std::vector<float> y(static_cast<size_t>(n) * rows);
  for (auto t : {GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K}) {
    if (only && std::strcmp(only, ggml_type_name(t)) != 0) continue;
    std::vector<uint8_t> data(ggml_row_size(t, cols) * rows);
    for (auto& b : data) b = static_cast<uint8_t>(rng());
    // Sane fp16 scales so results stay finite: zero the high byte of every 16-bit word, keep it small.
    for (size_t i = 1; i < data.size(); i += 2) data[i] &= 0x1F;
    Weight w{t, data.data(), rows, cols, ggml_row_size(t, cols)};
    PackedBuffer pc = pack_weight(w), pw = pack_weight(w, true);
    for (int pass = 0; pass < 2; ++pass) {
      Weight v = w;
      if (pass == 0) v.wide = nullptr;  // compact panels
      if (pass == 1 && !v.wide) continue;
      double best = 1e9;
      for (int r = 0; r < 30; ++r) {  // first runs warm clocks and the thread pool
        auto t0 = std::chrono::steady_clock::now();
        matmul(v, x.data(), xq.data(), n, y.data());
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      std::printf("%-5s %-7s n=%-4d %7.2f ms  %5.2f TMAC/s\n", ggml_type_name(t), pass ? "wide" : "compact", n,
                  best * 1e3, static_cast<double>(rows) * cols * n / best / 1e12);
    }
  }
}
