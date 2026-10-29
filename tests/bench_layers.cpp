// Decode bandwidth: n = 1 matmuls over L layers of distinct packed weights (Qwen3 shapes, larger than L3), best of R.
// bench_gemm repeats one matrix, which stays in the 96 MB L3 of a 9800X3D at n = 1.
// usage: bench_layers [type=Q8_0] [layers=36] [R=20] [E H*D KV*D F, default Qwen3-4B 2560 4096 1024 9728]
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "quant.h"

using namespace malkin;

int main(int argc, char** argv) {
  GgmlType t = GgmlType::Q8_0;
  for (auto c : {GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K})
    if (argc > 1 && std::strcmp(argv[1], ggml_type_name(c)) == 0) t = c;
  const int L = argc > 2 ? std::atoi(argv[2]) : 36, R = argc > 3 ? std::atoi(argv[3]) : 20;
  const int64_t E = argc > 7 ? std::atoll(argv[4]) : 2560, Q = argc > 7 ? std::atoll(argv[5]) : 4096,
                K = argc > 7 ? std::atoll(argv[6]) : 1024, F = argc > 7 ? std::atoll(argv[7]) : 9728;
  const int64_t shape[4][3] = {{Q + 2 * K, E, 0}, {E, Q, 0}, {2 * F, E, 0}, {E, F, 0}};  // qkv, wo, gate+up, down
  std::vector<Weight> ws;
  std::vector<PackedBuffer> packs;
  std::vector<uint8_t> raw;
  uint64_t s = 88172645463325252ull;
  double total = 0;
  for (int l = 0; l < L; ++l)
    for (const auto& sh : shape) {
      Weight w{t, nullptr, sh[0], sh[1], ggml_row_size(t, sh[1])};
      raw.resize(w.row_bytes * w.rows);
      for (size_t i = 0; i < raw.size(); i += 8) {  // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        std::memcpy(raw.data() + i, &s, std::min<size_t>(8, raw.size() - i));
      }
      for (size_t i = 1; i < raw.size(); i += 2) raw[i] &= 0x1F;  // finite fp16 scales
      w.data = raw.data();
      packs.push_back(pack_weight(w));
      if (!w.packed) {
        std::fprintf(stderr, "no panel kernel for %s\n", ggml_type_name(t));
        return 1;
      }
      total += static_cast<double>(raw.size());
      w.data = w.packed;  // raw rows are gone; matmul reads the panels only
      ws.push_back(w);
    }
  std::vector<float> x(F, 0.5f), y(2 * F);
  std::vector<BlockQ8> xq;
  quantize_acts(x.data(), 1, F, xq);
  double best = 1e9, stage[4] = {};
  for (int r = 0; r < R; ++r) {
    double st[4] = {};
    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < 4; ++i) {
        const auto a = std::chrono::steady_clock::now();
        matmul(ws[l * 4 + i], x.data(), xq.data(), 1, y.data());
        st[i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
      }
    const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (tt < best) {
      best = tt;
      std::copy(st, st + 4, stage);
    }
  }
  const char* names[4] = {"qkv", "wo", "gate+up", "down"};
  for (int i = 0; i < 4; ++i)
    std::printf("%-8s %7.2f ms %6.2f GB/s\n", names[i], stage[i] * 1e3,
                static_cast<double>(shape[i][0]) * ggml_row_size(t, shape[i][1]) * L / stage[i] / 1e9);
  std::printf("%s, %d layers, %.2f GB: %.2f ms/token, %.2f GB/s, %d threads\n", ggml_type_name(t), L, total / 1e9,
              best * 1e3, total / best / 1e9, omp_get_max_threads());
}
