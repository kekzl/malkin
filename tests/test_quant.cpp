// ISA dot kernels vs scalar dequantize + dot on random blocks.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu.h"
#include "kernels.h"
#include "quant.h"

using namespace malkin;

static std::mt19937 rng(42);

static uint16_t f16(float f) { return fp32_to_fp16(f); }

static void put16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }

static std::vector<uint8_t> random_row(GgmlType t, int64_t k) {
  std::uniform_real_distribution<float> uf(-1.0f, 1.0f), ud(0.001f, 0.02f);
  std::uniform_int_distribution<int> ub(0, 255);
  std::vector<uint8_t> row(ggml_row_size(t, k));
  for (auto& b : row) b = static_cast<uint8_t>(ub(rng));
  switch (t) {
    case GgmlType::F32:
      for (int64_t i = 0; i < k; ++i) {
        float f = uf(rng);
        std::memcpy(row.data() + 4 * i, &f, 4);
      }
      break;
    case GgmlType::F16:
      for (int64_t i = 0; i < k; ++i) put16(row.data() + 2 * i, f16(uf(rng)));
      break;
    case GgmlType::BF16:
      for (int64_t i = 0; i < k; ++i) put16(row.data() + 2 * i, fp32_to_bf16(uf(rng)));
      break;
    case GgmlType::Q8_0:
      for (int64_t i = 0; i < k / 32; ++i) put16(row.data() + 34 * i, f16(ud(rng)));
      break;
    case GgmlType::Q4_K:
    case GgmlType::Q5_K: {
      size_t bs = t == GgmlType::Q4_K ? 144 : 176;
      for (int64_t i = 0; i < k / 256; ++i) {
        put16(row.data() + bs * i, f16(ud(rng)));
        put16(row.data() + bs * i + 2, f16(ud(rng)));
      }
      break;
    }
    case GgmlType::Q6_K:
      for (int64_t i = 0; i < k / 256; ++i) put16(row.data() + 210 * i + 208, f16(ud(rng)));
      break;
    default:
      break;
  }
  return row;
}

static bool check(GgmlType t, int64_t k) {
  std::uniform_real_distribution<float> uf(-2.0f, 2.0f);
  std::vector<float> x(k);
  for (auto& v : x) v = uf(rng);
  std::vector<BlockQ8> xq(k / 32);
  quantize_row_q8(x.data(), xq.data(), k);
  std::vector<float> xr(k);
  for (int64_t i = 0; i < k; ++i) xr[i] = xq[i / 32].d * xq[i / 32].qs[i % 32];
  const std::vector<float>& xa = weight_type_quantized(t) ? xr : x;

  int bad = 0;
  double worst = 0;
  for (int trial = 0; trial < 64; ++trial) {
    auto row = random_row(t, k);
    std::vector<float> w(k);
    dequantize_row(t, row.data(), w.data(), k);
    double ref = 0, mag = 0;
    for (int64_t i = 0; i < k; ++i) {
      ref += static_cast<double>(w[i]) * xa[i];
      mag += std::fabs(static_cast<double>(w[i]) * xa[i]);
    }
    float got = vec_dot(t, k, row.data(), x.data(), xq.data());
    double err = std::fabs(got - ref) / (mag + 1e-6);
    worst = std::max(worst, err);
    if (!(err < 1e-5)) ++bad;
  }
  std::printf("%-5s k=%-5lld worst rel err %.2e %s\n", ggml_type_name(t), static_cast<long long>(k), worst,
              bad ? "FAIL" : "ok");
  return bad == 0;
}

static bool check_q8_sums() {
  std::uniform_real_distribution<float> uf(-3.0f, 3.0f);
  std::vector<float> x(256);
  for (auto& v : x) v = uf(rng);
  std::vector<BlockQ8> q(8);
  quantize_row_q8(x.data(), q.data(), 256);
  for (auto& b : q) {
    int s0 = 0, s1 = 0;
    for (int l = 0; l < 16; ++l) s0 += b.qs[l], s1 += b.qs[16 + l];
    if (s0 != b.s[0] || s1 != b.s[1]) {
      std::printf("q8 sums FAIL\n");
      return false;
    }
  }
  std::printf("q8 sums ok\n");
  return true;
}

static bool check_bf16_helpers() {
  std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
  const int n = 128;
  std::vector<float> q(n), y(n), y0(n);
  std::vector<uint16_t> kv(n);
  for (int i = 0; i < n; ++i) q[i] = uf(rng), kv[i] = fp32_to_bf16(uf(rng)), y[i] = y0[i] = uf(rng);
  double ref = 0;
  for (int i = 0; i < n; ++i) ref += static_cast<double>(q[i]) * bf16_to_fp32(kv[i]);
  bool ok = std::fabs(kern::dot_f32_bf16(q.data(), kv.data(), n) - ref) < 1e-4;
  kern::axpy_bf16(0.5f, kv.data(), y.data(), n);
  for (int i = 0; i < n; ++i) ok &= std::fabs(y[i] - (y0[i] + 0.5f * bf16_to_fp32(kv[i]))) < 1e-6;
  std::printf("bf16 attention helpers %s (%s)\n", ok ? "ok" : "FAIL", kern::isa_name());
  return ok;
}

// Panel GEMM vs row path: 19 rows (partial pair), 35 rows (pair + partial single), tokens across tile edges.
static bool check_gemm(GgmlType t, int64_t k, int rows, bool outliers = false) {
  if (kern::panel_bytes(t, k, false) == 0) return true;

  std::vector<uint8_t> data;
  for (int r = 0; r < rows; ++r) {
    auto row = random_row(t, k);
    data.insert(data.end(), row.begin(), row.end());
  }
  Weight w;
  w.type = t;
  w.data = data.data();
  w.rows = rows;
  w.cols = k;
  w.row_bytes = ggml_row_size(t, k);
  Weight wp = w;
  PackedBuffer buf = pack_weight(wp);
  PackedBuffer bufw = pack_weight(wp, true);  // n > kWideMinTokens takes these where the type has them
  std::uniform_real_distribution<float> uf(-2.0f, 2.0f);
  bool ok = true;
  double worst = 0;
  for (int n : {1, 3, 4, 6, 7, 11, 17, 23}) {
    std::vector<float> x(n * k);
    for (auto& v : x) v = uf(rng);
    // LLM-like activations: a few channels 100x larger, so most Q8 values in their blocks round to 0 or +-1.
    if (outliers)
      for (int64_t i = 0; i < n * k; i += 97) x[i] *= 100.0f;
    std::vector<BlockQ8> xq;
    quantize_acts(x.data(), n, k, xq);
    std::vector<float> y0(static_cast<size_t>(n) * rows), y1(static_cast<size_t>(n) * rows);
    matmul(w, x.data(), xq.data(), n, y0.data());
    matmul(wp, x.data(), xq.data(), n, y1.data());
    double rms = 0;
    for (float v : y0) rms += static_cast<double>(v) * v;
    rms = std::sqrt(rms / y0.size());
    for (int i = 0; i < n * rows; ++i) {
      double err = std::fabs(y0[i] - y1[i]) / (rms + 1e-6);
      worst = std::max(worst, err);
      ok &= err < 1e-5;
    }
  }
  std::printf("gemm %-5s k=%-5lld rows=%-3d worst rel err %.2e %s\n", ggml_type_name(t), static_cast<long long>(k),
              rows, worst, ok ? "ok" : "FAIL");
  return ok;
}

int main() {
  if (auto miss = cpu_missing_features(); !miss.empty()) {
    std::printf("skip: CPU lacks %s\n", miss.c_str());
    return 77;
  }
  bool ok = check_q8_sums() && check_bf16_helpers();
  for (auto t :
       {GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K}) {
    ok &= check(t, 2560);
  }
  ok &= check(GgmlType::Q8_0, 96);  // odd block count: 256-bit tail
  ok &= check(GgmlType::F32, 100);  // scalar tail
  for (auto t : {GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K})
    ok &= check_gemm(t, 512, 19) && check_gemm(t, 512, 35) && check_gemm(t, 5120, 35, true);
  return ok ? 0 : 1;
}
