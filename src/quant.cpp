#include "quant.h"

#include <omp.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace malkin {

using kern::ld16;
using kern::scale_min_k4;

float fp16_to_fp32(uint16_t h) {
  _Float16 f;
  std::memcpy(&f, &h, 2);
  return static_cast<float>(f);
}

uint16_t fp32_to_fp16(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

float bf16_to_fp32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t fp32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000) return static_cast<uint16_t>((u >> 16) | 64);  // quiet NaN
  u += 0x7fff + ((u >> 16) & 1);                                                    // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}

bool weight_type_supported(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::Q8_0:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
      return true;
    default:
      return false;
  }
}

bool weight_type_quantized(GgmlType t) {
  return t == GgmlType::Q8_0 || t == GgmlType::Q4_K || t == GgmlType::Q5_K || t == GgmlType::Q6_K;
}

namespace {

void dequant_q8_0(const uint8_t* b, float* y, int64_t k) {
  for (int64_t i = 0; i < k / 32; ++i, b += 34) {
    float d = fp16_to_fp32(ld16(b));
    for (int l = 0; l < 32; ++l) y[i * 32 + l] = d * static_cast<int8_t>(b[2 + l]);
  }
}

void dequant_q4_k(const uint8_t* b, float* y, int64_t k) {
  for (int64_t i = 0; i < k / 256; ++i, b += 144) {
    float d = fp16_to_fp32(ld16(b)), dmin = fp16_to_fp32(ld16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;
    for (int j = 0; j < 4; ++j, q += 32) {
      int s0, m0, s1, m1;
      scale_min_k4(2 * j, sc, s0, m0);
      scale_min_k4(2 * j + 1, sc, s1, m1);
      for (int l = 0; l < 32; ++l) *y++ = d * s0 * (q[l] & 0xF) - dmin * m0;
      for (int l = 0; l < 32; ++l) *y++ = d * s1 * (q[l] >> 4) - dmin * m1;
    }
  }
}

void dequant_q5_k(const uint8_t* b, float* y, int64_t k) {
  for (int64_t i = 0; i < k / 256; ++i, b += 176) {
    float d = fp16_to_fp32(ld16(b)), dmin = fp16_to_fp32(ld16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* q = b + 48;
    for (int j = 0; j < 4; ++j, q += 32) {
      int s0, m0, s1, m1;
      scale_min_k4(2 * j, sc, s0, m0);
      scale_min_k4(2 * j + 1, sc, s1, m1);
      for (int l = 0; l < 32; ++l) *y++ = d * s0 * ((q[l] & 0xF) + (((qh[l] >> (2 * j)) & 1) << 4)) - dmin * m0;
      for (int l = 0; l < 32; ++l) *y++ = d * s1 * ((q[l] >> 4) + (((qh[l] >> (2 * j + 1)) & 1) << 4)) - dmin * m1;
    }
  }
}

void dequant_q6_k(const uint8_t* b, float* y, int64_t k) {
  for (int64_t i = 0; i < k / 256; ++i, b += 210) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    float d = fp16_to_fp32(ld16(b + 208));
    for (int n = 0; n < 2; ++n, ql += 64, qh += 32, sc += 8, y += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l] = d * sc[is] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
    }
  }
}

}  // namespace

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k) { kern::quantize_row_q8(x, y, k); }

void dequantize_row(GgmlType t, const void* src, float* dst, int64_t k) {
  auto* b = static_cast<const uint8_t*>(src);
  switch (t) {
    case GgmlType::F32:
      std::memcpy(dst, src, k * 4);
      return;
    case GgmlType::F16:
      for (int64_t i = 0; i < k; ++i) dst[i] = fp16_to_fp32(ld16(b + 2 * i));
      return;
    case GgmlType::BF16:
      for (int64_t i = 0; i < k; ++i) dst[i] = bf16_to_fp32(ld16(b + 2 * i));
      return;
    case GgmlType::Q8_0:
      dequant_q8_0(b, dst, k);
      return;
    case GgmlType::Q4_K:
      dequant_q4_k(b, dst, k);
      return;
    case GgmlType::Q5_K:
      dequant_q5_k(b, dst, k);
      return;
    case GgmlType::Q6_K:
      dequant_q6_k(b, dst, k);
      return;
    default:
      throw std::runtime_error(std::string("dequantize: unsupported type ") + ggml_type_name(t));
  }
}

float vec_dot(GgmlType t, int64_t k, const void* w, const float* x, const BlockQ8* xq) {
  auto* b = static_cast<const uint8_t*>(w);
  switch (t) {
    case GgmlType::F32:
      return kern::dot_f32(k, static_cast<const float*>(w), x);
    case GgmlType::F16:
      return kern::dot_f16(k, static_cast<const uint16_t*>(w), x);
    case GgmlType::BF16:
      return kern::dot_bf16(k, static_cast<const uint16_t*>(w), x);
    case GgmlType::Q8_0:
      return kern::dot_q8_0(k, b, xq);
    case GgmlType::Q4_K:
      return kern::dot_q4_k(k, b, xq);
    case GgmlType::Q5_K:
      return kern::dot_q5_k(k, b, xq);
    case GgmlType::Q6_K:
      return kern::dot_q6_k(k, b, xq);
    default:
      throw std::runtime_error(std::string("vec_dot: unsupported type ") + ggml_type_name(t));
  }
}

void quantize_acts(const float* x, int n, int64_t k, std::vector<BlockQ8>& out) {
  const int64_t nb = k / 32;
  if (out.size() < static_cast<size_t>(n * nb)) out.resize(n * nb);
  BlockQ8* o = out.data();
#pragma omp parallel for schedule(static) if (n > 1)
  for (int t = 0; t < n; ++t) kern::quantize_row_q8(x + t * k, o + t * nb, k);
}

PackedBuffer::~PackedBuffer() { std::free(ptr); }

PackedBuffer pack_weight(Weight& w, bool wide) {
  PackedBuffer b;
  const size_t pb = kern::panel_bytes(w.type, w.cols, wide);
  if (pb == 0) return b;
  const int R = kern::panel_rows();
  const int64_t panels = (w.rows + R - 1) / R;
  constexpr size_t kHuge = 2u << 20;
  b.size = (static_cast<size_t>(panels) * pb + kHuge - 1) / kHuge * kHuge;
  b.ptr = static_cast<uint8_t*>(std::aligned_alloc(kHuge, b.size));
  if (!b.ptr) throw std::bad_alloc();
  madvise(b.ptr, b.size, MADV_HUGEPAGE);
#pragma omp parallel for schedule(dynamic, 4)
  for (int64_t p = 0; p < panels; ++p) {
    const int rv = static_cast<int>(std::min<int64_t>(R, w.rows - p * R));
    kern::pack_panel(w.type, wide, w.row(p * R), w.row_bytes, rv, w.cols, b.ptr + p * pb);
  }
  (wide ? w.wide : w.packed) = b.ptr;
  (wide ? w.wide_bytes : w.panel_bytes) = pb;
  return b;
}

void matmul(const Weight& w, const float* x, const BlockQ8* xq, int n, float* y) {
  const bool wide = w.wide && n > kWideMinTokens;
  if (wide || w.packed) {
    // Work items: (panel pair, token tile); static order keeps a pair's tiles on one thread.
    // Decode (n == 1) uses single panels: twice the items for the threads, no compute to share.
    const uint8_t* base = wide ? w.wide : w.packed;
    const size_t pb = wide ? w.wide_bytes : w.panel_bytes;
    const int nt_max = kern::gemm_tile(w.type, wide);
    const int R = kern::panel_rows();
    const int np = n > 1 ? 2 : 1;
    const int64_t panels = (w.rows + R - 1) / R;
    const int64_t groups = (panels + np - 1) / np;
    const int64_t nb = w.cols / 32;
    // Work item: one panel group x one token chunk, all its tiles in a row (the group stays in L2).
    // Dynamic hand-out: with other tenants on some cores, static splits wait for the slowest thread
    // (14B Q6_K gate+up: 1.57 TMAC/s in the model vs 3.0 in isolation).
    const int64_t threads = omp_get_max_threads();
    const int tiles = (n + nt_max - 1) / nt_max;
    const int chunks = static_cast<int>(std::clamp<int64_t>((threads * 4 + groups - 1) / groups, 1, tiles));
    const int per = (tiles + chunks - 1) / chunks;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t item = 0; item < groups * chunks; ++item) {
      const int64_t g = item / chunks;
      const int64_t p = g * np;
      const int gp = static_cast<int>(std::min<int64_t>(np, panels - p));
      const int rv = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(gp) * R, w.rows - p * R));
      const int tt_end = std::min(tiles, static_cast<int>(item % chunks + 1) * per);
      for (int tt = static_cast<int>(item % chunks) * per; tt < tt_end; ++tt) {
        const int t0 = tt * nt_max, nt = std::min(nt_max, n - t0);
        kern::gemm_panels(w.type, wide, base + p * pb, pb, gp, w.cols, xq + t0 * nb, nt, y + t0 * w.rows + p * R,
                          w.rows, rv);
      }
    }
    return;
  }
  // Tiles of up to 16 rows, at least 4 tiles per thread (1024 rows on 84 cores: tile 3).
  const int64_t threads = omp_get_max_threads();
  const int64_t tile_rows = std::clamp<int64_t>(w.rows / (threads * 4), 1, 16);
  const int64_t tiles = (w.rows + tile_rows - 1) / tile_rows;
  const int64_t k = w.cols, nb = k / 32;
  const GgmlType t = w.type;
#pragma omp parallel for schedule(static)
  for (int64_t tile = 0; tile < tiles; ++tile) {
    const int64_t r0 = tile * tile_rows, r1 = std::min(w.rows, r0 + tile_rows);
    for (int tok = 0; tok < n; ++tok) {
      const float* xr = x + tok * k;
      const BlockQ8* qr = xq ? xq + tok * nb : nullptr;
      float* yr = y + static_cast<int64_t>(tok) * w.rows;
      for (int64_t r = r0; r < r1; ++r) yr[r] = vec_dot(t, k, w.row(r), xr, qr);
    }
  }
}

}  // namespace malkin
