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

namespace {

// Panel GEMM of one weight, cut into work items: one panel group (2 panels for n > 1, else 1) x one token chunk.
struct PanelPlan {
  const Weight* w;
  float* y;
  const BlockQ8* xq;
  bool wide;
  const uint8_t* base;
  size_t pb;
  int nt_max, np, tiles, chunks, per;
  int64_t panels, items;
};

PanelPlan plan_panels(const Weight& w, float* y, const BlockQ8* xq, int n, int64_t threads) {
  PanelPlan p{};
  p.w = &w;
  p.y = y;
  p.xq = xq;
  p.wide = w.wide && n > kWideMinTokens;
  p.base = p.wide ? w.wide : w.packed;
  p.pb = p.wide ? w.wide_bytes : w.panel_bytes;
  p.nt_max = kern::gemm_tile(w.type, p.wide);
  p.np = n > 1 ? 2 : 1;
  p.panels = (w.rows + kern::panel_rows() - 1) / kern::panel_rows();
  const int64_t groups = (p.panels + p.np - 1) / p.np;
  p.tiles = (n + p.nt_max - 1) / p.nt_max;
  p.chunks = static_cast<int>(std::clamp<int64_t>((threads * 4 + groups - 1) / groups, 1, p.tiles));
  p.per = (p.tiles + p.chunks - 1) / p.chunks;
  p.items = groups * p.chunks;
  return p;
}

void run_item(const PanelPlan& pl, int64_t item, int n) {
  const Weight& w = *pl.w;
  const int R = kern::panel_rows();
  const int64_t nb = w.cols / 32;
  const int64_t p = item / pl.chunks * pl.np;
  const int gp = static_cast<int>(std::min<int64_t>(pl.np, pl.panels - p));
  const int rv = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(gp) * R, w.rows - p * R));
  const int tt_end = std::min(pl.tiles, static_cast<int>(item % pl.chunks + 1) * pl.per);
  for (int tt = static_cast<int>(item % pl.chunks) * pl.per; tt < tt_end; ++tt) {
    const int t0 = tt * pl.nt_max, nt = std::min(pl.nt_max, n - t0);
    kern::gemm_panels(w.type, pl.wide, pl.base + p * pl.pb, pl.pb, gp, w.cols, pl.xq + t0 * nb, nt,
                      pl.y + t0 * w.rows + p * R, w.rows, rv);
  }
}

}  // namespace

void matmul_group(std::initializer_list<MatmulOut> outs, const float* x, const BlockQ8* xq, int n) {
  for (const auto& o : outs) {
    if (!o.w->packed && !(o.w->wide && n > kWideMinTokens)) {
      for (const auto& m : outs) matmul(*m.w, x, xq, n, m.y);
      return;
    }
  }
  const int64_t threads = omp_get_max_threads();
  const int64_t nb = outs.begin()->w->cols / 32;
  const BlockQ8* uxq = nullptr;
  for (const auto& o : outs) {
    if (!kern::gemm_unsigned_acts(o.w->type, o.w->wide && n > kWideMinTokens) || uxq) continue;
    // Same blocks with qs ^ 0x80 (= qs + 128 as unsigned bytes); n * cols bytes, small next to the GEMM.
    static thread_local std::vector<BlockQ8> ux;
    if (ux.size() < static_cast<size_t>(n * nb)) ux.resize(n * nb);
    BlockQ8* u = ux.data();  // thread_local: resolve on this thread, not inside the parallel region
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n * nb; ++i) {
      u[i] = xq[i];
      for (int8_t& v : u[i].qs) v = static_cast<int8_t>(static_cast<uint8_t>(v) ^ 0x80);
    }
    uxq = u;
  }
  constexpr int kMax = 4;
  PanelPlan plans[kMax];
  int64_t end[kMax];
  int np = 0;
  for (const auto& o : outs) {
    const bool wide = o.w->wide && n > kWideMinTokens;
    plans[np] = plan_panels(*o.w, o.y, kern::gemm_unsigned_acts(o.w->type, wide) ? uxq : xq, n, threads);
    end[np] = (np ? end[np - 1] : 0) + plans[np].items;
    ++np;
  }
  // One region for all weights: one fork/join per group (q/k/v, gate/up). Dynamic hand-out: with other tenants on
  // some cores, static splits wait for the slowest thread (14B Q6_K gate+up: 1.57 TMAC/s in the model vs 3.0 alone).
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t item = 0; item < end[np - 1]; ++item) {
    int j = 0;
    while (item >= end[j]) ++j;
    run_item(plans[j], item - (j ? end[j - 1] : 0), n);
  }
}

void matmul(const Weight& w, const float* x, const BlockQ8* xq, int n, float* y) {
  if (w.packed || (w.wide && n > kWideMinTokens)) {
    matmul_group({{&w, y}}, x, xq, n);
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
