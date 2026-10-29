// AArch64: NEON + dotprod + fp16 (Armv8.2-A, Neoverse-N1 baseline).
#include <arm_neon.h>

#include <algorithm>
#include <cmath>

#include "kernels.h"

namespace malkin::kern {

namespace {

inline float32x4_t bf16x4(const uint16_t* p) { return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16)); }

inline int32_t dot32(int8x16_t a0, int8x16_t a1, const int8_t* x) {
  return vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), a0, vld1q_s8(x)), a1, vld1q_s8(x + 16)));
}

inline int32_t dot16(int8x16_t a, const int8_t* x) { return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), a, vld1q_s8(x))); }

template <bool kHigh>
float dot_qk_45(int64_t k, const uint8_t* w, const BlockQ8* x) {
  constexpr size_t kBytes = kHigh ? 176 : 144;
  const uint8x16_t low4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
  float sum = 0.0f;
  for (int64_t i = 0; i < k / 256; ++i, w += kBytes, x += 8) {
    const float d = fp16_to_fp32(ld16(w)), dmin = fp16_to_fp32(ld16(w + 2));
    const uint8_t* sc = w + 4;
    const uint8_t* q = w + (kHigh ? 48 : 16);
    uint8x16_t h0 = kHigh ? vld1q_u8(w + 16) : vdupq_n_u8(0);
    uint8x16_t h1 = kHigh ? vld1q_u8(w + 32) : vdupq_n_u8(0);
    for (int j = 0; j < 4; ++j, q += 32) {
      int s0, m0, s1, m1;
      scale_min_k4(2 * j, sc, s0, m0);
      scale_min_k4(2 * j + 1, sc, s1, m1);
      uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
      uint8x16_t lo0 = vandq_u8(q0, low4), lo1 = vandq_u8(q1, low4);
      uint8x16_t hi0 = vshrq_n_u8(q0, 4), hi1 = vshrq_n_u8(q1, 4);
      if constexpr (kHigh) {
        lo0 = vorrq_u8(lo0, vshlq_n_u8(vandq_u8(h0, one), 4));
        lo1 = vorrq_u8(lo1, vshlq_n_u8(vandq_u8(h1, one), 4));
        hi0 = vorrq_u8(hi0, vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 1), one), 4));
        hi1 = vorrq_u8(hi1, vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 1), one), 4));
        h0 = vshrq_n_u8(h0, 2);
        h1 = vshrq_n_u8(h1, 2);
      }
      const BlockQ8& x0 = x[2 * j];
      const BlockQ8& x1 = x[2 * j + 1];
      int32_t p0 = dot32(vreinterpretq_s8_u8(lo0), vreinterpretq_s8_u8(lo1), x0.qs);
      int32_t p1 = dot32(vreinterpretq_s8_u8(hi0), vreinterpretq_s8_u8(hi1), x1.qs);
      sum += d * (s0 * x0.d * p0 + s1 * x1.d * p1);
      sum -= dmin * (m0 * x0.d * (x0.s[0] + x0.s[1]) + m1 * x1.d * (x1.s[0] + x1.s[1]));
    }
  }
  return sum;
}

}  // namespace

const char* isa_name() { return "neon-dotprod"; }

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k) {
  for (int64_t i = 0; i < k / 32; ++i, x += 32) {
    float32x4_t v[8];
    float32x4_t mx = vdupq_n_f32(0.0f);
    for (int j = 0; j < 8; ++j) {
      v[j] = vld1q_f32(x + 4 * j);
      mx = vmaxq_f32(mx, vabsq_f32(v[j]));
    }
    const float d = vmaxvq_f32(mx) / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int32x4_t q[8];
    for (int j = 0; j < 8; ++j) q[j] = vcvtnq_s32_f32(vmulq_n_f32(v[j], id));
    for (int h = 0; h < 2; ++h) {
      int32x4_t* p = q + 4 * h;
      int16x8_t a = vcombine_s16(vqmovn_s32(p[0]), vqmovn_s32(p[1]));
      int16x8_t b = vcombine_s16(vqmovn_s32(p[2]), vqmovn_s32(p[3]));
      vst1q_s8(y[i].qs + 16 * h, vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)));
      y[i].s[h] = static_cast<int16_t>(vaddvq_s32(vaddq_s32(vaddq_s32(p[0], p[1]), vaddq_s32(p[2], p[3]))));
    }
    y[i].d = d;
  }
}

float dot_f32(int64_t k, const float* w, const float* x) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  int64_t i = 0;
  for (; i + 16 <= k; i += 16) {
    a0 = vfmaq_f32(a0, vld1q_f32(w + i), vld1q_f32(x + i));
    a1 = vfmaq_f32(a1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    a2 = vfmaq_f32(a2, vld1q_f32(w + i + 8), vld1q_f32(x + i + 8));
    a3 = vfmaq_f32(a3, vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  for (; i < k; ++i) s += w[i] * x[i];
  return s;
}

float dot_f16(int64_t k, const uint16_t* w, const float* x) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
  int64_t i = 0;
  for (; i + 8 <= k; i += 8) {
    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + i));
    a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
    a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
  }
  float s = vaddvq_f32(vaddq_f32(a0, a1));
  for (; i < k; ++i) s += fp16_to_fp32(w[i]) * x[i];
  return s;
}

float dot_bf16(int64_t k, const uint16_t* w, const float* x) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
  int64_t i = 0;
  for (; i + 8 <= k; i += 8) {
    a0 = vfmaq_f32(a0, bf16x4(w + i), vld1q_f32(x + i));
    a1 = vfmaq_f32(a1, bf16x4(w + i + 4), vld1q_f32(x + i + 4));
  }
  float s = vaddvq_f32(vaddq_f32(a0, a1));
  for (; i < k; ++i) s += bf16_to_fp32(w[i]) * x[i];
  return s;
}

float dot_q8_0(int64_t k, const uint8_t* w, const BlockQ8* x) {
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < k / 32; ++i, w += 34) {
    const int8_t* q = reinterpret_cast<const int8_t*>(w + 2);
    int32x4_t p =
        vdotq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(q), vld1q_s8(x[i].qs)), vld1q_s8(q + 16), vld1q_s8(x[i].qs + 16));
    acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p), fp16_to_fp32(ld16(w)) * x[i].d);
  }
  return vaddvq_f32(acc);
}

float dot_q4_k(int64_t k, const uint8_t* w, const BlockQ8* x) { return dot_qk_45<false>(k, w, x); }
float dot_q5_k(int64_t k, const uint8_t* w, const BlockQ8* x) { return dot_qk_45<true>(k, w, x); }

float dot_q6_k(int64_t k, const uint8_t* w, const BlockQ8* x) {
  const uint8x16_t low4 = vdupq_n_u8(0x0F), three = vdupq_n_u8(3);
  const int8x16_t off = vdupq_n_s8(32);
  float sum = 0.0f;
  for (int64_t i = 0; i < k / 256; ++i, w += 210) {
    const uint8_t* ql = w;
    const uint8_t* qh = w + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
    const float d = fp16_to_fp32(ld16(w + 208));
    for (int n = 0; n < 2; ++n, ql += 64, qh += 32, sc += 8, x += 4) {
      for (int half = 0; half < 2; ++half) {  // values l in [16*half, 16*half+16)
        uint8x16_t l0 = vld1q_u8(ql + 16 * half), l1 = vld1q_u8(ql + 32 + 16 * half), h = vld1q_u8(qh + 16 * half);
        auto q6 = [&](uint8x16_t lo4, int shift) {
          uint8x16_t hb = vshlq_n_u8(vandq_u8(vshlq_u8(h, vdupq_n_s8(static_cast<int8_t>(-shift))), three), 4);
          return vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo4, hb)), off);
        };
        int8x16_t q1 = q6(vandq_u8(l0, low4), 0);
        int8x16_t q2 = q6(vandq_u8(l1, low4), 2);
        int8x16_t q3 = q6(vshrq_n_u8(l0, 4), 4);
        int8x16_t q4 = q6(vshrq_n_u8(l1, 4), 6);
        sum += d * (x[0].d * sc[half] * dot16(q1, x[0].qs + 16 * half) +
                    x[1].d * sc[2 + half] * dot16(q2, x[1].qs + 16 * half) +
                    x[2].d * sc[4 + half] * dot16(q3, x[2].qs + 16 * half) +
                    x[3].d * sc[6 + half] * dot16(q4, x[3].qs + 16 * half));
      }
    }
  }
  return sum;
}

float dot_f32_bf16(const float* q, const uint16_t* k, int n) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
  for (int d = 0; d < n; d += 8) {
    a0 = vfmaq_f32(a0, vld1q_f32(q + d), bf16x4(k + d));
    a1 = vfmaq_f32(a1, vld1q_f32(q + d + 4), bf16x4(k + d + 4));
  }
  return vaddvq_f32(vaddq_f32(a0, a1));
}

void axpy_bf16(float a, const uint16_t* v, float* y, int n) {
  for (int d = 0; d < n; d += 4) vst1q_f32(y + d, vfmaq_n_f32(vld1q_f32(y + d), bf16x4(v + d), a));
}

}  // namespace malkin::kern

namespace malkin::kern {

namespace {

// e^x as exp512 (kernels_avx512.cpp): Cody-Waite reduction, degree-6 Taylor, rel err ~1e-7; x clamped to [-87, 88].
inline float32x4_t exp4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
  const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
  r = vfmsq_f32(r, n, vdupq_n_f32(1.42860682030941723e-6f));
  float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
  for (float c : {1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f, 1.0f, 1.0f}) p = vfmaq_f32(vdupq_n_f32(c), p, r);
  const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// s[i] = exp(s[i] - max(s)), returns 1 / sum.
float softmax_inplace(float* s, int n) {
  float32x4_t m = vdupq_n_f32(-INFINITY);
  int i = 0;
  for (; i + 4 <= n; i += 4) m = vmaxq_f32(m, vld1q_f32(s + i));
  float mx = vmaxvq_f32(m);
  for (; i < n; ++i) mx = std::max(mx, s[i]);
  const float32x4_t vm = vdupq_n_f32(mx);
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (i = 0; i + 4 <= n; i += 4) {
    const float32x4_t e = exp4(vsubq_f32(vld1q_f32(s + i), vm));
    vst1q_f32(s + i, e);
    sum = vaddq_f32(sum, e);
  }
  float tail = 0.0f;
  for (; i < n; ++i) tail += s[i] = std::exp(s[i] - mx);
  return 1.0f / (vaddvq_f32(sum) + tail);
}

// out[0..4C) = iv * sum_j p[j] * v_j[0..4C): C accumulators stay in registers.
template <int C, class Row>
void pv_pass(const float* p, int n, const uint16_t* v, Row row, float iv, float* out) {
  float32x4_t acc[C];
  for (auto& a : acc) a = vdupq_n_f32(0.0f);
  for (int j = 0; j < n; ++j) {
    const uint16_t* vr = v + row(j);
    for (int c = 0; c < C; ++c) acc[c] = vfmaq_n_f32(acc[c], bf16x4(vr + 4 * c), p[j]);
  }
  for (int c = 0; c < C; ++c) vst1q_f32(out + 4 * c, vmulq_n_f32(acc[c], iv));
}

}  // namespace

void silu_mul(int64_t n, float* g, const float* u) {
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t x = vld1q_f32(g + i);
    const float32x4_t e = exp4(vnegq_f32(x));
    vst1q_f32(g + i, vmulq_f32(vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e)), vld1q_f32(u + i)));
  }
  for (; i < n; ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
}

void attend_gqa(const float* q, int G, int D, const uint16_t* k, const uint16_t* v, size_t stride, int n0, float scale,
                float* out, float* scratch, const int* extra, int ne) {
  const int n = n0 + ne;
  auto row = [&](int j) { return static_cast<size_t>(j < n0 ? j : extra[j - n0]) * stride; };
  // Scores: each key row is read once for all G heads.
  for (int j = 0; j < n; ++j) {
    const uint16_t* kr = k + row(j);
    for (int h = 0; h < G; ++h) {
      const float* qh = q + static_cast<size_t>(h) * D;
      float32x4_t a[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
      for (int d = 0; d < D; d += 16)
        for (int c = 0; c < 4; ++c) a[c] = vfmaq_f32(a[c], vld1q_f32(qh + d + 4 * c), bf16x4(kr + d + 4 * c));
      scratch[static_cast<size_t>(h) * n + j] =
          vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]))) * scale;
    }
  }
  // Values: 64 output floats per pass (16 registers), 16 for the rest of D.
  for (int h = 0; h < G; ++h) {
    float* s = scratch + static_cast<size_t>(h) * n;
    const float iv = softmax_inplace(s, n);
    float* o = out + static_cast<size_t>(h) * D;
    int d = 0;
    for (; d + 64 <= D; d += 64) pv_pass<16>(s, n, v + d, row, iv, o + d);
    for (; d < D; d += 16) pv_pass<4>(s, n, v + d, row, iv, o + d);
  }
}

// 4 rows (token x head) per block: one K/V row conversion feeds 4 rows of FMAs.
int attend_block_tokens(int G) { return std::max(1, 4 / G); }

void attend_block(const float* q, size_t q_stride, int B, int G, int D, const uint16_t* k, const uint16_t* v,
                  size_t stride, const int* n, float scale, float* out, float* scratch) {
  const int rows = B * G;
  constexpr int kMaxRows = 64;
  if (rows > kMaxRows) {
    for (int b = 0; b < B; ++b)
      attend_gqa(q + b * q_stride, G, D, k, v, stride, n[b], scale, out + b * q_stride, scratch);
    return;
  }
  int hi = 0;
  for (int b = 0; b < B; ++b) hi = std::max(hi, n[b]);
  const auto off = [&](int r) { return static_cast<size_t>(r / G) * q_stride + static_cast<size_t>(r % G) * D; };
  const auto keys = [&](int r) { return n[r / G]; };
  float iv[kMaxRows];
  for (int r0 = 0; r0 < rows; r0 += 4) {
    const int rc = std::min(4, rows - r0);
    // Scores: 4 rows x 4 keys per step (16 accumulators), pairwise adds reduce them to 4 scores per row.
    const float* qr[4];
    for (int i = 0; i < 4; ++i) qr[i] = q + off(r0 + std::min(i, rc - 1));
    int j = 0;
    for (; j + 4 <= hi; j += 4) {
      float32x4_t acc[4][4];
      for (auto& a : acc)
        for (auto& x : a) x = vdupq_n_f32(0.0f);
      for (int d = 0; d < D; d += 4) {
        float32x4_t kf[4];
        for (int c = 0; c < 4; ++c) kf[c] = bf16x4(k + static_cast<size_t>(j + c) * stride + d);
        for (int i = 0; i < 4; ++i) {
          const float32x4_t qv = vld1q_f32(qr[i] + d);
          for (int c = 0; c < 4; ++c) acc[i][c] = vfmaq_f32(acc[i][c], qv, kf[c]);
        }
      }
      for (int i = 0; i < rc; ++i) {
        const float32x4_t s = vpaddq_f32(vpaddq_f32(acc[i][0], acc[i][1]), vpaddq_f32(acc[i][2], acc[i][3]));
        vst1q_f32(scratch + static_cast<size_t>(r0 + i) * hi + j, vmulq_n_f32(s, scale));
      }
    }
    for (; j < hi; ++j)
      for (int i = 0; i < rc; ++i)
        scratch[static_cast<size_t>(r0 + i) * hi + j] =
            dot_f32_bf16(qr[i], k + static_cast<size_t>(j) * stride, D) * scale;
    int lo = hi;
    for (int i = 0; i < rc; ++i) {
      iv[r0 + i] = softmax_inplace(scratch + static_cast<size_t>(r0 + i) * hi, keys(r0 + i));
      lo = std::min(lo, keys(r0 + i));
    }
    // Values: 16 output floats x 4 rows per pass; keys 0..lo-1 shared, each row's own tail after that.
    const float* p[4];
    for (int i = 0; i < 4; ++i) p[i] = scratch + static_cast<size_t>(r0 + std::min(i, rc - 1)) * hi;
    for (int d0 = 0; d0 < D; d0 += 16) {
      float32x4_t acc[4][4];
      for (auto& a : acc)
        for (auto& x : a) x = vdupq_n_f32(0.0f);
      for (int jj = 0; jj < lo; ++jj) {
        const uint16_t* vr = v + static_cast<size_t>(jj) * stride + d0;
        const float32x4_t vf[4] = {bf16x4(vr), bf16x4(vr + 4), bf16x4(vr + 8), bf16x4(vr + 12)};
        for (int i = 0; i < 4; ++i)
          for (int c = 0; c < 4; ++c) acc[i][c] = vfmaq_n_f32(acc[i][c], vf[c], p[i][jj]);
      }
      for (int i = 0; i < rc; ++i) {
        for (int jj = lo; jj < keys(r0 + i); ++jj) {
          const uint16_t* vr = v + static_cast<size_t>(jj) * stride + d0;
          for (int c = 0; c < 4; ++c) acc[i][c] = vfmaq_n_f32(acc[i][c], bf16x4(vr + 4 * c), p[i][jj]);
        }
        float* o = out + off(r0 + i) + d0;
        for (int c = 0; c < 4; ++c) vst1q_f32(o + 4 * c, vmulq_n_f32(acc[i][c], iv[r0 + i]));
      }
    }
  }
}

}  // namespace malkin::kern
