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

// No NEON panel kernels yet: panel_bytes() == 0 keeps every weight on the row path.
int panel_rows() { return 4; }
size_t panel_bytes(GgmlType, int64_t, bool) { return 0; }
void pack_panel(GgmlType, bool, const uint8_t*, size_t, int, int64_t, uint8_t*) {}
int gemm_tile(GgmlType, bool) { return 1; }
void gemm_panels(GgmlType, bool, const uint8_t*, size_t, int, int64_t, const BlockQ8*, int, float*, int64_t, int) {}

}  // namespace malkin::kern

namespace malkin::kern {

void attend_gqa(const float* q, int G, int D, const uint16_t* k, const uint16_t* v, size_t stride, int n, float scale,
                float* out, float* scratch, const int* extra, int ne) {
  const int nk = n + ne;
  auto key = [&](int j) { return static_cast<size_t>(j < n ? j : extra[j - n]) * stride; };
  for (int h = 0; h < G; ++h) {
    float* s = scratch + static_cast<size_t>(h) * nk;
    float mx = -INFINITY;
    for (int j = 0; j < nk; ++j) mx = std::max(mx, s[j] = dot_f32_bf16(q + h * D, k + key(j), D) * scale);
    float sum = 0.0f;
    for (int j = 0; j < nk; ++j) sum += s[j] = std::exp(s[j] - mx);
    float* o = out + h * D;
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    for (int j = 0; j < nk; ++j) axpy_bf16(s[j] / sum, v + key(j), o, D);
  }
}
}  // namespace malkin::kern
