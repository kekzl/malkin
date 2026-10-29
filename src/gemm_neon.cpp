// NEON dotprod panel GEMM: 16 weight rows per panel as 4 quads of 4 rows. A 16-byte weight vector holds one quad x
// 4 columns (lane = row), so vdotq_laneq_s32 multiplies it with 4 activation bytes picked by lane. Up to 2 tokens
// per call (32 q registers). No wide panels: compact panels serve prefill too.
#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "kernels.h"

namespace malkin::kern {

namespace {

constexpr int R = 16;
constexpr int kTile = 2;

// Panel sizes per 32 (Q8_0) or 256 (K-quants) columns. A group is 4 columns x 16 rows.
// Q8_0: d fp16 [16 rows]; 8 groups x 4 quads x 16 bytes (signed).
// Q4_K: d, dmin fp16 [16]; sc, m u8 [8 sub-blocks][16]; 64 nibble groups of 32 bytes: bytes 0-15 quads 0 (low
//       nibble) and 1 (high), bytes 16-31 quads 2 and 3.
// Q6_K: d fp16 [16]; sc i8 [16 halves][16]; low nibbles as Q4_K; 64 x 16 bytes high bits, quad u at bits 2u..2u+1.
//       q stays unsigned (0..63); the -32 offset comes from the activation half sums.
constexpr size_t kQ8Block = 32 + 8 * 64;
constexpr size_t kQ4KBlock = 64 + 256 + 64 * 32;
constexpr size_t kQ6KBlock = 32 + 256 + 64 * 32 + 64 * 16;

inline float32x4_t f16x4(const uint8_t* p) {
  uint16_t h[4];
  std::memcpy(h, p, 8);
  return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h)));
}

// 16 bytes (one per row) -> 4 quads of floats.
inline void u8_rows(const uint8_t* p, float32x4_t (&f)[4]) {
  const uint8x16_t v = vld1q_u8(p);
  const uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_high_u8(v);
  f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
  f[1] = vcvtq_f32_u32(vmovl_high_u16(lo));
  f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
  f[3] = vcvtq_f32_u32(vmovl_high_u16(hi));
}

inline void s8_rows(const uint8_t* p, float32x4_t (&f)[4]) {
  const int8x16_t v = vreinterpretq_s8_u8(vld1q_u8(p));
  const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_high_s8(v);
  f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
  f[1] = vcvtq_f32_s32(vmovl_high_s16(lo));
  f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
  f[3] = vcvtq_f32_s32(vmovl_high_s16(hi));
}

// f(lane) for lanes 0..3 as compile-time constants (vdotq_laneq_s32 needs one).
template <class F>
inline void for_lanes(F&& f) {
  f(std::integral_constant<int, 0>{});
  f(std::integral_constant<int, 1>{});
  f(std::integral_constant<int, 2>{});
  f(std::integral_constant<int, 3>{});
}

template <int L, int NT>
inline void sdot_quads(int32x4_t (&ai)[4][NT], const int8x16_t (&w)[4], const int8x16_t (&xv)[NT]) {
  for (int u = 0; u < 4; ++u)
    for (int t = 0; t < NT; ++t) ai[u][t] = vdotq_laneq_s32(ai[u][t], w[u], xv[t], L);
}

// Low nibbles of one Q4_K/Q6_K group -> 4 quads.
inline void nibble_quads(const uint8_t* g, uint8x16_t (&w)[4]) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F), v0 = vld1q_u8(g), v1 = vld1q_u8(g + 16);
  w[0] = vandq_u8(v0, m4);
  w[1] = vshrq_n_u8(v0, 4);
  w[2] = vandq_u8(v1, m4);
  w[3] = vshrq_n_u8(v1, 4);
}

template <int NT>
inline void store(float* y, int64_t ys, const float32x4_t (&acc)[4][NT], int rv) {
  for (int t = 0; t < NT; ++t) {
    if (rv == R) {
      for (int u = 0; u < 4; ++u) vst1q_f32(y + t * ys + 4 * u, acc[u][t]);
    } else {
      float tmp[R];
      for (int u = 0; u < 4; ++u) vst1q_f32(tmp + 4 * u, acc[u][t]);
      std::memcpy(y + t * ys, tmp, sizeof(float) * rv);
    }
  }
}

template <int NT>
void k_q8_0(const uint8_t* p, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  float32x4_t acc[4][NT];
  for (auto& a : acc)
    for (auto& v : a) v = vdupq_n_f32(0.0f);
  for (int64_t b = 0; b < nb; ++b, p += kQ8Block) {
    int32x4_t ai[4][NT];
    for (auto& a : ai)
      for (auto& v : a) v = vdupq_n_s32(0);
    for (int h = 0; h < 2; ++h) {
      int8x16_t xv[NT];
      for (int t = 0; t < NT; ++t) xv[t] = vld1q_s8(x[t * nb + b].qs + 16 * h);
      for_lanes([&](auto l) {
        constexpr int L = decltype(l)::value;
        const uint8_t* g = p + 32 + (4 * h + L) * 64;
        const int8x16_t w[4] = {vreinterpretq_s8_u8(vld1q_u8(g)), vreinterpretq_s8_u8(vld1q_u8(g + 16)),
                                vreinterpretq_s8_u8(vld1q_u8(g + 32)), vreinterpretq_s8_u8(vld1q_u8(g + 48))};
        sdot_quads<L, NT>(ai, w, xv);
      });
    }
    for (int u = 0; u < 4; ++u) {
      const float32x4_t dw = f16x4(p + 8 * u);
      for (int t = 0; t < NT; ++t)
        acc[u][t] = vfmaq_f32(acc[u][t], vcvtq_f32_s32(ai[u][t]), vmulq_n_f32(dw, x[t * nb + b].d));
    }
  }
  store<NT>(y, ys, acc, rv);
}

template <int NT>
void k_q4_k(const uint8_t* p, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  float32x4_t acc[4][NT];
  for (auto& a : acc)
    for (auto& v : a) v = vdupq_n_f32(0.0f);
  for (int64_t sb = 0; sb < cols / 256; ++sb, p += kQ4KBlock) {
    for (int j = 0; j < 8; ++j) {
      const BlockQ8* xb[NT];
      for (int t = 0; t < NT; ++t) xb[t] = x + t * nb + sb * 8 + j;
      int32x4_t ai[4][NT];
      for (auto& a : ai)
        for (auto& v : a) v = vdupq_n_s32(0);
      for (int h = 0; h < 2; ++h) {
        int8x16_t xv[NT];
        for (int t = 0; t < NT; ++t) xv[t] = vld1q_s8(xb[t]->qs + 16 * h);
        for_lanes([&](auto l) {
          constexpr int L = decltype(l)::value;
          uint8x16_t q[4];
          nibble_quads(p + 320 + (j * 8 + 4 * h + L) * 32, q);
          const int8x16_t w[4] = {vreinterpretq_s8_u8(q[0]), vreinterpretq_s8_u8(q[1]), vreinterpretq_s8_u8(q[2]),
                                  vreinterpretq_s8_u8(q[3])};
          sdot_quads<L, NT>(ai, w, xv);
        });
      }
      float32x4_t sc[4], mn[4];
      u8_rows(p + 64 + 16 * j, sc);
      u8_rows(p + 192 + 16 * j, mn);
      for (int u = 0; u < 4; ++u) {
        sc[u] = vmulq_f32(sc[u], f16x4(p + 8 * u));
        mn[u] = vmulq_f32(mn[u], f16x4(p + 32 + 8 * u));
      }
      for (int t = 0; t < NT; ++t) {
        const float dx = xb[t]->d, dsx = dx * static_cast<float>(xb[t]->s[0] + xb[t]->s[1]);
        for (int u = 0; u < 4; ++u) {
          acc[u][t] = vfmaq_f32(acc[u][t], vcvtq_f32_s32(ai[u][t]), vmulq_n_f32(sc[u], dx));
          acc[u][t] = vfmsq_f32(acc[u][t], mn[u], vdupq_n_f32(dsx));
        }
      }
    }
  }
  store<NT>(y, ys, acc, rv);
}

template <int NT>
void k_q6_k(const uint8_t* p, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  const uint8x16_t c30 = vdupq_n_u8(0x30);
  float32x4_t acc[4][NT];
  for (auto& a : acc)
    for (auto& v : a) v = vdupq_n_f32(0.0f);
  for (int64_t sb = 0; sb < cols / 256; ++sb, p += kQ6KBlock) {
    for (int j = 0; j < 8; ++j) {
      const BlockQ8* xb[NT];
      for (int t = 0; t < NT; ++t) xb[t] = x + t * nb + sb * 8 + j;
      for (int h = 0; h < 2; ++h) {
        int32x4_t ai[4][NT];
        for (auto& a : ai)
          for (auto& v : a) v = vdupq_n_s32(0);
        int8x16_t xv[NT];
        for (int t = 0; t < NT; ++t) xv[t] = vld1q_s8(xb[t]->qs + 16 * h);
        for_lanes([&](auto l) {
          constexpr int L = decltype(l)::value;
          const int grp = j * 8 + 4 * h + L;
          uint8x16_t q[4];
          nibble_quads(p + 288 + grp * 32, q);
          const uint8x16_t hb = vld1q_u8(p + 288 + 64 * 32 + grp * 16);
          const int8x16_t w[4] = {
              vreinterpretq_s8_u8(vorrq_u8(q[0], vandq_u8(vshlq_n_u8(hb, 4), c30))),
              vreinterpretq_s8_u8(vorrq_u8(q[1], vandq_u8(vshlq_n_u8(hb, 2), c30))),
              vreinterpretq_s8_u8(vorrq_u8(q[2], vandq_u8(hb, c30))),
              vreinterpretq_s8_u8(vorrq_u8(q[3], vandq_u8(vshrq_n_u8(hb, 2), c30))),
          };
          sdot_quads<L, NT>(ai, w, xv);
        });
        float32x4_t sc[4];
        s8_rows(p + 32 + 16 * (2 * j + h), sc);
        for (int u = 0; u < 4; ++u) sc[u] = vmulq_f32(sc[u], f16x4(p + 8 * u));
        for (int t = 0; t < NT; ++t) {
          const int32x4_t bias = vdupq_n_s32(32 * xb[t]->s[h]);
          for (int u = 0; u < 4; ++u)
            acc[u][t] = vfmaq_f32(acc[u][t], vcvtq_f32_s32(vsubq_s32(ai[u][t], bias)), vmulq_n_f32(sc[u], xb[t]->d));
        }
      }
    }
  }
  store<NT>(y, ys, acc, rv);
}

// ---- packing (scalar, once at load) ----

// Byte of row r (0..15), column k of a group in a 4-quad vector set: quad r / 4, element (r % 4) * 4 + k % 4.
inline int quad_elem(int r, int k) { return (r % 4) * 4 + k % 4; }

void pack_q8_0(const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  for (int64_t b = 0; b < cols / 32; ++b) {
    uint8_t* o = dst + b * kQ8Block;
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + b * 34;
      std::memcpy(o + 2 * r, s, 2);
      for (int k = 0; k < 32; ++k) o[32 + (k / 4) * 64 + (r / 4) * 16 + quad_elem(r, k)] = s[2 + k];
    }
  }
}

// Low nibble v of row r, column k (0..255) into the nibble groups at g.
inline void put_nibble(uint8_t* g, int r, int k, int v) {
  const int u = r / 4;
  g[(k / 4) * 32 + (u / 2) * 16 + quad_elem(r, k)] |= static_cast<uint8_t>((v & 15) << (4 * (u & 1)));
}

void pack_q4_k(const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    uint8_t* o = dst + sb * kQ4KBlock;
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * 144;
      std::memcpy(o + 2 * r, s, 2);
      std::memcpy(o + 32 + 2 * r, s + 2, 2);
      for (int j = 0; j < 8; ++j) {
        int sc, m;
        scale_min_k4(j, s + 4, sc, m);
        o[64 + 16 * j + r] = static_cast<uint8_t>(sc);
        o[192 + 16 * j + r] = static_cast<uint8_t>(m);
      }
      for (int k = 0; k < 256; ++k) {
        const uint8_t q = s[16 + (k / 64) * 32 + k % 32];
        put_nibble(o + 320, r, k, k % 64 < 32 ? q & 15 : q >> 4);
      }
    }
  }
}

int q6_value(const uint8_t* s, int k) {
  const int n = k / 128, rem = k % 128, c = rem / 32, l = rem % 32;
  const uint8_t lo = s[64 * n + l + (c & 1) * 32];
  const int nib = c >= 2 ? lo >> 4 : lo & 0xF;
  const int hb = (s[128 + 32 * n + l] >> (2 * c)) & 3;
  return nib | (hb << 4);
}

void pack_q6_k(const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    uint8_t* o = dst + sb * kQ6KBlock;
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * 210;
      std::memcpy(o + 2 * r, s + 208, 2);
      for (int h = 0; h < 16; ++h) o[32 + 16 * h + r] = s[192 + h];
      for (int k = 0; k < 256; ++k) {
        const int q = q6_value(s, k);
        put_nibble(o + 288, r, k, q);
        o[288 + 64 * 32 + (k / 4) * 16 + quad_elem(r, k)] |= static_cast<uint8_t>((q >> 4) << (2 * (r / 4)));
      }
    }
  }
}

using Kernel = void (*)(const uint8_t*, int64_t, const BlockQ8*, float*, int64_t, int);

Kernel kernel(GgmlType t, int n) {
  switch (t) {
    case GgmlType::Q8_0:
      return n == 1 ? k_q8_0<1> : k_q8_0<2>;
    case GgmlType::Q4_K:
      return n == 1 ? k_q4_k<1> : k_q4_k<2>;
    case GgmlType::Q6_K:
      return n == 1 ? k_q6_k<1> : k_q6_k<2>;
    default:
      return nullptr;
  }
}

}  // namespace

int panel_rows() { return R; }

size_t panel_bytes(GgmlType t, int64_t cols, bool wide) {
  if (wide) return 0;
  switch (t) {
    case GgmlType::Q8_0:
      return static_cast<size_t>(cols / 32) * kQ8Block;
    case GgmlType::Q4_K:
      return static_cast<size_t>(cols / 256) * kQ4KBlock;
    case GgmlType::Q6_K:
      return static_cast<size_t>(cols / 256) * kQ6KBlock;
    default:
      return 0;
  }
}

void pack_panel(GgmlType t, bool wide, const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  std::memset(dst, 0, panel_bytes(t, cols, wide));
  switch (t) {
    case GgmlType::Q8_0:
      pack_q8_0(src, ss, rv, cols, dst);
      break;
    case GgmlType::Q4_K:
      pack_q4_k(src, ss, rv, cols, dst);
      break;
    case GgmlType::Q6_K:
      pack_q6_k(src, ss, rv, cols, dst);
      break;
    default:
      break;
  }
}

int gemm_tile(GgmlType, bool) { return kTile; }

bool gemm_unsigned_acts(GgmlType, bool) { return false; }

void gemm_panels(GgmlType t, bool, const uint8_t* p, size_t pb, int np, int64_t cols, const BlockQ8* x, int n, float* y,
                 int64_t ys, int rv) {
  const Kernel k = kernel(t, n);
  for (int q = 0; q < np && rv > q * R; ++q) k(p + q * pb, cols, x, y + q * R, ys, std::min(R, rv - q * R));
}

}  // namespace malkin::kern
