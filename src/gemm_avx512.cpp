// AVX-512 VNNI panel GEMM: 16 weight rows per panel, one 32-bit lane per row, up to 8 tokens per call.
// Weights are stored unsigned (vpdpbusd's u8 operand), activations are the signed Q8 blocks.
#include <immintrin.h>

#include <array>
#include <cstring>
#include <utility>

#include "kernels.h"

namespace malkin::kern {

namespace {

constexpr int R = 16;

// Panel sizes per 32 (Q8_0) or 256 (K-quants) columns.
// Q8_0 block pairs, 64-byte aligned: 2 x 16 fp16 scales, then 2 x 8 groups x 64 bytes (w + 128).
constexpr size_t kQ8_0Pair = 64 + 2 * 8 * 64;
inline const uint8_t* q8_scales(const uint8_t* p, int64_t b) { return p + (b / 2) * kQ8_0Pair + 32 * (b & 1); }
inline const uint8_t* q8_weights(const uint8_t* p, int64_t b) { return p + (b / 2) * kQ8_0Pair + 64 + 512 * (b & 1); }
constexpr size_t kQ4KBlock = 64 + 256 + 64 * 32;          // d/dmin, per-sub-block sc/m, 64 nibble groups
constexpr size_t kQ5KBlock = kQ4KBlock + 64 * 8;          // + bit-4 mask per group
constexpr size_t kQ6KBlock = 32 + 256 + 64 * 32 + 64 * 16;  // d, int8 scale per 16, nibbles, bit-4/5 masks

inline int32_t ld32(const void* p) {
  int32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline uint64_t ld64(const void* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

// 32 nibble bytes -> 64 bytes: rows 0-7 from low nibbles, rows 8-15 from high nibbles.
inline __m512i expand_nibbles(const uint8_t* p) {
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  const __m256i m = _mm256_set1_epi8(0x0F);
  return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(v, m)),
                            _mm256_and_si256(_mm256_srli_epi16(v, 4), m), 1);
}

// Nibble group byte offset and shift for row r, k-in-group i.
inline void nib_pos(int r, int i, int& byte, int& shift) {
  byte = (r & 7) * 4 + i;
  shift = r < 8 ? 0 : 4;
}

inline void store_rows(float* y, __m512 v, int rows_valid) {
  if (rows_valid == R) {
    _mm512_storeu_ps(y, v);
  } else {
    _mm512_mask_storeu_ps(y, static_cast<__mmask16>((1u << rows_valid) - 1), v);
  }
}

// ---- kernels: NP panels (16 rows each) x NT tokens per call ----

inline void store_panels(float* y, int64_t ys, int t, int q, __m512 v, int rv) {
  const int r = rv - R * q;
  if (r > 0) store_rows(y + t * ys + R * q, v, r < R ? r : R);
}

template <int NT, int NP>
void k_q8_0(const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  __m512 acc[NP][NT];
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) acc[q][t] = _mm512_setzero_ps();
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* pq[NP];
    __m512 dw[NP];
    for (int q = 0; q < NP; ++q) {
      pq[q] = q8_weights(p + q * pb, b);
      dw[q] = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_scales(p + q * pb, b))));
    }
    __m512i ai[NP][NT];
    for (int q = 0; q < NP; ++q)
      for (int t = 0; t < NT; ++t) ai[q][t] = _mm512_setzero_si512();
    for (int g = 0; g < 8; ++g) {
      __m512i w[NP];
      for (int q = 0; q < NP; ++q) w[q] = _mm512_loadu_si512(pq[q] + 64 * g);
      for (int t = 0; t < NT; ++t) {
        const __m512i xv = _mm512_set1_epi32(ld32(x[t * nb + b].qs + 4 * g));
        for (int q = 0; q < NP; ++q) ai[q][t] = _mm512_dpbusd_epi32(ai[q][t], w[q], xv);
      }
    }
    for (int t = 0; t < NT; ++t) {
      const BlockQ8& xb = x[t * nb + b];
      const __m512i corr = _mm512_set1_epi32(128 * (xb.s[0] + xb.s[1]));
      const __m512 dx = _mm512_set1_ps(xb.d);
      for (int q = 0; q < NP; ++q)
        acc[q][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[q][t], corr)), _mm512_mul_ps(dw[q], dx), acc[q][t]);
    }
  }
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) store_panels(y, ys, t, q, acc[q][t], rv);
}

template <int NT, int NP, bool kHigh>
void k_q45_k(const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  constexpr size_t kBlock = kHigh ? kQ5KBlock : kQ4KBlock;
  const __m512i c16 = _mm512_set1_epi8(16);
  __m512 acc[NP][NT];
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) acc[q][t] = _mm512_setzero_ps();
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    const uint8_t* pq[NP];
    for (int q = 0; q < NP; ++q) pq[q] = p + q * pb + sb * kBlock;
    for (int j = 0; j < 8; ++j) {
      __m512 sc[NP], mn[NP];
      for (int q = 0; q < NP; ++q) {
        const uint8_t* sm = pq[q] + 64 + 32 * j;
        sc[q] = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pq[q]))),
                              _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sm)))));
        mn[q] = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pq[q] + 32))),
                              _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sm + 16)))));
      }
      const int64_t xb = sb * 8 + j;
      __m512i ai[NP][NT];
      for (int q = 0; q < NP; ++q)
        for (int t = 0; t < NT; ++t) ai[q][t] = _mm512_setzero_si512();
      for (int g = 0; g < 8; ++g) {
        __m512i w[NP];
        for (int q = 0; q < NP; ++q) {
          w[q] = expand_nibbles(pq[q] + 320 + (j * 8 + g) * 32);
          if constexpr (kHigh) w[q] = _mm512_mask_add_epi8(w[q], _cvtu64_mask64(ld64(pq[q] + kQ4KBlock + (j * 8 + g) * 8)), w[q], c16);
        }
        for (int t = 0; t < NT; ++t) {
          const __m512i xv = _mm512_set1_epi32(ld32(x[t * nb + xb].qs + 4 * g));
          for (int q = 0; q < NP; ++q) ai[q][t] = _mm512_dpbusd_epi32(ai[q][t], w[q], xv);
        }
      }
      for (int t = 0; t < NT; ++t) {
        const BlockQ8& xq = x[t * nb + xb];
        const __m512 dx = _mm512_set1_ps(xq.d);
        const __m512 dsx = _mm512_set1_ps(xq.d * static_cast<float>(xq.s[0] + xq.s[1]));
        for (int q = 0; q < NP; ++q) {
          acc[q][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai[q][t]), _mm512_mul_ps(sc[q], dx), acc[q][t]);
          acc[q][t] = _mm512_fnmadd_ps(mn[q], dsx, acc[q][t]);
        }
      }
    }
  }
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) store_panels(y, ys, t, q, acc[q][t], rv);
}

template <int NT, int NP>
void k_q6_k(const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  const __m512i c16 = _mm512_set1_epi8(16), c32 = _mm512_set1_epi8(32);
  __m512 acc[NP][NT];
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) acc[q][t] = _mm512_setzero_ps();
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    const uint8_t* pq[NP];
    for (int q = 0; q < NP; ++q) pq[q] = p + q * pb + sb * kQ6KBlock;
    for (int h = 0; h < 16; ++h) {
      __m512 dsc[NP];
      for (int q = 0; q < NP; ++q) {
        dsc[q] = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pq[q]))),
                               _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pq[q] + 32 + 16 * h)))));
      }
      const int64_t xb = sb * 8 + h / 2;
      const int off = (h & 1) * 16;
      __m512i ai[NP][NT];
      for (int q = 0; q < NP; ++q)
        for (int t = 0; t < NT; ++t) ai[q][t] = _mm512_setzero_si512();
      for (int gg = 0; gg < 4; ++gg) {
        const int G = h * 4 + gg;
        __m512i w[NP];
        for (int q = 0; q < NP; ++q) {
          w[q] = expand_nibbles(pq[q] + 288 + 32 * G);
          const uint8_t* mk = pq[q] + 288 + 2048 + 16 * G;
          w[q] = _mm512_mask_add_epi8(w[q], _cvtu64_mask64(ld64(mk)), w[q], c16);
          w[q] = _mm512_mask_add_epi8(w[q], _cvtu64_mask64(ld64(mk + 8)), w[q], c32);
        }
        for (int t = 0; t < NT; ++t) {
          const __m512i xv = _mm512_set1_epi32(ld32(x[t * nb + xb].qs + off + 4 * gg));
          for (int q = 0; q < NP; ++q) ai[q][t] = _mm512_dpbusd_epi32(ai[q][t], w[q], xv);
        }
      }
      for (int t = 0; t < NT; ++t) {
        const BlockQ8& xq = x[t * nb + xb];
        const __m512i bias = _mm512_set1_epi32(32 * xq.s[h & 1]);
        const __m512 dx = _mm512_set1_ps(xq.d);
        for (int q = 0; q < NP; ++q)
          acc[q][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[q][t], bias)), _mm512_mul_ps(dsc[q], dx), acc[q][t]);
      }
    }
  }
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) store_panels(y, ys, t, q, acc[q][t], rv);
}

// ---- wide kernels: one byte per weight, unpacked at load; prefill only (compute-bound) ----

// Q4_K/Q5_K wide: d, dmin f32[16]; per sub-block sc, m u8[16]; 64 groups x 64 bytes q (0..31).
constexpr size_t kQ45WBlock = 64 + 64 + 256 + 64 * 64;
// Q6_K wide: d f32[16]; int8 scale [16 h][16 rows]; 64 groups x 64 bytes q (0..63, bias 32).
constexpr size_t kQ6WBlock = 64 + 256 + 64 * 64;

template <int NT, int NP>
void k_q45_w(const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  __m512 acc[NP][NT];
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) acc[q][t] = _mm512_setzero_ps();
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    const uint8_t* pq[NP];
    __m512 d[NP], dmin[NP];
    for (int q = 0; q < NP; ++q) {
      pq[q] = p + q * pb + sb * kQ45WBlock;
      d[q] = _mm512_loadu_ps(pq[q]);
      dmin[q] = _mm512_loadu_ps(pq[q] + 64);
    }
    for (int j = 0; j < 8; ++j) {
      __m512 sc[NP], mn[NP];
      for (int q = 0; q < NP; ++q) {
        const uint8_t* sm = pq[q] + 128 + 32 * j;
        sc[q] = _mm512_mul_ps(d[q], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sm)))));
        mn[q] = _mm512_mul_ps(dmin[q], _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sm + 16)))));
      }
      const int64_t xb = sb * 8 + j;
      __m512i ai[NP][NT];
      for (int q = 0; q < NP; ++q)
        for (int t = 0; t < NT; ++t) ai[q][t] = _mm512_setzero_si512();
      for (int g = 0; g < 8; ++g) {
        __m512i w[NP];
        for (int q = 0; q < NP; ++q) w[q] = _mm512_loadu_si512(pq[q] + 384 + (j * 8 + g) * 64);
        for (int t = 0; t < NT; ++t) {
          const __m512i xv = _mm512_set1_epi32(ld32(x[t * nb + xb].qs + 4 * g));
          for (int q = 0; q < NP; ++q) ai[q][t] = _mm512_dpbusd_epi32(ai[q][t], w[q], xv);
        }
      }
      for (int t = 0; t < NT; ++t) {
        const BlockQ8& xq = x[t * nb + xb];
        const __m512 dx = _mm512_set1_ps(xq.d);
        const __m512 dsx = _mm512_set1_ps(xq.d * static_cast<float>(xq.s[0] + xq.s[1]));
        for (int q = 0; q < NP; ++q) {
          acc[q][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai[q][t]), _mm512_mul_ps(sc[q], dx), acc[q][t]);
          acc[q][t] = _mm512_fnmadd_ps(mn[q], dsx, acc[q][t]);
        }
      }
    }
  }
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) store_panels(y, ys, t, q, acc[q][t], rv);
}

template <int NT, int NP>
void k_q6_w(const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, float* y, int64_t ys, int rv) {
  const int64_t nb = cols / 32;
  __m512 acc[NP][NT];
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) acc[q][t] = _mm512_setzero_ps();
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    const uint8_t* pq[NP];
    __m512 d[NP];
    for (int q = 0; q < NP; ++q) {
      pq[q] = p + q * pb + sb * kQ6WBlock;
      d[q] = _mm512_loadu_ps(pq[q]);
    }
    for (int h = 0; h < 16; ++h) {
      __m512 dsc[NP];
      for (int q = 0; q < NP; ++q) {
        dsc[q] = _mm512_mul_ps(d[q], _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pq[q] + 64 + 16 * h)))));
      }
      const int64_t xb = sb * 8 + h / 2;
      const int off = (h & 1) * 16;
      __m512i ai[NP][NT];
      for (int q = 0; q < NP; ++q)
        for (int t = 0; t < NT; ++t) ai[q][t] = _mm512_setzero_si512();
      for (int gg = 0; gg < 4; ++gg) {
        __m512i w[NP];
        for (int q = 0; q < NP; ++q) w[q] = _mm512_loadu_si512(pq[q] + 320 + (h * 4 + gg) * 64);
        for (int t = 0; t < NT; ++t) {
          const __m512i xv = _mm512_set1_epi32(ld32(x[t * nb + xb].qs + off + 4 * gg));
          for (int q = 0; q < NP; ++q) ai[q][t] = _mm512_dpbusd_epi32(ai[q][t], w[q], xv);
        }
      }
      for (int t = 0; t < NT; ++t) {
        const BlockQ8& xq = x[t * nb + xb];
        const __m512i bias = _mm512_set1_epi32(32 * xq.s[h & 1]);
        const __m512 dx = _mm512_set1_ps(xq.d);
        for (int q = 0; q < NP; ++q)
          acc[q][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ai[q][t], bias)), _mm512_mul_ps(dsc[q], dx), acc[q][t]);
      }
    }
  }
  for (int q = 0; q < NP; ++q)
    for (int t = 0; t < NT; ++t) store_panels(y, ys, t, q, acc[q][t], rv);
}

using Kernel = void (*)(const uint8_t*, size_t, int64_t, const BlockQ8*, float*, int64_t, int);

// Token tile per kernel (register budget: NP=2 x NT int + float accumulators within 32 zmm).
constexpr int kTileQ8_0 = 6, kTileQK = 4, kTileWide = 6;

template <template <int, int> class K, int NP, int... N>
constexpr auto table(std::integer_sequence<int, N...>) { return std::array<Kernel, sizeof...(N)>{&K<N + 1, NP>::run...}; }

#define MALKIN_KERNEL(name, fn) \
  template <int NT, int NP>     \
  struct name {                 \
    static void run(const uint8_t* p, size_t pb, int64_t c, const BlockQ8* x, float* y, int64_t ys, int rv) { fn; } \
  };
MALKIN_KERNEL(KQ8_0, (k_q8_0<NT, NP>(p, pb, c, x, y, ys, rv)))
MALKIN_KERNEL(KQ4K, (k_q45_k<NT, NP, false>(p, pb, c, x, y, ys, rv)))
MALKIN_KERNEL(KQ5K, (k_q45_k<NT, NP, true>(p, pb, c, x, y, ys, rv)))
MALKIN_KERNEL(KQ6K, (k_q6_k<NT, NP>(p, pb, c, x, y, ys, rv)))
MALKIN_KERNEL(KQ45W, (k_q45_w<NT, NP>(p, pb, c, x, y, ys, rv)))
MALKIN_KERNEL(KQ6W, (k_q6_w<NT, NP>(p, pb, c, x, y, ys, rv)))
#undef MALKIN_KERNEL

template <int NP>
struct Tables {
  static constexpr auto q8_0 = table<KQ8_0, NP>(std::make_integer_sequence<int, kTileQ8_0>{});
  static constexpr auto q4k = table<KQ4K, NP>(std::make_integer_sequence<int, kTileQK>{});
  static constexpr auto q5k = table<KQ5K, NP>(std::make_integer_sequence<int, kTileQK>{});
  static constexpr auto q6k = table<KQ6K, NP>(std::make_integer_sequence<int, kTileQK>{});
  static constexpr auto q45w = table<KQ45W, NP>(std::make_integer_sequence<int, kTileWide>{});
  static constexpr auto q6w = table<KQ6W, NP>(std::make_integer_sequence<int, kTileWide>{});
};

template <int NP>
void dispatch(GgmlType t, bool wide, const uint8_t* p, size_t pb, int64_t cols, const BlockQ8* x, int n, float* y,
              int64_t ys, int rv) {
  using T = Tables<NP>;
  switch (t) {
    case GgmlType::Q8_0: T::q8_0[n - 1](p, pb, cols, x, y, ys, rv); break;
    case GgmlType::Q4_K: wide ? T::q45w[n - 1](p, pb, cols, x, y, ys, rv) : T::q4k[n - 1](p, pb, cols, x, y, ys, rv); break;
    case GgmlType::Q5_K: wide ? T::q45w[n - 1](p, pb, cols, x, y, ys, rv) : T::q5k[n - 1](p, pb, cols, x, y, ys, rv); break;
    case GgmlType::Q6_K: wide ? T::q6w[n - 1](p, pb, cols, x, y, ys, rv) : T::q6k[n - 1](p, pb, cols, x, y, ys, rv); break;
    default: break;
  }
}

// ---- packing ----

void pack_q8_0(const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  for (int64_t b = 0; b < cols / 32; ++b) {
    uint8_t* ps = const_cast<uint8_t*>(q8_scales(dst, b));
    uint8_t* pw = const_cast<uint8_t*>(q8_weights(dst, b));
    std::memset(ps, 0, 32);
    std::memset(pw, 0x80, 8 * 64);
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + b * 34;
      std::memcpy(ps + 2 * r, s, 2);
      for (int k = 0; k < 32; ++k) pw[(k / 4) * 64 + 4 * r + k % 4] = s[2 + k] ^ 0x80;
    }
  }
}

void pack_q45_k(bool high, const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  const size_t sbytes = high ? 176 : 144, block = high ? kQ5KBlock : kQ4KBlock;
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    uint8_t* pb = dst + sb * block;
    std::memset(pb, 0, block);
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * sbytes;
      std::memcpy(pb + 2 * r, s, 2);
      std::memcpy(pb + 32 + 2 * r, s + 2, 2);
      const uint8_t* qh = s + 16;
      const uint8_t* qs = s + (high ? 48 : 16);
      for (int j = 0; j < 8; ++j) {
        int sc, m;
        scale_min_k4(j, s + 4, sc, m);
        pb[64 + 32 * j + r] = static_cast<uint8_t>(sc);
        pb[64 + 32 * j + 16 + r] = static_cast<uint8_t>(m);
        for (int k = 0; k < 32; ++k) {
          const uint8_t byte = qs[32 * (j / 2) + k];
          const int q = (j & 1) ? byte >> 4 : byte & 0xF;
          int at, shift;
          nib_pos(r, k % 4, at, shift);
          pb[320 + (j * 8 + k / 4) * 32 + at] |= static_cast<uint8_t>(q << shift);
          if (high && ((qh[k] >> j) & 1)) {
            uint64_t mk = ld64(pb + kQ4KBlock + (j * 8 + k / 4) * 8) | (1ull << (4 * r + k % 4));
            std::memcpy(pb + kQ4KBlock + (j * 8 + k / 4) * 8, &mk, 8);
          }
        }
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
    uint8_t* pb = dst + sb * kQ6KBlock;
    std::memset(pb, 0, kQ6KBlock);
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * 210;
      std::memcpy(pb + 2 * r, s + 208, 2);
      for (int h = 0; h < 16; ++h) pb[32 + 16 * h + r] = s[192 + h];
      for (int k = 0; k < 256; ++k) {
        const int q = q6_value(s, k), G = k / 4;
        int at, shift;
        nib_pos(r, k % 4, at, shift);
        pb[288 + 32 * G + at] |= static_cast<uint8_t>((q & 0xF) << shift);
        const uint64_t bit = 1ull << (4 * r + k % 4);
        for (int hb = 0; hb < 2; ++hb) {
          if ((q >> (4 + hb)) & 1) {
            uint8_t* mp = pb + 288 + 2048 + 16 * G + 8 * hb;
            uint64_t mk = ld64(mp) | bit;
            std::memcpy(mp, &mk, 8);
          }
        }
      }
    }
  }
}

void put_f32(uint8_t* p, float v) { std::memcpy(p, &v, 4); }

void pack_q45_w(bool high, const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  const size_t sbytes = high ? 176 : 144;
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    uint8_t* pb = dst + sb * kQ45WBlock;
    std::memset(pb, 0, kQ45WBlock);
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * sbytes;
      put_f32(pb + 4 * r, fp16_to_fp32(ld16(s)));
      put_f32(pb + 64 + 4 * r, fp16_to_fp32(ld16(s + 2)));
      const uint8_t* qh = s + 16;
      const uint8_t* qs = s + (high ? 48 : 16);
      for (int j = 0; j < 8; ++j) {
        int sc, m;
        scale_min_k4(j, s + 4, sc, m);
        pb[128 + 32 * j + r] = static_cast<uint8_t>(sc);
        pb[128 + 32 * j + 16 + r] = static_cast<uint8_t>(m);
        for (int k = 0; k < 32; ++k) {
          const uint8_t byte = qs[32 * (j / 2) + k];
          int q = (j & 1) ? byte >> 4 : byte & 0xF;
          if (high) q |= ((qh[k] >> j) & 1) << 4;
          pb[384 + (j * 8 + k / 4) * 64 + 4 * r + k % 4] = static_cast<uint8_t>(q);
        }
      }
    }
  }
}

void pack_q6_w(const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  for (int64_t sb = 0; sb < cols / 256; ++sb) {
    uint8_t* pb = dst + sb * kQ6WBlock;
    std::memset(pb, 0, kQ6WBlock);
    for (int r = 0; r < rv; ++r) {
      const uint8_t* s = src + r * ss + sb * 210;
      put_f32(pb + 4 * r, fp16_to_fp32(ld16(s + 208)));
      for (int h = 0; h < 16; ++h) pb[64 + 16 * h + r] = s[192 + h];
      for (int k = 0; k < 256; ++k) pb[320 + (k / 4) * 64 + 4 * r + k % 4] = static_cast<uint8_t>(q6_value(s, k));
    }
  }
}

}  // namespace

int panel_rows() { return R; }

size_t panel_bytes(GgmlType t, int64_t cols, bool wide) {
  const auto sb = static_cast<size_t>(cols / 256);
  switch (t) {
    case GgmlType::Q8_0: return wide ? 0 : static_cast<size_t>((cols / 32 + 1) / 2) * kQ8_0Pair;
    case GgmlType::Q4_K: return sb * (wide ? kQ45WBlock : kQ4KBlock);
    case GgmlType::Q5_K: return sb * (wide ? kQ45WBlock : kQ5KBlock);
    case GgmlType::Q6_K: return sb * (wide ? kQ6WBlock : kQ6KBlock);
    default: return 0;
  }
}

void pack_panel(GgmlType t, bool wide, const uint8_t* src, size_t ss, int rv, int64_t cols, uint8_t* dst) {
  switch (t) {
    case GgmlType::Q8_0: pack_q8_0(src, ss, rv, cols, dst); break;
    case GgmlType::Q4_K: wide ? pack_q45_w(false, src, ss, rv, cols, dst) : pack_q45_k(false, src, ss, rv, cols, dst); break;
    case GgmlType::Q5_K: wide ? pack_q45_w(true, src, ss, rv, cols, dst) : pack_q45_k(true, src, ss, rv, cols, dst); break;
    case GgmlType::Q6_K: wide ? pack_q6_w(src, ss, rv, cols, dst) : pack_q6_k(src, ss, rv, cols, dst); break;
    default: break;
  }
}

int gemm_tile(GgmlType t, bool wide) { return wide ? kTileWide : t == GgmlType::Q8_0 ? kTileQ8_0 : kTileQK; }

void gemm_panels(GgmlType t, bool wide, const uint8_t* p, size_t pb, int np, int64_t cols, const BlockQ8* x, int n,
                 float* y, int64_t ys, int rv) {
  if (np == 2) {
    dispatch<2>(t, wide, p, pb, cols, x, n, y, ys, rv);
  } else {
    dispatch<1>(t, wide, p, pb, cols, x, n, y, ys, rv);
  }
}

}  // namespace malkin::kern
