// x86-64: AVX-512 (F, BW, VL, DQ) + VNNI.
#include <immintrin.h>

#include <cmath>

#include "kernels.h"

namespace malkin::kern {

namespace {

inline __m256i ld256(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }

inline __m512i join(__m256i lo, __m256i hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); }

inline __m512 join_ps(float lo, float hi) { return _mm512_insertf32x8(_mm512_set1_ps(lo), _mm256_set1_ps(hi), 1); }

inline __m512 load_bf16(const uint16_t* p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(ld256(p)), 16));
}

// Q4_K and Q5_K: unsigned 4/5-bit q per 32-value sub-block, value = d*sc*q - dmin*m.
template <bool kHigh>
float dot_qk_45(int64_t k, const uint8_t* w, const BlockQ8* x) {
  constexpr size_t kBytes = kHigh ? 176 : 144;
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i one = _mm256_set1_epi8(1);
  __m512 acc = _mm512_setzero_ps();
  float mins = 0.0f;
  for (int64_t i = 0; i < k / 256; ++i, w += kBytes, x += 8) {
    const float d = fp16_to_fp32(ld16(w)), dmin = fp16_to_fp32(ld16(w + 2));
    const uint8_t* sc = w + 4;
    const uint8_t* q = w + (kHigh ? 48 : 16);
    __m256i qh = kHigh ? ld256(w + 16) : _mm256_setzero_si256();
    for (int j = 0; j < 4; ++j, q += 32) {
      int s0, m0, s1, m1;
      scale_min_k4(2 * j, sc, s0, m0);
      scale_min_k4(2 * j + 1, sc, s1, m1);
      __m256i qv = ld256(q);
      __m256i lo = _mm256_and_si256(qv, low4);
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), low4);
      if constexpr (kHigh) {
        lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(qh, one), 4));
        hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 1), one), 4));
        qh = _mm256_srli_epi16(qh, 2);
      }
      const BlockQ8& x0 = x[2 * j];
      const BlockQ8& x1 = x[2 * j + 1];
      __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), join(lo, hi), join(ld256(x0.qs), ld256(x1.qs)));
      acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), join_ps(d * s0 * x0.d, d * s1 * x1.d), acc);
      mins += dmin * (m0 * x0.d * (x0.s[0] + x0.s[1]) + m1 * x1.d * (x1.s[0] + x1.s[1]));
    }
  }
  return _mm512_reduce_add_ps(acc) - mins;
}

}  // namespace

const char* isa_name() { return "avx512-vnni"; }

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k) {
  for (int64_t i = 0; i < k / 32; ++i, x += 32) {
    __m512 a = _mm512_loadu_ps(x), b = _mm512_loadu_ps(x + 16);
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a), _mm512_abs_ps(b)));
    float d = amax / 127.0f;
    __m512 id = _mm512_set1_ps(d != 0.0f ? 1.0f / d : 0.0f);
    __m512i qa = _mm512_cvtps_epi32(_mm512_mul_ps(a, id));
    __m512i qb = _mm512_cvtps_epi32(_mm512_mul_ps(b, id));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs), _mm512_cvtsepi32_epi8(qa));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + 16), _mm512_cvtsepi32_epi8(qb));
    y[i].d = d;
    y[i].s[0] = static_cast<int16_t>(_mm512_reduce_add_epi32(qa));
    y[i].s[1] = static_cast<int16_t>(_mm512_reduce_add_epi32(qb));
  }
}

float dot_f32(int64_t k, const float* w, const float* x) {
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 16 <= k; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), acc);
  float s = _mm512_reduce_add_ps(acc);
  for (; i < k; ++i) s += w[i] * x[i];
  return s;
}

float dot_f16(int64_t k, const uint16_t* w, const float* x) {
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 16 <= k; i += 16) acc = _mm512_fmadd_ps(_mm512_cvtph_ps(ld256(w + i)), _mm512_loadu_ps(x + i), acc);
  float s = _mm512_reduce_add_ps(acc);
  for (; i < k; ++i) s += fp16_to_fp32(w[i]) * x[i];
  return s;
}

float dot_bf16(int64_t k, const uint16_t* w, const float* x) {
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 16 <= k; i += 16) acc = _mm512_fmadd_ps(load_bf16(w + i), _mm512_loadu_ps(x + i), acc);
  float s = _mm512_reduce_add_ps(acc);
  for (; i < k; ++i) s += bf16_to_fp32(w[i]) * x[i];
  return s;
}

// Q8_0: signed x signed via |w| (unsigned) times x with the sign of w.
float dot_q8_0(int64_t k, const uint8_t* w, const BlockQ8* x) {
  const int64_t nb = k / 32;
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 2 <= nb; i += 2) {
    const uint8_t* b0 = w + i * 34;
    const uint8_t* b1 = b0 + 34;
    __m512i wv = join(ld256(b0 + 2), ld256(b1 + 2));
    __m512i xv = join(ld256(x[i].qs), ld256(x[i + 1].qs));
    __mmask64 neg = _mm512_movepi8_mask(wv);
    __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
    __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
    __m512 sc = join_ps(fp16_to_fp32(ld16(b0)) * x[i].d, fp16_to_fp32(ld16(b1)) * x[i + 1].d);
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), sc, acc);
  }
  float s = _mm512_reduce_add_ps(acc);
  for (; i < nb; ++i) {
    const uint8_t* b = w + i * 34;
    __m256i wv = ld256(b + 2);
    __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(wv), _mm256_sign_epi8(ld256(x[i].qs), wv));
    s += static_cast<float>(_mm512_reduce_add_epi32(_mm512_zextsi256_si512(dp))) * fp16_to_fp32(ld16(b)) * x[i].d;
  }
  return s;
}

float dot_q4_k(int64_t k, const uint8_t* w, const BlockQ8* x) { return dot_qk_45<false>(k, w, x); }
float dot_q5_k(int64_t k, const uint8_t* w, const BlockQ8* x) { return dot_qk_45<true>(k, w, x); }

// Q6_K: unsigned 6-bit q (stored +32), int8 scale per 16 values.
float dot_q6_k(int64_t k, const uint8_t* w, const BlockQ8* x) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i three = _mm256_set1_epi8(3);
  const __m128i rep_a = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
  const __m128i rep_b = _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
  __m512 acc = _mm512_setzero_ps();
  float bias = 0.0f;
  for (int64_t i = 0; i < k / 256; ++i, w += 210) {
    const uint8_t* ql = w;
    const uint8_t* qhp = w + 128;
    const int8_t* scp = reinterpret_cast<const int8_t*>(w + 192);
    const float d = fp16_to_fp32(ld16(w + 208));
    for (int n = 0; n < 2; ++n, ql += 64, qhp += 32, scp += 8, x += 4) {
      __m256i l0 = ld256(ql), l1 = ld256(ql + 32), h = ld256(qhp);
      auto hbits = [&](int shift) {
        return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), three), 4);
      };
      __m256i q1 = _mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0));
      __m256i q2 = _mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2));
      __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4));
      __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6));
      __m128i sc8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(scp));
      __m512i sca = _mm512_cvtepi8_epi32(_mm_shuffle_epi8(sc8, rep_a));
      __m512i scb = _mm512_cvtepi8_epi32(_mm_shuffle_epi8(sc8, rep_b));
      __m512i dpa = _mm512_dpbusd_epi32(_mm512_setzero_si512(), join(q1, q2), join(ld256(x[0].qs), ld256(x[1].qs)));
      __m512i dpb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), join(q3, q4), join(ld256(x[2].qs), ld256(x[3].qs)));
      acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dpa, sca)), join_ps(d * x[0].d, d * x[1].d), acc);
      acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_mullo_epi32(dpb, scb)), join_ps(d * x[2].d, d * x[3].d), acc);
      for (int g = 0; g < 4; ++g) bias += d * x[g].d * 32.0f * (scp[2 * g] * x[g].s[0] + scp[2 * g + 1] * x[g].s[1]);
    }
  }
  return _mm512_reduce_add_ps(acc) - bias;
}

float dot_f32_bf16(const float* q, const uint16_t* k, int n) {
  __m512 s = _mm512_setzero_ps();
  for (int d = 0; d < n; d += 16) s = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), load_bf16(k + d), s);
  return _mm512_reduce_add_ps(s);
}

void axpy_bf16(float a, const uint16_t* v, float* y, int n) {
  const __m512 av = _mm512_set1_ps(a);
  for (int d = 0; d < n; d += 16)
    _mm512_storeu_ps(y + d, _mm512_fmadd_ps(av, load_bf16(v + d), _mm512_loadu_ps(y + d)));
}

namespace {

// e^x for x <= 0 (softmax after max subtraction): Cody-Waite range reduction, degree-6 Taylor, rel err ~1e-7.
inline __m512 exp512(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
  const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693145751953125f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.42860682030941723e-6f), r);
  __m512 p = _mm512_set1_ps(1.0f / 720.0f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
  return _mm512_scalef_ps(p, n);
}

// s[i] = exp(s[i] - max(s)), returns 1 / sum.
float softmax_inplace(float* s, int n) {
  __m512 m = _mm512_set1_ps(-INFINITY);
  int i = 0;
  for (; i + 16 <= n; i += 16) m = _mm512_max_ps(m, _mm512_loadu_ps(s + i));
  if (i < n)
    m = _mm512_mask_max_ps(m, static_cast<__mmask16>((1u << (n - i)) - 1), m,
                           _mm512_maskz_loadu_ps(static_cast<__mmask16>((1u << (n - i)) - 1), s + i));
  const __m512 mx = _mm512_set1_ps(_mm512_reduce_max_ps(m));
  __m512 sum = _mm512_setzero_ps();
  for (i = 0; i + 16 <= n; i += 16) {
    const __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(s + i), mx));
    _mm512_storeu_ps(s + i, e);
    sum = _mm512_add_ps(sum, e);
  }
  if (i < n) {
    const auto k = static_cast<__mmask16>((1u << (n - i)) - 1);
    const __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(k, s + i), mx));
    _mm512_mask_storeu_ps(s + i, k, e);
    sum = _mm512_mask_add_ps(sum, k, sum, e);
  }
  return 1.0f / _mm512_reduce_add_ps(sum);
}

template <int DC>
void attend_t(const float* q, int G, const uint16_t* k, const uint16_t* v, size_t stride, int n0, const int* extra,
              int ne, float scale, float* out, float* sc) {
  constexpr int D = DC * 16;
  constexpr int HP = DC <= 8 ? 2 : 1;  // heads per PV pass: HP x DC accumulators stay in registers
  const int n = n0 + ne;               // keys 0..n0-1, then extra[]
  for (int j = 0; j < n; ++j) {
    const uint16_t* kr = k + static_cast<size_t>(j < n0 ? j : extra[j - n0]) * stride;
    __m512 kf[DC];
    for (int d = 0; d < DC; ++d) kf[d] = load_bf16(kr + 16 * d);
    for (int h = 0; h < G; ++h) {
      __m512 s = _mm512_mul_ps(_mm512_loadu_ps(q + h * D), kf[0]);
      for (int d = 1; d < DC; ++d) s = _mm512_fmadd_ps(_mm512_loadu_ps(q + h * D + 16 * d), kf[d], s);
      sc[h * n + j] = _mm512_reduce_add_ps(s) * scale;
    }
  }
  float inv[16];
  for (int h = 0; h < G; ++h) inv[h] = softmax_inplace(sc + h * n, n);
  for (int h0 = 0; h0 < G; h0 += HP) {
    const int hp = G - h0 < HP ? G - h0 : HP;
    __m512 acc[HP][DC];
    for (int hh = 0; hh < HP; ++hh)
      for (int d = 0; d < DC; ++d) acc[hh][d] = _mm512_setzero_ps();
    for (int j = 0; j < n; ++j) {
      const uint16_t* vr = v + static_cast<size_t>(j < n0 ? j : extra[j - n0]) * stride;
      __m512 pj[HP];
      for (int hh = 0; hh < HP; ++hh) pj[hh] = _mm512_set1_ps(hh < hp ? sc[(h0 + hh) * n + j] : 0.0f);
      for (int d = 0; d < DC; ++d) {
        const __m512 vf = load_bf16(vr + 16 * d);
        for (int hh = 0; hh < HP; ++hh) acc[hh][d] = _mm512_fmadd_ps(pj[hh], vf, acc[hh][d]);
      }
    }
    for (int hh = 0; hh < hp; ++hh) {
      const __m512 iv = _mm512_set1_ps(inv[h0 + hh]);
      for (int d = 0; d < DC; ++d) _mm512_storeu_ps(out + (h0 + hh) * D + 16 * d, _mm512_mul_ps(acc[hh][d], iv));
    }
  }
}

}  // namespace

void attend_gqa(const float* q, int G, int D, const uint16_t* k, const uint16_t* v, size_t stride, int n, float scale,
                float* out, float* scratch, const int* extra, int ne) {
  if (G <= 16) {
    switch (D) {
      case 64:
        return attend_t<4>(q, G, k, v, stride, n, extra, ne, scale, out, scratch);
      case 128:
        return attend_t<8>(q, G, k, v, stride, n, extra, ne, scale, out, scratch);
      case 256:
        return attend_t<16>(q, G, k, v, stride, n, extra, ne, scale, out, scratch);
      default:
        break;
    }
  }
  const int nk = n + ne;
  auto key = [&](int j) { return static_cast<size_t>(j < n ? j : extra[j - n]) * stride; };
  for (int h = 0; h < G; ++h) {  // other head sizes: one head at a time
    float* s = scratch + static_cast<size_t>(h) * nk;
    for (int j = 0; j < nk; ++j) s[j] = dot_f32_bf16(q + h * D, k + key(j), D) * scale;
    const float iv = softmax_inplace(s, nk);
    float* o = out + h * D;
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    for (int j = 0; j < nk; ++j) axpy_bf16(s[j] * iv, v + key(j), o, D);
  }
}

}  // namespace malkin::kern
