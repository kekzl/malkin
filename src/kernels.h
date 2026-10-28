#pragma once
// ISA kernels: one implementation per target (kernels_avx512.cpp, kernels_neon.cpp).
#include <cstdint>
#include <cstring>

#include "quant.h"

namespace malkin::kern {

inline uint16_t ld16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// Q4_K / Q5_K packed 6-bit scale and min of sub-block j (0..7).
inline void scale_min_k4(int j, const uint8_t* q, int& sc, int& m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k);

float dot_f32(int64_t k, const float* w, const float* x);
float dot_f16(int64_t k, const uint16_t* w, const float* x);
float dot_bf16(int64_t k, const uint16_t* w, const float* x);
float dot_q8_0(int64_t k, const uint8_t* w, const BlockQ8* x);
float dot_q4_k(int64_t k, const uint8_t* w, const BlockQ8* x);
float dot_q5_k(int64_t k, const uint8_t* w, const BlockQ8* x);
float dot_q6_k(int64_t k, const uint8_t* w, const BlockQ8* x);

// Attention helpers on BF16 KV rows; n is a multiple of 16.
float dot_f32_bf16(const float* q, const uint16_t* k, int n);
void axpy_bf16(float a, const uint16_t* v, float* y, int n);

// Name of the ISA this binary was built for.
const char* isa_name();

}  // namespace malkin::kern
