#pragma once
#include <cstdint>
#include <vector>

#include "gguf.h"

namespace malkin {

// Activation block: 32 int8 values, scale d, and the int sum of each 16-value half.
struct BlockQ8 {
  float d;
  int16_t s[2];
  int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40);

// A 2-D weight view into the mapped GGUF: rows x cols, cols contiguous.
struct Weight {
  GgmlType type = GgmlType::F32;
  const uint8_t* data = nullptr;
  int64_t rows = 0, cols = 0;
  size_t row_bytes = 0;
  const uint8_t* row(int64_t r) const { return data + static_cast<size_t>(r) * row_bytes; }
  explicit operator bool() const { return data != nullptr; }
};

bool weight_type_supported(GgmlType t);
// True for types whose dot product runs on Q8 activations (otherwise on float activations).
bool weight_type_quantized(GgmlType t);

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k);
void dequantize_row(GgmlType t, const void* src, float* dst, int64_t k);

// Dot product of one weight row (k elements) with one activation row.
float vec_dot(GgmlType t, int64_t k, const void* w, const float* x, const BlockQ8* xq);

// Quantize n activation rows of k values each (parallel). out holds n * k/32 blocks.
void quantize_acts(const float* x, int n, int64_t k, std::vector<BlockQ8>& out);

// y[t][r] = sum_c W[r][c] * x[t][c] for t < n. xq from quantize_acts, unused for float weights.
void matmul(const Weight& w, const float* x, const BlockQ8* xq, int n, float* y);

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(uint16_t h);
uint16_t fp32_to_bf16(float f);

}  // namespace malkin
