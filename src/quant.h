#pragma once
#include <cstdint>
#include <initializer_list>
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
  // ISA panel layout (kern::panel_rows() rows interleaved), owned by the model; null = row path.
  const uint8_t* packed = nullptr;
  size_t panel_bytes = 0;
  const uint8_t* wide = nullptr;  // wide panels for prefill (n > kWideMinTokens); null = use packed
  size_t wide_bytes = 0;
  const uint8_t* row(int64_t r) const { return data + static_cast<size_t>(r) * row_bytes; }
  explicit operator bool() const { return data != nullptr; }
};

// Panel storage, 2 MiB aligned for huge pages.
struct PackedBuffer {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  PackedBuffer() = default;
  explicit PackedBuffer(size_t bytes);
  PackedBuffer(PackedBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }
  PackedBuffer& operator=(PackedBuffer&&) = delete;
  ~PackedBuffer();
};
// Panel bytes of w, rounded up to 64; 0 = the ISA has no such panel kernel for w.type.
size_t packed_bytes(const Weight& w, bool wide);
// Packs w into dst (packed_bytes(w, wide) bytes, 64-byte aligned) and points w.packed / w.wide at it.
void pack_weight_into(Weight& w, bool wide, uint8_t* dst);
// One weight in a buffer of its own (tests, benches); the model packs all weights into one buffer.
PackedBuffer pack_weight(Weight& w, bool wide = false);
// Token count above which matmul prefers wide panels.
inline constexpr int kWideMinTokens = 16;

bool weight_type_supported(GgmlType t);
// True for types whose dot product runs on Q8 activations (otherwise on float activations).
bool weight_type_quantized(GgmlType t);

void quantize_row_q8(const float* x, BlockQ8* y, int64_t k);
// KV head row: q[i] = round(x[i] / scale), scale = max|x| / 127 (returned; 0 for an all-zero row).
float quantize_i8(const float* x, int n, int8_t* q);
void dequantize_row(GgmlType t, const void* src, float* dst, int64_t k);

// Dot product of one weight row (k elements) with one activation row.
float vec_dot(GgmlType t, int64_t k, const void* w, const float* x, const BlockQ8* xq);

// Quantize n activation rows of k values each (parallel). out holds n * k/32 blocks.
void quantize_acts(const float* x, int n, int64_t k, std::vector<BlockQ8>& out);

// y[t][r] = sum_c W[r][c] * x[t][c] for t < n. xq from quantize_acts, unused for float weights.
void matmul(const Weight& w, const float* x, const BlockQ8* xq, int n, float* y);

// Up to 4 weights with the same input (q/k/v, gate/up) in one parallel region.
struct MatmulOut {
  const Weight* w;
  float* y;
};
void matmul_group(std::initializer_list<MatmulOut> outs, const float* x, const BlockQ8* xq, int n);

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(uint16_t h);
uint16_t fp32_to_bf16(float f);

}  // namespace malkin
