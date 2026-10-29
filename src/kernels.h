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

// SwiGLU gate: g[i] = g[i] / (1 + e^-g[i]) * u[i].
void silu_mul(int64_t n, float* g, const float* u);

// Attention helpers on BF16 KV rows; n is a multiple of 16.
float dot_f32_bf16(const float* q, const uint16_t* k, int n);
void axpy_bf16(float a, const uint16_t* v, float* y, int n);

// Attention of one query token for the G query heads sharing one KV head (GQA).
// q: G x D floats; keys/values 0..n-1 at k + j * stride, v + j * stride (BF16), then the positions in extra[0..ne)
// (tree verification: the ancestors of a draft node); out: G x D.
// scratch: at least G * (n + ne) floats. D is a multiple of 16.
void attend_gqa(const float* q, int G, int D, const uint16_t* k, const uint16_t* v, size_t stride, int n, float scale,
                float* out, float* scratch, const int* extra = nullptr, int ne = 0);

// B consecutive query tokens of one sequence (prompt reading), same G heads: token b sees keys 0..n[b]-1.
// Head h of token b at q + b * q_stride + h * D, out likewise; scratch: at least B * G * max(n) floats.
// Shares each K/V row conversion across the B * G rows.
void attend_block(const float* q, size_t q_stride, int B, int G, int D, const uint16_t* k, const uint16_t* v,
                  size_t stride, const int* n, float scale, float* out, float* scratch);
// Tokens per attend_block call for G heads per KV head; 1 = attend_gqa per token.
int attend_block_tokens(int G);

// Panel GEMM. A panel holds panel_rows() weight rows interleaved for the ISA's dot instruction.
int panel_rows();
// Two panel formats: compact (about the GGUF size, all n) and wide (one byte per weight, unpacked at load, for
// compute-bound prefill). Bytes of one panel for cols columns; 0 = no such panel for t.
size_t panel_bytes(GgmlType t, int64_t cols, bool wide);
// rows_valid (<= panel_rows()) source rows, row stride src_stride bytes; missing rows are zero.
void pack_panel(GgmlType t, bool wide, const uint8_t* src, size_t src_stride, int rows_valid, int64_t cols,
                uint8_t* dst);
// Tokens per gemm_panels call for t (<= 8).
int gemm_tile(GgmlType t, bool wide);
// True: gemm_panels for (t, wide) reads activation bytes as unsigned qs + 128 (caller passes qs ^ 0x80).
bool gemm_unsigned_acts(GgmlType t, bool wide);
// np (1 or 2) consecutive panels, panel_bytes apart: y[t * y_stride + r] = row r . x[t] for t < n (n <= gemm_tile),
// r < rows_valid (<= np * panel_rows()); x has cols/32 blocks per token.
void gemm_panels(GgmlType t, bool wide, const uint8_t* panel, size_t panel_bytes, int np, int64_t cols,
                 const BlockQ8* x, int n, float* y, int64_t y_stride, int rows_valid);

// Name of the ISA this binary was built for.
const char* isa_name();

}  // namespace malkin::kern
