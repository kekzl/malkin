#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "quant.h"

namespace malkin {

struct ModelConfig {
  std::string arch, name;
  int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
  int n_ctx_train = 0, n_vocab = 0;
  float rope_base = 10000.0f, rms_eps = 1e-6f;
};

ModelConfig read_config(const GgufFile& f);
// KV cache bytes per token position (BF16 K and V, all layers).
size_t kv_bytes_per_token(const ModelConfig& c);

// Qwen3 dense decoder over one sequence with a BF16 KV cache.
class Model {
 public:
  static constexpr int kMaxBatch = 256;

  Model(const GgufFile& f, int n_ctx);
  const ModelConfig& cfg() const { return cfg_; }
  int n_ctx() const { return n_ctx_; }

  // Evaluate n tokens at positions pos..pos+n-1 (n <= kMaxBatch); KV for those positions is overwritten.
  // logits: n_vocab floats for the last token, or n * n_vocab when all_logits.
  void forward(const int32_t* tokens, int n, int pos, float* logits, bool all_logits);

 private:
  struct Layer {
    const float* attn_norm;
    const float* ffn_norm;
    const float* q_norm;
    const float* k_norm;
    Weight wq, wk, wv, wo, w_gate, w_up, w_down;
  };

  void attention(int il, int n, int pos);

  ModelConfig cfg_;
  int n_ctx_;
  Weight tok_embd_, output_;
  const float* output_norm_ = nullptr;
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
  std::vector<uint16_t> kcache_, vcache_;  // [layer][pos][n_head_kv * head_dim], BF16

  // Scratch for up to kMaxBatch tokens.
  std::vector<float> x_, h_, q_, k_, v_, att_, gate_, up_, rope_cos_, rope_sin_;
  std::vector<BlockQ8> xq_;
  std::vector<PackedBuffer> packs_;  // panel copies of the matmul weights
};

}  // namespace malkin
