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

// One token of a multi-sequence batch: KV row seq * n_ctx + pos, attends to positions 0..pos of its sequence.
struct BatchToken {
  int32_t id;
  int seq, pos;
  bool logits;  // write n_vocab logits for this token
};

// Qwen3 dense decoder over n_seq sequences of n_ctx positions each, BF16 KV cache.
class Model {
 public:
  static constexpr int kMaxBatch = 256;

  Model(const GgufFile& f, int n_ctx, int n_seq = 1);
  const ModelConfig& cfg() const { return cfg_; }
  int n_ctx() const { return n_ctx_; }
  int n_seq() const { return n_seq_; }

  // Tokens of several sequences in one pass (weights read once). KV of each sequence must be valid below pos.
  // logits: n_vocab floats per token with .logits set, in batch order.
  void forward(const BatchToken* batch, int n, float* logits);

  // Evaluate n tokens of sequence seq at positions pos..pos+n-1 (n <= kMaxBatch); KV for those positions is
  // overwritten. logits: n_vocab floats for the last token, or n * n_vocab when all_logits. parent (optional, tree
  // verification): batch index of each token's parent, -1 = the token after the prefix; parents precede children. Token
  // t then sits at RoPE position pos + depth(t), KV slot pos + t, and attends to the prefix plus its ancestors.
  void forward(const int32_t* tokens, int n, int pos, float* logits, bool all_logits, const int* parent = nullptr,
               int seq = 0);
  // Copy the KV rows of position from to position to (all layers, sequence seq): compacts an accepted tree path.
  void move_kv(int from, int to, int seq = 0);

 private:
  // Evaluates tokens with the per-token layout below filled in.
  void run(const int32_t* tokens, int n, float* logits);
  // Per token: RoPE position, KV row written, first KV row attended, contiguous keys from there.
  std::vector<int> rpos_, kv_at_, kv_base_, n_keys_;
  std::vector<int> out_;  // tokens whose logits are written
  std::vector<int32_t> ids_;
  struct Layer {
    const float* attn_norm;
    const float* ffn_norm;
    const float* q_norm;
    const float* k_norm;
    Weight wq, wk, wv, wo, w_gate, w_up, w_down;
  };

  void attention(int il, int n);
  std::vector<int> depth_, anc_, anc_off_;  // tree mode: per-token depth, flattened ancestor KV rows

  ModelConfig cfg_;
  int n_ctx_, n_seq_;
  Weight tok_embd_, output_;
  const float* output_norm_ = nullptr;
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
  std::vector<uint16_t> kcache_, vcache_;  // [layer][seq * n_ctx + pos][n_head_kv * head_dim], BF16

  // Scratch for up to kMaxBatch tokens.
  std::vector<float> x_, h_, q_, k_, v_, att_, gate_, up_, rope_cos_, rope_sin_;
  std::vector<BlockQ8> xq_;
  std::vector<PackedBuffer> packs_;  // panel copies of the matmul weights
};

}  // namespace malkin
