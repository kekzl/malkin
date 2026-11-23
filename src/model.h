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
// KV cache bytes per token position (int8 K and V with a scale per head, all layers).
size_t kv_bytes_per_token(const ModelConfig& c);

// One token of a multi-sequence batch. Plain (parent < 0): RoPE position and KV row pos, attends to positions
// 0..pos of its sequence. Tree (parent = batch index of a token of the same sequence): KV row kv, attends to the
// keys of its parent plus itself, in that order (same key order as plain decode of the accepted path).
// base: first KV row of the token's span inside its sequence; the span holds rows base..base+pos and is its own
// context (probes: one span per document, each starting with a copy of the shared prefix).
struct BatchToken {
  int32_t id;
  int seq, pos;
  bool logits;  // write n_vocab logits for this token
  int kv = -1, parent = -1;
  int base = 0;
};

// Qwen3 dense decoder over n_seq sequences of n_ctx positions each, int8 KV cache.
class Model {
 public:
  static constexpr int kMaxBatch = 256;

  // n_ctx 0: weights only; alloc_kv before the first forward.
  Model(const GgufFile& f, int n_ctx, int n_seq = 1);
  void alloc_kv(int n_ctx);
  // Wide prefill panels: anonymous bytes they take (0 = none for this ISA / these types, or MALKIN_NO_WIDE), and
  // the pack itself. The caller checks the RAM first.
  size_t wide_bytes();
  void pack_wide();
  // First-touch the weights forward reads from the mapped file (those without panels); token embeddings fault lazily.
  void prefetch();
  // While dst is set, forward also copies the final-norm state of each logits token there (n_embd floats each).
  void capture_hidden(float* dst) { hidden_dst_ = dst; }
  const ModelConfig& cfg() const { return cfg_; }
  int n_ctx() const { return n_ctx_; }
  int n_seq() const { return n_seq_; }

  // Tokens of several sequences in one pass (weights read once). KV of each sequence must be valid below pos.
  // logits: n_vocab floats per token with .logits set, in batch order; null skips the LM head (capture_hidden).
  void forward(const BatchToken* batch, int n, float* logits);
  // Logits of n vocabulary ids for one captured hidden state: n rows of the LM head instead of n_vocab.
  void logits_of(const float* hidden, const int32_t* ids, int n, float* out);
  // Copy n KV rows from row `from` to row `to` of sequence seq (all layers); the ranges must not overlap.
  void copy_kv(int from, int to, int n, int seq = 0);

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
  float* hidden_dst_ = nullptr;
  std::vector<int> blocks_;  // attention token blocks: start indices, then n
  std::vector<int32_t> ids_;
  struct Layer {
    const float* attn_norm;
    const float* ffn_norm;
    const float* q_norm;
    const float* k_norm;
    Weight wq, wk, wv, wo, w_gate, w_up, w_down;
  };

  void attention(int il, int n);
  std::vector<Weight*> matmul_weights();
  // Packs all matmul weights into one buffer, then unmaps their file pages.
  void pack(bool wide);
  std::vector<int> depth_, anc_, anc_off_;  // tree mode: per-token depth, flattened ancestor KV rows

  ModelConfig cfg_;
  int n_ctx_, n_seq_;
  Weight tok_embd_, output_;
  const float* output_norm_ = nullptr;
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
  LazyBytes kcache_, vcache_;  // [layer][seq * n_ctx + pos]: n_head_kv x head_dim int8, n_head_kv float scales

  // Scratch for up to kMaxBatch tokens.
  std::vector<float> x_, h_, q_, k_, v_, att_, gate_, up_, rope_cos_, rope_sin_;
  std::vector<BlockQ8> xq_;
  std::vector<PackedBuffer> packs_;  // panel copies of the matmul weights: compact, then wide
};

}  // namespace malkin
