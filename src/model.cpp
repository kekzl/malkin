#include "model.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels.h"

namespace malkin {

namespace {

Weight weight(const GgufFile& f, const std::string& name, bool required = true) {
  const GgufTensor* t = f.tensor(name);
  if (!t) {
    if (required) throw std::runtime_error("model: tensor " + name + " missing");
    return {};
  }
  if (!weight_type_supported(t->type)) {
    throw std::runtime_error("model: tensor " + name + " has type " + ggml_type_name(t->type) +
                             ", supported: F32 F16 BF16 Q8_0 Q4_K Q5_K Q6_K");
  }
  Weight w;
  w.type = t->type;
  w.data = t->data;
  w.cols = t->ne[0];
  w.rows = t->ne.size() > 1 ? t->nelements() / t->ne[0] : 1;
  w.row_bytes = ggml_row_size(t->type, t->ne[0]);
  return w;
}

const float* norm(const GgufFile& f, const std::string& name, int64_t n) {
  const GgufTensor* t = f.tensor(name);
  if (!t) throw std::runtime_error("model: tensor " + name + " missing");
  if (t->type != GgmlType::F32 || t->nelements() != n)
    throw std::runtime_error("model: norm " + name + " not F32[" + std::to_string(n) + "]");
  return reinterpret_cast<const float*>(t->data);
}

void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
  const float s = 1.0f / std::sqrt(kern::dot_f32(n, x, x) / n + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * s * w[i];
}

// MALKIN_PROFILE=1: wall time per forward stage, printed at exit.
struct Prof {
  bool on = std::getenv("MALKIN_PROFILE") != nullptr;
  double ms[8] = {};
  ~Prof() {
    if (!on) return;
    static const char* names[8] = {"norm+qkv", "qk-norm/rope/kv", "attention", "wo+res+norm",
                                   "gate+up",  "silu+quant",      "down",      "output"};
    double total = 0;
    for (double v : ms) total += v;
    for (int i = 0; i < 8; ++i)
      std::fprintf(stderr, "profile %-16s %9.1f ms %5.1f %%\n", names[i], ms[i], 100 * ms[i] / total);
  }
} g_prof;

struct Lap {
  std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
  void operator()(int i) {
    if (!g_prof.on) return;
    auto now = std::chrono::steady_clock::now();
    g_prof.ms[i] += std::chrono::duration<double, std::milli>(now - t).count();
    t = now;
  }
};

}  // namespace

ModelConfig read_config(const GgufFile& f) {
  ModelConfig c;
  c.arch = f.get_str("general.architecture").value_or("");
  c.name = f.get_str("general.name").value_or("model");
  if (c.arch != "qwen3") throw std::runtime_error("architecture '" + c.arch + "' unsupported (supported: qwen3)");
  auto key = [&](const std::string& k) { return c.arch + "." + k; };
  auto need = [&](const std::string& k) {
    auto v = f.get_int(key(k));
    if (!v) throw std::runtime_error("model: metadata " + key(k) + " missing");
    return static_cast<int>(*v);
  };
  c.n_layer = need("block_count");
  c.n_embd = need("embedding_length");
  c.n_ff = need("feed_forward_length");
  c.n_head = need("attention.head_count");
  c.n_head_kv = static_cast<int>(f.get_int(key("attention.head_count_kv")).value_or(c.n_head));
  c.head_dim = static_cast<int>(f.get_int(key("attention.key_length")).value_or(c.n_embd / c.n_head));
  c.n_ctx_train = static_cast<int>(f.get_int(key("context_length")).value_or(4096));
  c.rope_base = static_cast<float>(f.get_float(key("rope.freq_base")).value_or(10000.0));
  c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")).value_or(1e-6));
  auto* toks = f.get_str_array("tokenizer.ggml.tokens");
  c.n_vocab = toks ? static_cast<int>(toks->size()) : 0;
  return c;
}

size_t kv_bytes_per_token(const ModelConfig& c) {
  return static_cast<size_t>(c.n_layer) * c.n_head_kv * c.head_dim * 2 * sizeof(uint16_t);
}

Model::Model(const GgufFile& f, int n_ctx, int n_seq) : cfg_(read_config(f)), n_ctx_(n_ctx), n_seq_(n_seq) {
  const auto& c = cfg_;
  if (c.head_dim % 32 || c.n_head % c.n_head_kv) throw std::runtime_error("model: unsupported head layout");
  tok_embd_ = weight(f, "token_embd.weight");
  output_ = weight(f, "output.weight", false);
  if (!output_) output_ = tok_embd_;  // tied embeddings
  cfg_.n_vocab = static_cast<int>(output_.rows);
  output_norm_ = norm(f, "output_norm.weight", c.n_embd);
  for (int i = 0; i < c.n_layer; ++i) {
    auto p = "blk." + std::to_string(i) + ".";
    Layer l;
    l.attn_norm = norm(f, p + "attn_norm.weight", c.n_embd);
    l.ffn_norm = norm(f, p + "ffn_norm.weight", c.n_embd);
    l.q_norm = norm(f, p + "attn_q_norm.weight", c.head_dim);
    l.k_norm = norm(f, p + "attn_k_norm.weight", c.head_dim);
    l.wq = weight(f, p + "attn_q.weight");
    l.wk = weight(f, p + "attn_k.weight");
    l.wv = weight(f, p + "attn_v.weight");
    l.wo = weight(f, p + "attn_output.weight");
    l.w_gate = weight(f, p + "ffn_gate.weight");
    l.w_up = weight(f, p + "ffn_up.weight");
    l.w_down = weight(f, p + "ffn_down.weight");
    layers_.push_back(l);
  }
  // MALKIN_NO_PACK: row path only. MALKIN_NO_WIDE: no wide prefill panels (saves ~1 byte per K-quant weight).
  if (!std::getenv("MALKIN_NO_PACK")) {
    const bool wide = !std::getenv("MALKIN_NO_WIDE");
    for (auto& l : layers_) {
      for (Weight* w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down}) {
        packs_.push_back(pack_weight(*w));
        if (wide) packs_.push_back(pack_weight(*w, true));
      }
    }
    packs_.push_back(pack_weight(output_));
    if (wide) packs_.push_back(pack_weight(output_, true));
  }
  for (int i = 0; i < c.head_dim / 2; ++i) inv_freq_.push_back(std::pow(c.rope_base, -2.0f * i / c.head_dim));

  if (n_ctx > 0) alloc_kv(n_ctx);
  const size_t kv_row = static_cast<size_t>(c.n_head_kv) * c.head_dim;
  const size_t B = kMaxBatch;
  x_.resize(B * c.n_embd);
  h_.resize(B * c.n_embd);
  q_.resize(B * c.n_head * c.head_dim);
  att_.resize(q_.size());
  k_.resize(B * kv_row);
  v_.resize(B * kv_row);
  gate_.resize(B * c.n_ff);
  up_.resize(B * c.n_ff);
  rope_cos_.resize(B * c.head_dim / 2);
  rope_sin_.resize(rope_cos_.size());
}

void Model::alloc_kv(int n_ctx) {
  const auto& c = cfg_;
  n_ctx_ = n_ctx;
  const size_t kv_row = static_cast<size_t>(c.n_head_kv) * c.head_dim;
  kcache_.assign(static_cast<size_t>(c.n_layer) * n_seq_ * n_ctx * kv_row, 0);
  vcache_.assign(kcache_.size(), 0);
}

void Model::attention(int il, int n) {
  const auto& c = cfg_;
  const int D = c.head_dim, H = c.n_head, KV = c.n_head_kv, G = H / KV;
  const size_t kv_row = static_cast<size_t>(KV) * D;
  const size_t layer = static_cast<size_t>(il) * n_seq_ * n_ctx_ * kv_row;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  // Work items: (token, KV head); the G query heads of a KV head share every K/V row load.
#pragma omp parallel
  {
    thread_local std::vector<float> scratch;
    if (scratch.size() < static_cast<size_t>(G) * n_ctx_) scratch.resize(static_cast<size_t>(G) * n_ctx_);
#pragma omp for collapse(2) schedule(dynamic, 1)
    for (int t = 0; t < n; ++t) {
      for (int g = 0; g < KV; ++g) {
        const size_t qoff = (static_cast<size_t>(t) * H + static_cast<size_t>(g) * G) * D;
        const size_t kv = layer + static_cast<size_t>(kv_base_[t]) * kv_row + static_cast<size_t>(g) * D;
        const bool tree = !anc_off_.empty();
        kern::attend_gqa(q_.data() + qoff, G, D, kcache_.data() + kv, vcache_.data() + kv, kv_row, n_keys_[t], scale,
                         att_.data() + qoff, scratch.data(), tree ? anc_.data() + anc_off_[t] : nullptr,
                         tree ? anc_off_[t + 1] - anc_off_[t] : 0);
      }
    }
  }
}

void Model::forward(const int32_t* tokens, int n, int pos, float* logits, bool all_logits, const int* parent, int seq) {
  if (n <= 0 || n > kMaxBatch) throw std::runtime_error("forward: batch size out of range");
  if (pos + n > n_ctx_) throw std::runtime_error("forward: context full");
  if (seq < 0 || seq >= n_seq_) throw std::runtime_error("forward: bad sequence");
  depth_.clear();
  anc_.clear();
  anc_off_.clear();
  if (parent) {  // tree: depth and ancestor KV positions (self included) per token
    depth_.resize(n);
    for (int t = 0; t < n; ++t) {
      depth_[t] = parent[t] < 0 ? 0 : depth_[parent[t]] + 1;
      anc_off_.push_back(static_cast<int>(anc_.size()));
      for (int a = t; a >= 0; a = parent[a]) anc_.push_back(pos + a);
      std::reverse(anc_.begin() + anc_off_.back(), anc_.end());  // root first: same key order as plain decode
    }
    anc_off_.push_back(static_cast<int>(anc_.size()));
  }
  rpos_.resize(n);
  kv_at_.resize(n);
  kv_base_.assign(n, seq * n_ctx_);
  n_keys_.resize(n);
  out_.clear();
  for (int t = 0; t < n; ++t) {
    rpos_[t] = pos + (parent ? depth_[t] : t);
    kv_at_[t] = seq * n_ctx_ + pos + t;
    n_keys_[t] = parent ? pos : pos + t + 1;  // tree: prefix only, ancestors come as extra rows
    if (all_logits || t == n - 1) out_.push_back(t);
  }
  run(tokens, n, logits);
}

void Model::forward(const BatchToken* batch, int n, float* logits) {
  if (n <= 0 || n > kMaxBatch) throw std::runtime_error("forward: batch size out of range");
  depth_.clear();
  anc_.clear();
  anc_off_.clear();
  ids_.resize(n);
  rpos_.resize(n);
  kv_at_.resize(n);
  kv_base_.resize(n);
  n_keys_.resize(n);
  out_.clear();
  bool tree = false;
  for (int t = 0; t < n; ++t) {
    const BatchToken& b = batch[t];
    const int kv = b.parent < 0 ? b.pos : b.kv;
    if (b.seq < 0 || b.seq >= n_seq_ || b.pos < 0 || b.pos >= n_ctx_ || kv < 0 || kv >= n_ctx_)
      throw std::runtime_error("forward: bad slot");
    if (b.parent >= t || (b.parent >= 0 && batch[b.parent].seq != b.seq))
      throw std::runtime_error("forward: bad tree parent");
    ids_[t] = b.id;
    rpos_[t] = b.pos;
    kv_base_[t] = b.seq * n_ctx_;
    kv_at_[t] = kv_base_[t] + kv;
    tree = tree || b.parent >= 0;
    if (b.logits) out_.push_back(t);
  }
  // Keys: plain tokens 0..pos; tree tokens the parent's keys (contiguous part, then extra rows) plus their own row.
  if (tree) anc_off_.push_back(0);
  for (int t = 0; t < n; ++t) {
    const BatchToken& b = batch[t];
    if (b.parent < 0) {
      n_keys_[t] = b.pos + 1;
    } else {
      n_keys_[t] = n_keys_[b.parent];
      for (int i = anc_off_[b.parent]; i < anc_off_[b.parent + 1]; ++i) {
        const int row = anc_[i];  // copy: push_back may reallocate
        anc_.push_back(row);
      }
      anc_.push_back(kv_at_[t] - kv_base_[t]);
    }
    if (tree) anc_off_.push_back(static_cast<int>(anc_.size()));
  }
  run(ids_.data(), n, logits);
}

void Model::run(const int32_t* tokens, int n, float* logits) {
  const auto& c = cfg_;
  const int E = c.n_embd, D = c.head_dim, H = c.n_head, KV = c.n_head_kv, F = c.n_ff;
  const size_t kv_row = static_cast<size_t>(KV) * D;

  // Token embeddings, then RoPE angles per token.
  for (int t = 0; t < n; ++t) {
    if (tokens[t] < 0 || tokens[t] >= tok_embd_.rows) throw std::runtime_error("forward: token id out of range");
    dequantize_row(tok_embd_.type, tok_embd_.row(tokens[t]), x_.data() + static_cast<size_t>(t) * E, E);
  }
  for (int t = 0; t < n; ++t) {
    for (int i = 0; i < D / 2; ++i) {
      float a = static_cast<float>(rpos_[t]) * inv_freq_[i];
      rope_cos_[t * D / 2 + i] = std::cos(a);
      rope_sin_[t * D / 2 + i] = std::sin(a);
    }
  }
  auto rope = [&](float* v, int t) {
    const float* cs = rope_cos_.data() + t * D / 2;
    const float* sn = rope_sin_.data() + t * D / 2;
    for (int i = 0; i < D / 2; ++i) {
      float x0 = v[i], x1 = v[i + D / 2];
      v[i] = x0 * cs[i] - x1 * sn[i];
      v[i + D / 2] = x0 * sn[i] + x1 * cs[i];
    }
  };
  // x += h over n tokens, threaded.
  auto add_residual = [&](int n_tok) {
    const size_t ne = static_cast<size_t>(n_tok) * E;
#pragma omp parallel for schedule(static) if (n_tok > 1)
    for (size_t i = 0; i < ne; ++i) x_[i] += h_[i];
  };

  Lap lap;
  // Decoder layers.
  for (int il = 0; il < c.n_layer; ++il) {
    const Layer& L = layers_[il];
#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t) rmsnorm(h_.data() + t * E, x_.data() + t * E, L.attn_norm, E, c.rms_eps);
    quantize_acts(h_.data(), n, E, xq_);
    // Pre-attention RMSNorm, Q8 activations shared by the Q, K, V projections.
    matmul(L.wq, h_.data(), xq_.data(), n, q_.data());
    matmul(L.wk, h_.data(), xq_.data(), n, k_.data());
    matmul(L.wv, h_.data(), xq_.data(), n, v_.data());
    lap(0);

    // Q/K head norms, RoPE, then K and V into the cache rows kv_at_.
    const size_t layer = static_cast<size_t>(il) * n_seq_ * n_ctx_;
#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t) {
      for (int h = 0; h < H; ++h) {
        float* q = q_.data() + (static_cast<size_t>(t) * H + h) * D;
        rmsnorm(q, q, L.q_norm, D, c.rms_eps);
        rope(q, t);
      }
      for (int h = 0; h < KV; ++h) {
        float* k = k_.data() + (static_cast<size_t>(t) * KV + h) * D;
        rmsnorm(k, k, L.k_norm, D, c.rms_eps);
        rope(k, t);
      }
      for (size_t i = 0; i < kv_row; ++i) {
        kcache_[(layer + kv_at_[t]) * kv_row + i] = fp32_to_bf16(k_[t * kv_row + i]);
        vcache_[(layer + kv_at_[t]) * kv_row + i] = fp32_to_bf16(v_[t * kv_row + i]);
      }
    }

    lap(1);
    // Attention over the BF16 KV cache, output projection, residual add.
    attention(il, n);
    lap(2);
    quantize_acts(att_.data(), n, static_cast<int64_t>(H) * D, xq_);
    matmul(L.wo, att_.data(), xq_.data(), n, h_.data());
    lap(3);
    add_residual(n);

#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t) rmsnorm(h_.data() + t * E, x_.data() + t * E, L.ffn_norm, E, c.rms_eps);
    quantize_acts(h_.data(), n, E, xq_);
    // SwiGLU FFN: down(silu(gate(h)) * up(h)), residual add.
    matmul(L.w_gate, h_.data(), xq_.data(), n, gate_.data());
    matmul(L.w_up, h_.data(), xq_.data(), n, up_.data());
    lap(4);
#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t)
      kern::silu_mul(F, gate_.data() + static_cast<size_t>(t) * F, up_.data() + static_cast<size_t>(t) * F);
    quantize_acts(gate_.data(), n, F, xq_);
    lap(5);
    matmul(L.w_down, gate_.data(), xq_.data(), n, h_.data());
    lap(6);
    add_residual(n);
  }

  // Final norm and LM head for the tokens in out_.
  const int m = static_cast<int>(out_.size());
  if (m == 0) return;
  for (int i = 0; i < m; ++i) rmsnorm(h_.data() + i * E, x_.data() + out_[i] * E, output_norm_, E, c.rms_eps);
  quantize_acts(h_.data(), m, E, xq_);
  matmul(output_, h_.data(), xq_.data(), m, logits);
  lap(7);
}

}  // namespace malkin

namespace malkin {

void Model::move_kv(int from, int to, int seq) {
  if (from == to) return;
  const size_t row = static_cast<size_t>(cfg_.n_head_kv) * cfg_.head_dim;
  for (int il = 0; il < cfg_.n_layer; ++il) {
    const size_t base = static_cast<size_t>(il) * n_seq_ * n_ctx_ + static_cast<size_t>(seq) * n_ctx_;
    std::memcpy(kcache_.data() + (base + to) * row, kcache_.data() + (base + from) * row, row * sizeof(uint16_t));
    std::memcpy(vcache_.data() + (base + to) * row, vcache_.data() + (base + from) * row, row * sizeof(uint16_t));
  }
}

}  // namespace malkin
