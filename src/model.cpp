#include "model.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
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
  if (t->type != GgmlType::F32 || t->nelements() != n) throw std::runtime_error("model: norm " + name + " not F32[" + std::to_string(n) + "]");
  return reinterpret_cast<const float*>(t->data);
}

void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
  const float s = 1.0f / std::sqrt(kern::dot_f32(n, x, x) / n + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * s * w[i];
}

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

Model::Model(const GgufFile& f, int n_ctx) : cfg_(read_config(f)), n_ctx_(n_ctx) {
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
  for (int i = 0; i < c.head_dim / 2; ++i) inv_freq_.push_back(std::pow(c.rope_base, -2.0f * i / c.head_dim));

  const size_t kv_row = static_cast<size_t>(c.n_head_kv) * c.head_dim;
  kcache_.assign(static_cast<size_t>(c.n_layer) * n_ctx * kv_row, 0);
  vcache_.assign(kcache_.size(), 0);
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

void Model::attention(int il, int n, int pos) {
  const auto& c = cfg_;
  const int D = c.head_dim, H = c.n_head, G = c.n_head / c.n_head_kv;
  const size_t kv_row = static_cast<size_t>(c.n_head_kv) * D;
  const uint16_t* kc = kcache_.data() + static_cast<size_t>(il) * n_ctx_ * kv_row;
  const uint16_t* vc = vcache_.data() + static_cast<size_t>(il) * n_ctx_ * kv_row;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
#pragma omp parallel
  {
    thread_local std::vector<float> scores;
    thread_local std::vector<float> acc;
    if (scores.size() < static_cast<size_t>(n_ctx_)) scores.resize(n_ctx_);
    if (acc.size() < static_cast<size_t>(D)) acc.resize(D);
#pragma omp for collapse(2) schedule(dynamic, 1)
    for (int t = 0; t < n; ++t) {
      for (int h = 0; h < H; ++h) {
        const int p_end = pos + t + 1;
        const float* q = q_.data() + (static_cast<size_t>(t) * H + h) * D;
        const size_t koff = static_cast<size_t>(h / G) * D;
        float mx = -INFINITY;
        for (int j = 0; j < p_end; ++j) {
          float sc = kern::dot_f32_bf16(q, kc + j * kv_row + koff, D) * scale;
          scores[j] = sc;
          mx = std::max(mx, sc);
        }
        float sum = 0.0f;
        for (int j = 0; j < p_end; ++j) {
          scores[j] = std::exp(scores[j] - mx);
          sum += scores[j];
        }
        std::fill(acc.begin(), acc.begin() + D, 0.0f);
        for (int j = 0; j < p_end; ++j) kern::axpy_bf16(scores[j], vc + j * kv_row + koff, acc.data(), D);
        float* out = att_.data() + (static_cast<size_t>(t) * H + h) * D;
        const float inv = 1.0f / sum;
        for (int d = 0; d < D; ++d) out[d] = acc[d] * inv;
      }
    }
  }
}

void Model::forward(const int32_t* tokens, int n, int pos, float* logits, bool all_logits) {
  const auto& c = cfg_;
  if (n <= 0 || n > kMaxBatch) throw std::runtime_error("forward: batch size out of range");
  if (pos + n > n_ctx_) throw std::runtime_error("forward: context full");
  const int E = c.n_embd, D = c.head_dim, H = c.n_head, KV = c.n_head_kv, F = c.n_ff;
  const size_t kv_row = static_cast<size_t>(KV) * D;

  for (int t = 0; t < n; ++t) {
    if (tokens[t] < 0 || tokens[t] >= tok_embd_.rows) throw std::runtime_error("forward: token id out of range");
    dequantize_row(tok_embd_.type, tok_embd_.row(tokens[t]), x_.data() + static_cast<size_t>(t) * E, E);
  }
  for (int t = 0; t < n; ++t) {
    for (int i = 0; i < D / 2; ++i) {
      float a = static_cast<float>(pos + t) * inv_freq_[i];
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

  for (int il = 0; il < c.n_layer; ++il) {
    const Layer& L = layers_[il];
#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t) rmsnorm(h_.data() + t * E, x_.data() + t * E, L.attn_norm, E, c.rms_eps);
    quantize_acts(h_.data(), n, E, xq_);
    matmul(L.wq, h_.data(), xq_.data(), n, q_.data());
    matmul(L.wk, h_.data(), xq_.data(), n, k_.data());
    matmul(L.wv, h_.data(), xq_.data(), n, v_.data());

    uint16_t* kc = kcache_.data() + (static_cast<size_t>(il) * n_ctx_ + pos) * kv_row;
    uint16_t* vc = vcache_.data() + (static_cast<size_t>(il) * n_ctx_ + pos) * kv_row;
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
        kc[t * kv_row + i] = fp32_to_bf16(k_[t * kv_row + i]);
        vc[t * kv_row + i] = fp32_to_bf16(v_[t * kv_row + i]);
      }
    }

    attention(il, n, pos);
    quantize_acts(att_.data(), n, static_cast<int64_t>(H) * D, xq_);
    matmul(L.wo, att_.data(), xq_.data(), n, h_.data());
    for (size_t i = 0; i < static_cast<size_t>(n) * E; ++i) x_[i] += h_[i];

#pragma omp parallel for schedule(static) if (n > 1)
    for (int t = 0; t < n; ++t) rmsnorm(h_.data() + t * E, x_.data() + t * E, L.ffn_norm, E, c.rms_eps);
    quantize_acts(h_.data(), n, E, xq_);
    matmul(L.w_gate, h_.data(), xq_.data(), n, gate_.data());
    matmul(L.w_up, h_.data(), xq_.data(), n, up_.data());
    const size_t nf = static_cast<size_t>(n) * F;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nf; ++i) {
      float g = gate_[i];
      gate_[i] = g / (1.0f + std::exp(-g)) * up_[i];
    }
    quantize_acts(gate_.data(), n, F, xq_);
    matmul(L.w_down, gate_.data(), xq_.data(), n, h_.data());
    for (size_t i = 0; i < static_cast<size_t>(n) * E; ++i) x_[i] += h_[i];
  }

  const int first = all_logits ? 0 : n - 1;
  const int m = n - first;
  for (int t = first; t < n; ++t) rmsnorm(h_.data() + (t - first) * E, x_.data() + t * E, output_norm_, E, c.rms_eps);
  quantize_acts(h_.data(), m, E, xq_);
  matmul(output_, h_.data(), xq_.data(), m, logits);
}

}  // namespace malkin
