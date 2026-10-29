#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace malkin {

struct SamplingParams {
  float temperature = 0.8f;  // <= 0: greedy
  float top_p = 0.95f;
  int top_k = 40;  // <= 0: whole vocabulary
  float min_p = 0.05f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0;  // 0: random
};

// Sparse probability distribution, highest first.
using Dist = std::vector<std::pair<float, int32_t>>;

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p, bool penalties = true);
  // Final sampling distribution after penalties, temperature, top-k, min-p, top-p (greedy: one entry).
  // logits are modified in place.
  void distribution(float* logits, int n_vocab, Dist& out);
  int32_t draw(const Dist& d);
  int32_t sample(float* logits, int n_vocab);
  void accept(int32_t id) { ++counts_[id]; }
  // Speculative sampling step (Leviathan et al. 2023): draft token d was drawn from q. Returns d with
  // probability min(1, p(d)/q(d)), otherwise a draw from max(0, p - q); the result is distributed exactly as p.
  int32_t verify(const Dist& p, const Dist& q, int32_t d, bool& accepted);

 private:
  SamplingParams p_;
  bool penalties_;
  std::mt19937_64 rng_;
  std::unordered_map<int32_t, int> counts_;
  Dist cand_;
};

}  // namespace malkin
