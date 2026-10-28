#pragma once
#include <cstdint>
#include <random>
#include <unordered_map>
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

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);
  // logits are modified in place.
  int32_t sample(float* logits, int n_vocab);
  void accept(int32_t id) { ++counts_[id]; }

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
  std::unordered_map<int32_t, int> counts_;
  std::vector<std::pair<float, int32_t>> cand_;
};

}  // namespace malkin
