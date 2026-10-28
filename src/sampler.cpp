#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace malkin {

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

int32_t Sampler::sample(float* logits, int n_vocab) {
  for (auto [id, cnt] : counts_) {
    logits[id] -= p_.presence_penalty + p_.frequency_penalty * static_cast<float>(cnt);
  }
  if (p_.temperature <= 0.0f) {
    return static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits);
  }
  cand_.clear();
  cand_.reserve(n_vocab);
  for (int i = 0; i < n_vocab; ++i) cand_.emplace_back(logits[i], i);
  auto gt = [](const auto& a, const auto& b) { return a.first > b.first; };
  size_t k = p_.top_k > 0 ? std::min<size_t>(p_.top_k, cand_.size()) : cand_.size();
  std::partial_sort(cand_.begin(), cand_.begin() + static_cast<ptrdiff_t>(k), cand_.end(), gt);
  cand_.resize(k);

  const float mx = cand_[0].first;
  double sum = 0;
  for (auto& c : cand_) {
    c.first = std::exp((c.first - mx) / p_.temperature);
    sum += c.first;
  }
  // min_p relative to the top probability, then nucleus.
  const double floor = p_.min_p * cand_[0].first;
  double kept = 0;
  size_t n = 0;
  for (; n < cand_.size(); ++n) {
    if (n > 0 && cand_[n].first < floor) break;
    kept += cand_[n].first;
    if (kept >= p_.top_p * sum) {
      ++n;
      break;
    }
  }
  cand_.resize(std::max<size_t>(n, 1));
  double total = 0;
  for (auto& c : cand_) total += c.first;
  double r = std::uniform_real_distribution<double>(0.0, total)(rng_);
  for (auto& c : cand_) {
    r -= c.first;
    if (r <= 0) return c.second;
  }
  return cand_.back().second;
}

}  // namespace malkin
