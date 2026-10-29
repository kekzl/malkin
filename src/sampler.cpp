#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace malkin {

namespace {

float prob_of(const Dist& d, int32_t id) {
  for (const auto& [p, i] : d) {
    if (i == id) return p;
  }
  return 0.0f;
}

}  // namespace

Sampler::Sampler(const SamplingParams& p, bool penalties)
    : p_(p), penalties_(penalties), rng_(p.seed ? p.seed : std::random_device{}()) {}

void Sampler::distribution(float* logits, int n_vocab, Dist& out) {
  out.clear();
  if (penalties_) {
    for (auto [id, cnt] : counts_) logits[id] -= p_.presence_penalty + p_.frequency_penalty * static_cast<float>(cnt);
  }
  if (p_.temperature <= 0.0f) {
    out.emplace_back(1.0f, static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits));
    return;
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
  n = std::max<size_t>(n, 1);
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += cand_[i].first;
  out.assign(cand_.begin(), cand_.begin() + static_cast<ptrdiff_t>(n));
  for (auto& c : out) c.first = static_cast<float>(c.first / total);
}

int32_t Sampler::draw(const Dist& d) {
  if (d.size() == 1) return d[0].second;
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
  for (const auto& [p, id] : d) {
    r -= p;
    if (r <= 0) return id;
  }
  return d.back().second;
}

int32_t Sampler::sample(float* logits, int n_vocab) {
  Dist d;
  distribution(logits, n_vocab, d);
  return draw(d);
}

int32_t Sampler::verify(const Dist& p, const Dist& q, int32_t d, bool& accepted) {
  const float pd = prob_of(p, d), qd = prob_of(q, d);
  const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
  accepted = qd > 0.0f && u * qd < pd;  // u < pd / qd
  if (accepted) return d;
  Dist resid;
  double total = 0;
  for (const auto& [pp, id] : p) {
    const float r = pp - prob_of(q, id);
    if (r > 0.0f) {
      resid.emplace_back(r, id);
      total += r;
    }
  }
  if (resid.empty()) return draw(p);  // p == q up to rounding
  for (auto& c : resid) c.first = static_cast<float>(c.first / total);
  return draw(resid);
}

}  // namespace malkin
