// Speculative sampling is exact: d ~ q, then Sampler::verify must yield tokens distributed as p.
#include <cmath>
#include <cstdio>
#include <random>

#include "sampler.h"

using namespace malkin;

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  bool ok = true;
  for (int trial = 0; trial < 5; ++trial) {
    const int V = 8;
    Dist p, q;
    float sp = 0, sq = 0;
    for (int i = 0; i < V; ++i) {
      float a = u(rng), b = i == trial ? 0.0f : u(rng);  // q misses one token p has
      p.emplace_back(a, i);
      q.emplace_back(b, i);
      sp += a;
      sq += b;
    }
    for (auto& e : p) e.first /= sp;
    for (auto& e : q) e.first /= sq;
    SamplingParams sp1;
    sp1.seed = 11 + trial;
    Sampler target(sp1), draft(SamplingParams{.seed = 99u + trial});
    std::vector<double> hist(V, 0);
    const int N = 400000;
    int acc = 0;
    for (int n = 0; n < N; ++n) {
      const int32_t d = draft.draw(q);
      bool a = false;
      hist[target.verify(p, q, d, a)] += 1;
      acc += a;
    }
    double tv = 0;
    for (const auto& [pp, id] : p) tv += std::fabs(hist[id] / N - pp);
    tv /= 2;
    std::printf("trial %d: total variation %.4f, accepted %.1f %% %s\n", trial, tv, 100.0 * acc / N,
                tv < 0.005 ? "ok" : "FAIL");
    ok &= tv < 0.005;
  }
  return ok ? 0 : 1;
}
