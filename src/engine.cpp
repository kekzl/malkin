#include "engine.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <stdexcept>

#include "cpu.h"
#include "kernels.h"

namespace malkin {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Bytes at the end of s that form an incomplete UTF-8 sequence.
size_t utf8_incomplete_tail(const std::string& s) {
  for (size_t back = 1; back <= std::min<size_t>(3, s.size()); ++back) {
    unsigned char c = s[s.size() - back];
    if ((c & 0xC0) == 0x80) continue;  // continuation byte
    size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return need > back ? back : 0;
  }
  return 0;
}

}  // namespace

Engine::Engine(const EngineOptions& o) {
  auto t0 = Clock::now();
  file_ = std::make_unique<GgufFile>(o.model_path);
  auto cfg = read_config(*file_);
  // Physical cores for both phases (9800X3D, 4B Q8_0): decode 13.2 tok/s on 8 threads vs 7.3 on 16; panel-GEMM
  // prefill 348.1 vs 188.7 (VNNI saturates a core; SMT siblings add barrier stragglers). ICVs are per thread.
  const int procs = omp_get_num_procs();
  threads_ = o.threads > 0 ? o.threads : std::min(cpu_physical_cores(), procs);
  threads_batch_ = o.threads_batch > 0 ? o.threads_batch : threads_;
  omp_set_num_threads(threads_);

  int n_ctx = o.n_ctx > 0 ? o.n_ctx : std::min(cfg.n_ctx_train, 32768);
  if (o.n_ctx <= 0) {
    // Auto: KV may take half of the RAM left after the weights.
    size_t avail = mem_available_bytes();
    size_t budget = avail > file_->file_size() ? (avail - file_->file_size()) / 2 : 0;
    n_ctx = std::max(512, std::min<int>(n_ctx, static_cast<int>(budget / kv_bytes_per_token(cfg))));
  }
  n_ctx = std::min(n_ctx, cfg.n_ctx_train);

  tok_ = std::make_unique<Tokenizer>(*file_);
  chat_ = std::make_unique<ChatFormat>(*tok_, file_->get_str("tokenizer.chat_template").value_or(""));
  model_ = std::make_unique<Model>(*file_, n_ctx);
  if (!o.draft_path.empty()) {
    dfile_ = std::make_unique<GgufFile>(o.draft_path);
    Tokenizer dtok(*dfile_);
    bool same = dtok.size() == tok_->size();
    for (int32_t id : {0, 1000, 100000, static_cast<int32_t>(tok_->size()) - 1}) same = same && dtok.piece(id) == tok_->piece(id);
    if (!same) throw std::runtime_error("draft model tokenizer differs from the target");
    draft_ = std::make_unique<Model>(*dfile_, n_ctx);
    dlogits_.resize(static_cast<size_t>(draft_->cfg().n_vocab));
    if (o.n_draft > 0) n_draft_ = std::min(o.n_draft, Model::kMaxBatch - 1);
    vlogits_.resize(static_cast<size_t>(2 * n_draft_ + 1) * model_->cfg().n_vocab);  // tree: chain + alternatives
    std::fprintf(stderr, "malkin: draft %s, %d layers, up to %d tokens per step\n",
                 std::filesystem::path(o.draft_path).stem().c_str(), draft_->cfg().n_layer, n_draft_);
  }
  logits_.resize(static_cast<size_t>(cfg.n_vocab));
  worker_ = std::thread([this] { worker_loop(); });
  on_worker([this] {
    omp_set_num_threads(threads_batch_);
    file_->prefetch();
  });
  model_id_ = std::filesystem::path(o.model_path).stem().string();
  std::fprintf(stderr,
               "malkin: %s | %s | %d layers, %d embd, %d heads (%d kv), vocab %d | ctx %d (KV %.2f GiB BF16) | "
               "threads %d decode / %d prefill | loaded in %.1f s\n",
               model_id_.c_str(), cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv,
               cfg.n_vocab, n_ctx, static_cast<double>(kv_bytes_per_token(cfg)) * n_ctx / (1 << 30), threads_, threads_batch_,
               ms_since(t0) / 1000.0);
  std::map<std::string, int> types;
  for (const auto& [name, t] : file_->tensors()) ++types[ggml_type_name(t.type)];
  std::string hist;
  for (const auto& [ty, n] : types) hist += " " + ty + " x" + std::to_string(n);
  std::fprintf(stderr, "malkin: isa %s | %.2f GiB weights:%s\n", kern::isa_name(),
               static_cast<double>(file_->file_size()) / (1 << 30), hist.c_str());
}

Engine::~Engine() {
  {
    std::lock_guard l(qmu_);
    stop_ = true;
  }
  qcv_.notify_one();
  if (worker_.joinable()) worker_.join();
}

void Engine::worker_loop() {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock l(qmu_);
      qcv_.wait(l, [this] { return stop_ || job_; });
      if (stop_) return;
      job = std::move(job_);
      job_ = nullptr;
    }
    job();
  }
}

void Engine::prefill(const std::vector<int32_t>& toks, size_t from, float* logits) {
  omp_set_num_threads(toks.size() - from > 16 ? threads_batch_ : threads_);
  for (size_t i = from; i < toks.size(); i += Model::kMaxBatch) {
    int n = static_cast<int>(std::min<size_t>(Model::kMaxBatch, toks.size() - i));
    model_->forward(toks.data() + i, n, static_cast<int>(i), logits, false);
  }
}

GenResult Engine::generate(const GenRequest& req, const OnText& on_text) {
  std::lock_guard lock(mu_);
  return on_worker([&] { return generate_locked(req, on_text); });
}

void Engine::draft_sync(const std::vector<int32_t>& ctx) {
  size_t common = 0;
  while (common < dcache_.size() && common < ctx.size() && dcache_[common] == ctx[common]) ++common;
  common = std::min(common, ctx.size() - 1);
  dcache_.resize(common);
  for (size_t i = common; i < ctx.size(); i += Model::kMaxBatch) {
    const int n = static_cast<int>(std::min<size_t>(Model::kMaxBatch, ctx.size() - i));
    draft_->forward(ctx.data() + i, n, static_cast<int>(i), dlogits_.data(), false);
  }
  dcache_ = ctx;
}

GenResult Engine::generate_locked(const GenRequest& req, const OnText& on_text) {
  GenResult r;
  const auto& prompt = req.prompt;
  const int n_ctx = model_->n_ctx();
  const int n_vocab = model_->cfg().n_vocab;
  if (prompt.empty()) throw std::invalid_argument("empty prompt");
  if (static_cast<int>(prompt.size()) >= n_ctx) {
    throw std::invalid_argument("prompt has " + std::to_string(prompt.size()) + " tokens, context is " +
                                std::to_string(n_ctx));
  }
  r.prompt_tokens = static_cast<int>(prompt.size());

  size_t common = 0;
  while (common < cache_.size() && common < prompt.size() && cache_[common] == prompt[common]) ++common;
  common = std::min(common, prompt.size() - 1);  // the last prompt token is always evaluated for its logits
  r.cached_tokens = static_cast<int>(common);
  cache_.resize(common);

  auto t0 = Clock::now();
  prefill(prompt, common, logits_.data());
  cache_ = prompt;
  r.prefill_ms = ms_since(t0);

  Sampler sampler(req.sampling);
  std::string text;
  size_t emitted = 0;
  const int budget = req.max_tokens > 0 ? req.max_tokens : n_ctx;
  omp_set_num_threads(threads_);
  auto t1 = Clock::now();
  r.finish_reason = "length";
  bool cancelled = false;

  // Appends one generated token; false = generation ends (EOG, stop string, budget, cancel).
  auto emit = [&](int32_t id) {
    if (tok_->is_eog(id)) {
      r.finish_reason = "stop";
      return false;
    }
    sampler.accept(id);
    ++r.completion_tokens;
    text += tok_->piece(id);
    size_t stop_at = std::string::npos;
    for (const auto& s : req.stop) {
      if (s.empty()) continue;
      auto p = text.find(s, emitted > s.size() ? emitted - s.size() : 0);
      if (p != std::string::npos) stop_at = std::min(stop_at, p);
    }
    if (stop_at != std::string::npos) {
      text.resize(stop_at);
      r.finish_reason = "stop";
    }
    // Hold back an incomplete UTF-8 tail and any suffix that could start a stop string.
    size_t safe = text.size() - utf8_incomplete_tail(text);
    if (stop_at == std::string::npos) {
      for (const auto& s : req.stop) {
        if (s.empty()) continue;
        for (size_t len = std::min(s.size() - 1, text.size()); len > 0; --len) {
          if (text.compare(text.size() - len, len, s, 0, len) == 0) {
            safe = std::min(safe, text.size() - len);
            break;
          }
        }
      }
    }
    if (safe > emitted) {
      if (on_text && !on_text(text.substr(emitted, safe - emitted))) cancelled = true;
      emitted = safe;
    }
    if (stop_at != std::string::npos || cancelled) return false;
    if (r.completion_tokens >= budget) return false;
    return static_cast<int>(cache_.size()) + 1 < n_ctx;
  };

  int32_t next = sampler.sample(logits_.data(), n_vocab);
  if (!draft_) {
    while (emit(next)) {
      model_->forward(&next, 1, static_cast<int>(cache_.size()), logits_.data(), false);
      cache_.push_back(next);
      next = sampler.sample(logits_.data(), n_vocab);
    }
  } else if (req.sampling.temperature <= 0.0f) {
    // Greedy tree speculation: the draft's top-1 chain plus its top-2 alternative at every depth (free, same
    // draft logits), verified in one target pass. Accepts the longest path the target's argmax follows.
    std::vector<int32_t> batch, chain, alt;
    std::vector<int> parent;
    Dist p;
    int k = std::min(4, n_draft_);
    const int dv = draft_->cfg().n_vocab;
    bool go = emit(next);
    while (go) {
      const int room = std::min(budget - r.completion_tokens, (n_ctx - static_cast<int>(cache_.size()) - 2) / 2);
      const int kk = std::max(0, std::min(k, room));
      chain.clear();
      alt.clear();
      if (kk > 0) {
        std::vector<int32_t> ctx = cache_;
        ctx.push_back(next);
        draft_sync(ctx);
        for (int i = 0; i < kk; ++i) {
          int32_t a = 0, b = 1;
          if (dlogits_[b] > dlogits_[a]) std::swap(a, b);
          for (int32_t v = 2; v < dv; ++v) {
            if (dlogits_[v] > dlogits_[a]) b = a, a = v;
            else if (dlogits_[v] > dlogits_[b]) b = v;
          }
          chain.push_back(a);
          alt.push_back(b);
          if (i + 1 < kk) {
            draft_->forward(&a, 1, static_cast<int>(dcache_.size()), dlogits_.data(), false);
            dcache_.push_back(a);
          }
        }
      }
      // Layout: [next, chain0, alt0, chain1, alt1, ...]; chain i at 1 + 2i, alt i at 2 + 2i.
      batch.assign(1, next);
      parent.assign(1, -1);
      for (int i = 0; i < kk; ++i) {
        const int par = i == 0 ? 0 : 2 * i - 1;
        batch.push_back(chain[i]);
        parent.push_back(par);
        batch.push_back(alt[i]);
        parent.push_back(par);
      }
      const int pos0 = static_cast<int>(cache_.size());
      model_->forward(batch.data(), static_cast<int>(batch.size()), pos0, vlogits_.data(), true, parent.data());
      cache_.push_back(next);
      r.drafted += kk;
      int cur = 0, i = 0;
      bool bonus_ready = false;
      for (; i < kk; ++i) {
        sampler.distribution(vlogits_.data() + static_cast<size_t>(cur) * n_vocab, n_vocab, p);
        const int32_t a = p[0].second;
        int slot = -1;
        if (a == chain[i]) slot = 1 + 2 * i;
        else if (a == alt[i]) slot = 2 + 2 * i;
        if (slot < 0) {
          next = a;
          bonus_ready = true;
          break;
        }
        model_->move_kv(pos0 + slot, static_cast<int>(cache_.size()));
        cache_.push_back(a);
        ++r.accepted;
        cur = slot;
        if (!(go = emit(a))) break;
        if (slot == 2 + 2 * i) {  // alternative leaf: nothing below it
          ++i;
          break;
        }
      }
      if (!go) break;
      if (!bonus_ready) {
        sampler.distribution(vlogits_.data() + static_cast<size_t>(cur) * n_vocab, n_vocab, p);
        next = p[0].second;
      }
      if (i == kk && kk == k && cur == 2 * kk - 1) k = std::min(k + 1, n_draft_);
      else if (2 * i < kk) k = std::max(1, k - 1);
      go = emit(next);
    }
  } else {
    // Speculative decoding. Invariant: cache_ holds the tokens with target KV; next is sampled, not in KV.
    SamplingParams dp = req.sampling;
    dp.seed = dp.seed ? dp.seed + 1 : 0;
    Sampler dsampler(dp, false);
    std::vector<int32_t> batch;
    std::vector<Dist> qd;
    Dist p;
    int k = std::min(4, n_draft_);
    bool go = emit(next);
    while (go) {
      // Draft up to k tokens after cache_ + next (no further than the budget and context allow).
      const int room = std::min(budget - r.completion_tokens, n_ctx - static_cast<int>(cache_.size()) - 2);
      const int kk = std::max(0, std::min(k, room));
      batch.assign(1, next);
      qd.resize(kk);
      if (kk > 0) {
        std::vector<int32_t> ctx = cache_;
        ctx.push_back(next);
        draft_sync(ctx);
        for (int i = 0; i < kk; ++i) {
          dsampler.distribution(dlogits_.data(), draft_->cfg().n_vocab, qd[i]);
          const int32_t d = dsampler.draw(qd[i]);
          batch.push_back(d);
          if (i + 1 < kk) {
            draft_->forward(&d, 1, static_cast<int>(dcache_.size()), dlogits_.data(), false);
            dcache_.push_back(d);
          }
        }
      }
      // One target pass over next + drafts: logits for every position.
      const int nb = static_cast<int>(batch.size());
      model_->forward(batch.data(), nb, static_cast<int>(cache_.size()), vlogits_.data(), true);
      cache_.push_back(next);
      r.drafted += nb - 1;
      int i = 0;
      for (; i < nb - 1; ++i) {
        sampler.distribution(vlogits_.data() + static_cast<size_t>(i) * n_vocab, n_vocab, p);
        bool ok = false;
        const int32_t t = sampler.verify(p, qd[i], batch[i + 1], ok);
        if (!ok) {
          next = t;
          break;
        }
        ++r.accepted;
        cache_.push_back(t);
        if (!(go = emit(t))) break;
      }
      if (!go) break;
      if (i == nb - 1) next = sampler.sample(vlogits_.data() + static_cast<size_t>(nb - 1) * n_vocab, n_vocab);
      // Adapt the draft length: grow after a fully accepted step, shrink when most drafts fail.
      if (i == nb - 1 && nb - 1 == k) k = std::min(k + 1, n_draft_);
      else if (2 * i < nb - 1) k = std::max(1, k - 1);
      go = emit(next);
    }
  }
  if (cancelled) r.finish_reason = "cancelled";
  if (!cancelled && emitted < text.size() && on_text) on_text(text.substr(emitted));
  r.decode_ms = ms_since(t1);
  r.text = std::move(text);
  return r;
}

double Engine::perplexity(const std::string& text, int window, int* scored) {
  std::lock_guard lock(mu_);
  return on_worker([&] { return perplexity_locked(text, window, scored); });
}

double Engine::perplexity_locked(const std::string& text, int window, int* scored) {
  omp_set_num_threads(threads_batch_);
  auto toks = tok_->encode(text, false);
  const int n_vocab = model_->cfg().n_vocab;
  window = std::min(window, model_->n_ctx());
  const int n_chunks = static_cast<int>(toks.size()) / window;
  if (n_chunks == 0) throw std::invalid_argument("text shorter than one window");
  std::vector<float> logits(static_cast<size_t>(Model::kMaxBatch) * n_vocab);
  double nll = 0;
  int count = 0;
  const int first = window / 2;
  for (int c = 0; c < n_chunks; ++c) {
    std::vector<int32_t> chunk(toks.begin() + c * window, toks.begin() + (c + 1) * window);
    if (tok_->add_bos()) chunk[0] = tok_->bos();
    for (int i = 0; i < window; i += Model::kMaxBatch) {
      int n = std::min(Model::kMaxBatch, window - i);
      model_->forward(chunk.data() + i, n, i, logits.data(), true);
      for (int t = 0; t < n; ++t) {
        int j = i + t;
        if (j < first || j + 1 >= window) continue;
        const float* l = logits.data() + static_cast<size_t>(t) * n_vocab;
        float mx = *std::max_element(l, l + n_vocab);
        double sum = 0;
        for (int v = 0; v < n_vocab; ++v) sum += std::exp(static_cast<double>(l[v] - mx));
        nll += std::log(sum) + mx - l[chunk[j + 1]];
        ++count;
      }
    }
    std::fprintf(stderr, "[%d/%d] ppl so far %.4f\n", c + 1, n_chunks, std::exp(nll / count));
  }
  cache_.clear();
  if (scored) *scored = count;
  return nll / count;
}

}  // namespace malkin
