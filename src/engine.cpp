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
// One generation in flight. Invariant while decoding: the slot's cache holds the tokens with KV; next is sampled
// and emitted but not in the KV yet.
struct Engine::Request {
  Request(const GenRequest& rq, const OnText& cb, int n_ctx)
      : req(rq), on_text(cb), sampler(rq.sampling), budget(rq.max_tokens > 0 ? rq.max_tokens : n_ctx) {
    SamplingParams dp = rq.sampling;
    dp.seed = dp.seed ? dp.seed + 1 : 0;
    dsampler = std::make_unique<Sampler>(dp, false);
  }
  const GenRequest& req;
  const OnText& on_text;
  std::promise<GenResult> done;
  Sampler sampler;
  std::unique_ptr<Sampler> dsampler;  // speculative decoding: draws the draft tokens
  std::string text;
  size_t emitted = 0;
  GenResult r;
  bool cancelled = false, decoding = false;
  int budget, k = 4;  // k: adaptive draft length
  int32_t next = 0;
  Clock::time_point t0 = Clock::now(), t1;

  // Appends one generated token; false = generation ends (EOG, stop string, budget, cancel, context).
  bool emit(int32_t id, const Tokenizer& tok, size_t cached, int n_ctx) {
    if (tok.is_eog(id)) {
      r.finish_reason = "stop";
      return false;
    }
    sampler.accept(id);
    ++r.completion_tokens;
    text += tok.piece(id);
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
    return static_cast<int>(cached) + 1 < n_ctx;
  }
};

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

  const int n_seq = std::clamp(o.parallel, 1, Model::kMaxBatch);
  int n_ctx = o.n_ctx > 0 ? o.n_ctx : std::min(cfg.n_ctx_train, 32768);
  if (o.n_ctx <= 0) {
    // Auto: KV of all sequences may take half of the RAM left after the weights.
    size_t avail = mem_available_bytes();
    size_t budget = avail > file_->file_size() ? (avail - file_->file_size()) / 2 : 0;
    n_ctx = std::max(512, std::min<int>(n_ctx, static_cast<int>(budget / n_seq / kv_bytes_per_token(cfg))));
  }
  n_ctx = std::min(n_ctx, cfg.n_ctx_train);

  tok_ = std::make_unique<Tokenizer>(*file_);
  chat_ = std::make_unique<ChatFormat>(*tok_, file_->get_str("tokenizer.chat_template").value_or(""));
  model_ = std::make_unique<Model>(*file_, n_ctx, n_seq);
  slots_.resize(n_seq);
  if (!o.draft_path.empty()) {
    dfile_ = std::make_unique<GgufFile>(o.draft_path);
    Tokenizer dtok(*dfile_);
    bool same = dtok.size() == tok_->size();
    for (int32_t id : {0, 1000, 100000, static_cast<int32_t>(tok_->size()) - 1})
      same = same && dtok.piece(id) == tok_->piece(id);
    if (!same) throw std::runtime_error("draft model tokenizer differs from the target");
    draft_ = std::make_unique<Model>(*dfile_, n_ctx);
    dlogits_.resize(static_cast<size_t>(draft_->cfg().n_vocab));
    if (o.n_draft > 0) n_draft_ = std::min(o.n_draft, Model::kMaxBatch - 1);
    vlogits_.resize(static_cast<size_t>(2 * n_draft_ + 1) * model_->cfg().n_vocab);  // tree: chain + alternatives
    std::fprintf(stderr, "malkin: draft %s, %d layers, up to %d tokens per step\n",
                 std::filesystem::path(o.draft_path).stem().c_str(), draft_->cfg().n_layer, n_draft_);
  }
  logits_.resize(static_cast<size_t>(n_seq) * cfg.n_vocab);
  worker_ = std::thread([this] { worker_loop(); });
  on_worker([this] {
    omp_set_num_threads(threads_batch_);
    file_->prefetch();
  });
  model_id_ = std::filesystem::path(o.model_path).stem().string();
  std::fprintf(
      stderr,
      "malkin: %s | %s | %d layers, %d embd, %d heads (%d kv), vocab %d | ctx %d x %d seq (KV %.2f GiB BF16) | "
      "threads %d decode / %d prefill | loaded in %.1f s\n",
      model_id_.c_str(), cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.n_vocab, n_ctx,
      n_seq, static_cast<double>(kv_bytes_per_token(cfg)) * n_ctx * n_seq / (1 << 30), threads_, threads_batch_,
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
    std::vector<std::unique_ptr<Request>> arrived;
    {
      std::unique_lock l(qmu_);
      qcv_.wait(l, [this] { return stop_ || job_ || !pending_.empty() || busy_ > 0; });
      if (stop_) return;
      if (job_ && busy_ == 0) {
        job = std::move(job_);
        job_ = nullptr;
      } else {
        const size_t take = std::min(pending_.size(), slots_.size() - static_cast<size_t>(busy_));
        for (size_t i = 0; i < take; ++i) arrived.push_back(std::move(pending_[i]));
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<ptrdiff_t>(take));
      }
    }
    if (job) {
      job();
      continue;
    }
    try {
      for (auto& r : arrived) admit(std::move(r));
      if (busy_ > 0) step();
    } catch (...) {  // fail every request in flight; their KV is in an unknown state
      for (Slot& s : slots_) {
        if (!s.req) continue;
        s.req->done.set_exception(std::current_exception());
        s.req.reset();
        s.cache.clear();
      }
      busy_ = 0;
    }
  }
}

GenResult Engine::generate(const GenRequest& req, const OnText& on_text) {
  const int n_ctx = model_->n_ctx();
  if (req.prompt.empty()) throw std::invalid_argument("empty prompt");
  if (static_cast<int>(req.prompt.size()) >= n_ctx) {
    throw std::invalid_argument("prompt has " + std::to_string(req.prompt.size()) + " tokens, context is " +
                                std::to_string(n_ctx));
  }
  auto r = std::make_unique<Request>(req, on_text, n_ctx);
  auto fut = r->done.get_future();
  {
    std::lock_guard l(qmu_);
    pending_.push_back(std::move(r));
  }
  qcv_.notify_one();
  return fut.get();
}

void Engine::admit(std::unique_ptr<Request> r) {
  const auto& prompt = r->req.prompt;
  int best = -1;
  size_t best_common = 0;
  for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
    const Slot& s = slots_[i];
    if (s.req) continue;
    size_t c = 0;
    while (c < s.cache.size() && c < prompt.size() && s.cache[c] == prompt[c]) ++c;
    if (best < 0 || c > best_common || (c == best_common && s.used < slots_[best].used)) {
      best = i;
      best_common = c;
    }
  }
  Slot& s = slots_[best];
  best_common = std::min(best_common, prompt.size() - 1);  // the last prompt token is always evaluated for its logits
  s.cache.resize(best_common);
  r->r.prompt_tokens = static_cast<int>(prompt.size());
  r->r.cached_tokens = static_cast<int>(best_common);
  r->r.finish_reason = "length";
  s.used = ++admitted_;
  r->k = std::min(4, n_draft_);
  s.req = std::move(r);
  ++busy_;
}

void Engine::step() {
  const int n_ctx = model_->n_ctx(), n_vocab = model_->cfg().n_vocab;
  const int n_slots = static_cast<int>(slots_.size());
  if (draft_ && busy_ == 1) {
    for (int i = 0; i < n_slots; ++i) {
      if (slots_[i].req && slots_[i].req->decoding) return spec_step(i);
    }
  }
  batch_.clear();
  batch_seq_.clear();
  for (int i = 0; i < n_slots; ++i) {  // decode tokens first: one per decoding slot
    const Slot& s = slots_[i];
    if (!s.req || !s.req->decoding) continue;
    batch_.push_back({s.req->next, i, static_cast<int>(s.cache.size()), true});
    batch_seq_.push_back(i);
  }
  for (int i = 0; i < n_slots; ++i) {  // then prompt chunks in slot order, up to kMaxBatch tokens
    const Slot& s = slots_[i];
    if (!s.req || s.req->decoding) continue;
    const auto& prompt = s.req->req.prompt;
    const int from = static_cast<int>(s.cache.size());
    const int take =
        std::min(Model::kMaxBatch - static_cast<int>(batch_.size()), static_cast<int>(prompt.size()) - from);
    for (int pos = from; pos < from + take; ++pos) {
      const bool last = pos + 1 == static_cast<int>(prompt.size());
      batch_.push_back({prompt[pos], i, pos, last});
      if (last) batch_seq_.push_back(i);
    }
  }
  const int n = static_cast<int>(batch_.size());
  if (n == 0) return;
  omp_set_num_threads(n > 16 ? threads_batch_ : threads_);
  model_->forward(batch_.data(), n, logits_.data());
  for (const BatchToken& b : batch_) slots_[b.seq].cache.push_back(b.id);
  for (size_t row = 0; row < batch_seq_.size(); ++row) {
    Slot& s = slots_[batch_seq_[row]];
    Request& q = *s.req;
    if (!q.decoding) {
      q.r.prefill_ms = ms_since(q.t0);
      q.t1 = Clock::now();
      q.decoding = true;
    }
    q.next = q.sampler.sample(logits_.data() + row * n_vocab, n_vocab);
    if (!q.emit(q.next, *tok_, s.cache.size(), n_ctx)) finish(s);
  }
}

void Engine::finish(Slot& s) {
  Request& q = *s.req;
  if (q.cancelled) q.r.finish_reason = "cancelled";
  if (!q.cancelled && q.emitted < q.text.size() && q.on_text) q.on_text(q.text.substr(q.emitted));
  q.r.decode_ms = ms_since(q.t1);
  q.r.text = std::move(q.text);
  q.done.set_value(std::move(q.r));
  s.req.reset();
  --busy_;
}

void Engine::spec_step(int seq) {
  Slot& s = slots_[seq];
  Request& q = *s.req;
  auto& cache = s.cache;
  const int n_ctx = model_->n_ctx(), n_vocab = model_->cfg().n_vocab;
  auto emit = [&](int32_t id) { return q.emit(id, *tok_, cache.size(), n_ctx); };
  omp_set_num_threads(threads_);
  Dist p;
  bool go = true;
  if (q.req.sampling.temperature <= 0.0f) {
    // Greedy tree speculation: the draft's top-1 chain plus its top-2 alternative at every depth (free, same
    // draft logits), verified in one target pass. Accepts the longest path the target's argmax follows.
    std::vector<int32_t> batch, chain, alt;
    std::vector<int> parent;
    const int dv = draft_->cfg().n_vocab;
    const int room = std::min(q.budget - q.r.completion_tokens, (n_ctx - static_cast<int>(cache.size()) - 2) / 2);
    const int kk = std::max(0, std::min(q.k, room));
    if (kk > 0) {
      std::vector<int32_t> ctx = cache;
      ctx.push_back(q.next);
      draft_sync(ctx);
      for (int i = 0; i < kk; ++i) {
        int32_t a = 0, b = 1;
        if (dlogits_[b] > dlogits_[a]) std::swap(a, b);
        for (int32_t v = 2; v < dv; ++v) {
          if (dlogits_[v] > dlogits_[a])
            b = a, a = v;
          else if (dlogits_[v] > dlogits_[b])
            b = v;
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
    batch.assign(1, q.next);
    parent.assign(1, -1);
    for (int i = 0; i < kk; ++i) {
      const int par = i == 0 ? 0 : 2 * i - 1;
      batch.push_back(chain[i]);
      parent.push_back(par);
      batch.push_back(alt[i]);
      parent.push_back(par);
    }
    const int pos0 = static_cast<int>(cache.size());
    model_->forward(batch.data(), static_cast<int>(batch.size()), pos0, vlogits_.data(), true, parent.data(), seq);
    cache.push_back(q.next);
    q.r.drafted += kk;
    int cur = 0, i = 0;
    bool bonus_ready = false;
    for (; i < kk; ++i) {
      q.sampler.distribution(vlogits_.data() + static_cast<size_t>(cur) * n_vocab, n_vocab, p);
      const int32_t a = p[0].second;
      int slot = -1;
      if (a == chain[i])
        slot = 1 + 2 * i;
      else if (a == alt[i])
        slot = 2 + 2 * i;
      if (slot < 0) {
        q.next = a;
        bonus_ready = true;
        break;
      }
      model_->move_kv(pos0 + slot, static_cast<int>(cache.size()), seq);
      cache.push_back(a);
      ++q.r.accepted;
      cur = slot;
      go = emit(a);
      if (!go) break;
      if (slot == 2 + 2 * i) {  // alternative leaf: nothing below it
        ++i;
        break;
      }
    }
    if (go) {
      if (!bonus_ready) {
        q.sampler.distribution(vlogits_.data() + static_cast<size_t>(cur) * n_vocab, n_vocab, p);
        q.next = p[0].second;
      }
      if (i == kk && kk == q.k && cur == 2 * kk - 1)
        q.k = std::min(q.k + 1, n_draft_);
      else if (2 * i < kk)
        q.k = std::max(1, q.k - 1);
      go = emit(q.next);
    }
  } else {
    // Speculative sampling: draft up to k tokens after cache + next (no further than the budget and context allow).
    std::vector<int32_t> batch;
    std::vector<Dist> qd;
    const int room = std::min(q.budget - q.r.completion_tokens, n_ctx - static_cast<int>(cache.size()) - 2);
    const int kk = std::max(0, std::min(q.k, room));
    batch.assign(1, q.next);
    qd.resize(kk);
    if (kk > 0) {
      std::vector<int32_t> ctx = cache;
      ctx.push_back(q.next);
      draft_sync(ctx);
      for (int i = 0; i < kk; ++i) {
        q.dsampler->distribution(dlogits_.data(), draft_->cfg().n_vocab, qd[i]);
        const int32_t d = q.dsampler->draw(qd[i]);
        batch.push_back(d);
        if (i + 1 < kk) {
          draft_->forward(&d, 1, static_cast<int>(dcache_.size()), dlogits_.data(), false);
          dcache_.push_back(d);
        }
      }
    }
    // One target pass over next + drafts: logits for every position.
    const int nb = static_cast<int>(batch.size());
    model_->forward(batch.data(), nb, static_cast<int>(cache.size()), vlogits_.data(), true, nullptr, seq);
    cache.push_back(q.next);
    q.r.drafted += nb - 1;
    int i = 0;
    for (; i < nb - 1; ++i) {
      q.sampler.distribution(vlogits_.data() + static_cast<size_t>(i) * n_vocab, n_vocab, p);
      bool ok = false;
      const int32_t t = q.sampler.verify(p, qd[i], batch[i + 1], ok);
      if (!ok) {
        q.next = t;
        break;
      }
      ++q.r.accepted;
      cache.push_back(t);
      go = emit(t);
      if (!go) break;
    }
    if (go) {
      if (i == nb - 1) q.next = q.sampler.sample(vlogits_.data() + static_cast<size_t>(nb - 1) * n_vocab, n_vocab);
      // Adapt the draft length: grow after a fully accepted step, shrink when most drafts fail.
      if (i == nb - 1 && nb - 1 == q.k)
        q.k = std::min(q.k + 1, n_draft_);
      else if (2 * i < nb - 1)
        q.k = std::max(1, q.k - 1);
      go = emit(q.next);
    }
  }
  if (!go) finish(s);
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
    std::vector<int32_t> chunk(toks.begin() + static_cast<ptrdiff_t>(c) * window,
                               toks.begin() + static_cast<ptrdiff_t>(c + 1) * window);
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
  slots_[0].cache.clear();  // perplexity used sequence 0
  if (scored) *scored = count;
  return nll / count;
}

}  // namespace malkin
