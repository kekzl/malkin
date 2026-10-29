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
  tok_ = std::make_unique<Tokenizer>(*file_);
  chat_ = std::make_unique<ChatFormat>(*tok_, file_->get_str("tokenizer.chat_template").value_or(""));
  model_ = std::make_unique<Model>(*file_, 0, n_seq);
  slots_.resize(n_seq);
  if (!o.draft_path.empty()) {
    dfile_ = std::make_unique<GgufFile>(o.draft_path);
    Tokenizer dtok(*dfile_);
    bool same = dtok.size() == tok_->size();
    for (int32_t id : {0, 1000, 100000, static_cast<int32_t>(tok_->size()) - 1})
      same = same && dtok.piece(id) == tok_->piece(id);
    if (!same) throw std::runtime_error("draft model tokenizer differs from the target");
    draft_ = std::make_unique<Model>(*dfile_, 0, n_seq);
    dlogits_.resize(static_cast<size_t>(n_seq) * draft_->cfg().n_vocab);
    dout_.resize(dlogits_.size());
    if (o.n_draft > 0) n_draft_ = std::min(o.n_draft, Model::kMaxBatch - 1);
    std::fprintf(stderr, "malkin: draft %s, %d layers, up to %d tokens per step\n",
                 std::filesystem::path(o.draft_path).stem().c_str(), draft_->cfg().n_layer, n_draft_);
  }
  int n_ctx = o.n_ctx > 0 ? o.n_ctx : std::min(cfg.n_ctx_train, 32768);
  if (o.n_ctx <= 0) {
    // Auto: KV (target + draft, all sequences) may take half of the RAM left once the packed weights are resident
    // (wide panels: ~3x a Q4_K_M file); cgroup limits count.
    const size_t per_token = kv_bytes_per_token(cfg) + (draft_ ? kv_bytes_per_token(draft_->cfg()) : 0);
    n_ctx = std::max(512, std::min<int>(n_ctx, static_cast<int>(mem_available_bytes() / 2 / n_seq / per_token)));
  }
  n_ctx = std::min(n_ctx, cfg.n_ctx_train);
  model_->alloc_kv(n_ctx);
  if (draft_) draft_->alloc_kv(n_ctx);
  // Logits rows: every decode token (greedy tree: 2 * n_draft + 1 per slot, at most kMaxBatch) plus one per
  // prompt that finishes in the step.
  const int rows = std::min(Model::kMaxBatch, n_seq * (draft_ ? 2 * n_draft_ + 1 : 1)) + n_seq;
  logits_.resize(static_cast<size_t>(rows) * cfg.n_vocab);
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

void Engine::propose() {
  const int dv = draft_->cfg().n_vocab;
  // Runs dbatch_ on the draft; copies the logits of flagged tokens to their slot's row of dlogits_.
  auto flush = [&] {
    if (dbatch_.empty()) return;
    draft_->forward(dbatch_.data(), static_cast<int>(dbatch_.size()), dout_.data());
    size_t row = 0;
    for (const BatchToken& b : dbatch_) {
      if (!b.logits) continue;
      std::copy_n(dout_.data() + row++ * dv, dv, dlogits_.data() + static_cast<size_t>(b.seq) * dv);
    }
    dbatch_.clear();
  };
  // Catch up: draft KV of each proposing slot to cache + next (common prefix reuse), in passes of kMaxBatch tokens.
  for (const Proposal& p : props_) {
    if (p.kk == 0) continue;
    Slot& s = slots_[p.seq];
    const size_t len = s.cache.size() + 1;
    size_t common = 0;
    while (common < s.dcache.size() && common < s.cache.size() && s.dcache[common] == s.cache[common]) ++common;
    if (common == s.cache.size() && common < s.dcache.size() && s.dcache[common] == s.req->next) ++common;
    common = std::min(common, len - 1);  // the last token is evaluated for its logits
    s.dcache.resize(common);
    for (size_t pos = common; pos < len; ++pos) {
      const int32_t id = pos < s.cache.size() ? s.cache[pos] : s.req->next;
      dbatch_.push_back({id, p.seq, static_cast<int>(pos), pos + 1 == len});
      s.dcache.push_back(id);
      if (static_cast<int>(dbatch_.size()) == Model::kMaxBatch) flush();
    }
  }
  flush();
  // Rounds: every slot picks its next draft token from its logits, then all feed it in one pass.
  for (int i = 0;; ++i) {
    bool more = false;
    for (Proposal& p : props_) {
      if (i >= p.kk) continue;
      Slot& s = slots_[p.seq];
      float* dl = dlogits_.data() + static_cast<size_t>(p.seq) * dv;
      int32_t a = 0;
      if (p.greedy) {  // top-1 chain, top-2 alternative
        int32_t b = 1;
        if (dl[b] > dl[a]) std::swap(a, b);
        for (int32_t v = 2; v < dv; ++v) {
          if (dl[v] > dl[a])
            b = a, a = v;
          else if (dl[v] > dl[b])
            b = v;
        }
        p.alt.push_back(b);
      } else {
        p.qd.emplace_back();
        s.req->dsampler->distribution(dl, dv, p.qd.back());
        a = s.req->dsampler->draw(p.qd.back());
      }
      p.chain.push_back(a);
      if (i + 1 < p.kk) {
        dbatch_.push_back({a, p.seq, static_cast<int>(s.dcache.size()), true});
        s.dcache.push_back(a);
        more = true;
      }
    }
    flush();
    if (!more) break;
  }
}

void Engine::step() {
  const int n_ctx = model_->n_ctx(), n_vocab = model_->cfg().n_vocab;
  const int n_slots = static_cast<int>(slots_.size());
  props_.clear();
  for (int i = 0; i < n_slots; ++i) {
    if (!slots_[i].req || !slots_[i].req->decoding) continue;
    props_.emplace_back();
    props_.back().seq = i;
  }
  const int nd = static_cast<int>(props_.size());
  // Draft length per decoding slot: adaptive k, within budget and context; all trees fit one target pass.
  bool drafting = false;
  for (Proposal& p : props_) {
    const Slot& s = slots_[p.seq];
    const Request& q = *s.req;
    p.greedy = q.req.sampling.temperature <= 0.0f;
    if (!draft_) continue;
    const int left = q.budget - q.r.completion_tokens, free = n_ctx - static_cast<int>(s.cache.size()) - 2;
    const int cap = Model::kMaxBatch / nd - 1;  // tokens per slot after its next token
    p.kk = std::max(0, std::min({q.k, left, p.greedy ? free / 2 : free, p.greedy ? cap / 2 : cap}));
    drafting = drafting || p.kk > 0;
  }
  if (drafting) {
    omp_set_num_threads(threads_);
    propose();
  }

  // Target batch. Greedy layout per slot: [next, chain0, alt0, chain1, alt1, ...] at KV rows P, P + 1, ...;
  // chain i and alt i sit at RoPE position P + 1 + i under chain i - 1. Sampled: [next, d0, d1, ...] in a row.
  batch_.clear();
  batch_seq_.clear();
  for (Proposal& p : props_) {
    const Slot& s = slots_[p.seq];
    const int P = static_cast<int>(s.cache.size());
    p.root = static_cast<int>(batch_.size());
    batch_.push_back({s.req->next, p.seq, P, true});
    for (int i = 0; i < p.kk; ++i) {
      if (p.greedy) {
        const int par = i == 0 ? p.root : p.root + 2 * i - 1;
        batch_.push_back({p.chain[i], p.seq, P + 1 + i, true, P + 1 + 2 * i, par});
        batch_.push_back({p.alt[i], p.seq, P + 1 + i, true, P + 2 + 2 * i, par});
      } else {
        batch_.push_back({p.chain[i], p.seq, P + 1 + i, true});
      }
    }
  }
  const int decode_rows = static_cast<int>(batch_.size());
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
  for (int t = decode_rows; t < n; ++t) slots_[batch_[t].seq].cache.push_back(batch_[t].id);

  // Prompts that finished: first token.
  for (size_t j = 0; j < batch_seq_.size(); ++j) {
    Slot& s = slots_[batch_seq_[j]];
    Request& q = *s.req;
    q.r.prefill_ms = ms_since(q.t0);
    q.t1 = Clock::now();
    q.decoding = true;
    q.next = q.sampler.sample(logits_.data() + (decode_rows + j) * n_vocab, n_vocab);
    if (!q.emit(q.next, *tok_, s.cache.size(), n_ctx)) finish(s);
  }

  // Decoding slots: accept the longest verified path, then the next token.
  Dist p;
  for (const Proposal& pr : props_) {
    Slot& s = slots_[pr.seq];
    Request& q = *s.req;
    auto& cache = s.cache;
    const int P = static_cast<int>(cache.size());
    auto row = [&](int t) { return logits_.data() + static_cast<size_t>(pr.root + t) * n_vocab; };
    auto emit = [&](int32_t id) { return q.emit(id, *tok_, cache.size(), n_ctx); };
    cache.push_back(q.next);
    q.r.drafted += pr.kk;
    bool go = true;
    if (pr.kk == 0) {
      q.next = q.sampler.sample(row(0), n_vocab);
    } else if (pr.greedy) {
      int cur = 0, i = 0;
      bool bonus_ready = false;
      for (; i < pr.kk; ++i) {
        q.sampler.distribution(row(cur), n_vocab, p);
        const int32_t a = p[0].second;
        int slot = -1;
        if (a == pr.chain[i])
          slot = 1 + 2 * i;
        else if (a == pr.alt[i])
          slot = 2 + 2 * i;
        if (slot < 0) {
          q.next = a;
          bonus_ready = true;
          break;
        }
        model_->move_kv(P + slot, static_cast<int>(cache.size()), pr.seq);
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
      if (go && !bonus_ready) {
        q.sampler.distribution(row(cur), n_vocab, p);
        q.next = p[0].second;
      }
      if (i == pr.kk && pr.kk == q.k && cur == 2 * pr.kk - 1)
        q.k = std::min(q.k + 1, n_draft_);
      else if (2 * i < pr.kk)
        q.k = std::max(1, q.k - 1);
    } else {
      int i = 0;
      for (; i < pr.kk; ++i) {
        q.sampler.distribution(row(i), n_vocab, p);
        bool ok = false;
        const int32_t t = q.sampler.verify(p, pr.qd[i], pr.chain[i], ok);
        if (!ok) {
          q.next = t;
          break;
        }
        ++q.r.accepted;
        cache.push_back(t);
        go = emit(t);
        if (!go) break;
      }
      if (go && i == pr.kk) q.next = q.sampler.sample(row(pr.kk), n_vocab);
      // Adapt the draft length: grow after a fully accepted step, shrink when most drafts fail.
      if (i == pr.kk && pr.kk == q.k)
        q.k = std::min(q.k + 1, n_draft_);
      else if (2 * i < pr.kk)
        q.k = std::max(1, q.k - 1);
    }
    if (go) go = emit(q.next);
    if (!go) finish(s);
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
