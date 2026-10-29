#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "gguf.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace malkin {

struct EngineOptions {
  std::string model_path;
  int n_ctx = 0;           // 0: auto
  int threads = 0;         // decode; 0: physical cores
  int threads_batch = 0;   // prefill; 0: same as threads
  std::string draft_path;  // speculative decoding draft model (same tokenizer); empty: off
  int n_draft = 0;         // max draft tokens per step; 0: default 8 (adaptive below it)
  int parallel = 1;        // sequences decoded together (continuous batching); n_ctx is per sequence
};

struct GenRequest {
  std::vector<int32_t> prompt;
  int max_tokens = 0;  // 0: until context end
  SamplingParams sampling;
  std::vector<std::string> stop;
};

struct GenResult {
  std::string text;
  std::string finish_reason;  // stop | length | cancelled
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  double prefill_ms = 0, decode_ms = 0;
  int drafted = 0, accepted = 0;  // speculative decoding: draft tokens proposed / accepted
};

// Text delta callback; return false to cancel.
using OnText = std::function<bool(const std::string&)>;

// One model, up to `parallel` requests in flight: each step evaluates every active sequence in one forward pass
// (decode tokens plus prompt chunks). A request alone with a draft model decodes speculatively. Each slot reuses
// the KV of the longest common token prefix with its previous request.
class Engine {
 public:
  explicit Engine(const EngineOptions& o);
  ~Engine();
  // Blocks until the request finishes; on_text runs on the engine thread.
  GenResult generate(const GenRequest& req, const OnText& on_text);
  // Mean negative log-likelihood per token over text, in windows of n_ctx tokens (second half scored).
  double perplexity(const std::string& text, int window, int* scored);

  const Tokenizer& tokenizer() const { return *tok_; }
  const ChatFormat& chat() const { return *chat_; }
  const ModelConfig& cfg() const { return model_->cfg(); }
  int n_ctx() const { return model_->n_ctx(); }
  int threads() const { return threads_; }
  int threads_batch() const { return threads_batch_; }
  int parallel() const { return static_cast<int>(slots_.size()); }
  const std::string& model_id() const { return model_id_; }

 private:
  struct Request;  // one generation in flight (engine.cpp)
  struct Slot {
    std::vector<int32_t> cache;    // tokens whose KV is valid in this sequence
    std::unique_ptr<Request> req;  // null: idle
    uint64_t used = 0;             // admission counter, for least-recently-used reuse
  };

  // All compute runs on one worker thread: one OpenMP pool. A pool per HTTP thread ran decode at
  // 7.2 tok/s instead of 13.2 (4B Q8_0, 9800X3D) from the second request on. Jobs run when no slot is busy.
  template <typename F>
  auto on_worker(F&& f) -> decltype(f()) {
    std::packaged_task<decltype(f())()> task(std::forward<F>(f));
    auto fut = task.get_future();
    {
      std::lock_guard l(qmu_);
      job_ = [&task] { task(); };
    }
    qcv_.notify_one();
    return fut.get();
  }
  void worker_loop();
  // Puts r into the idle slot with the longest common prompt prefix.
  void admit(std::unique_ptr<Request> r);
  // One forward pass over all busy slots: a decode token each, then prompt chunks up to kMaxBatch tokens.
  void step();
  // One speculative round for the only busy slot, which is decoding (needs a draft model).
  void spec_step(int seq);
  void finish(Slot& s);
  double perplexity_locked(const std::string& text, int window, int* scored);
  // Bring the draft KV to ctx (common prefix reuse); its logits then predict the token after ctx.
  void draft_sync(const std::vector<int32_t>& ctx);

  std::mutex qmu_;
  std::condition_variable qcv_;
  std::function<void()> job_;
  std::mutex mu_;  // one on_worker job at a time
  std::vector<std::unique_ptr<Request>> pending_;
  bool stop_ = false;
  std::thread worker_;

  std::unique_ptr<GgufFile> file_;
  std::unique_ptr<Tokenizer> tok_;
  std::unique_ptr<ChatFormat> chat_;
  std::unique_ptr<Model> model_;
  std::vector<Slot> slots_;  // one per model sequence
  int busy_ = 0;             // slots with a request (worker thread only)
  uint64_t admitted_ = 0;
  std::vector<BatchToken> batch_;
  std::vector<int> batch_seq_;  // slot of each logits row in the last step
  std::vector<float> logits_;
  std::unique_ptr<GgufFile> dfile_;
  std::unique_ptr<Model> draft_;
  std::vector<int32_t> dcache_;
  std::vector<float> dlogits_, vlogits_;
  int n_draft_ = 8;
  int threads_ = 0, threads_batch_ = 0;
  std::string model_id_;
};

}  // namespace malkin
