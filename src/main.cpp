#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cpu.h"
#include "engine.h"
#include "kernels.h"
#include "server.h"

using namespace malkin;

namespace {

constexpr const char* kUsage = R"(malkin )" MALKIN_VERSION R"( - LLM inference for AVX-512 and Neoverse CPUs

usage: malkin [serve] --model PATH [options]     OpenAI + Anthropic API server (default)
       malkin run     --model PATH -p TEXT        one chat turn to stdout
       malkin bench   --model PATH [-pp N] [-tg N]
       malkin ppl     --model PATH -f FILE [--window N]

options:
  --model PATH      GGUF file
  --host ADDR       listen address (default 0.0.0.0)
  --port N          listen port (default 8080)
  --ctx N           context length (default: auto from RAM, max 32768)
  --threads N       decode threads (default: physical cores)
  --threads-batch N prefill threads (default: same as --threads)
  --draft PATH      draft GGUF for speculative decoding (same tokenizer, e.g. Qwen3-0.6B for Qwen3)
  --n-draft N       max draft tokens per step (default 8)
  -p TEXT           prompt (run)
  -n N              max new tokens (run, default 512)
  --temp F          temperature (run, default 0 = greedy)
  --no-think        disable thinking for hybrid reasoning models (run)
  -pp N, -tg N      bench prompt / generation tokens (default 512 / 128)
  -f FILE           text for perplexity
  --window N        perplexity window (default 512)
)";

struct Args {
  std::string cmd = "serve", model, host = "0.0.0.0", prompt, file;
  std::string draft;
  int n_draft = 0;
  int port = 8080, ctx = 0, threads = 0, threads_batch = 0, n = 512, pp = 512, tg = 128, window = 512;
  float temp = 0.0f;
  bool think = true;
};

Args parse(int argc, char** argv) {
  Args a;
  int i = 1;
  if (argc > 1 && argv[1][0] != '-') a.cmd = argv[i++];
  auto val = [&](const char* name) -> std::string {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
    return argv[++i];
  };
  for (; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--model" || s == "-m") a.model = val("--model");
    else if (s == "--host") a.host = val("--host");
    else if (s == "--port") a.port = std::stoi(val("--port"));
    else if (s == "--ctx" || s == "-c") a.ctx = std::stoi(val("--ctx"));
    else if (s == "--threads" || s == "-t") a.threads = std::stoi(val("--threads"));
    else if (s == "--threads-batch" || s == "-tb") a.threads_batch = std::stoi(val("--threads-batch"));
    else if (s == "-p" || s == "--prompt") a.prompt = val("-p");
    else if (s == "-n") a.n = std::stoi(val("-n"));
    else if (s == "--temp") a.temp = std::stof(val("--temp"));
    else if (s == "--no-think") a.think = false;
    else if (s == "--draft") a.draft = val("--draft");
    else if (s == "--n-draft") a.n_draft = std::stoi(val("--n-draft"));
    else if (s == "-pp") a.pp = std::stoi(val("-pp"));
    else if (s == "-tg") a.tg = std::stoi(val("-tg"));
    else if (s == "-f") a.file = val("-f");
    else if (s == "--window") a.window = std::stoi(val("--window"));
    else if (s == "-h" || s == "--help") a.cmd = "help";
    else throw std::invalid_argument("unknown option " + s);
  }
  return a;
}

int run(Engine& e, const Args& a) {
  GenRequest req;
  req.prompt = e.chat().supported() ? e.chat().tokenize({{"user", a.prompt}}, a.think) : e.tokenizer().encode(a.prompt, true);
  req.max_tokens = a.n;
  req.sampling.temperature = a.temp;
  GenResult r = e.generate(req, [](const std::string& d) {
    std::fwrite(d.data(), 1, d.size(), stdout);
    std::fflush(stdout);
    return true;
  });
  std::fprintf(stderr, "\n[prompt %d tok, %.1f tok/s | gen %d tok, %.1f tok/s | %s]\n", r.prompt_tokens,
               r.prompt_tokens * 1000.0 / r.prefill_ms, r.completion_tokens, r.completion_tokens * 1000.0 / r.decode_ms,
               r.finish_reason.c_str());
  if (r.drafted) std::fprintf(stderr, "[draft: %d proposed, %d accepted (%.1f %%)]\n", r.drafted, r.accepted, 100.0 * r.accepted / r.drafted);
  return 0;
}

int bench(Engine& e, const Args& a) {
  // Fixed token ids so runs are comparable; EOG is excluded by ignoring it.
  GenRequest req;
  for (int i = 0; i < a.pp; ++i) req.prompt.push_back(1000 + (i * 7919) % 20000);
  req.max_tokens = 1;
  req.sampling.temperature = 0;
  GenResult pp = e.generate(req, nullptr);
  req.prompt[0] += 1;  // defeat the prefix cache for the timed generation run
  req.max_tokens = a.tg;
  GenResult tg = e.generate(req, nullptr);
  std::printf("| model | isa | threads tg/pp | pp%d tok/s | tg%d tok/s |\n|---|---|---:|---:|---:|\n", a.pp, a.tg);
  std::printf("| %s | %s | %d/%d | %.1f | %.2f |\n", e.model_id().c_str(), kern::isa_name(), e.threads(), e.threads_batch(),
              a.pp * 1000.0 / pp.prefill_ms,
              tg.completion_tokens * 1000.0 / tg.decode_ms);
  if (tg.completion_tokens < a.tg) std::fprintf(stderr, "note: generation stopped early at %d tokens (%s)\n", tg.completion_tokens, tg.finish_reason.c_str());
  return 0;
}

int ppl(Engine& e, const Args& a) {
  std::ifstream f(a.file, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot read " + a.file);
  std::stringstream ss;
  ss << f.rdbuf();
  int scored = 0;
  double nll = e.perplexity(ss.str(), a.window, &scored);
  std::printf("perplexity %.4f over %d tokens (window %d)\n", std::exp(nll), scored, a.window);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    if (a.cmd == "help" || a.model.empty()) {
      std::fputs(kUsage, a.cmd == "help" ? stdout : stderr);
      return a.cmd == "help" ? 0 : 2;
    }
    if (auto miss = cpu_missing_features(); !miss.empty()) {
      std::fprintf(stderr, "malkin: this CPU (%s) lacks %s. malkin needs AVX-512 with VNNI and BF16 (Zen 4+, Sapphire Rapids+).\n",
                   cpu_model_name().c_str(), miss.c_str());
      return 3;
    }
    EngineOptions o{a.model, a.ctx, a.threads, a.threads_batch, a.draft, a.n_draft};
    if (a.cmd == "ppl" && a.ctx == 0) o.n_ctx = a.window;
    Engine e(o);
    if (a.cmd == "serve") return serve(e, a.host, a.port);
    if (a.cmd == "run") return run(e, a);
    if (a.cmd == "bench") return bench(e, a);
    if (a.cmd == "ppl") return ppl(e, a);
    std::fprintf(stderr, "unknown command %s\n%s", a.cmd.c_str(), kUsage);
    return 2;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "malkin: %s\n", ex.what());
    return 1;
  }
}
