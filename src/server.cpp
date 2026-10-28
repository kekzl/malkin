#include "server.h"

#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <random>

namespace malkin {

using json = nlohmann::json;

namespace {

struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

int64_t now_s() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string rand_id(const char* prefix) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(rng()));
  return std::string(prefix) + buf;
}

void send_error(httplib::Response& res, int status, const std::string& msg) {
  res.status = status;
  json e = {{"error", {{"message", msg}, {"type", status == 400 ? "invalid_request_error" : "server_error"}}}};
  res.set_content(e.dump(), "application/json");
}

// Text of an OpenAI/Anthropic content field: string or array of text parts.
std::string content_text(const json& c) {
  if (c.is_null()) return {};
  if (c.is_string()) return c.get<std::string>();
  if (!c.is_array()) throw BadRequest("content must be a string or an array");
  std::string s;
  for (const auto& p : c) {
    auto type = p.value("type", "text");
    if (type != "text") throw BadRequest("content part type '" + type + "' unsupported (text only)");
    s += p.value("text", "");
  }
  return s;
}

std::vector<std::string> stop_list(const json& j, const char* key) {
  std::vector<std::string> out;
  if (!j.contains(key) || j[key].is_null()) return out;
  if (j[key].is_string()) return {j[key].get<std::string>()};
  for (const auto& s : j[key]) out.push_back(s.get<std::string>());
  return out;
}

SamplingParams sampling_from(const json& j) {
  SamplingParams p;
  p.temperature = j.value("temperature", p.temperature);
  p.top_p = j.value("top_p", p.top_p);
  p.top_k = j.value("top_k", p.top_k);
  p.min_p = j.value("min_p", p.min_p);
  p.presence_penalty = j.value("presence_penalty", p.presence_penalty);
  p.frequency_penalty = j.value("frequency_penalty", p.frequency_penalty);
  if (j.contains("seed") && j["seed"].is_number_integer()) p.seed = j["seed"].get<uint64_t>();
  return p;
}

// Splits streamed text into reasoning (<think>...</think>) and content deltas.
class ReasoningSplitter {
 public:
  explicit ReasoningSplitter(bool enabled) : enabled_(enabled) {}
  // Returns {reasoning delta, content delta}.
  std::pair<std::string, std::string> feed(const std::string& delta, bool final = false) {
    text_ += delta;
    auto [r, c] = split(final);
    std::pair<std::string, std::string> out;
    if (r.size() > sent_r_) out.first = r.substr(sent_r_), sent_r_ = r.size();
    if (c.size() > sent_c_) out.second = c.substr(sent_c_), sent_c_ = c.size();
    return out;
  }
  std::pair<std::string, std::string> all() const { return split(true); }

 private:
  static std::string ltrim_nl(const std::string& s) {
    auto p = s.find_first_not_of('\n');
    return p == std::string::npos ? std::string() : s.substr(p);
  }
  std::pair<std::string, std::string> split(bool final) const {
    if (!enabled_) return {"", text_};
    static const std::string open = "<think>", close = "</think>";
    size_t lead = text_.find_first_not_of(" \n");
    if (lead == std::string::npos) return {"", final ? text_ : ""};
    std::string_view head(text_.data() + lead, text_.size() - lead);
    if (head.starts_with(open)) {
      std::string body(head.substr(open.size()));
      auto e = body.find(close);
      if (e != std::string::npos) return {ltrim_nl(body.substr(0, e)), ltrim_nl(body.substr(e + close.size()))};
      if (!final) {  // withhold a partial closing tag
        for (size_t len = std::min(close.size() - 1, body.size()); len > 0; --len) {
          if (body.compare(body.size() - len, len, close, 0, len) == 0) {
            body.resize(body.size() - len);
            break;
          }
        }
      }
      return {ltrim_nl(body), ""};
    }
    if (!final && open.starts_with(head)) return {"", ""};
    return {"", text_};
  }

  bool enabled_;
  std::string text_;
  size_t sent_r_ = 0, sent_c_ = 0;
};

struct Prepared {
  GenRequest req;
  bool stream = false;
  bool include_usage = false;
  bool thinking = false;
};

std::vector<ChatMessage> openai_messages(const json& body) {
  if (!body.contains("messages") || !body["messages"].is_array()) throw BadRequest("messages must be an array");
  std::vector<ChatMessage> msgs;
  for (const auto& m : body["messages"]) {
    auto role = m.value("role", "");
    if (role == "developer") role = "system";
    if (role != "system" && role != "user" && role != "assistant") throw BadRequest("role '" + role + "' unsupported");
    msgs.push_back({role, content_text(m.contains("content") ? m["content"] : json())});
  }
  return msgs;
}

void check_no_tools(const json& body) {
  if (body.contains("tools") && !body["tools"].empty()) throw BadRequest("tools are not supported yet");
}

json usage_json(const GenResult& r) {
  return {{"prompt_tokens", r.prompt_tokens},
          {"completion_tokens", r.completion_tokens},
          {"total_tokens", r.prompt_tokens + r.completion_tokens},
          {"prompt_tokens_details", {{"cached_tokens", r.cached_tokens}}}};
}

json timings_json(const GenResult& r) {
  int prefilled = r.prompt_tokens - r.cached_tokens;
  return {{"prompt_n", prefilled},
          {"prompt_ms", r.prefill_ms},
          {"prompt_per_second", r.prefill_ms > 0 ? prefilled * 1000.0 / r.prefill_ms : 0.0},
          {"predicted_n", r.completion_tokens},
          {"predicted_ms", r.decode_ms},
          {"predicted_per_second", r.decode_ms > 0 ? r.completion_tokens * 1000.0 / r.decode_ms : 0.0}};
}

void log_request(const char* path, const GenResult& r) {
  json t = timings_json(r);
  std::fprintf(stderr, "%s: prompt %d (cached %d) %.1f tok/s | gen %d %.1f tok/s | %s\n", path, r.prompt_tokens,
               r.cached_tokens, t["prompt_per_second"].get<double>(), r.completion_tokens,
               t["predicted_per_second"].get<double>(), r.finish_reason.c_str());
}

std::string sse(const json& j) { return "data: " + j.dump() + "\n\n"; }

std::string sse_event(const char* event, const json& j) {
  return std::string("event: ") + event + "\ndata: " + j.dump() + "\n\n";
}

class Server {
 public:
  explicit Server(Engine& e) : e_(e) {}

  void routes(httplib::Server& svr) {
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"status":"ok"})", "application/json");
    });
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
      json m = {{"object", "list"},
                {"data", json::array({{{"id", e_.model_id()},
                                       {"object", "model"},
                                       {"created", now_s()},
                                       {"owned_by", "malkin"},
                                       {"context_length", e_.n_ctx()}}})}};
      res.set_content(m.dump(), "application/json");
    });
    svr.Post("/v1/chat/completions", wrap([this](const json& b, httplib::Response& res) { chat(b, res); }));
    svr.Post("/v1/completions", wrap([this](const json& b, httplib::Response& res) { completion(b, res); }));
    svr.Post("/v1/messages", wrap([this](const json& b, httplib::Response& res) { messages(b, res); }));
  }

 private:
  template <typename F>
  httplib::Server::Handler wrap(F f) {
    return [f](const httplib::Request& req, httplib::Response& res) {
      try {
        f(json::parse(req.body), res);
      } catch (const json::exception& ex) {
        send_error(res, 400, std::string("bad json: ") + ex.what());
      } catch (const BadRequest& ex) {
        send_error(res, 400, ex.what());
      } catch (const std::invalid_argument& ex) {
        send_error(res, 400, ex.what());
      } catch (const std::exception& ex) {
        send_error(res, 500, ex.what());
      }
    };
  }

  Prepared prepare_chat(const json& b, const std::vector<ChatMessage>& msgs) {
    if (!e_.chat().supported()) throw BadRequest("this model has no supported chat format, use /v1/completions");
    Prepared p;
    bool thinking = true;
    if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].contains("enable_thinking")) {
      thinking = b["chat_template_kwargs"]["enable_thinking"].get<bool>();
    }
    p.thinking = thinking && e_.chat().has_thinking();
    p.req.prompt = e_.chat().tokenize(msgs, thinking);
    p.req.sampling = sampling_from(b);
    p.req.max_tokens = b.value("max_completion_tokens", b.value("max_tokens", 0));
    p.req.stop = stop_list(b, "stop");
    p.stream = b.value("stream", false);
    p.include_usage = b.contains("stream_options") && b["stream_options"].value("include_usage", false);
    return p;
  }

  void chat(const json& b, httplib::Response& res) {
    check_no_tools(b);
    Prepared p = prepare_chat(b, openai_messages(b));
    const std::string id = rand_id("chatcmpl-");
    const int64_t created = now_s();
    const std::string model = e_.model_id();
    if (!p.stream) {
      GenResult r = e_.generate(p.req, nullptr);
      log_request("chat", r);
      ReasoningSplitter sp(p.thinking);
      sp.feed(r.text, true);
      auto [reasoning, content] = sp.all();
      json msg = {{"role", "assistant"}, {"content", content}};
      if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
      json out = {{"id", id},
                  {"object", "chat.completion"},
                  {"created", created},
                  {"model", model},
                  {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", r.finish_reason}}})},
                  {"usage", usage_json(r)},
                  {"timings", timings_json(r)}};
      res.set_content(out.dump(), "application/json");
      return;
    }
    res.set_chunked_content_provider("text/event-stream", [this, p, id, created, model](size_t, httplib::DataSink& sink) {
      auto chunk = [&](const json& delta, const json& finish) {
        return json{{"id", id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model},
                    {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
      };
      auto write = [&](const std::string& s) { return sink.is_writable() && sink.write(s.data(), s.size()); };
      write(sse(chunk({{"role", "assistant"}, {"content", ""}}, nullptr)));
      ReasoningSplitter sp(p.thinking);
      auto emit = [&](const std::pair<std::string, std::string>& d) {
        bool ok = true;
        if (!d.first.empty()) ok &= write(sse(chunk({{"reasoning_content", d.first}}, nullptr)));
        if (!d.second.empty()) ok &= write(sse(chunk({{"content", d.second}}, nullptr)));
        return ok;
      };
      GenResult r;
      try {
        r = e_.generate(p.req, [&](const std::string& delta) { return emit(sp.feed(delta)); });
      } catch (const std::exception& ex) {
        write(sse({{"error", {{"message", ex.what()}}}}));
        sink.done();
        return true;
      }
      log_request("chat", r);
      emit(sp.feed("", true));
      write(sse(chunk(json::object(), r.finish_reason)));
      if (p.include_usage) {
        json u = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model},
                  {"choices", json::array()}, {"usage", usage_json(r)}, {"timings", timings_json(r)}};
        write(sse(u));
      }
      write("data: [DONE]\n\n");
      sink.done();
      return true;
    });
  }

  void completion(const json& b, httplib::Response& res) {
    if (!b.contains("prompt") || !b["prompt"].is_string()) throw BadRequest("prompt must be a string");
    GenRequest req;
    const auto& tok = e_.tokenizer();
    if (tok.add_bos()) req.prompt.push_back(tok.bos());
    auto ids = tok.encode(b["prompt"].get<std::string>(), true);
    req.prompt.insert(req.prompt.end(), ids.begin(), ids.end());
    req.sampling = sampling_from(b);
    req.max_tokens = b.value("max_tokens", 16);
    req.stop = stop_list(b, "stop");
    const bool stream = b.value("stream", false);
    const std::string id = rand_id("cmpl-");
    const int64_t created = now_s();
    const std::string model = e_.model_id();
    auto body = [&](const std::string& text, const json& finish) {
      return json{{"id", id},
                  {"object", "text_completion"},
                  {"created", created},
                  {"model", model},
                  {"choices", json::array({{{"index", 0}, {"text", text}, {"finish_reason", finish}}})}};
    };
    if (!stream) {
      GenResult r = e_.generate(req, nullptr);
      log_request("completion", r);
      json out = body(r.text, r.finish_reason);
      out["usage"] = usage_json(r);
      out["timings"] = timings_json(r);
      res.set_content(out.dump(), "application/json");
      return;
    }
    res.set_chunked_content_provider("text/event-stream", [this, req, body](size_t, httplib::DataSink& sink) {
      auto write = [&](const std::string& s) { return sink.is_writable() && sink.write(s.data(), s.size()); };
      try {
        GenResult r = e_.generate(req, [&](const std::string& d) { return write(sse(body(d, nullptr))); });
        log_request("completion", r);
        write(sse(body("", r.finish_reason)));
      } catch (const std::exception& ex) {
        write(sse({{"error", {{"message", ex.what()}}}}));
      }
      write("data: [DONE]\n\n");
      sink.done();
      return true;
    });
  }

  void messages(const json& b, httplib::Response& res) {
    check_no_tools(b);
    std::vector<ChatMessage> msgs;
    if (b.contains("system")) msgs.push_back({"system", content_text(b["system"])});
    if (!b.contains("messages") || !b["messages"].is_array()) throw BadRequest("messages must be an array");
    for (const auto& m : b["messages"]) {
      auto role = m.value("role", "");
      if (role != "user" && role != "assistant") throw BadRequest("role '" + role + "' unsupported");
      msgs.push_back({role, content_text(m.contains("content") ? m["content"] : json())});
    }
    json sb = b;
    sb.erase("stop");
    if (b.contains("stop_sequences")) sb["stop"] = b["stop_sequences"];
    bool thinking = b.contains("thinking") && b["thinking"].value("type", "") == "enabled";
    sb["chat_template_kwargs"] = {{"enable_thinking", thinking}};
    Prepared p = prepare_chat(sb, msgs);
    const std::string id = rand_id("msg_");
    const std::string model = b.value("model", e_.model_id());
    auto stop_reason = [](const GenResult& r) { return r.finish_reason == "length" ? "max_tokens" : "end_turn"; };
    auto usage = [](const GenResult& r) {
      return json{{"input_tokens", r.prompt_tokens - r.cached_tokens},
                  {"cache_read_input_tokens", r.cached_tokens},
                  {"output_tokens", r.completion_tokens}};
    };
    if (!p.stream) {
      GenResult r = e_.generate(p.req, nullptr);
      log_request("messages", r);
      ReasoningSplitter sp(p.thinking);
      sp.feed(r.text, true);
      auto [reasoning, content] = sp.all();
      json blocks = json::array();
      if (!reasoning.empty()) blocks.push_back({{"type", "thinking"}, {"thinking", reasoning}, {"signature", ""}});
      blocks.push_back({{"type", "text"}, {"text", content}});
      json out = {{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", model},
                  {"content", blocks}, {"stop_reason", stop_reason(r)}, {"stop_sequence", nullptr},
                  {"usage", usage(r)}};
      res.set_content(out.dump(), "application/json");
      return;
    }
    res.set_chunked_content_provider("text/event-stream", [this, p, id, model, stop_reason, usage](size_t, httplib::DataSink& sink) {
      auto write = [&](const std::string& s) { return sink.is_writable() && sink.write(s.data(), s.size()); };
      write(sse_event("message_start",
                      {{"type", "message_start"},
                       {"message", {{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", model},
                                    {"content", json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                                    {"usage", {{"input_tokens", p.req.prompt.size()}, {"output_tokens", 0}}}}}}));
      int index = -1;
      std::string open;  // "thinking" or "text"
      auto block = [&](const std::string& kind) {
        if (open == kind) return;
        if (!open.empty()) write(sse_event("content_block_stop", {{"type", "content_block_stop"}, {"index", index}}));
        open = kind;
        ++index;
        json cb = kind == "thinking" ? json{{"type", "thinking"}, {"thinking", ""}} : json{{"type", "text"}, {"text", ""}};
        write(sse_event("content_block_start", {{"type", "content_block_start"}, {"index", index}, {"content_block", cb}}));
      };
      ReasoningSplitter sp(p.thinking);
      auto emit = [&](const std::pair<std::string, std::string>& d) {
        if (!d.first.empty()) {
          block("thinking");
          write(sse_event("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                  {"delta", {{"type", "thinking_delta"}, {"thinking", d.first}}}}));
        }
        if (!d.second.empty()) {
          block("text");
          write(sse_event("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                  {"delta", {{"type", "text_delta"}, {"text", d.second}}}}));
        }
        return sink.is_writable();
      };
      GenResult r;
      try {
        r = e_.generate(p.req, [&](const std::string& delta) { return emit(sp.feed(delta)); });
      } catch (const std::exception& ex) {
        write(sse_event("error", {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", ex.what()}}}}));
        sink.done();
        return true;
      }
      log_request("messages", r);
      emit(sp.feed("", true));
      block("text");
      write(sse_event("content_block_stop", {{"type", "content_block_stop"}, {"index", index}}));
      write(sse_event("message_delta", {{"type", "message_delta"},
                                        {"delta", {{"stop_reason", stop_reason(r)}, {"stop_sequence", nullptr}}},
                                        {"usage", usage(r)}}));
      write(sse_event("message_stop", {{"type", "message_stop"}}));
      sink.done();
      return true;
    });
  }

  Engine& e_;
};

}  // namespace

int serve(Engine& engine, const std::string& host, int port) {
  httplib::Server svr;
  Server s(engine);
  s.routes(svr);
  svr.set_payload_max_length(64 << 20);
  svr.set_read_timeout(600);
  svr.set_write_timeout(600);
  std::fprintf(stderr, "malkin: listening on http://%s:%d\n", host.c_str(), port);
  if (!svr.listen(host, port)) {
    std::fprintf(stderr, "malkin: cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}

}  // namespace malkin
