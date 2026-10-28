#include "chat.h"

namespace malkin {

namespace {

std::string strip_think(const std::string& s) {
  auto p = s.find("</think>");
  if (p == std::string::npos) return s;
  auto rest = s.substr(p + 8);
  auto q = rest.find_first_not_of("\n");
  return q == std::string::npos ? std::string() : rest.substr(q);
}

}  // namespace

ChatFormat::ChatFormat(const Tokenizer& tok, const std::string& template_src) : tok_(tok) {
  supported_ = tok.find("<|im_start|>") >= 0 && tok.find("<|im_end|>") >= 0;
  thinking_ = template_src.find("enable_thinking") != std::string::npos;
}

std::vector<int32_t> ChatFormat::tokenize(const std::vector<ChatMessage>& msgs, bool enable_thinking) const {
  std::vector<int32_t> out;
  if (tok_.add_bos()) out.push_back(tok_.bos());
  auto add = [&](const std::string& s, bool special) {
    auto ids = tok_.encode(s, special);
    out.insert(out.end(), ids.begin(), ids.end());
  };
  for (const auto& m : msgs) {
    add("<|im_start|>" + m.role + "\n", true);
    add(m.role == "assistant" ? strip_think(m.content) : m.content, false);
    add("<|im_end|>\n", true);
  }
  add("<|im_start|>assistant\n", true);
  if (thinking_ && !enable_thinking) add("<think>\n\n</think>\n\n", true);
  return out;
}

bool split_reasoning(const std::string& text, std::string& reasoning, std::string& content) {
  auto start = text.find("<think>");
  auto end = text.find("</think>");
  if (end == std::string::npos) return false;
  auto from = start == std::string::npos || start > end ? 0 : start + 7;
  reasoning = text.substr(from, end - from);
  auto a = reasoning.find_first_not_of('\n');
  auto b = reasoning.find_last_not_of('\n');
  reasoning = a == std::string::npos ? std::string() : reasoning.substr(a, b - a + 1);
  content = strip_think(text);
  return true;
}

}  // namespace malkin
