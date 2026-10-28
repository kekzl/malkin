#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.h"

namespace malkin {

struct ChatMessage {
  std::string role;
  std::string content;
};

// ChatML (<|im_start|>role ... <|im_end|>), the Qwen family format.
class ChatFormat {
 public:
  ChatFormat(const Tokenizer& tok, const std::string& template_src);
  bool supported() const { return supported_; }
  bool has_thinking() const { return thinking_; }
  // Tokens of the conversation plus the open assistant turn. Message content never parses control tokens.
  std::vector<int32_t> tokenize(const std::vector<ChatMessage>& msgs, bool enable_thinking) const;

 private:
  const Tokenizer& tok_;
  bool supported_ = false;
  bool thinking_ = false;
};

// Split "<think>...</think>rest" into reasoning and content. Returns false if no think block.
bool split_reasoning(const std::string& text, std::string& reasoning, std::string& content);

}  // namespace malkin
