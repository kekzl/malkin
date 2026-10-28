#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace malkin {

// Byte-level BPE (GGUF tokenizer.ggml.model = "gpt2").
class Tokenizer {
 public:
  explicit Tokenizer(const GgufFile& f);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // parse_special: control/added token strings in text become their ids.
  std::vector<int32_t> encode(std::string_view text, bool parse_special) const;
  // Raw bytes of one token (may be a partial UTF-8 sequence).
  std::string piece(int32_t id) const;
  std::string decode(const std::vector<int32_t>& ids) const;

  bool is_eog(int32_t id) const;
  bool is_control(int32_t id) const;
  int32_t find(const std::string& text) const;  // -1 if absent
  int32_t bos() const { return bos_; }
  bool add_bos() const { return add_bos_; }
  size_t size() const { return tokens_.size(); }

 private:
  void bpe(const std::string& word, std::vector<int32_t>& out) const;
  void encode_plain(std::string_view text, std::vector<int32_t>& out) const;

  std::vector<std::string> tokens_;
  std::vector<int32_t> types_;
  std::unordered_map<std::string, int32_t> ids_;
  std::unordered_map<std::string, int32_t> merges_;
  std::vector<int32_t> specials_;  // ids of control + user-defined tokens, longest text first
  std::vector<bool> eog_;
  std::string byte_enc_[256];
  std::unordered_map<uint32_t, uint8_t> byte_dec_;
  void* re_ = nullptr;  // pcre2_code*
  int32_t bos_ = -1;
  bool add_bos_ = false;
};

}  // namespace malkin
