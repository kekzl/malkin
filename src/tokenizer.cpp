#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>

namespace malkin {

namespace {

// Pre-tokenizer regexes by GGUF tokenizer.ggml.pre.
constexpr const char* kRegexQwen2 =
    R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr const char* kRegexLlama3 =
    R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

enum TokenType : int32_t { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

void append_utf8(std::string& s, uint32_t cp) {
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (cp >> 18));
    s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// Length of the UTF-8 sequence starting with byte c (1 for invalid lead bytes).
size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xE) return 3;
  if ((c >> 3) == 0x1E) return 4;
  return 1;
}

uint32_t next_cp(const std::string& s, size_t& i) {
  unsigned char c = s[i];
  size_t n = std::min(utf8_len(c), s.size() - i);
  uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
  for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  i += n;
  return cp;
}

}  // namespace

Tokenizer::Tokenizer(const GgufFile& f) {
  auto model = f.get_str("tokenizer.ggml.model").value_or("");
  if (model != "gpt2") throw std::runtime_error("tokenizer model '" + model + "' unsupported (need gpt2 byte-level BPE)");
  auto* toks = f.get_str_array("tokenizer.ggml.tokens");
  auto* merges = f.get_str_array("tokenizer.ggml.merges");
  if (!toks || !merges) throw std::runtime_error("tokenizer: tokens or merges missing");
  tokens_ = *toks;
  auto* types = f.get_int_array("tokenizer.ggml.token_type");
  types_.assign(tokens_.size(), kNormal);
  if (types) std::copy(types->begin(), types->end(), types_.begin());
  ids_.reserve(tokens_.size());
  for (size_t i = 0; i < tokens_.size(); ++i) ids_.emplace(tokens_[i], static_cast<int32_t>(i));
  merges_.reserve(merges->size());
  for (size_t i = 0; i < merges->size(); ++i) merges_.emplace((*merges)[i], static_cast<int32_t>(i));

  for (size_t i = 0; i < tokens_.size(); ++i) {
    if (types_[i] == kControl || types_[i] == kUserDefined) specials_.push_back(static_cast<int32_t>(i));
  }
  std::ranges::sort(specials_, [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });

  // GPT-2 byte <-> unicode mapping: printable bytes map to themselves, the rest to U+0100 onwards.
  uint32_t extra = 256;
  for (uint32_t b = 0; b < 256; ++b) {
    bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    uint32_t cp = printable ? b : extra++;
    append_utf8(byte_enc_[b], cp);
    byte_dec_[cp] = static_cast<uint8_t>(b);
  }

  bos_ = static_cast<int32_t>(f.get_int("tokenizer.ggml.bos_token_id").value_or(-1));
  add_bos_ = f.get_int("tokenizer.ggml.add_bos_token").value_or(0) != 0 && bos_ >= 0;
  eog_.assign(tokens_.size(), false);
  for (auto key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
    auto id = f.get_int(key);
    if (id && *id >= 0 && static_cast<size_t>(*id) < tokens_.size()) eog_[*id] = true;
  }
  for (auto name : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|return|>",
                    "<|call|>", "<|end|>"}) {
    int32_t id = find(name);
    if (id >= 0 && types_[id] == kControl) eog_[id] = true;
  }

  auto pre = f.get_str("tokenizer.ggml.pre").value_or("default");
  const char* pattern = kRegexQwen2;
  if (pre == "llama-bpe" || pre == "llama3" || pre == "smaug-bpe") {
    pattern = kRegexLlama3;
  } else if (pre != "qwen2") {
    std::fprintf(stderr, "malkin: tokenizer pre '%s' unknown, using the qwen2 pre-tokenizer\n", pre.c_str());
  }
  int err = 0;
  PCRE2_SIZE erroff = 0;
  auto* re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                           PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &err, &erroff, nullptr);
  if (!re) throw std::runtime_error("tokenizer: regex compile failed");
  pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
  re_ = re;
}

Tokenizer::~Tokenizer() { pcre2_code_free(static_cast<pcre2_code*>(re_)); }

int32_t Tokenizer::find(const std::string& text) const {
  auto it = ids_.find(text);
  return it == ids_.end() ? -1 : it->second;
}

bool Tokenizer::is_eog(int32_t id) const { return id >= 0 && static_cast<size_t>(id) < eog_.size() && eog_[id]; }

bool Tokenizer::is_control(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < types_.size() && types_[id] == kControl;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const {
  if (auto it = ids_.find(word); it != ids_.end()) {
    out.push_back(it->second);
    return;
  }
  std::vector<std::string> sym;
  for (size_t i = 0; i < word.size();) {
    size_t n = std::min(utf8_len(static_cast<unsigned char>(word[i])), word.size() - i);
    sym.push_back(word.substr(i, n));
    i += n;
  }
  std::string key;
  while (sym.size() > 1) {
    int32_t best = INT32_MAX;
    size_t at = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      key.assign(sym[i]).append(" ").append(sym[i + 1]);
      auto it = merges_.find(key);
      if (it != merges_.end() && it->second < best) best = it->second, at = i;
    }
    if (best == INT32_MAX) break;
    sym[at] += sym[at + 1];
    sym.erase(sym.begin() + static_cast<ptrdiff_t>(at) + 1);
  }
  for (auto& s : sym) {
    auto it = ids_.find(s);
    if (it != ids_.end()) {
      out.push_back(it->second);
    } else {  // not in vocab: fall back to single byte-chars
      for (size_t i = 0; i < s.size();) {
        size_t n = std::min(utf8_len(static_cast<unsigned char>(s[i])), s.size() - i);
        auto jt = ids_.find(s.substr(i, n));
        if (jt != ids_.end()) out.push_back(jt->second);
        i += n;
      }
    }
  }
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t>& out) const {
  auto* re = static_cast<pcre2_code*>(re_);
  auto* md = pcre2_match_data_create_from_pattern(re, nullptr);
  auto emit = [&](size_t a, size_t b) {
    std::string w;
    for (size_t i = a; i < b; ++i) w += byte_enc_[static_cast<unsigned char>(text[i])];
    bpe(w, out);
  };
  size_t off = 0;
  const auto* s = reinterpret_cast<PCRE2_SPTR>(text.data());
  while (off < text.size()) {
    int rc = pcre2_match(re, s, text.size(), off, 0, md, nullptr);
    if (rc < 0) {
      emit(off, text.size());
      break;
    }
    PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
    if (ov[0] > off) emit(off, ov[0]);
    if (ov[1] == ov[0]) {  // empty match: consume one byte
      emit(ov[0], ov[0] + 1);
      off = ov[0] + 1;
      continue;
    }
    emit(ov[0], ov[1]);
    off = ov[1];
  }
  pcre2_match_data_free(md);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
  std::vector<int32_t> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best = std::string_view::npos;
    int32_t best_id = -1;
    for (int32_t id : specials_) {
      if (!parse_special && types_[id] == kControl) continue;
      size_t p = text.find(tokens_[id], pos);
      if (p < best) best = p, best_id = id;  // longest-first order wins ties
    }
    if (best_id < 0) {
      encode_plain(text.substr(pos), out);
      break;
    }
    if (best > pos) encode_plain(text.substr(pos, best - pos), out);
    out.push_back(best_id);
    pos = best + tokens_[best_id].size();
  }
  return out;
}

std::string Tokenizer::piece(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return {};
  const std::string& t = tokens_[id];
  if (types_[id] == kControl || types_[id] == kUserDefined) return t;
  std::string out;
  for (size_t i = 0; i < t.size();) {
    uint32_t cp = next_cp(t, i);
    auto it = byte_dec_.find(cp);
    if (it != byte_dec_.end()) {
      out += static_cast<char>(it->second);
    } else {
      append_utf8(out, cp);
    }
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string s;
  for (auto id : ids) s += piece(id);
  return s;
}

}  // namespace malkin
