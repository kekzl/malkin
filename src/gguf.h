#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace malkin {

// ggml tensor type ids as stored in GGUF.
enum class GgmlType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15, BF16 = 30, MXFP4 = 39,
};
const char* ggml_type_name(GgmlType t);

using GgufArray = std::variant<std::vector<int64_t>, std::vector<double>, std::vector<std::string>>;
using GgufValue = std::variant<int64_t, double, bool, std::string, GgufArray>;

struct GgufTensor {
  std::string name;
  GgmlType type;
  std::vector<int64_t> ne;  // ne[0] is the contiguous (innermost) dimension
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  int64_t nelements() const;
};

class GgufFile {
 public:
  explicit GgufFile(const std::string& path);
  ~GgufFile();
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;

  const GgufValue* find(const std::string& key) const;
  std::optional<int64_t> get_int(const std::string& key) const;
  std::optional<double> get_float(const std::string& key) const;
  std::optional<std::string> get_str(const std::string& key) const;
  const std::vector<std::string>* get_str_array(const std::string& key) const;
  const std::vector<int64_t>* get_int_array(const std::string& key) const;
  const std::vector<double>* get_float_array(const std::string& key) const;

  const GgufTensor* tensor(const std::string& name) const;
  const std::map<std::string, GgufTensor>& tensors() const { return tensors_; }
  size_t file_size() const { return size_; }
  // Fault weights into RAM and advise the kernel to keep them (no-op on failure).
  void prefetch() const;

 private:
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, GgufValue> kv_;
  std::map<std::string, GgufTensor> tensors_;
};

// Bytes for n elements of type t (n must be a multiple of the block size).
size_t ggml_row_size(GgmlType t, int64_t n);

}  // namespace malkin
