#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace malkin {

const char* ggml_type_name(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
      return "F32";
    case GgmlType::F16:
      return "F16";
    case GgmlType::BF16:
      return "BF16";
    case GgmlType::Q4_0:
      return "Q4_0";
    case GgmlType::Q4_1:
      return "Q4_1";
    case GgmlType::Q5_0:
      return "Q5_0";
    case GgmlType::Q5_1:
      return "Q5_1";
    case GgmlType::Q8_0:
      return "Q8_0";
    case GgmlType::Q8_1:
      return "Q8_1";
    case GgmlType::Q2_K:
      return "Q2_K";
    case GgmlType::Q3_K:
      return "Q3_K";
    case GgmlType::Q4_K:
      return "Q4_K";
    case GgmlType::Q5_K:
      return "Q5_K";
    case GgmlType::Q6_K:
      return "Q6_K";
    case GgmlType::Q8_K:
      return "Q8_K";
    case GgmlType::MXFP4:
      return "MXFP4";
  }
  return "unknown";
}

namespace {

struct TypeTraits {
  int64_t block;
  size_t bytes;
};

TypeTraits traits(GgmlType t) {
  switch (t) {
    case GgmlType::F32:
      return {1, 4};
    case GgmlType::F16:
    case GgmlType::BF16:
      return {1, 2};
    case GgmlType::Q4_0:
      return {32, 18};
    case GgmlType::Q4_1:
      return {32, 20};
    case GgmlType::Q5_0:
      return {32, 22};
    case GgmlType::Q5_1:
      return {32, 24};
    case GgmlType::Q8_0:
      return {32, 34};
    case GgmlType::Q8_1:
      return {32, 36};
    case GgmlType::Q2_K:
      return {256, 84};
    case GgmlType::Q3_K:
      return {256, 110};
    case GgmlType::Q4_K:
      return {256, 144};
    case GgmlType::Q5_K:
      return {256, 176};
    case GgmlType::Q6_K:
      return {256, 210};
    case GgmlType::Q8_K:
      return {256, 292};
    case GgmlType::MXFP4:
      return {32, 17};
  }
  throw std::runtime_error("unknown ggml type " + std::to_string(static_cast<uint32_t>(t)));
}

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
  template <typename T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  std::string str() {
    auto n = get<uint64_t>();
    need(n);
    std::string s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }
  size_t offset(const uint8_t* base) const { return static_cast<size_t>(p_ - base); }

 private:
  void need(size_t n) const {
    if (static_cast<size_t>(end_ - p_) < n) throw std::runtime_error("gguf: truncated header");
  }
  const uint8_t* p_;
  const uint8_t* end_;
};

enum : uint32_t { kU8, kI8, kU16, kI16, kU32, kI32, kF32, kBool, kStr, kArr, kU64, kI64, kF64 };

int64_t read_int(Reader& r, uint32_t type) {
  switch (type) {
    case kU8:
      return r.get<uint8_t>();
    case kI8:
      return r.get<int8_t>();
    case kU16:
      return r.get<uint16_t>();
    case kI16:
      return r.get<int16_t>();
    case kU32:
      return r.get<uint32_t>();
    case kI32:
      return r.get<int32_t>();
    case kU64:
      return static_cast<int64_t>(r.get<uint64_t>());
    case kI64:
      return r.get<int64_t>();
    default:
      break;
  }
  throw std::runtime_error("gguf: not an int type");
}

bool is_int(uint32_t t) { return t <= kI32 || t == kU64 || t == kI64; }

GgufValue read_value(Reader& r, uint32_t type) {
  if (is_int(type)) return read_int(r, type);
  switch (type) {
    case kF32:
      return static_cast<double>(r.get<float>());
    case kF64:
      return r.get<double>();
    case kBool:
      return r.get<uint8_t>() != 0;
    case kStr:
      return r.str();
    case kArr: {
      auto et = r.get<uint32_t>();
      auto n = r.get<uint64_t>();
      if (et == kStr) {
        std::vector<std::string> v;
        v.reserve(n);
        for (uint64_t i = 0; i < n; ++i) v.push_back(r.str());
        return GgufArray{std::move(v)};
      }
      if (et == kF32 || et == kF64) {
        std::vector<double> v(n);
        for (auto& x : v) x = et == kF32 ? r.get<float>() : r.get<double>();
        return GgufArray{std::move(v)};
      }
      if (is_int(et) || et == kBool) {
        std::vector<int64_t> v(n);
        for (auto& x : v) x = et == kBool ? r.get<uint8_t>() : read_int(r, et);
        return GgufArray{std::move(v)};
      }
      throw std::runtime_error("gguf: nested arrays unsupported");
    }
    default:
      break;
  }
  throw std::runtime_error("gguf: bad value type " + std::to_string(type));
}

}  // namespace

size_t ggml_row_size(GgmlType t, int64_t n) {
  auto tr = traits(t);
  if (n % tr.block) throw std::runtime_error("row not a multiple of the block size");
  return static_cast<size_t>(n / tr.block) * tr.bytes;
}

int64_t GgufTensor::nelements() const {
  int64_t n = 1;
  for (auto d : ne) n *= d;
  return n;
}

GgufFile::GgufFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  struct stat st{};
  fstat(fd_, &st);
  size_ = static_cast<size_t>(st.st_size);
  void* m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
  base_ = static_cast<uint8_t*>(m);

  Reader r(base_, size_);
  if (r.get<uint32_t>() != 0x46554747) throw std::runtime_error(path + " is not a GGUF file");
  auto version = r.get<uint32_t>();
  if (version < 3) throw std::runtime_error("GGUF v" + std::to_string(version) + " unsupported, need v3");
  auto n_tensors = r.get<uint64_t>();
  auto n_kv = r.get<uint64_t>();
  for (uint64_t i = 0; i < n_kv; ++i) {
    auto key = r.str();
    auto type = r.get<uint32_t>();
    kv_[key] = read_value(r, type);
  }
  struct Info {
    GgufTensor t;
    uint64_t off;
  };
  std::vector<Info> infos;
  for (uint64_t i = 0; i < n_tensors; ++i) {
    Info in;
    in.t.name = r.str();
    auto nd = r.get<uint32_t>();
    for (uint32_t d = 0; d < nd; ++d) in.t.ne.push_back(static_cast<int64_t>(r.get<uint64_t>()));
    in.t.type = static_cast<GgmlType>(r.get<uint32_t>());
    in.off = r.get<uint64_t>();
    infos.push_back(std::move(in));
  }
  size_t align = static_cast<size_t>(get_int("general.alignment").value_or(32));
  size_t data_start = (r.offset(base_) + align - 1) / align * align;
  for (auto& in : infos) {
    auto& t = in.t;
    t.nbytes = ggml_row_size(t.type, t.ne[0]) * static_cast<size_t>(t.nelements() / t.ne[0]);
    if (data_start + in.off + t.nbytes > size_) throw std::runtime_error("gguf: tensor " + t.name + " out of bounds");
    t.data = base_ + data_start + in.off;
    tensors_[t.name] = std::move(t);
  }
}

GgufFile::~GgufFile() {
  if (base_) munmap(base_, size_);
  if (fd_ >= 0) ::close(fd_);
}

// Parallel first touch: on NUMA hosts (EPYC NPS2/NPS4) page-cache pages land near the threads that read them.
void GgufFile::prefetch() const {
  madvise(base_, size_, MADV_WILLNEED);
  const int64_t pages = static_cast<int64_t>((size_ + 4095) / 4096);
  unsigned sum = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (int64_t i = 0; i < pages; ++i) sum += base_[i * 4096];
  volatile unsigned sink = sum;
  (void)sink;
}

const GgufValue* GgufFile::find(const std::string& key) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}

std::optional<int64_t> GgufFile::get_int(const std::string& key) const {
  auto* v = find(key);
  if (!v) return std::nullopt;
  if (auto* i = std::get_if<int64_t>(v)) return *i;
  if (auto* b = std::get_if<bool>(v)) return *b ? 1 : 0;
  return std::nullopt;
}

std::optional<double> GgufFile::get_float(const std::string& key) const {
  auto* v = find(key);
  if (!v) return std::nullopt;
  if (auto* d = std::get_if<double>(v)) return *d;
  if (auto* i = std::get_if<int64_t>(v)) return static_cast<double>(*i);
  return std::nullopt;
}

std::optional<std::string> GgufFile::get_str(const std::string& key) const {
  auto* v = find(key);
  if (!v) return std::nullopt;
  if (auto* s = std::get_if<std::string>(v)) return *s;
  return std::nullopt;
}

template <typename T>
static const std::vector<T>* array_of(const GgufValue* v) {
  if (!v) return nullptr;
  auto* a = std::get_if<GgufArray>(v);
  return a ? std::get_if<std::vector<T>>(a) : nullptr;
}

const std::vector<std::string>* GgufFile::get_str_array(const std::string& key) const {
  return array_of<std::string>(find(key));
}
const std::vector<int64_t>* GgufFile::get_int_array(const std::string& key) const {
  return array_of<int64_t>(find(key));
}
const std::vector<double>* GgufFile::get_float_array(const std::string& key) const {
  return array_of<double>(find(key));
}

const GgufTensor* GgufFile::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

}  // namespace malkin
