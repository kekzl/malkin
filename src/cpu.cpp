#include "cpu.h"

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <thread>

namespace malkin {

std::string cpu_missing_features() {
  std::string missing;
  auto need = [&](bool ok, const char* name) {
    if (!ok) missing += std::string(missing.empty() ? "" : " ") + name;
  };
#if defined(__x86_64__)
  __builtin_cpu_init();
  need(__builtin_cpu_supports("avx512f"), "avx512f");
  need(__builtin_cpu_supports("avx512bw"), "avx512bw");
  need(__builtin_cpu_supports("avx512vl"), "avx512vl");
  need(__builtin_cpu_supports("avx512dq"), "avx512dq");
  need(__builtin_cpu_supports("avx512vnni"), "avx512vnni");
  need(__builtin_cpu_supports("avx512bf16"), "avx512bf16");
#elif defined(__aarch64__)
  const unsigned long hw = getauxval(AT_HWCAP);
  need(hw & HWCAP_ASIMD, "asimd");
  need(hw & HWCAP_ASIMDDP, "asimddp");
  need(hw & HWCAP_FPHP, "fphp");
  need(hw & HWCAP_ASIMDHP, "asimdhp");
#endif
  return missing;
}

std::string cpu_model_name() {
  std::ifstream f("/proc/cpuinfo");
  for (std::string line; std::getline(f, line);) {
    auto p = line.find(':');
    if (p == std::string::npos) continue;
    if (line.starts_with("model name")) return line.substr(p + 2);
    if (line.starts_with("CPU part")) {  // Arm: no model name, map the part number
      auto part = std::stoi(line.substr(p + 1), nullptr, 16);
      switch (part) {
        case 0xd0c:
          return "Arm Neoverse-N1";
        case 0xd40:
          return "Arm Neoverse-V1";
        case 0xd49:
          return "Arm Neoverse-N2";
        case 0xd4f:
          return "Arm Neoverse-V2";
        case 0xd8e:
          return "Arm Neoverse-N3";
        case 0xd84:
          return "Arm Neoverse-V3";
        default:
          break;
      }
      return "Arm part " + line.substr(p + 2);
    }
  }
  return "unknown";
}

// Distinct (physical id, core id) pairs; SMT siblings do not add memory bandwidth.
int cpu_physical_cores() {
  std::ifstream f("/proc/cpuinfo");
  std::set<std::pair<int, int>> cores;
  int phys = 0;
  for (std::string line; std::getline(f, line);) {
    auto p = line.find(':');
    if (p == std::string::npos) continue;
    if (line.starts_with("physical id")) phys = std::stoi(line.substr(p + 1));
    if (line.starts_with("core id")) cores.insert({phys, std::stoi(line.substr(p + 1))});
  }
  if (cores.empty()) return static_cast<int>(std::thread::hardware_concurrency());
  return static_cast<int>(cores.size());
}

namespace {

// SIZE_MAX: no such file or "max".
size_t read_size(const std::string& path) {
  std::ifstream f(path);
  std::string s;
  if (!(f >> s) || s == "max") return SIZE_MAX;
  return std::stoull(s);
}

}  // namespace

size_t mem_available_bytes() {
  size_t avail = 0;
  std::ifstream f("/proc/meminfo");
  for (std::string line; std::getline(f, line);) {
    if (line.starts_with("MemAvailable:")) avail = std::stoull(line.substr(13)) * 1024;
  }
  // cgroup v2 limits (docker --memory, systemd MemoryMax), tightest ancestor: limit - (usage - page cache).
  std::ifstream cg("/proc/self/cgroup");
  std::string dir;
  for (std::string line; std::getline(cg, line);) {
    if (line.starts_with("0::")) dir = line.substr(3);
  }
  while (!dir.empty()) {
    const std::string d = "/sys/fs/cgroup" + (dir == "/" ? std::string() : dir);
    const size_t max = read_size(d + "/memory.max"), cur = read_size(d + "/memory.current");
    if (max != SIZE_MAX && cur != SIZE_MAX) {
      size_t file = 0;
      std::ifstream st(d + "/memory.stat");
      for (std::string line; std::getline(st, line);) {
        if (line.starts_with("file ")) file = std::stoull(line.substr(5));
      }
      const size_t used = cur - std::min(file, cur);
      avail = std::min(avail, max > used ? max - used : 0);
    }
    if (dir == "/") break;
    const size_t slash = dir.rfind('/');
    dir = slash == 0 ? "/" : dir.substr(0, slash);
  }
  return avail;
}

}  // namespace malkin
