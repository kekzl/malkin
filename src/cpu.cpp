#include "cpu.h"

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

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
        case 0xd0c: return "Arm Neoverse-N1";
        case 0xd40: return "Arm Neoverse-V1";
        case 0xd49: return "Arm Neoverse-N2";
        case 0xd4f: return "Arm Neoverse-V2";
        case 0xd8e: return "Arm Neoverse-N3";
        case 0xd84: return "Arm Neoverse-V3";
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

size_t mem_available_bytes() {
  std::ifstream f("/proc/meminfo");
  for (std::string line; std::getline(f, line);) {
    if (line.starts_with("MemAvailable:")) return std::stoull(line.substr(13)) * 1024;
  }
  return 0;
}

}  // namespace malkin
