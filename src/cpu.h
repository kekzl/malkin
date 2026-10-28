#pragma once
#include <cstddef>
#include <string>

namespace malkin {

// Empty string = this CPU runs malkin; otherwise the missing features.
std::string cpu_missing_features();
std::string cpu_model_name();
int cpu_physical_cores();
size_t mem_available_bytes();

}  // namespace malkin
