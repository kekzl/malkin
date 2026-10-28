#pragma once
#include <string>

#include "engine.h"

namespace malkin {

// Blocks serving the OpenAI and Anthropic HTTP API until the process ends.
int serve(Engine& engine, const std::string& host, int port);

}  // namespace malkin
