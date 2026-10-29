#!/usr/bin/env bash
# llama.cpp with its own speculative decoding (llama-speculative-simple, same draft, greedy) over the 4 prompts of
# tools/bench_decode.sh. usage: tools/bench_spec_llama.sh HOST_TARGET_GGUF HOST_DRAFT_GGUF [N_TOKENS=200] [DRAFT_MAX=8]
set -euo pipefail
T=$1; D=$2; N=${3:-200}; K=${4:-8}
prompts=(
  "Explain in three short paragraphs how a CPU cache hierarchy works."
  "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling."
  "List ten practical tips for reducing household energy use, one sentence each."
  "Summarize the plot of Romeo and Juliet in about 150 words."
)
for p in "${prompts[@]}"; do
  # Same ChatML turn malkin builds with --no-think (Qwen3 hybrid models: empty think block).
  chat="<|im_start|>user
$p<|im_end|>
<|im_start|>assistant
<think>

</think>

"
  docker run --rm -v "$(dirname "$T"):/t:ro" -v "$(dirname "$D"):/d:ro" --entrypoint /app/llama-speculative-simple \
    ghcr.io/ggml-org/llama.cpp:full -m "/t/$(basename "$T")" -md "/d/$(basename "$D")" -t 8 -td 8 --temp 0 --spec-type draft-simple \
    -n "$N" --spec-draft-n-max "$K" -c 4096 -p "$chat" 2>&1 |
    sed -n 's/.*decoded *\([0-9]*\) tokens in *\([0-9.]*\) seconds.*/\1 \2/p'
done | awk '{tok += $1; s += $2} END {printf "llama.cpp + draft (4 prompts) %6.2f tok/s\n", tok / s}'
