#!/usr/bin/env bash
# Decode speed over 4 prompt types, malkin plain vs malkin + draft (greedy, lossless) vs llama.cpp tg128.
# usage: tools/bench_decode.sh HOST_TARGET_GGUF HOST_DRAFT_GGUF [N_TOKENS=200]
set -euo pipefail
T=$1; D=$2; N=${3:-200}
cd "$(dirname "$0")/.."
prompts=(
  "Explain in three short paragraphs how a CPU cache hierarchy works."
  "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling."
  "List ten practical tips for reducing household energy use, one sentence each."
  "Summarize the plot of Romeo and Juliet in about 150 words."
)
run() {  # $1 = extra args; prints "tokens ms" per prompt
  for p in "${prompts[@]}"; do
    # shellcheck disable=SC2086  # $1 is a word-split argument list
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/src" -v "$(dirname "$T"):/t:ro" -v "$(dirname "$D"):/d:ro" -w /src \
      malkin-dev:local ./build-clang/malkin run --model "/t/$(basename "$T")" $1 --no-think -p "$p" -n "$N" 2>&1 >/dev/null |
      sed -n 's/.*gen \([0-9]*\) tok, \([0-9.]*\) tok\/s.*/\1 \2/p'
  done | awk '{tok += $1; ms += $1 / $2 * 1000} END {printf "%.2f", tok / ms * 1000}'
}
plain=$(run "")
spec=$(run "--draft /d/$(basename "$D")")
llama=$(docker run --rm -v "$(dirname "$T"):/t:ro" --entrypoint /app/llama-bench ghcr.io/ggml-org/llama.cpp:full \
  -m "/t/$(basename "$T")" -p 0 -n 128 -t 8 -r 2 -o csv 2>/dev/null | tr -d '"' |
  awk -F, 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next}{print $c["avg_ts"]}' | tail -1)
printf 'llama.cpp tg128         %7.2f tok/s\n' "$llama"
printf 'malkin plain (4 prompts) %6.2f tok/s  %.2fx\n' "$plain" "$(awk -v a="$plain" -v b="$llama" 'BEGIN{print a / b}')"
printf 'malkin + draft           %6.2f tok/s  %.2fx\n' "$spec" "$(awk -v a="$spec" -v b="$llama" 'BEGIN{print a / b}')"
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
