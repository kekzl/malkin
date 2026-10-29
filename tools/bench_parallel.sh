#!/usr/bin/env bash
# Aggregate generation throughput with N concurrent chat requests against a running server.
# usage: tools/bench_parallel.sh URL [N=4] [MAX_TOKENS=128]
set -euo pipefail
URL=$1; N=${2:-4}; MAXT=${3:-128}
topics=("the history of the printing press" "how a compiler optimizes loops" "the water cycle on Earth"
  "the rules of chess" "how vaccines train the immune system" "the economics of renewable energy"
  "the life of honey bees" "how public-key cryptography works")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
t0=$(date +%s.%N)
for i in $(seq 0 $((N - 1))); do
  body=$(jq -n --arg t "${topics[i % ${#topics[@]}]}" --argjson m "$MAXT" \
    '{messages: [{role: "user", content: ("Write a long, detailed essay about " + $t + ".")}], max_tokens: $m, temperature: 0}')
  curl -s -H 'Content-Type: application/json' -d "$body" "$URL/v1/chat/completions" > "$tmp/$i.json" &
done
wait
t1=$(date +%s.%N)
toks=$(jq -s 'map(.usage.completion_tokens) | add' "$tmp"/*.json)
printf 'requests %d, completion tokens %d, wall %.2f s, aggregate %.2f tok/s\n' "$N" "$toks" \
  "$(echo "$t1 - $t0" | bc -l)" "$(echo "$toks / ($t1 - $t0)" | bc -l)"
