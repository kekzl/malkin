#!/usr/bin/env bash
# Greedy answers of N concurrent requests must equal the same requests one at a time (batching is lossless).
# usage: tools/check_parallel.sh URL [N=4] [MAX_TOKENS=64]
set -euo pipefail
URL=$1; N=${2:-4}; MAXT=${3:-64}
topics=("the printing press" "loop unrolling" "the water cycle" "chess openings" "vaccines" "solar power"
  "honey bees" "public-key cryptography")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ask() {
  jq -n --arg t "${topics[$1 % ${#topics[@]}]}" --argjson m "$MAXT" \
    '{messages: [{role: "user", content: ("Explain " + $t + " in detail.")}], max_tokens: $m, temperature: 0}' |
    curl -s -H 'Content-Type: application/json' -d @- "$URL/v1/chat/completions" | jq -r '.choices[0].message | (.reasoning_content // "") + (.content // "")'
}
for i in $(seq 0 $((N - 1))); do ask "$i" > "$tmp/par$i" & done
wait
fail=0
for i in $(seq 0 $((N - 1))); do
  ask "$i" > "$tmp/seq$i"
  if cmp -s "$tmp/par$i" "$tmp/seq$i"; then echo "request $i: same ($(wc -c < "$tmp/seq$i") bytes)"; else
    echo "request $i: DIFFERENT"; diff <(fold -w 80 "$tmp/par$i") <(fold -w 80 "$tmp/seq$i") | head -6; fail=1; fi
done
exit $fail
