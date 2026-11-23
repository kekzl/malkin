#!/usr/bin/env bash
# Latency of /v1/rerank against a running server (Qwen3-Reranker GGUF).
# Usage: tools/bench_rerank.sh URL [DOCS=20] [WORDS=30] [ROUNDS=10]
# Every round asks a different query over DOCS synthetic documents of WORDS words, so only the system prompt
# and the instruction come from the KV of the previous request. Prints min / median / max ms and the scores hash.
set -euo pipefail
url=${1:?server url}
docs=${2:-20}
words=${3:-30}
rounds=${4:-10}

vocab='["lock","gpu","server","port","model","cache","disk","note","vault","index","query","token","thread","docker","image","build","kernel","memory","search","backup","network","config","script","path","release"]'
payload() {  # round
  jq -cn --argjson v "$vocab" --argjson n "$docs" --argjson w "$words" --argjson r "$1" '
    ($v | length) as $l
    | {model: "m", query: "which note explains the \($v[$r % $l]) and the \($v[($r * 7 + 3) % $l])",
       documents: [range(0; $n) as $d
         | [range(0; $w) as $i | $v[($d * 31 + $i * ($d + 7) + $i * $i) % $l]] | join(" ") | "Note \($d): \(.)."]}'
}

times=()
hash=""
for r in $(seq 1 "$rounds"); do
  out=$(curl -sf -o /dev/stdout -w '\n%{time_total}' "$url/v1/rerank" -H 'Content-Type: application/json' -d "$(payload "$r")")
  times+=("$(awk -v t="${out##*$'\n'}" 'BEGIN { printf "%.1f", t * 1000 }')")
  hash+=$(jq -c '.results | sort_by(.index) | map(.relevance_score * 1000000 | round)' <<<"${out%$'\n'*}")
  tokens=$(jq '.usage.prompt_tokens' <<<"${out%$'\n'*}")
done
sorted=$(printf '%s\n' "${times[@]}" | sort -n)
echo "docs $docs, words $words, rounds $rounds, prompt tokens (last round) $tokens"
echo "ms min $(head -1 <<<"$sorted"), median $(sed -n "$(((rounds + 1) / 2))p" <<<"$sorted"), max $(tail -1 <<<"$sorted")"
echo "scores sha256 $(sha256sum <<<"$hash" | cut -c1-16)"
