#!/usr/bin/env bash
# Embeddings and rerank scores against llama.cpp on the same GGUF files.
# Usage: tools/check_embed.sh EMBED_GGUF RERANK_GGUF   (host paths under /home/kekz/models; needs build/malkin)
# Prints the cosine similarity per input and both engines' rerank scores. Exit 1 if a cosine is below 0.97
# or the engines rank the documents differently.
set -euo pipefail
embed=${1:?embedding gguf}
rerank=${2:?reranker gguf}
root="$(cd "$(dirname "$0")/.." && pwd)"
port=18091
base="http://127.0.0.1:$port"
name=malkin-check-embed
tmp="$(mktemp -d)"
trap 'docker rm -f $name >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT

inputs='["The GPU lock must be taken before every job.","Wie verkleinere ich die virtuelle Festplatte?","Paris is the capital of France.","x"]'
query='how do I take the GPU lock'
docs='["Run gpu-lock.sh acquire before a GPU job and release it afterwards.","The cat sat on the mat.","Port 8080 serves the chat model.","Vor jedem GPU-Job muss das Lock geholt werden."]'

start() {  # engine model extra-args...
  local engine=$1 model=$2
  shift 2
  docker rm -f "$name" >/dev/null 2>&1 || true
  if [ "$engine" = malkin ]; then
    docker run -d --name "$name" --user "$(id -u):$(id -g)" -v "$root":/src -v /home/kekz/models:/models:ro \
      -p "127.0.0.1:$port:8080" malkin-dev:local /src/build/malkin --model "/models/$(basename "$model")" "$@" >/dev/null
  else
    docker run -d --name "$name" -v /home/kekz/models:/models:ro -p "127.0.0.1:$port:8080" \
      ghcr.io/ggml-org/llama.cpp:full --server -m "/models/$(basename "$model")" --host 0.0.0.0 --port 8080 \
      -c 2048 "$@" >/dev/null
  fi
  for _ in $(seq 1 120); do
    curl -sf "$base/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  docker logs "$name" 2>&1 | tail -20
  return 1
}

post() { curl -sf "$base$1" -H 'Content-Type: application/json' -d "$2"; }

start malkin "$embed"
post /v1/embeddings "{\"model\":\"m\",\"input\":$inputs}" >"$tmp/emb_malkin.json"
post /v1/embeddings "{\"model\":\"m\",\"input\":$inputs,\"encoding_format\":\"base64\",\"dimensions\":256}" >"$tmp/emb_b64.json"
start llama "$embed" --embedding --pooling last
post /v1/embeddings "{\"model\":\"m\",\"input\":$inputs}" >"$tmp/emb_llama.json"

start malkin "$rerank"
post /v1/rerank "{\"model\":\"m\",\"query\":\"$query\",\"documents\":$docs}" >"$tmp/rr_malkin.json"
start llama "$rerank" --reranking
post /v1/rerank "{\"model\":\"m\",\"query\":\"$query\",\"documents\":$docs}" >"$tmp/rr_llama.json"

fail=0
echo "embedding dims: malkin $(jq '.data[0].embedding | length' "$tmp/emb_malkin.json"), llama.cpp $(jq '.data[0].embedding | length' "$tmp/emb_llama.json")"
echo "embedding prompt tokens: malkin $(jq '.usage.prompt_tokens' "$tmp/emb_malkin.json"), llama.cpp $(jq '.usage.prompt_tokens' "$tmp/emb_llama.json")"
echo "base64, dimensions 256: $(jq -r '.data[0].embedding' "$tmp/emb_b64.json" | base64 -d | wc -c) bytes"
cos=$(jq -rn --slurpfile a "$tmp/emb_malkin.json" --slurpfile b "$tmp/emb_llama.json" '
  def dot(x; y): [x, y] | transpose | map(.[0] * .[1]) | add;
  ($a[0].data | sort_by(.index)) as $da | ($b[0].data | sort_by(.index)) as $db
  | [range(0; $da | length)] | map(. as $i | $da[$i].embedding as $x | $db[$i].embedding as $y
    | dot($x; $y) / ((dot($x; $x) | sqrt) * (dot($y; $y) | sqrt))) | map(tostring) | join(" ")')
echo "cosine malkin vs llama.cpp per input: $cos"
for c in $cos; do
  awk -v c="$c" 'BEGIN { exit !(c >= 0.97) }' || fail=1
done

order() { jq -c '.results | sort_by(-.relevance_score) | map(.index)' "$1"; }
scores() { jq -c '.results | sort_by(.index) | map(.relevance_score * 10000 | round / 10000)' "$1"; }
echo "rerank scores by document: malkin $(scores "$tmp/rr_malkin.json"), llama.cpp $(scores "$tmp/rr_llama.json")"
echo "rerank order: malkin $(order "$tmp/rr_malkin.json"), llama.cpp $(order "$tmp/rr_llama.json")"
[ "$(order "$tmp/rr_malkin.json")" = "$(order "$tmp/rr_llama.json")" ] || fail=1
[ $fail = 0 ] && echo "OK" || echo "FAIL"
exit $fail
