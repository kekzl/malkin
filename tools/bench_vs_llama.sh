#!/usr/bin/env bash
# malkin vs llama.cpp, back to back, best of N runs (other tenants share this host).
# usage: tools/bench_vs_llama.sh HOST_GGUF [PP] [TG] [RUNS] [extra malkin env, e.g. MALKIN_NO_PACK=1]
set -euo pipefail
F=$1; M=/m/$(basename "$1"); PP=${2:-512}; TG=${3:-128}; RUNS=${4:-2}; ENVV=${5:-}
cd "$(dirname "$0")/.."
best() { sort -g | tail -1; }
mp=(); mt=(); lp=(); lt=()
for _ in $(seq "$RUNS"); do
  line=$(docker run --rm --user "$(id -u):$(id -g)" ${ENVV:+-e $ENVV} -v "$PWD:/src" -v "$(dirname "$F"):/m:ro" -w /src \
    malkin-dev:local ./"${MALKIN_BUILD:-build-clang}"/malkin bench --model "$M" --ctx 4096 -pp "$PP" -tg "$TG" 2>/dev/null | tail -1)
  mp+=("$(echo "$line" | awk -F'|' '{print $5}')"); mt+=("$(echo "$line" | awk -F'|' '{print $6}')")
  out=$(docker run --rm -v "$(dirname "$F"):/m:ro" --entrypoint /app/llama-bench ghcr.io/ggml-org/llama.cpp:full \
    -m "$M" -p "$PP" -n "$TG" -t 8 -r 1 -o csv 2>/dev/null)
  lp+=("$(echo "$out" | tr -d '"' | awk -F, -v want=pp 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next} (want=="pp"&&$c["n_gen"]==0)||(want=="tg"&&$c["n_prompt"]==0){print $c["avg_ts"]}')")
  lt+=("$(echo "$out" | tr -d '"' | awk -F, -v want=tg 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next} (want=="pp"&&$c["n_gen"]==0)||(want=="tg"&&$c["n_prompt"]==0){print $c["avg_ts"]}')")
done
printf '%s\n' "${mp[@]}" | best | xargs printf 'malkin    pp%s %8.2f\n' "$PP"
printf '%s\n' "${lp[@]}" | best | xargs printf 'llama.cpp pp%s %8.2f\n' "$PP"
printf '%s\n' "${mt[@]}" | best | xargs printf 'malkin    tg%s %8.2f\n' "$TG"
printf '%s\n' "${lt[@]}" | best | xargs printf 'llama.cpp tg%s %8.2f\n' "$TG"
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
