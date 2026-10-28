#!/usr/bin/env bash
# A/B of malkin builds/envs on one GGUF, interleaved, best of RUNS. usage: tools/ab_decode.sh HOST_GGUF PP TG RUNS "ENV_A" "ENV_B"
set -euo pipefail
F=$1; PP=$2; TG=$3; RUNS=$4; A=$5; B=$6
cd "$(dirname "$0")/.."
run() {
  docker run --rm --user "$(id -u):$(id -g)" ${1:+-e $1} -v "$PWD:/src" -v "$(dirname "$F"):/m:ro" -w /src malkin-dev:local \
    ./build-clang/malkin bench --model "/m/$(basename "$F")" --ctx 4096 -pp "$PP" -tg "$TG" 2>/dev/null | tail -1 |
    awk -F'|' '{gsub(/ /, "", $5); gsub(/ /, "", $6); print $5, $6}'
}
for v in A B; do : > "/tmp/ab_$v.$$"; done
for _ in $(seq "$RUNS"); do run "$A" >> "/tmp/ab_A.$$"; run "$B" >> "/tmp/ab_B.$$"; done
for v in A B; do
  env=$([ $v = A ] && echo "$A" || echo "$B")
  printf '%-22s pp%s %8.2f  tg%s %7.2f\n' "${env:-default}" "$PP" "$(sort -k1 -g "/tmp/ab_$v.$$" | tail -1 | cut -d' ' -f1)" \
    "$TG" "$(sort -k2 -g "/tmp/ab_$v.$$" | tail -1 | cut -d' ' -f2)"
  rm -f "/tmp/ab_$v.$$"
done
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
