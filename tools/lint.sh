#!/usr/bin/env bash
# Code quality gate: clang-format, clang-tidy, shellcheck. Runs in the deps image:
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/src" -w /src malkin-dev:local tools/lint.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LLVM=${LLVM:-23}
mapfile -t cxx < <(find src tests -name '*.cpp' -o -name '*.h' | sort)

echo "== clang-format (${#cxx[@]} files)"
clang-format-"$LLVM" --dry-run -Werror "${cxx[@]}"

echo "== clang-tidy"
cmake -B build-lint -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null
# kernels_neon.cpp only compiles for aarch64.
mapfile -t tus < <(find src tests -name '*.cpp' ! -name 'kernels_neon.cpp' | sort)
clang-tidy-"$LLVM" -p build-lint --quiet "${tus[@]}"

echo "== shellcheck"
shellcheck tools/*.sh tests/*.sh docker/*.sh
echo "lint ok"
