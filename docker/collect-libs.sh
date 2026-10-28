#!/bin/sh
# Copy the LLVM runtime libs malkin needs (libc++, libc++abi, libomp, libunwind) for the target arch into $2.
# Everything else it links (glibc, libgcc_s, PCRE2) is in debian:trixie-slim.
set -eu
bin=$1
out=$2
triple=$3
mkdir -p "$out"
need() { llvm-readelf-"$LLVM" -d "$1" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'; }
queue=$(need "$bin")
seen=""
while [ -n "$queue" ]; do
  set -- $queue
  lib=$1; shift; queue="$*"
  case " $seen " in *" $lib "*) continue ;; esac
  seen="$seen $lib"
  case "$lib" in libc++*|libomp*|libunwind*) ;; *) continue ;; esac
  f=""
  for d in /usr/lib/"$triple" /usr/lib/llvm-"$LLVM"/lib /usr/lib/llvm-"$LLVM"/lib/"$triple"; do
    [ -e "$d/$lib" ] && file_arch_ok=$(llvm-readelf-"$LLVM" -h "$d/$lib" | grep -c "$( [ "$triple" = aarch64-linux-gnu ] && echo AArch64 || echo X86-64)") && \
      [ "$file_arch_ok" -gt 0 ] && { f="$d/$lib"; break; }
  done
  [ -n "$f" ] || { echo "collect-libs: $lib for $triple not found"; exit 1; }
  cp -L "$f" "$out/"
  queue="$queue $(need "$f")"
done
ls "$out"
