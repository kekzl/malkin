#!/bin/sh
# Toolchain for the deps stage: LLVM (clang, lld, libc++, libomp), CMake, Ninja, PCRE2.
# Runtime libs are installed for TARGETARCH only; amd64 and arm64 LLVM dev packages are not co-installable.
set -eu
: "${BUILDARCH:?}" "${TARGETARCH:?}" "${LLVM:?}" "${CMAKE_VERSION:?}" "${NINJA_VERSION:?}"
export DEBIAN_FRONTEND=noninteractive

case "$BUILDARCH" in
  amd64) cm=x86_64; cm_sha=ed85135c5a5135b45e50423cb406dea04ca3338da0b8ef3c9ede963559680be0
         nz=ninja-linux.zip; nz_sha=5749cbc4e668273514150a80e387a957f933c6ed3f5f11e03fb30955e2bbead6 ;;
  arm64) cm=aarch64; cm_sha=301d5063ab259577c1b57ce5490b1aab9ea5c56dc4c822f73499553dab94925f
         nz=ninja-linux-aarch64.zip; nz_sha=fd2cacc8050a7f12a16a2e48f9e06fca5c14fc4c2bee2babb67b58be17a607fc ;;
  *) echo "unsupported build arch $BUILDARCH"; exit 1 ;;
esac

# apt.llvm.org drops DNS and connections for seconds at a time: up to 5 attempts, 10-40 s apart.
# Error-Mode=any makes a failed index fetch fatal, otherwise `apt-get update` exits 0 with stale lists.
retry() {
  n=1
  until "$@"; do
    [ "$n" -lt 5 ] || return 1
    echo "retry $n of 4 in $((n * 10)) s: $*" >&2
    sleep $((n * 10))
    n=$((n + 1))
  done
}
printf 'Acquire::Retries "5";\nAPT::Update::Error-Mode "any";\n' > /etc/apt/apt.conf.d/80retries
fetch() { curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors "$@"; }

retry apt-get update
retry apt-get install -y --no-install-recommends ca-certificates curl unzip pkg-config

fetch -o /tmp/cmake.tgz "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$cm.tar.gz"
echo "$cm_sha  /tmp/cmake.tgz" | sha256sum -c -
tar -xzf /tmp/cmake.tgz -C /opt
ln -s "/opt/cmake-$CMAKE_VERSION-linux-$cm"/bin/* /usr/local/bin/
fetch -o /tmp/ninja.zip "https://github.com/ninja-build/ninja/releases/download/v$NINJA_VERSION/$nz"
echo "$nz_sha  /tmp/ninja.zip" | sha256sum -c -
unzip -q /tmp/ninja.zip -d /usr/local/bin

t=""
extra=""
arches="$BUILDARCH"
if [ "$TARGETARCH" != "$BUILDARCH" ]; then
  [ "$BUILDARCH/$TARGETARCH" = "amd64/arm64" ] || { echo "cross build only amd64 -> arm64"; exit 1; }
  dpkg --add-architecture arm64
  t=":arm64"
  arches="amd64,arm64"
  extra="qemu-user libgcc-14-dev:arm64 libc6-dev:arm64"
fi
mkdir -p /etc/apt/keyrings
fetch https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/keyrings/llvm.asc
echo "deb [arch=$arches signed-by=/etc/apt/keyrings/llvm.asc] http://apt.llvm.org/trixie/ llvm-toolchain-trixie-$LLVM main" \
  > /etc/apt/sources.list.d/llvm.list
retry apt-get update
# shellcheck disable=SC2086
retry apt-get install -y --no-install-recommends clang-$LLVM lld-$LLVM llvm-$LLVM \
  clang-format-$LLVM clang-tidy-$LLVM shellcheck \
  libc++-$LLVM-dev$t libc++abi-$LLVM-dev$t libomp-$LLVM-dev$t libpcre2-dev$t $extra
rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/80retries /tmp/cmake.tgz /tmp/ninja.zip
