# syntax=docker/dockerfile:1
# Targets linux/amd64 (AVX-512) and linux/arm64 (Neoverse-N1+). arm64 is cross-compiled on an amd64 build host,
# so no emulation is needed: docker buildx build --platform linux/amd64,linux/arm64 .
# Toolchain: LLVM 23 (clang, lld, libc++, libomp), CMake 4.4, Ninja 1.13, C++26.
FROM --platform=$BUILDPLATFORM debian:trixie-slim AS deps
ARG TARGETARCH
ARG BUILDARCH
ARG LLVM=23
ARG CMAKE_VERSION=4.4.5
ARG NINJA_VERSION=1.13.2
COPY docker/deps.sh /usr/local/bin/malkin-deps
RUN BUILDARCH=$BUILDARCH TARGETARCH=$TARGETARCH LLVM=$LLVM CMAKE_VERSION=$CMAKE_VERSION NINJA_VERSION=$NINJA_VERSION sh /usr/local/bin/malkin-deps
ENV LLVM=${LLVM} CC=clang-${LLVM} CXX=clang++-${LLVM}
WORKDIR /src

FROM deps AS build
ARG TARGETARCH
ARG BUILDARCH
COPY CMakeLists.txt .
COPY cmake cmake
COPY docker docker
COPY src src
COPY tests tests
RUN set -e; tc=""; triple=x86_64-linux-gnu; \
    if [ "$TARGETARCH" = arm64 ]; then triple=aarch64-linux-gnu; fi; \
    if [ "$TARGETARCH" != "$BUILDARCH" ]; then tc="-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake"; fi; \
    cmake -B build -G Ninja $tc && cmake --build build; \
    mkdir -p /out && cp build/malkin /out/ && sh docker/collect-libs.sh build/malkin /out/lib $triple

# Runs the kernel tests (through qemu for arm64): docker build --target test .
FROM build AS test
RUN ctest --test-dir build --output-on-failure

# COPY only, no RUN: the target-arch stage never executes code at build time.
FROM debian:trixie-slim AS runtime
COPY --from=build /out/lib/ /usr/local/lib/malkin/
COPY --from=build /out/malkin /usr/local/bin/malkin
ENV LD_LIBRARY_PATH=/usr/local/lib/malkin
EXPOSE 8080
ENTRYPOINT ["malkin"]
CMD ["--help"]
