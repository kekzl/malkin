# syntax=docker/dockerfile:1
# Targets linux/amd64 (AVX-512) and linux/arm64 (Neoverse-N1+). arm64 is cross-compiled on an amd64 build host,
# so no emulation is needed: docker buildx build --platform linux/amd64,linux/arm64 .
FROM --platform=$BUILDPLATFORM debian:trixie-slim AS deps
ARG TARGETARCH
ARG BUILDARCH
RUN set -e; pkgs="g++ cmake ninja-build pkg-config libpcre2-dev ca-certificates"; \
    if [ "$TARGETARCH" != "$BUILDARCH" ]; then \
      [ "$BUILDARCH/$TARGETARCH" = "amd64/arm64" ] || { echo "cross build only amd64 -> arm64"; exit 1; }; \
      dpkg --add-architecture arm64; pkgs="$pkgs g++-aarch64-linux-gnu libpcre2-dev:arm64 qemu-user"; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends $pkgs && rm -rf /var/lib/apt/lists/*
WORKDIR /src

FROM deps AS build
ARG TARGETARCH
ARG BUILDARCH
COPY CMakeLists.txt .
COPY cmake cmake
COPY src src
COPY tests tests
RUN set -e; tc=""; cxx=g++; \
    if [ "$TARGETARCH" != "$BUILDARCH" ]; then tc="-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake"; cxx=aarch64-linux-gnu-g++; fi; \
    cmake -B build -G Ninja $tc && cmake --build build; \
    mkdir -p /out && cp build/malkin /out/ && cp -L "$($cxx -print-file-name=libgomp.so.1)" /out/

# Runs the kernel tests (through qemu for arm64): docker build --target test .
FROM build AS test
RUN ctest --test-dir build --output-on-failure

# COPY only, no RUN: the target-arch stage never executes code at build time.
FROM debian:trixie-slim AS runtime
COPY --from=build /out/libgomp.so.1 /usr/local/lib/malkin/
COPY --from=build /out/malkin /usr/local/bin/malkin
ENV LD_LIBRARY_PATH=/usr/local/lib/malkin
EXPOSE 8080
ENTRYPOINT ["malkin"]
CMD ["--help"]
