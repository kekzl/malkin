cmake_minimum_required(VERSION 4.0)
project(malkin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)  # no modules; the scan adds @modmap flags clang-tidy cannot read
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Primary toolchain: clang + libc++ + lld (Dockerfile pins LLVM 23). GCC 16 builds too, for comparison.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++)
  set(CMAKE_LINKER_TYPE LLD)
endif()

# Two ISAs, no generic fallback:
#   x86-64:  AVX-512 (x86-64-v4) + VNNI + BF16, Zen 4 / Zen 5 / Sapphire Rapids and newer
#   aarch64: Armv8.2-A + dotprod + fp16, Neoverse-N1 and newer
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(MALKIN_ARCH_FLAGS -march=x86-64-v4 -mavx512vnni -mavx512bf16 -mtune=znver4)
  set(MALKIN_KERNELS src/kernels_avx512.cpp src/gemm_avx512.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(MALKIN_ARCH_FLAGS -mcpu=neoverse-n1)
  set(MALKIN_KERNELS src/kernels_neon.cpp)
else()
  message(FATAL_ERROR "unsupported CPU architecture ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

include(FetchContent)
FetchContent_Declare(httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.63.0.tar.gz
  URL_HASH SHA256=44ab42e16926e95f7641d286bc90d739ab1c16886114d7cc5bc7a9f905c5edf8
  SOURCE_SUBDIR none)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
  URL_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
  DOWNLOAD_NO_EXTRACT TRUE
  SOURCE_SUBDIR none)
FetchContent_MakeAvailable(httplib nlohmann_json)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/deps/nlohmann)
file(COPY ${nlohmann_json_SOURCE_DIR}/json.hpp DESTINATION ${CMAKE_BINARY_DIR}/deps/nlohmann)

add_library(malkin_core STATIC
  src/cpu.cpp
  src/gguf.cpp
  src/quant.cpp
  src/tokenizer.cpp
  src/chat.cpp
  src/model.cpp
  src/sampler.cpp
  src/engine.cpp
  ${MALKIN_KERNELS})
target_include_directories(malkin_core PUBLIC src)
target_compile_options(malkin_core PUBLIC ${MALKIN_ARCH_FLAGS} -Wall -Wextra)
target_link_libraries(malkin_core PUBLIC OpenMP::OpenMP_CXX PkgConfig::PCRE2)

add_executable(malkin src/main.cpp src/server.cpp)
target_include_directories(malkin PRIVATE ${httplib_SOURCE_DIR} ${CMAKE_BINARY_DIR}/deps)
set(MALKIN_VERSION_STRING "${PROJECT_VERSION}" CACHE STRING "version shown by malkin --help")
if(MALKIN_VERSION_STRING STREQUAL "dev" OR MALKIN_VERSION_STRING STREQUAL "")
  set(MALKIN_VERSION_STRING "${PROJECT_VERSION}-dev" CACHE STRING "" FORCE)
endif()
target_compile_definitions(malkin PRIVATE MALKIN_VERSION="${MALKIN_VERSION_STRING}")
target_link_libraries(malkin PRIVATE malkin_core)

enable_testing()
add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE malkin_core)
add_test(NAME quant COMMAND test_quant)
set_tests_properties(quant PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE malkin_core)
add_test(NAME sampler COMMAND test_sampler)
set_tests_properties(sampler PROPERTIES SKIP_RETURN_CODE 77)
add_executable(bench_gemm tests/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE malkin_core)

add_executable(bench_layers tests/bench_layers.cpp)
target_link_libraries(bench_layers PRIVATE malkin_core)

install(TARGETS malkin RUNTIME DESTINATION bin)
