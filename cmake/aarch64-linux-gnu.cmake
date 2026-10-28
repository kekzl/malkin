# Cross build for Neoverse-N1 from x86-64 with clang (Debian multiarch, sysroot /); ctest runs through qemu-user.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CXX_COMPILER clang++-$ENV{LLVM})
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(ENV{PKG_CONFIG_LIBDIR} /usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig)
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64;-cpu;neoverse-n1)
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-stdlib=libc++ -fuse-ld=lld")
