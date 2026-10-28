# Cross build for Neoverse-N1 from x86-64; ctest runs through qemu-user.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(ENV{PKG_CONFIG_LIBDIR} /usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig)
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64;-cpu;neoverse-n1;-L;/usr/aarch64-linux-gnu)
