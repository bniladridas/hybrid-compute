# Disable benchmarks on Windows due to linking issues with Google Benchmark
set(ENABLE_BENCHMARK
    OFF
    CACHE BOOL "Disable benchmarks on Windows" FORCE)

# Use static MSVC runtime to avoid linking conflicts
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
