# Windows-specific configuration Benchmarks are disabled on Windows due to linking issues with Google Benchmark tests.
set(ENABLE_BENCHMARK
    OFF
    CACHE BOOL "Disable benchmarks on Windows" FORCE)

# Set MSVC runtime to static to avoid linking conflicts with benchmark and gtest
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
