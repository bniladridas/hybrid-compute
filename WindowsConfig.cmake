# Windows-specific configuration Benchmarks are enabled on Windows with dynamic MSVC runtime to avoid conflicts.
set(ENABLE_BENCHMARK
    ON
    CACHE BOOL "Enable benchmarks on Windows" FORCE)

# Set MSVC runtime to dynamic to avoid linking conflicts with benchmark and gtest
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
