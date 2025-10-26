# Windows-specific configuration Benchmarks are enabled on Windows with static MSVC runtime to avoid conflicts.
set(ENABLE_BENCHMARK
    ON
    CACHE BOOL "Enable benchmarks on Windows" FORCE)

# Set MSVC runtime to static to avoid linking conflicts with benchmark and gtest
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
