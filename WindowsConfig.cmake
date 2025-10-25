# Windows-specific configuration To activate benchmarks on Windows, change OFF
# to ON below. Note: May cause MSVC runtime conflicts; test carefully.
set(ENABLE_BENCHMARK
    ON
    CACHE BOOL "Enable benchmarks on Windows" FORCE)

# Set MSVC runtime to static to avoid linking conflicts with benchmark and gtest
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
