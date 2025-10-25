# Windows-specific configuration To activate benchmarks on Windows, change OFF
# to ON below. Note: May cause MSVC runtime conflicts; test carefully.
set(ENABLE_BENCHMARK
    ON
    CACHE BOOL "Enable benchmarks on Windows" FORCE)
