# Windows-specific configuration
# To activate benchmarks on Windows, change OFF to ON below.
# Note: May cause MSVC runtime conflicts; test carefully.
set(ENABLE_BENCHMARK OFF CACHE BOOL "Disable benchmarks on Windows due to MSVC runtime issues" FORCE)
