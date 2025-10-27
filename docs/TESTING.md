# Testing and Benchmarking Guide

This document provides information on how to run tests and benchmarks for the Metal shim implementation.

## Prerequisites

- Xcode command line tools
- CMake 3.10+
- C++17 compatible compiler
- Metal-capable macOS device

## Running Tests

### Unit Tests

To build and run the unit tests:

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make
ctest --output-on-failure
```

### Performance Benchmarks

To run the performance benchmarks:

```bash
cd build
./tests/benchmark_metal_shim --benchmark_min_time=1s
```

### Test Coverage

To generate a test coverage report (requires `gcov` and `lcov`):

```bash
mkdir -p build_coverage && cd build_coverage
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
make
test_metal_shim
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage
open coverage/index.html
```

## Writing Tests

### Unit Tests

- Place unit tests in `tests/unit/`
- Use Google Test framework
- Test files should be named `test_*.cpp`
- Test cases should be small and focused

### Benchmark Tests

- Place benchmark tests in `tests/performance/`
- Use Google Benchmark framework
- Test files should be named `benchmark_*.cpp`
- Include a range of input sizes

## CI/CD Integration

Tests are automatically run on pull requests and pushes to the main branch. The CI pipeline includes:

- Build verification
- Unit tests
- Code style checks
- Documentation generation

## Performance Profiling

To profile the Metal shim:

1. Use Xcode's Instruments
2. Select the Time Profiler
3. Run your benchmark or test
4. Analyze the results

## Memory Management

Use the following tools to check for memory issues:

- Xcode's Memory Graph Debugger
- Address Sanitizer (add `-fsanitize=address` to compiler flags)
- Leak Sanitizer (add `-fsanitize=leak` to compiler flags)

## Troubleshooting

### Test Executable Not Found

If `ctest` reports "Could not find executable", build first:

```bash
cmake --build build
cd build && ctest -R user_counters_tabular_test -V
```

## Best Practices

- Write tests for all new features
- Update tests when modifying existing code
- Keep tests independent and isolated
- Use meaningful test names
- Include assertions for expected behavior
- Test edge cases and error conditions
- Document test assumptions and requirements
