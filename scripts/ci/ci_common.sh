#!/bin/bash
set -euo pipefail

# Common environment variables
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC

# Build configuration
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CTEST_PARALLEL_LEVEL=2

# Limit memory usage
export NINJA_STATUS="[%f/%t %o/sec] "
export NINJA_FLAGS="-j2"

# Print build information
print_build_info() {
    echo "=== Build Information ==="
    echo "Build OS: $(uname -s)"
    echo "Build Arch: $(uname -m)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not installed')"
    echo "CMake: $(cmake --version 2>/dev/null | head -n1 || echo 'Not installed')"
    echo "GCC: $(gcc --version 2>/dev/null | head -n1 || echo 'Not installed')"
    echo "CUDA: $(nvcc --version 2>/dev/null | head -n1 || echo 'Not installed')
    echo "========================="
}

# Run CMake configuration
configure_cmake() {
    local build_type=${1:-Release}
    local build_dir=${2:-build}
    local extra_args="${@:3}"

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_C_STANDARD=11 \
        $extra_args

    cd ..
}

# Build the project
build_project() {
    local build_dir=${1:-build}

    cmake --build "$build_dir" --config Release --target all -- -j2
}

# Run tests
run_tests() {
    local build_dir=${1:-build}

    cd "$build_dir"
    ctest --output-on-failure
    cd ..
}

# Main build function
main() {
    print_build_info

    # Install dependencies
    ./scripts/ci/install_dependencies.sh

    # Configure and build
    configure_cmake "Release" "build" "$@"
    build_project "build"

    # Run tests
    run_tests "build"
}

# Run the main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
