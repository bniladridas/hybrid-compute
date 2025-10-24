# Add Google Test
include(FetchContent)

# Add Google Test (using a more recent version)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Add Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

FetchContent_MakeAvailable(googletest benchmark)

# Add test for Metal shim (only if shim is available)
if(TARGET cudart_shim)
    add_executable(test_metal_shim
        unit/test_metal_shim.cpp
    )

    # Link against the Metal shim library and Google Test
    target_link_libraries(test_metal_shim PRIVATE
        cudart_shim
        gtest_main
        ${METAL_FRAMEWORKS}
        ${OpenCV_LIBS}
    )
endif()

# Add benchmark executable (only if shim is available)
if(TARGET cudart_shim)
    add_executable(benchmark_metal_shim
        performance/benchmark_metal_shim.cpp
    )

    # Link benchmark executable
    target_link_libraries(benchmark_metal_shim PRIVATE
        cudart_shim
        benchmark::benchmark
        ${METAL_FRAMEWORKS}
    )
endif()

# Link against the Metal shim library and Google Test
target_link_libraries(test_metal_shim PRIVATE
    cudart_shim
    gtest_main
    ${METAL_FRAMEWORKS}
    ${OpenCV_LIBS}
)

# Link benchmark executable
target_link_libraries(benchmark_metal_shim PRIVATE
    cudart_shim
    benchmark::benchmark
    ${METAL_FRAMEWORKS}
)

# Set C++ standard
target_compile_features(test_metal_shim PRIVATE cxx_std_17)
target_compile_features(benchmark_metal_shim PRIVATE cxx_std_17)

# Set include directories
target_include_directories(test_metal_shim PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers
    ${gtest_SOURCE_DIR}/googletest/include
    ${gtest_SOURCE_DIR}/googlemock/include
)

target_include_directories(benchmark_metal_shim PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers
)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_metal_shim PRIVATE
        -fobjc-arc
        -fmodules
        -fcxx-modules
    )
endif()

# Add test
if(TARGET test_metal_shim)
    add_test(NAME test_metal_shim COMMAND test_metal_shim)
endif()

# Add test data
set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_data)
file(MAKE_DIRECTORY ${TEST_DATA_DIR})

# Add a custom command to generate test data
add_custom_command(
    TARGET test_metal_shim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DATA_DIR}
    COMMENT "Creating test data directory"
)

# Enable testing
enable_testing()
