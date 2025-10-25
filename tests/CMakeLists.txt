# Add Google Test
include(FetchContent)

# Set platform-specific args
if(WIN32)
  set(MSVC_ARGS -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
                -DCMAKE_CXX_FLAGS_RELEASE="/MT")
else()
  set(MSVC_ARGS "")
endif()

# Add Google Test (using a more recent version)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  CMAKE_ARGS ${MSVC_ARGS})

# Add Google Benchmark (conditionally)
if(ENABLE_BENCHMARK)
  # Set benchmark testing flags based on options
  if(ENABLE_BENCHMARK_TESTING)
    set(BENCHMARK_TESTING_FLAG -DBENCHMARK_ENABLE_TESTING=ON)
  else()
    set(BENCHMARK_TESTING_FLAG -DBENCHMARK_ENABLE_TESTING=OFF)
  endif()

  if(ENABLE_BENCHMARK_GTEST_TESTS)
    set(BENCHMARK_GTEST_FLAG -DBENCHMARK_ENABLE_GTEST_TESTS=ON)
  else()
    set(BENCHMARK_GTEST_FLAG -DBENCHMARK_ENABLE_GTEST_TESTS=OFF)
  endif()

  if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()

  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
    CMAKE_ARGS ${MSVC_ARGS} ${BENCHMARK_TESTING_FLAG} ${BENCHMARK_GTEST_FLAG})

  FetchContent_MakeAvailable(googletest benchmark)
endif()

# Add test for Metal shim (only if shim is available)
if(TARGET cudart_shim)
  add_executable(test_metal_shim unit/test_metal_shim.cpp)

  # Link against the Metal shim library and Google Test
  target_link_libraries(
    test_metal_shim PRIVATE cudart_shim gtest_main ${METAL_FRAMEWORKS}
                            ${OpenCV_LIBS})

  # Set C++ standard
  target_compile_features(test_metal_shim PRIVATE cxx_std_17)

  # Set include directories
  target_include_directories(
    test_metal_shim
    PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers
      ${gtest_SOURCE_DIR}/googletest/include
      ${gtest_SOURCE_DIR}/googlemock/include)

  # Set compiler flags
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test_metal_shim PRIVATE -fobjc-arc -fmodules
                                                   -fcxx-modules)
  endif()

  # Add test
  add_test(NAME test_metal_shim COMMAND $<TARGET_FILE:test_metal_shim>)

  # Add test data
  set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_data)
  file(MAKE_DIRECTORY ${TEST_DATA_DIR})

  # Add a custom command to generate test data
  add_custom_command(
    TARGET test_metal_shim
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DATA_DIR}
    COMMENT "Creating test data directory")
endif()

# Add benchmark executable (only if shim and benchmarks are enabled)
if(ENABLE_BENCHMARK AND TARGET cudart_shim)
  add_executable(benchmark_metal_shim performance/benchmark_metal_shim.cpp)

  # Link benchmark executable
  target_link_libraries(
    benchmark_metal_shim PRIVATE cudart_shim benchmark::benchmark
                                 ${METAL_FRAMEWORKS})

  # Set C++ standard
  target_compile_features(benchmark_metal_shim PRIVATE cxx_std_17)

  # Set include directories
  target_include_directories(
    benchmark_metal_shim
    PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers)
endif()
