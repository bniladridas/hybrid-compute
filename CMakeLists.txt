cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0077 NEW)

# Set MSVC runtime library to static to avoid conflicts
if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Set project name and version
file(READ VERSION VERSION_CONTENT)
string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
string(REPLACE "\n" ";" VERSION_LIST ${VERSION_CONTENT})
list(FILTER VERSION_LIST EXCLUDE REGEX "^#")
list(GET VERSION_LIST 0 PROJECT_VERSION)

if(APPLE)
  project(
    hybrid-compute
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "Hybrid Compute - CUDA to Metal Shim"
    LANGUAGES CXX C OBJC OBJCXX)
else()
  project(
    hybrid-compute
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "Hybrid Compute - CUDA to Metal Shim"
    LANGUAGES CXX C)
  # Disable ObjC compilers on non-Apple platforms
  set(CMAKE_OBJC_COMPILER NOTFOUND)
  set(CMAKE_OBJCXX_COMPILER NOTFOUND)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  include(WindowsConfig.cmake)
endif()
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Include benchmark and CUDA control
include(benchmark.cmake)
include(cuda.cmake)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(APPLE)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  find_library(METALKIT_LIBRARY MetalKit REQUIRED)
  find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
  find_library(IOKIT_LIBRARY IOKit REQUIRED)
  find_library(CORE_GRAPHICS_LIBRARY CoreGraphics REQUIRED)

  # Set Metal framework paths
  set(METAL_FRAMEWORKS
      ${METAL_LIBRARY} ${FOUNDATION_LIBRARY} ${METALKIT_LIBRARY}
      ${QUARTZCORE_LIBRARY} ${IOKIT_LIBRARY} ${CORE_GRAPHICS_LIBRARY})

  # Set include directories
  include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Foundation.framework/Headers)

  # Set framework search paths
  set(CMAKE_FRAMEWORK_PATH ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks)

  # Set compiler and linker flags for Metal
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fmodules -fcxx-modules -mmacosx-version-min=11.0")
  set(CMAKE_OBJCXX_FLAGS
      "${CMAKE_OBJCXX_FLAGS} -fmodules -fcxx-modules -mmacosx-version-min=11.0")

  # Enable Objective-C and Objective-C++ for Metal shim
  enable_language(OBJC)
  enable_language(OBJCXX)

  # Set Objective-C/Objective-C++ standard and properties
  set(CMAKE_OBJC_STANDARD 17)
  set(CMAKE_OBJC_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

  # Set deployment target
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "11.0"
      CACHE STRING "Minimum OS X deployment version")

  # Set framework search path
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
else()
  set(METAL_FRAMEWORKS "")
endif()

# ARC is enabled per-target as needed

# Find and link required frameworks
if(APPLE)
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  find_library(METALKIT_LIBRARY MetalKit REQUIRED)
  find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
  find_library(IOKIT_LIBRARY IOKit REQUIRED)
  find_library(COREGRAPHICS_LIBRARY CoreGraphics REQUIRED)

  # Set Metal framework paths
  set(METAL_FRAMEWORKS
      ${COCOA_LIBRARY}
      ${METAL_LIBRARY}
      ${FOUNDATION_LIBRARY}
      ${METALKIT_LIBRARY}
      ${QUARTZCORE_LIBRARY}
      ${IOKIT_LIBRARY}
      ${COREGRAPHICS_LIBRARY})
endif()

# CUDA support controlled by cuda.cmake
if(USE_CUDA)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    message(STATUS "CUDA found, building with CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    add_compile_options(${CUDA_CUDA_FLAGS})

    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY}
                       ${CUDA_cudadevrt_LIBRARY})

    file(GLOB_RECURSE CUDA_SOURCES "cloud_gpu/*.cu")

    add_library(cuda_objects OBJECT ${CUDA_SOURCES})
    target_include_directories(cuda_objects PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(WITH_OPENCV)
      target_include_directories(cuda_objects PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()

    set_target_properties(
      cuda_objects PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Find OpenMP (only needed for CUDA targets)
    find_package(OpenMP)
    if(OpenMP_FOUND)
      message(STATUS "OpenMP found, enabling OpenMP support")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
      message(STATUS "OpenMP not found, building without OpenMP support")
    endif()
  else()
    message(WARNING "CUDA not found, building without CUDA support")
  endif()
else()
  message(STATUS "CUDA support disabled by user")
endif()

if(APPLE)
  # Enable Objective-C and Objective-C++ for Metal shim
  enable_language(OBJC)
  enable_language(OBJCXX)

  # Set Objective-C/Objective-C++ standard and properties
  set(CMAKE_OBJC_STANDARD 17)
  set(CMAKE_OBJC_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

  # Set deployment target
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "11.0"
      CACHE STRING "Minimum OS X deployment version")

  # Set framework search path
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

  # Set macOS deployment target to ensure compatibility
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "11.0"
      CACHE STRING "Minimum OS X deployment version")

  # Enable ARC for Objective-C++ files
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES")

  # Create output directory for Metal shaders
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

  # Add Metal shader compilation
  set(METAL_SHADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Upscale.metal)

  # Compile Metal shaders into a Metal library
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.metallib
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_FILES} -o
            ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.air
    COMMAND
      xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.air
      -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.metallib
    DEPENDS ${METAL_SHADER_FILES}
    COMMENT "Compiling Metal shaders..."
    VERBATIM)

  # Add the Metal library as a build target
  add_custom_target(
    metal_shaders ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.metallib)

  # Set the Metal library output path
  set(METAL_LIBRARY_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.metallib)

  # Install the Metal library
  install(
    FILES ${METAL_LIBRARY_OUTPUT}
    DESTINATION share/hybrid-compute/shaders
    COMPONENT Runtime)

  # Add Metal include directory
  include_directories(
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers)
  include_directories(
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/MetalKit.framework/Headers)
  include_directories(
    ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/MetalPerformanceShaders.framework/Headers
  )

  # Add Metal compiler flags
  add_compile_definitions(HAVE_METAL=1)

  # Add Metal source files
  set(METAL_SOURCES src/metal/MetalUpscaler.cpp)

  # Create a Metal upscaler library
  add_library(metal_upscaler ${METAL_SOURCES})

  # Set source file properties to ensure proper compilation
  set_source_files_properties(
    ${METAL_SOURCES} PROPERTIES COMPILE_FLAGS "-x objective-c++ -fobjc-arc")

  # Set include directories
  target_include_directories(metal_upscaler
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Link against required frameworks
  target_link_libraries(metal_upscaler PRIVATE -framework Metal -framework
                                               Foundation)

  # Set properties for the target
  set_target_properties(
    metal_upscaler PROPERTIES FRAMEWORK FALSE
                              XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES")

  # Add dependencies
  if(TARGET metal_shaders)
    add_dependencies(metal_upscaler metal_shaders)
  endif()

  # Set Objective-C++ standard
  set_property(TARGET metal_upscaler PROPERTY CXX_STANDARD 17)
  set_property(TARGET metal_upscaler PROPERTY CXX_STANDARD_REQUIRED ON)

  # Add Metal compiler flags
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(metal_upscaler PRIVATE -fobjc-arc -fmodules
                                                  -fcxx-modules)
  endif()

  # Add Metal upscaler to the build
  add_library(upscaler_metal ALIAS metal_upscaler)

  # Add Metal shim implementation
  set(METAL_SHIM_SOURCES src/metal/MetalShim.mm)

  add_library(metal_shim OBJECT ${METAL_SHIM_SOURCES})
  target_include_directories(
    metal_shim
    PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Metal.framework/Headers
      ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/MetalKit.framework/Headers
      ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Foundation.framework/Headers
  )

  # Disable ARC for Metal shim as it uses manual memory management
  set_source_files_properties(
    ${METAL_SHIM_SOURCES}
    PROPERTIES
      COMPILE_FLAGS
      "-fno-objc-arc -fmodules -mmacosx-version-min=11.0 -Wno-unguarded-availability"
  )

  # Set Objective-C++ standard
  set_source_files_properties(${METAL_SHIM_SOURCES} PROPERTIES COMPILE_FLAGS
                                                               "-std=gnu++17")

  # Add Metal shader compilation
  set(METAL_SHADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Upscale.metal)

  # Create output directory for Metal shaders
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

  # Set the Metal library output path
  set(METAL_LIBRARY_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.metallib)

  # Compile Metal shaders into a Metal library
  add_custom_command(
    OUTPUT ${METAL_LIBRARY_OUTPUT}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_FILES} -o
            ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.air
    COMMAND
      xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.air
      -o ${METAL_LIBRARY_OUTPUT}
    DEPENDS ${METAL_SHADER_FILES}
    COMMENT "Compiling Metal shaders..."
    VERBATIM)

  # Add a custom target for the Metal shaders
  add_custom_target(metal_shader_compilation ALL
                    DEPENDS ${METAL_LIBRARY_OUTPUT})

  # Add dependency of metal_shim on the shader compilation
  add_dependencies(metal_shim metal_shader_compilation)

  # Create a shared library for the Metal shim
  add_library(cudart_shim SHARED $<TARGET_OBJECTS:metal_shim>)
  target_link_libraries(
    cudart_shim PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
                        ${METALKIT_LIBRARY})
  set_target_properties(
    cudart_shim
    PROPERTIES OUTPUT_NAME "cudart"
               VERSION ${PROJECT_VERSION}
               SOVERSION 1
               MACOSX_RPATH ON
               INSTALL_RPATH "@loader_path/../lib"
               BUILD_WITH_INSTALL_RPATH TRUE)

  # Add dependency on Metal shaders
  add_dependencies(cudart_shim metal_shaders)

  # Set the output directory for the Metal shim
  set_target_properties(
    cudart_shim
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
               RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

  # Install the Metal shim library
  install(
    TARGETS cudart_shim
    EXPORT hybrid-compute-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include)

  # Create a convenience interface library for the Metal shim
  add_library(cuda_shim INTERFACE)
  target_include_directories(
    cuda_shim INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

  # Create a combined shim library that can be used as a drop-in replacement for
  # CUDA
  add_library(hybrid_compute_shim INTERFACE)
  target_link_libraries(hybrid_compute_shim INTERFACE cuda_shim)

  # Export targets for use in other projects
  install(
    TARGETS cuda_shim hybrid_compute_shim
    EXPORT hybrid-compute-targets
    INCLUDES
    DESTINATION include)

  # Export the Metal shim object library
  install(
    TARGETS metal_shim
    EXPORT hybrid-compute-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include)

  # Export the targets file
  install(
    EXPORT hybrid-compute-targets
    FILE hybrid-compute-targets.cmake
    NAMESPACE hybrid::
    DESTINATION lib/cmake/hybrid-compute-${PROJECT_VERSION})

  # Create and install the config file
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hybrid-compute-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-config.cmake @ONLY)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-config.cmake
          DESTINATION lib/cmake/hybrid-compute-${PROJECT_VERSION})

  # Create and install the version file
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-config-version.cmake
          DESTINATION lib/cmake/hybrid-compute-${PROJECT_VERSION})

  # Create a package configuration file for find_package
  export(
    EXPORT hybrid-compute-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-targets.cmake
    NAMESPACE hybrid::)

  # Install the version file
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hybrid-compute-config-version.cmake
          DESTINATION lib/cmake/hybrid-compute-${PROJECT_VERSION})

  # Create an uninstall target
  if(NOT TARGET uninstall)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake IMMEDIATE @ONLY)

    add_custom_target(
      uninstall
      COMMAND ${CMAKE_COMMAND} -P
              ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
      COMMENT "Uninstalling...")
  endif()

  message(STATUS "Hybrid compute shim configured for Metal")
endif()

# Make OpenCV optional
option(WITH_OPENCV "Build with OpenCV support" OFF)

# Code coverage
option(COVERAGE "Enable code coverage" OFF)

# Add conda environment to CMAKE_PREFIX_PATH for OpenCV
if(WITH_OPENCV)
  if(APPLE AND DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
  endif()
  find_package(OpenCV QUIET)
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found, some features will be disabled")
    set(WITH_OPENCV OFF)
  else()
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
  endif()
endif()

find_package(CUDAToolkit)
find_package(OpenMP)

# Code coverage
if(COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  add_executable(upscale cloud_gpu/upscale.cu)
  target_link_libraries(upscale CUDA::cudart)
  target_include_directories(upscale PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(upscale OpenMP::OpenMP_CXX)
  endif()
  if(WITH_OPENCV)
    target_link_libraries(upscale ${OpenCV_LIBS})
    target_include_directories(upscale PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  # Additional CUDA executables
  add_executable(filters cloud_gpu/filters.cu)
  target_link_libraries(filters CUDA::cudart)
  target_include_directories(filters PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(filters OpenMP::OpenMP_CXX)
  endif()
  if(WITH_OPENCV)
    target_link_libraries(filters ${OpenCV_LIBS})
    target_include_directories(filters PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(rotation cloud_gpu/rotation.cu)
  target_link_libraries(rotation CUDA::cudart)
  target_include_directories(rotation PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(rotation OpenMP::OpenMP_CXX)
  endif()
  if(WITH_OPENCV)
    target_link_libraries(rotation ${OpenCV_LIBS})
    target_include_directories(rotation PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(resize cloud_gpu/resize.cu)
  target_link_libraries(resize CUDA::cudart)
  target_include_directories(resize PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(resize OpenMP::OpenMP_CXX)
  endif()
  if(WITH_OPENCV)
    target_link_libraries(resize ${OpenCV_LIBS})
    target_include_directories(resize PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  # Additional CUDA tools (placeholders)
  add_executable(colorspace cloud_gpu/colorspace.cu)
  target_link_libraries(colorspace CUDA::cudart)
  target_include_directories(colorspace PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(colorspace ${OpenCV_LIBS})
    target_include_directories(colorspace PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(histogram cloud_gpu/histogram.cu)
  target_link_libraries(histogram CUDA::cudart)
  target_include_directories(histogram PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(histogram ${OpenCV_LIBS})
    target_include_directories(histogram PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(morphology cloud_gpu/morphology.cu)
  target_link_libraries(morphology CUDA::cudart)
  target_include_directories(morphology PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(morphology ${OpenCV_LIBS})
    target_include_directories(morphology PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(median cloud_gpu/median.cu)
  target_link_libraries(median CUDA::cudart)
  target_include_directories(median PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(median ${OpenCV_LIBS})
    target_include_directories(median PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(sharpen cloud_gpu/sharpen.cu)
  target_link_libraries(sharpen CUDA::cudart)
  target_include_directories(sharpen PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(sharpen ${OpenCV_LIBS})
    target_include_directories(sharpen PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(threshold cloud_gpu/threshold.cu)
  target_link_libraries(threshold CUDA::cudart)
  target_include_directories(threshold PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(threshold ${OpenCV_LIBS})
    target_include_directories(threshold PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(canny cloud_gpu/canny.cu)
  target_link_libraries(canny CUDA::cudart)
  target_include_directories(canny PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(canny ${OpenCV_LIBS})
    target_include_directories(canny PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()

  add_executable(blend cloud_gpu/blend.cu)
  target_link_libraries(blend CUDA::cudart)
  target_include_directories(blend PRIVATE ${CMAKE_SOURCE_DIR}/include)
  if(WITH_OPENCV)
    target_link_libraries(blend ${OpenCV_LIBS})
    target_include_directories(blend PRIVATE ${OpenCV_INCLUDE_DIRS})
  endif()
endif()

# Add preprocess_c executable (always available, no OpenCV dependency)
add_executable(preprocess_c src/preprocess.c)
target_include_directories(preprocess_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(NOT MSVC)
  target_link_libraries(preprocess_c m)
endif()

# Add test_preprocess executable
add_executable(test_preprocess tests/test_preprocess.c)
target_include_directories(test_preprocess PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(NOT MSVC)
  target_link_libraries(test_preprocess m)
endif()

# Add preprocess library and tests (OpenCV dependent)
if(WITH_OPENCV AND OpenCV_FOUND)
  # Add the preprocess library
  add_library(preprocess src/preprocess.cpp)
  target_link_libraries(preprocess ${OpenCV_LIBS})
  target_include_directories(preprocess PRIVATE ${OpenCV_INCLUDE_DIRS}
                                                ${CMAKE_SOURCE_DIR}/include)

  # Add test_utils if it exists
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_utils.cpp")
    add_executable(test_utils tests/test_utils.cpp)
    target_link_libraries(test_utils ${OpenCV_LIBS})
    target_include_directories(test_utils PRIVATE ${OpenCV_INCLUDE_DIRS}
                                                  ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME test_utils COMMAND $<TARGET_FILE:test_utils>)
  endif()

endif()

add_test(NAME test_preprocess COMMAND $<TARGET_FILE:test_preprocess>)

# Add tests subdirectory if testing is enabled
option(BUILD_TESTING "Build the testing tree." ON)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  enable_testing()
  add_subdirectory(tests)
endif()
