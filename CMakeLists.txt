cmake_minimum_required(VERSION 3.10)
project(hybrid-compute)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS})

if(CUDAToolkit_FOUND)
    add_executable(upscale cloud_gpu/upscale.cu)
    target_link_libraries(upscale ${OpenCV_LIBS} CUDA::cudart)
else()
    message(WARNING "CUDA not found. Skipping upscale executable build. Use cloud with CUDA for upscaling.")
endif()

# Unit tests
enable_testing()
add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils ${OpenCV_LIBS})
add_test(NAME test_utils COMMAND test_utils)