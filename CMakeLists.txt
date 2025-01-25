cmake_minimum_required(VERSION 3.10)
project(HybridCompute)

set(CUDAToolkit_ROOT /usr/local/cuda)  # Update this line with the correct path

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS})

add_executable(upscale cloud_gpu/upscale.cu)
target_link_libraries(upscale ${OpenCV_LIBS} CUDA::cudart)