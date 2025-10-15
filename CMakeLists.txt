cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)
project(hybrid-compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add conda environment to CMAKE_PREFIX_PATH for OpenCV
if(APPLE AND DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit)
find_package(OpenMP)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS})
target_include_directories(preprocess PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    add_executable(upscaler cloud_gpu/upscale.cu)
    target_link_libraries(upscaler CUDA::cudart ${OpenCV_LIBS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(upscaler OpenMP::OpenMP_CXX)
    endif()
endif()

# Unit tests
enable_testing()
add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils ${OpenCV_LIBS})
target_include_directories(test_utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_utils COMMAND test_utils)

add_executable(test_upscaler tests/test_upscaler.cpp)
target_link_libraries(test_upscaler ${OpenCV_LIBS})
target_include_directories(test_upscaler PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_upscaler COMMAND test_upscaler)