cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)
project(hybrid-compute LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add conda environment to CMAKE_PREFIX_PATH for OpenCV
if(APPLE AND DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit)
find_package(OpenMP)

option(CUDA_ARCH "CUDA architecture to target" "75")
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    add_executable(upscale cloud_gpu/upscale.cu)
    target_link_libraries(upscale CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(upscale PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(upscale OpenMP::OpenMP_CXX)
    endif()

    # Additional CUDA executables
    add_executable(filters cloud_gpu/filters.cu)
    target_link_libraries(filters CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(filters PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(filters OpenMP::OpenMP_CXX)
    endif()

    add_executable(rotation cloud_gpu/rotation.cu)
    target_link_libraries(rotation CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(rotation PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rotation OpenMP::OpenMP_CXX)
    endif()

    add_executable(resize cloud_gpu/resize.cu)
    target_link_libraries(resize CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(resize PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(resize OpenMP::OpenMP_CXX)
    endif()

    # Additional CUDA tools (placeholders)
    add_executable(colorspace cloud_gpu/colorspace.cu)
    target_link_libraries(colorspace CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(colorspace PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(histogram cloud_gpu/histogram.cu)
    target_link_libraries(histogram CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(histogram PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(morphology cloud_gpu/morphology.cu)
    target_link_libraries(morphology CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(morphology PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(median cloud_gpu/median.cu)
    target_link_libraries(median CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(median PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(sharpen cloud_gpu/sharpen.cu)
    target_link_libraries(sharpen CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(sharpen PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(threshold cloud_gpu/threshold.cu)
    target_link_libraries(threshold CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(threshold PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(canny cloud_gpu/canny.cu)
    target_link_libraries(canny CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(canny PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

    add_executable(blend cloud_gpu/blend.cu)
    target_link_libraries(blend CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(blend PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
endif()

# Unit tests
enable_testing()
add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils ${OpenCV_LIBS})
target_include_directories(test_utils PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_utils COMMAND test_utils)

add_executable(test_upscaler tests/test_upscaler.cpp)
target_link_libraries(test_upscaler ${OpenCV_LIBS})
target_include_directories(test_upscaler PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_upscaler COMMAND test_upscaler)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS})
target_include_directories(preprocess PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_executable(preprocess_c src/preprocess.c)
target_include_directories(preprocess_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(NOT MSVC)
    target_link_libraries(preprocess_c m)
endif()

add_executable(test_preprocess tests/test_preprocess.c)
target_include_directories(test_preprocess PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(NOT MSVC)
    target_link_libraries(test_preprocess m)
endif()
add_test(NAME test_preprocess COMMAND test_preprocess)
