cmake_minimum_required(VERSION 3.10)
project(test_gemm LANGUAGES CXX CUDA)

# Set CUDA architectures policy and target
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 89)  # Assuming you're using RTX 4050 (compute_89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")

# Set CUDA as a language and specify standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

# Clean up include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${glog_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../include
)

# Add CUDA source files
set(CUDA_SOURCES
    ../../../../src/cuda_op/detail/cuBlas/gemm.cu
    ../../../../src/base/memory_pool.cu
)

# Create executable with both C++ and CUDA sources
add_executable(test_gemm 
    test_gemm.cpp
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(test_gemm
    ${CUDA_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
    glog::glog
    cublas
    cudart
    #cudnn
)

enable_testing()
add_test(NAME test_gemm COMMAND test_gemm)