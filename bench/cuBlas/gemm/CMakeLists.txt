cmake_minimum_required(VERSION 3.14)
project(bench_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)
if(NOT GLOG_LIBRARIES)
    set(GLOG_LIBRARIES -lglog)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/../../../include
    ${CUDA_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

add_executable(
    bench_gemm
    bench_gemm.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/cuda_op/detail/cuBlas/gemm.cu
    ${CMAKE_SOURCE_DIR}/../../../src/base/memory_pool.cu
)

target_link_libraries(bench_gemm
    ${CUDA_LIBRARIES}
    cublas
    cudart
    ${GLOG_LIBRARIES}
)