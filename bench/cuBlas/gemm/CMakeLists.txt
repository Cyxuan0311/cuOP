cmake_minimum_required(VERSION 3.18)  # 确保支持CUDA 12.8及compute_89架构
project(bench_gemm LANGUAGES CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 解决CMake策略问题并设置CUDA架构
cmake_policy(SET CMP0104 NEW)  # 解决CUDA_ARCHITECTURES空警告
set(CMAKE_CUDA_ARCHITECTURES 89)  # 明确指定RTX 4050的计算能力8.9
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")

# 强制要求CUDA版本≥12.8
find_package(CUDA 12.8 REQUIRED)
message(STATUS "Found CUDA: ${CUDA_VERSION}")

# 查找Glog库
find_package(Glog REQUIRED)
if(NOT GLOG_LIBRARIES)
    set(GLOG_LIBRARIES -lglog)
endif()

# 添加包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/../../../include
    ${CUDA_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(
    bench_gemm
    bench_gemm.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/cuda_op/detail/cuBlas/gemm.cu
    ${CMAKE_SOURCE_DIR}/../../../src/base/memory_pool.cu
)

# 链接库文件
target_link_libraries(bench_gemm
    ${CUDA_LIBRARIES}
    cublas
    cudart
    ${GLOG_LIBRARIES}
)

# 打印CUDA架构配置（用于调试）
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")