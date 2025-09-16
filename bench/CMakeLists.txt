cmake_minimum_required(VERSION 3.18)
project(cu_op_mem_benchmarks LANGUAGES CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 解决CMake策略问题并设置CUDA架构
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 89)
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
    ${CMAKE_SOURCE_DIR}/../include
    ${CUDA_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

# 添加cuBlas基准测试子目录
add_subdirectory(cuBlas/gemm)
add_subdirectory(cuBlas/gemv)
add_subdirectory(cuBlas/scal)
add_subdirectory(cuBlas/axpy)
add_subdirectory(cuBlas/copy)
add_subdirectory(cuBlas/dot)
add_subdirectory(cuBlas/symm)
add_subdirectory(cuBlas/trsm)

# 添加cuDNN基准测试子目录
add_subdirectory(cuDNN/Relu)
add_subdirectory(cuDNN/Softmax)
add_subdirectory(cuDNN/BatchNorm)
add_subdirectory(cuDNN/LayerNorm)
add_subdirectory(cuDNN/Convolution)
add_subdirectory(cuDNN/MaxPool)
add_subdirectory(cuDNN/AveragePool)
add_subdirectory(cuDNN/MatMul)

# 打印CUDA架构配置（用于调试）
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# 创建基准测试运行脚本
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_all_benchmarks.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/run_all_benchmarks.sh
    @ONLY
)

# 设置运行脚本的执行权限
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/run_all_benchmarks.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
)
