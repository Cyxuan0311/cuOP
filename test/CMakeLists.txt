cmake_minimum_required(VERSION 3.10)
project(cu_op_mem_tests LANGUAGES CXX CUDA)

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
    ${CMAKE_SOURCE_DIR}/../include
)

# 添加cuBlas单元测试子目录
add_subdirectory(cuBlas/test_gemm)
add_subdirectory(cuBlas/test_gemv)
add_subdirectory(cuBlas/test_scal)
add_subdirectory(cuBlas/test_axpy)
add_subdirectory(cuBlas/test_copy)
add_subdirectory(cuBlas/test_dot)
add_subdirectory(cuBlas/test_symm)
add_subdirectory(cuBlas/test_trsm)

# 添加cuDNN单元测试子目录
add_subdirectory(cuDNN/test_relu)
add_subdirectory(cuDNN/test_softmax)
add_subdirectory(cuDNN/test_maxpool2D)
add_subdirectory(cuDNN/test_averagepool2D)
add_subdirectory(cuDNN/test_globalmaxpool2D)
add_subdirectory(cuDNN/test_globalaverpool2D)
add_subdirectory(cuDNN/test_matmul)
add_subdirectory(cuDNN/test_batchnorm)
add_subdirectory(cuDNN/test_layernorm)
add_subdirectory(cuDNN/test_convolution)
add_subdirectory(cuDNN/test_flatten)
add_subdirectory(cuDNN/test_view)

# 添加其他测试子目录（如果存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JIT_test")
    add_subdirectory(JIT_test)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/performance")
    add_subdirectory(performance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/util")
    add_subdirectory(util)
endif()

enable_testing()

# 创建运行所有测试的脚本
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/run_all_tests.sh
    @ONLY
)

# 设置运行脚本的执行权限
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/run_all_tests.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE
                      WORLD_READ WORLD_EXECUTE
)
