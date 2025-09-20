cmake_minimum_required(VERSION 3.16)
project(cuOP_Examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# 设置CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/build)
include_directories(${CUDA_INCLUDE_DIRS})

# 链接目录
link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(${CUDA_LIBRARIES})

# 添加子目录
add_subdirectory(basic_operations)
add_subdirectory(deep_learning)
add_subdirectory(performance_benchmark)
add_subdirectory(advanced_features)
add_subdirectory(model_inference)

# 创建示例目标
add_custom_target(examples ALL)
add_dependencies(examples 
    basic_gemm
    basic_gemv
    basic_dot
    basic_axpy
    relu_demo
    softmax_demo
    conv2d_demo
    performance_test
    jit_demo
    memory_pool_demo
    simple_cnn_inference
)

