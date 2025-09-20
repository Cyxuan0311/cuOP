cmake_minimum_required(VERSION 3.18)
project(ModelInference)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")

# 模型推理示例
add_executable(simple_cnn_inference simple_cnn_inference.cpp)

# 设置包含目录
target_include_directories(simple_cnn_inference PRIVATE
    /mnt/f/My__StudyStack/My_Project/cu_op_mem/include
    ${CUDA_INCLUDE_DIRS}
)

# 链接目录
target_link_directories(simple_cnn_inference PRIVATE
    /mnt/f/My__StudyStack/My_Project/cu_op_mem/lib
    ${CUDA_LIBRARIES}
)

# 链接必要的库
target_link_libraries(simple_cnn_inference
    cuop_cublas
    cuop_cudnn
    cudart
    cublas
    cudnn
    glog
)

# 设置编译选项
target_compile_features(simple_cnn_inference PRIVATE cxx_std_17)
