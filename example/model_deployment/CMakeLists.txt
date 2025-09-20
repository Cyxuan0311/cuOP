cmake_minimum_required(VERSION 3.18)
project(ModelDeployment)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 模型部署示例
add_executable(cnn_deployment cnn_deployment.cpp)

# 设置包含目录
target_include_directories(cnn_deployment PRIVATE
    /mnt/f/My__StudyStack/My_Project/cu_op_mem/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接目录
target_link_directories(cnn_deployment PRIVATE
    /mnt/f/My__StudyStack/My_Project/cu_op_mem/lib
    ${CUDA_LIBRARIES}
)

# 链接必要的库
target_link_libraries(cnn_deployment
    cuop_cublas
    cuop_cudnn
    cudart
    cublas
    cudnn
    glog
    ${OpenCV_LIBS}
)

# 设置编译选项
target_compile_features(cnn_deployment PRIVATE cxx_std_17)
