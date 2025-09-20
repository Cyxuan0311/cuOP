# 深度学习示例
add_executable(relu_demo relu_demo.cpp)
target_link_libraries(relu_demo 
    cuop_cudnn 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(softmax_demo softmax_demo.cpp)
target_link_libraries(softmax_demo 
    cuop_cudnn 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(conv2d_demo conv2d_demo.cpp)
target_link_libraries(conv2d_demo 
    cuop_cudnn 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

