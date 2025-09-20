# 性能基准测试
add_executable(performance_test performance_test.cpp)
target_link_libraries(performance_test 
    cuop_cublas 
    cuop_cudnn 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

