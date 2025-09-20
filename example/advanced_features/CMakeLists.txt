# 高级功能示例
add_executable(jit_demo jit_demo.cpp)
target_link_libraries(jit_demo 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(memory_pool_demo memory_pool_demo.cpp)
target_link_libraries(memory_pool_demo 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

