# 基础操作示例
add_executable(basic_gemm basic_gemm.cpp)
target_link_libraries(basic_gemm 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(basic_gemv basic_gemv.cpp)
target_link_libraries(basic_gemv 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(basic_dot basic_dot.cpp)
target_link_libraries(basic_dot 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

add_executable(basic_axpy basic_axpy.cpp)
target_link_libraries(basic_axpy 
    cuop_cublas 
    cuop_jit 
    cuop_performance 
    glog 
    cuda 
    cudart
)

