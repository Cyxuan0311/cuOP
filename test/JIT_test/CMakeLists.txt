# JIT系统测试CMakeLists.txt

# JIT系统基础测试
add_executable(test_jit_system test_jit_system.cpp)
target_link_libraries(test_jit_system PRIVATE 
    cuop_jit 
    cuop_cublas 
    Glog::glog
)

# 设置测试属性
set_tests_properties(test_jit_system PROPERTIES
    TIMEOUT 300
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

# 添加测试
add_test(NAME JIT_System_Test COMMAND test_jit_system)

# 安装测试程序
install(TARGETS test_jit_system
    RUNTIME DESTINATION bin/tests/jit
)

# 复制测试数据（如果有的话）
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_data/* 
#         DESTINATION share/tests/jit/test_data) 