# JIT系统测试CMakeLists.txt

# JIT系统基础测试
add_executable(test_jit_system test_jit_system.cpp)
target_link_libraries(test_jit_system PRIVATE 
    cuop_jit 
    cuop_cublas 
    CUDA::cudart
    CUDA::nvrtc
    ${GLOG_LIBRARIES}
)
target_include_directories(test_jit_system PRIVATE ${CMAKE_SOURCE_DIR}/include)

# JIT持久化缓存测试
add_executable(test_persistent_cache test_persistent_cache.cpp)
target_link_libraries(test_persistent_cache PRIVATE 
    cuop_jit 
    cuop_cublas 
    CUDA::cudart
    CUDA::nvrtc
    ${GLOG_LIBRARIES}
    z
    crypto
)
target_include_directories(test_persistent_cache PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 添加测试
add_test(NAME JIT_System_Test COMMAND test_jit_system)
add_test(NAME JIT_Persistent_Cache_Test COMMAND test_persistent_cache)

# 设置测试属性
set_tests_properties(JIT_System_Test PROPERTIES
    TIMEOUT 300
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

set_tests_properties(JIT_Persistent_Cache_Test PROPERTIES
    TIMEOUT 600
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

# 安装测试程序
install(TARGETS test_jit_system test_persistent_cache
    RUNTIME DESTINATION bin/tests/jit
)

# 复制测试数据（如果有的话）
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_data/* 
#         DESTINATION share/tests/jit/test_data) 