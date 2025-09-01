# 错误码系统测试CMakeLists.txt

# 错误码系统测试
add_executable(test_status_code test_status_code.cpp)
target_link_libraries(test_status_code PRIVATE 
    GTest::gtest 
    GTest::gtest_main
)

# 设置测试属性
set_tests_properties(test_status_code PROPERTIES
    TIMEOUT 60
)

# 添加测试
add_test(NAME Status_Code_Test COMMAND test_status_code)

# 安装测试程序
install(TARGETS test_status_code
    RUNTIME DESTINATION bin/tests/util
) 