#include "util/status_code.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

using namespace cu_op_mem;

class StatusCodeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 测试基本错误码功能
TEST_F(StatusCodeTest, BasicStatusCodeTest) {
    // 测试成功状态
    EXPECT_EQ(static_cast<int>(StatusCode::SUCCESS), 0);
    
    // 测试系统错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::SYSTEM_ERROR), 1000);
    EXPECT_LT(static_cast<int>(StatusCode::SYSTEM_ERROR), 2000);
    
    // 测试CUDA错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::CUDA_ERROR), 2000);
    EXPECT_LT(static_cast<int>(StatusCode::CUDA_ERROR), 3000);
    
    // 测试内存错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::MEMORY_ERROR), 3000);
    EXPECT_LT(static_cast<int>(StatusCode::MEMORY_ERROR), 4000);
    
    // 测试验证错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::VALIDATION_ERROR), 4000);
    EXPECT_LT(static_cast<int>(StatusCode::VALIDATION_ERROR), 5000);
    
    // 测试编译错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::COMPILATION_ERROR), 5000);
    EXPECT_LT(static_cast<int>(StatusCode::COMPILATION_ERROR), 6000);
    
    // 测试执行错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::EXECUTION_ERROR), 6000);
    EXPECT_LT(static_cast<int>(StatusCode::EXECUTION_ERROR), 7000);
    
    // 测试JIT错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::JIT_ERROR), 7000);
    EXPECT_LT(static_cast<int>(StatusCode::JIT_ERROR), 8000);
    
    // 测试缓存错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::CACHE_ERROR), 8000);
    EXPECT_LT(static_cast<int>(StatusCode::CACHE_ERROR), 9000);
    
    // 测试插件错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::PLUGIN_ERROR), 9000);
    EXPECT_LT(static_cast<int>(StatusCode::PLUGIN_ERROR), 10000);
    
    // 测试张量错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::TENSOR_ERROR), 10000);
    EXPECT_LT(static_cast<int>(StatusCode::TENSOR_ERROR), 11000);
    
    // 测试算子错误码范围
    EXPECT_GE(static_cast<int>(StatusCode::OPERATOR_ERROR), 11000);
    EXPECT_LT(static_cast<int>(StatusCode::OPERATOR_ERROR), 12000);
}

// 测试错误分类功能
TEST_F(StatusCodeTest, ErrorCategoryTest) {
    // 测试成功分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::SUCCESS), ErrorCategory::SUCCESS);
    
    // 测试系统错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::SYSTEM_ERROR), ErrorCategory::SYSTEM);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::NOT_IMPLEMENTED), ErrorCategory::SYSTEM);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::TIMEOUT), ErrorCategory::SYSTEM);
    
    // 测试CUDA错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CUDA_ERROR), ErrorCategory::CUDA);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CUDA_DRIVER_ERROR), ErrorCategory::CUDA);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CUDA_MEMORY_ERROR), ErrorCategory::CUDA);
    
    // 测试内存错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::MEMORY_ERROR), ErrorCategory::MEMORY);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::OUT_OF_MEMORY), ErrorCategory::MEMORY);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::MEMORY_ALLOCATION_FAILED), ErrorCategory::MEMORY);
    
    // 测试验证错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::VALIDATION_ERROR), ErrorCategory::VALIDATION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::INVALID_ARGUMENT), ErrorCategory::VALIDATION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::INVALID_CONFIGURATION), ErrorCategory::VALIDATION);
    
    // 测试编译错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::COMPILATION_ERROR), ErrorCategory::COMPILATION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::COMPILATION_FAILED), ErrorCategory::COMPILATION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::COMPILATION_TIMEOUT), ErrorCategory::COMPILATION);
    
    // 测试执行错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::EXECUTION_ERROR), ErrorCategory::EXECUTION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::EXECUTION_FAILED), ErrorCategory::EXECUTION);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::EXECUTION_TIMEOUT), ErrorCategory::EXECUTION);
    
    // 测试JIT错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::JIT_ERROR), ErrorCategory::JIT);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::JIT_NOT_INITIALIZED), ErrorCategory::JIT);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::JIT_COMPILATION_ERROR), ErrorCategory::JIT);
    
    // 测试缓存错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CACHE_ERROR), ErrorCategory::CACHE);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CACHE_NOT_FOUND), ErrorCategory::CACHE);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::CACHE_CORRUPTED), ErrorCategory::CACHE);
    
    // 测试插件错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::PLUGIN_ERROR), ErrorCategory::PLUGIN);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::PLUGIN_NOT_FOUND), ErrorCategory::PLUGIN);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::PLUGIN_LOAD_ERROR), ErrorCategory::PLUGIN);
    
    // 测试张量错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::TENSOR_ERROR), ErrorCategory::TENSOR);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::TENSOR_DIMENSION_MISMATCH), ErrorCategory::TENSOR);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::TENSOR_SHAPE_MISMATCH), ErrorCategory::TENSOR);
    
    // 测试算子错误分类
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::OPERATOR_ERROR), ErrorCategory::OPERATOR);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::OPERATOR_NOT_SUPPORTED), ErrorCategory::OPERATOR);
    EXPECT_EQ(ErrorUtils::GetErrorCategory(StatusCode::OPERATOR_INVALID_INPUT), ErrorCategory::OPERATOR);
}

// 测试错误码字符串表示
TEST_F(StatusCodeTest, StatusCodeStringTest) {
    // 测试成功状态
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::SUCCESS), "SUCCESS");
    
    // 测试系统错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::SYSTEM_ERROR), "SYSTEM_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::NOT_IMPLEMENTED), "NOT_IMPLEMENTED");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::TIMEOUT), "TIMEOUT");
    
    // 测试CUDA错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CUDA_ERROR), "CUDA_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CUDA_DRIVER_ERROR), "CUDA_DRIVER_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CUDA_MEMORY_ERROR), "CUDA_MEMORY_ERROR");
    
    // 测试内存错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::MEMORY_ERROR), "MEMORY_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::OUT_OF_MEMORY), "OUT_OF_MEMORY");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::MEMORY_ALLOCATION_FAILED), "MEMORY_ALLOCATION_FAILED");
    
    // 测试验证错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::VALIDATION_ERROR), "VALIDATION_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::INVALID_ARGUMENT), "INVALID_ARGUMENT");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::INVALID_CONFIGURATION), "INVALID_CONFIGURATION");
    
    // 测试编译错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::COMPILATION_ERROR), "COMPILATION_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::COMPILATION_FAILED), "COMPILATION_FAILED");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::COMPILATION_TIMEOUT), "COMPILATION_TIMEOUT");
    
    // 测试执行错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::EXECUTION_ERROR), "EXECUTION_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::EXECUTION_FAILED), "EXECUTION_FAILED");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::EXECUTION_TIMEOUT), "EXECUTION_TIMEOUT");
    
    // 测试JIT错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::JIT_ERROR), "JIT_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::JIT_NOT_INITIALIZED), "JIT_NOT_INITIALIZED");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::JIT_COMPILATION_ERROR), "JIT_COMPILATION_ERROR");
    
    // 测试缓存错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CACHE_ERROR), "CACHE_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CACHE_NOT_FOUND), "CACHE_NOT_FOUND");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::CACHE_CORRUPTED), "CACHE_CORRUPTED");
    
    // 测试插件错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::PLUGIN_ERROR), "PLUGIN_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::PLUGIN_NOT_FOUND), "PLUGIN_NOT_FOUND");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::PLUGIN_LOAD_ERROR), "PLUGIN_LOAD_ERROR");
    
    // 测试张量错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::TENSOR_ERROR), "TENSOR_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::TENSOR_DIMENSION_MISMATCH), "TENSOR_DIMENSION_MISMATCH");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::TENSOR_SHAPE_MISMATCH), "TENSOR_SHAPE_MISMATCH");
    
    // 测试算子错误
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::OPERATOR_ERROR), "OPERATOR_ERROR");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::OPERATOR_NOT_SUPPORTED), "OPERATOR_NOT_SUPPORTED");
    EXPECT_EQ(ErrorUtils::GetStatusCodeString(StatusCode::OPERATOR_INVALID_INPUT), "OPERATOR_INVALID_INPUT");
}

// 测试错误分类字符串表示
TEST_F(StatusCodeTest, ErrorCategoryStringTest) {
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::SUCCESS), "SUCCESS");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::SYSTEM), "SYSTEM");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::CUDA), "CUDA");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::MEMORY), "MEMORY");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::VALIDATION), "VALIDATION");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::COMPILATION), "COMPILATION");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::EXECUTION), "EXECUTION");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::JIT), "JIT");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::CACHE), "CACHE");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::PLUGIN), "PLUGIN");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::TENSOR), "TENSOR");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::OPERATOR), "OPERATOR");
    EXPECT_EQ(ErrorUtils::GetErrorCategoryString(ErrorCategory::UNKNOWN), "UNKNOWN");
}

// 测试错误描述
TEST_F(StatusCodeTest, ErrorDescriptionTest) {
    // 测试成功状态描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::SUCCESS), "Operation completed successfully");
    
    // 测试系统错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::SYSTEM_ERROR), "General system error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::NOT_IMPLEMENTED), "Requested functionality is not implemented");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::TIMEOUT), "Operation timed out");
    
    // 测试CUDA错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CUDA_ERROR), "General CUDA error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CUDA_DRIVER_ERROR), "CUDA driver error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CUDA_MEMORY_ERROR), "CUDA memory operation failed");
    
    // 测试内存错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::MEMORY_ERROR), "General memory error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::OUT_OF_MEMORY), "Insufficient memory available");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::MEMORY_ALLOCATION_FAILED), "Memory allocation failed");
    
    // 测试验证错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::VALIDATION_ERROR), "General validation error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::INVALID_ARGUMENT), "Invalid argument provided");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::INVALID_CONFIGURATION), "Invalid configuration specified");
    
    // 测试编译错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::COMPILATION_ERROR), "General compilation error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::COMPILATION_FAILED), "Compilation process failed");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::COMPILATION_TIMEOUT), "Compilation timed out");
    
    // 测试执行错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::EXECUTION_ERROR), "General execution error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::EXECUTION_FAILED), "Execution process failed");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::EXECUTION_TIMEOUT), "Execution timed out");
    
    // 测试JIT错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::JIT_ERROR), "General JIT system error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::JIT_NOT_INITIALIZED), "JIT system not initialized");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::JIT_COMPILATION_ERROR), "JIT compilation failed");
    
    // 测试缓存错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CACHE_ERROR), "General cache error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CACHE_NOT_FOUND), "Requested cache entry not found");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::CACHE_CORRUPTED), "Cache data is corrupted");
    
    // 测试插件错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::PLUGIN_ERROR), "General plugin error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::PLUGIN_NOT_FOUND), "Requested plugin not found");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::PLUGIN_LOAD_ERROR), "Plugin loading failed");
    
    // 测试张量错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::TENSOR_ERROR), "General tensor error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::TENSOR_DIMENSION_MISMATCH), "Tensor dimension mismatch");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::TENSOR_SHAPE_MISMATCH), "Tensor shape mismatch");
    
    // 测试算子错误描述
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::OPERATOR_ERROR), "General operator error occurred");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::OPERATOR_NOT_SUPPORTED), "Operator not supported for given input");
    EXPECT_EQ(ErrorUtils::GetErrorDescription(StatusCode::OPERATOR_INVALID_INPUT), "Operator received invalid input");
}

// 测试错误严重性判断
TEST_F(StatusCodeTest, ErrorSeverityTest) {
    // 测试严重错误
    EXPECT_TRUE(ErrorUtils::IsCriticalError(StatusCode::OUT_OF_MEMORY));
    EXPECT_TRUE(ErrorUtils::IsCriticalError(StatusCode::CUDA_DRIVER_ERROR));
    EXPECT_TRUE(ErrorUtils::IsCriticalError(StatusCode::CUDA_CONTEXT_ERROR));
    EXPECT_TRUE(ErrorUtils::IsCriticalError(StatusCode::CACHE_CORRUPTED));
    EXPECT_TRUE(ErrorUtils::IsCriticalError(StatusCode::PLUGIN_INCOMPATIBLE));
    
    // 测试非严重错误
    EXPECT_FALSE(ErrorUtils::IsCriticalError(StatusCode::SUCCESS));
    EXPECT_FALSE(ErrorUtils::IsCriticalError(StatusCode::TIMEOUT));
    EXPECT_FALSE(ErrorUtils::IsCriticalError(StatusCode::INVALID_ARGUMENT));
    EXPECT_FALSE(ErrorUtils::IsCriticalError(StatusCode::CACHE_NOT_FOUND));
    
    // 测试可恢复错误
    EXPECT_TRUE(ErrorUtils::IsRecoverableError(StatusCode::TIMEOUT));
    EXPECT_TRUE(ErrorUtils::IsRecoverableError(StatusCode::INTERRUPTED));
    EXPECT_TRUE(ErrorUtils::IsRecoverableError(StatusCode::RESOURCE_UNAVAILABLE));
    EXPECT_TRUE(ErrorUtils::IsRecoverableError(StatusCode::CACHE_NOT_FOUND));
    EXPECT_TRUE(ErrorUtils::IsRecoverableError(StatusCode::EXECUTION_TIMEOUT));
    
    // 测试不可恢复错误
    EXPECT_FALSE(ErrorUtils::IsRecoverableError(StatusCode::SUCCESS));
    EXPECT_FALSE(ErrorUtils::IsRecoverableError(StatusCode::OUT_OF_MEMORY));
    EXPECT_FALSE(ErrorUtils::IsRecoverableError(StatusCode::CUDA_DRIVER_ERROR));
    EXPECT_FALSE(ErrorUtils::IsRecoverableError(StatusCode::INVALID_ARGUMENT));
}

// 测试Status类
TEST_F(StatusCodeTest, StatusClassTest) {
    // 测试成功状态
    Status success_status = ErrorUtils::Success();
    EXPECT_TRUE(success_status.IsSuccess());
    EXPECT_FALSE(success_status.IsError());
    EXPECT_EQ(success_status.GetCode(), StatusCode::SUCCESS);
    EXPECT_EQ(success_status.GetCategory(), ErrorCategory::SUCCESS);
    EXPECT_EQ(success_status.ToString(), "SUCCESS");
    
    // 测试错误状态
    Status error_status = ErrorUtils::Error(StatusCode::INVALID_ARGUMENT, "Invalid input", "TestFunction", "test.cpp", 42);
    EXPECT_FALSE(error_status.IsSuccess());
    EXPECT_TRUE(error_status.IsError());
    EXPECT_EQ(error_status.GetCode(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(error_status.GetCategory(), ErrorCategory::VALIDATION);
    EXPECT_EQ(error_status.GetMessage(), "Invalid input");
    EXPECT_EQ(error_status.GetFunction(), "TestFunction");
    EXPECT_EQ(error_status.GetFile(), "test.cpp");
    EXPECT_EQ(error_status.GetLine(), 42);
    
    // 测试布尔转换
    EXPECT_TRUE(static_cast<bool>(success_status));
    EXPECT_FALSE(static_cast<bool>(error_status));
    
    // 测试详细错误信息
    std::string detailed = error_status.ToDetailedString();
    EXPECT_NE(detailed.find("ERROR["), std::string::npos);
    EXPECT_NE(detailed.find("Invalid input"), std::string::npos);
    EXPECT_NE(detailed.find("TestFunction"), std::string::npos);
    EXPECT_NE(detailed.find("test.cpp:42"), std::string::npos);
}

// 测试工具函数
TEST_F(StatusCodeTest, UtilityFunctionsTest) {
    // 测试系统错误创建
    Status system_error = ErrorUtils::SystemError("System failure", "TestFunction", "test.cpp", 100);
    EXPECT_EQ(system_error.GetCode(), StatusCode::SYSTEM_ERROR);
    EXPECT_EQ(system_error.GetCategory(), ErrorCategory::SYSTEM);
    EXPECT_EQ(system_error.GetMessage(), "System failure");
    
    // 测试验证错误创建
    Status validation_error = ErrorUtils::ValidationError("Invalid parameter", "TestFunction", "test.cpp", 101);
    EXPECT_EQ(validation_error.GetCode(), StatusCode::VALIDATION_ERROR);
    EXPECT_EQ(validation_error.GetCategory(), ErrorCategory::VALIDATION);
    EXPECT_EQ(validation_error.GetMessage(), "Invalid parameter");
    
    // 测试内存错误创建
    Status memory_error = ErrorUtils::MemoryError("Memory allocation failed", "TestFunction", "test.cpp", 102);
    EXPECT_EQ(memory_error.GetCode(), StatusCode::MEMORY_ERROR);
    EXPECT_EQ(memory_error.GetCategory(), ErrorCategory::MEMORY);
    EXPECT_EQ(memory_error.GetMessage(), "Memory allocation failed");
    
    // 测试编译错误创建
    Status compilation_error = ErrorUtils::CompilationError("Compilation failed", "TestFunction", "test.cpp", 103);
    EXPECT_EQ(compilation_error.GetCode(), StatusCode::COMPILATION_ERROR);
    EXPECT_EQ(compilation_error.GetCategory(), ErrorCategory::COMPILATION);
    EXPECT_EQ(compilation_error.GetMessage(), "Compilation failed");
    
    // 测试执行错误创建
    Status execution_error = ErrorUtils::ExecutionError("Execution failed", "TestFunction", "test.cpp", 104);
    EXPECT_EQ(execution_error.GetCode(), StatusCode::EXECUTION_ERROR);
    EXPECT_EQ(execution_error.GetCategory(), ErrorCategory::EXECUTION);
    EXPECT_EQ(execution_error.GetMessage(), "Execution failed");
}

// 测试流操作符
TEST_F(StatusCodeTest, StreamOperatorTest) {
    std::stringstream ss;
    
    // 测试StatusCode流操作符
    ss << StatusCode::SUCCESS;
    EXPECT_EQ(ss.str(), "SUCCESS");
    
    ss.str("");
    ss << StatusCode::INVALID_ARGUMENT;
    EXPECT_EQ(ss.str(), "INVALID_ARGUMENT");
    
    // 测试ErrorCategory流操作符
    ss.str("");
    ss << ErrorCategory::SYSTEM;
    EXPECT_EQ(ss.str(), "SYSTEM");
    
    ss.str("");
    ss << ErrorCategory::CUDA;
    EXPECT_EQ(ss.str(), "CUDA");
    
    // 测试Status流操作符
    ss.str("");
    Status status = ErrorUtils::Error(StatusCode::TIMEOUT, "Operation timed out", "TestFunction");
    ss << status;
    EXPECT_NE(ss.str().find("TIMEOUT"), std::string::npos);
}

// 测试向后兼容性
TEST_F(StatusCodeTest, BackwardCompatibilityTest) {
    // 测试旧错误码仍然可用
    EXPECT_EQ(static_cast<int>(StatusCode::SHAPE_MISMATCH), static_cast<int>(StatusCode::TENSOR_SHAPE_MISMATCH));
    EXPECT_EQ(static_cast<int>(StatusCode::UNSUPPORTED_TYPE), static_cast<int>(StatusCode::OPERATOR_NOT_SUPPORTED));
    EXPECT_EQ(static_cast<int>(StatusCode::UNKNOWN_ERROR), static_cast<int>(StatusCode::SYSTEM_ERROR));
    EXPECT_EQ(static_cast<int>(StatusCode::TENSOR_DIMONSION_MISMATCH), static_cast<int>(StatusCode::TENSOR_DIMENSION_MISMATCH));
    EXPECT_EQ(static_cast<int>(StatusCode::NOT_INITIALIZED), static_cast<int>(StatusCode::JIT_NOT_INITIALIZED));
    EXPECT_EQ(static_cast<int>(StatusCode::INITIALIZATION_ERROR), static_cast<int>(StatusCode::JIT_INITIALIZATION_ERROR));
    EXPECT_EQ(static_cast<int>(StatusCode::KERNEL_NOT_FOUND), static_cast<int>(StatusCode::EXECUTION_KERNEL_NOT_FOUND));
    EXPECT_EQ(static_cast<int>(StatusCode::NOT_COMPILED), static_cast<int>(StatusCode::JIT_NOT_COMPILED));
    
    // 测试旧函数仍然可用
    EXPECT_STREQ(StatusCodeToString(StatusCode::SUCCESS), "SUCCESS");
    EXPECT_STREQ(StatusCodeToString(StatusCode::CUDA_ERROR), "CUDA_ERROR");
    EXPECT_STREQ(StatusCodeToString(StatusCode::SHAPE_MISMATCH), "SHAPE_MISMATCH");
}

// 测试错误信息设置
TEST_F(StatusCodeTest, ErrorInfoTest) {
    Status status = ErrorUtils::Error(StatusCode::CACHE_ERROR, "Cache operation failed", "TestFunction", "test.cpp", 200);
    
    // 设置详细错误信息
    status.SetDetails("Cache file was corrupted during serialization");
    
    // 验证详细错误信息
    EXPECT_EQ(status.GetDetails(), "Cache file was corrupted during serialization");
    
    // 测试详细字符串输出
    std::string detailed = status.ToDetailedString();
    EXPECT_NE(detailed.find("Cache file was corrupted during serialization"), std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== cuOP 错误码系统测试 ===" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\n✅ 所有测试通过！" << std::endl;
    } else {
        std::cout << "\n❌ 部分测试失败！" << std::endl;
    }
    
    return result;
} 