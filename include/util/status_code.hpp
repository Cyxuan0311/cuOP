#pragma once

#include <string>
#include <system_error>
#include <ostream>

namespace cu_op_mem {

// 错误码分类枚举
enum class ErrorCategory {
    SUCCESS = 0,           // 成功
    SYSTEM,                // 系统级错误
    CUDA,                  // CUDA相关错误
    MEMORY,                // 内存相关错误
    VALIDATION,            // 验证错误
    COMPILATION,           // 编译错误
    EXECUTION,             // 执行错误
    JIT,                   // JIT系统错误
    CACHE,                 // 缓存相关错误
    PLUGIN,                // 插件相关错误
    TENSOR,                // 张量相关错误
    OPERATOR,              // 算子相关错误
    UNKNOWN                // 未知错误
};

// 主状态码枚举
enum class StatusCode {
    // ==================== 成功状态 ====================
    SUCCESS = 0,                           // 操作成功
    
    // ==================== 系统级错误 (1000-1999) ====================
    SYSTEM_ERROR = 1000,                   // 系统错误
    NOT_IMPLEMENTED = 1001,                // 功能未实现
    UNSUPPORTED_OPERATION = 1002,          // 不支持的操作
    TIMEOUT = 1003,                        // 操作超时
    INTERRUPTED = 1004,                    // 操作被中断
    RESOURCE_UNAVAILABLE = 1005,           // 资源不可用
    FILE_NOT_FOUND = 1006,                 // 文件未找到
    PERMISSION_DENIED = 1007,              // 权限被拒绝
    
    // ==================== CUDA相关错误 (2000-2999) ====================
    CUDA_ERROR = 2000,                     // CUDA错误
    CUDA_DRIVER_ERROR = 2001,              // CUDA驱动错误
    CUDA_RUNTIME_ERROR = 2002,             // CUDA运行时错误
    CUDA_MEMORY_ERROR = 2003,              // CUDA内存错误
    CUDA_KERNEL_ERROR = 2004,              // CUDA内核错误
    CUDA_LAUNCH_ERROR = 2005,              // CUDA启动错误
    CUDA_SYNC_ERROR = 2006,                // CUDA同步错误
    CUDA_DEVICE_ERROR = 2007,              // CUDA设备错误
    CUDA_CONTEXT_ERROR = 2008,             // CUDA上下文错误
    CUDA_STREAM_ERROR = 2009,              // CUDA流错误
    CUDA_EVENT_ERROR = 2010,               // CUDA事件错误
    CUDA_INVALID_DEVICE = 2011,            // CUDA无效设备
    CUDA_INVALID_VALUE = 2012,             // CUDA无效值
    CUDA_OUT_OF_MEMORY = 2013,             // CUDA内存不足
    CUDA_LAUNCH_FAILED = 2014,             // CUDA启动失败
    
    // ==================== 内存相关错误 (3000-3999) ====================
    MEMORY_ERROR = 3000,                   // 内存错误
    OUT_OF_MEMORY = 3001,                  // 内存不足
    MEMORY_ALLOCATION_FAILED = 3002,       // 内存分配失败
    MEMORY_DEALLOCATION_FAILED = 3003,     // 内存释放失败
    MEMORY_COPY_FAILED = 3004,             // 内存拷贝失败
    MEMORY_ALIGNMENT_ERROR = 3005,         // 内存对齐错误
    MEMORY_POOL_ERROR = 3006,              // 内存池错误
    MEMORY_FREE_FAILED = 3007,             // 内存释放失败
    
    // ==================== 验证错误 (4000-4999) ====================
    VALIDATION_ERROR = 4000,               // 验证错误
    INVALID_ARGUMENT = 4001,               // 无效参数
    INVALID_CONFIGURATION = 4002,          // 无效配置
    INVALID_STATE = 4003,                  // 无效状态
    INVALID_FORMAT = 4004,                 // 无效格式
    INVALID_DIMENSION = 4005,              // 无效维度
    INVALID_TYPE = 4006,                   // 无效类型
    INVALID_OPERATION = 4007,              // 无效操作
    INVALID_INPUT = 4008,                  // 无效输入
    INVALID_OUTPUT = 4009,                 // 无效输出
    INVALID_DIMENSIONS = 4010,             // 无效维度
    INVALID_DATA_TYPE = 4011,              // 无效数据类型
    
    // ==================== 编译错误 (5000-5999) ====================
    COMPILATION_ERROR = 5000,              // 编译错误
    COMPILATION_FAILED = 5001,             // 编译失败
    COMPILATION_TIMEOUT = 5002,            // 编译超时
    COMPILATION_INVALID_CODE = 5003,       // 无效的编译代码
    COMPILATION_INVALID_OPTIONS = 5004,    // 无效的编译选项
    COMPILATION_LINKING_ERROR = 5005,      // 链接错误
    COMPILATION_OPTIMIZATION_ERROR = 5006, // 优化错误
    JIT_COMPILATION_FAILED = 5007,         // JIT编译失败
    
    // ==================== 执行错误 (6000-6999) ====================
    EXECUTION_ERROR = 6000,                // 执行错误
    EXECUTION_FAILED = 6001,               // 执行失败
    EXECUTION_TIMEOUT = 6002,              // 执行超时
    EXECUTION_INTERRUPTED = 6003,          // 执行被中断
    EXECUTION_INVALID_KERNEL = 6004,       // 无效的内核
    EXECUTION_KERNEL_NOT_FOUND = 6005,     // 内核未找到
    KERNEL_LAUNCH_FAILED = 6006,           // 内核启动失败
    
    // ==================== JIT系统错误 (7000-7999) ====================
    JIT_ERROR = 7000,                      // JIT系统错误
    JIT_NOT_INITIALIZED = 7001,            // JIT未初始化
    JIT_INITIALIZATION_ERROR = 7002,       // JIT初始化错误
    JIT_NOT_COMPILED = 7003,               // JIT未编译
    JIT_COMPILATION_ERROR = 7004,          // JIT编译错误
    JIT_EXECUTION_ERROR = 7005,            // JIT执行错误
    JIT_PLUGIN_ERROR = 7006,               // JIT插件错误
    JIT_TEMPLATE_ERROR = 7007,             // JIT模板错误
    JIT_CACHE_ERROR = 7008,                // JIT缓存错误
    
    // ==================== 缓存相关错误 (8000-8999) ====================
    CACHE_ERROR = 8000,                    // 缓存错误
    CACHE_NOT_FOUND = 8001,                // 缓存未找到
    CACHE_CORRUPTED = 8002,                // 缓存损坏
    CACHE_FULL = 8003,                     // 缓存已满
    CACHE_ACCESS_ERROR = 8004,             // 缓存访问错误
    CACHE_SERIALIZATION_ERROR = 8005,      // 缓存序列化错误
    CACHE_DESERIALIZATION_ERROR = 8006,    // 缓存反序列化错误
    CACHE_VERSION_MISMATCH = 8007,         // 缓存版本不匹配
    CACHE_MISS = 8008,                     // 缓存未命中
    
    // ==================== 插件相关错误 (9000-9999) ====================
    PLUGIN_ERROR = 9000,                   // 插件错误
    PLUGIN_NOT_FOUND = 9001,               // 插件未找到
    PLUGIN_LOAD_ERROR = 9002,              // 插件加载错误
    PLUGIN_INITIALIZATION_ERROR = 9003,    // 插件初始化错误
    PLUGIN_EXECUTION_ERROR = 9004,         // 插件执行错误
    PLUGIN_VERSION_MISMATCH = 9005,        // 插件版本不匹配
    PLUGIN_INCOMPATIBLE = 9006,            // 插件不兼容
    
    // ==================== 张量相关错误 (10000-10999) ====================
    TENSOR_ERROR = 10000,                  // 张量错误
    TENSOR_DIMENSION_MISMATCH = 10001,     // 张量维度不匹配
    TENSOR_SHAPE_MISMATCH = 10002,         // 张量形状不匹配
    TENSOR_TYPE_MISMATCH = 10003,          // 张量类型不匹配
    TENSOR_SIZE_MISMATCH = 10004,          // 张量大小不匹配
    TENSOR_INDEX_OUT_OF_BOUNDS = 10005,    // 张量索引越界
    TENSOR_STRIDE_ERROR = 10006,           // 张量步长错误
    TENSOR_LAYOUT_ERROR = 10007,           // 张量布局错误
    
    // ==================== 算子相关错误 (11000-11999) ====================
    OPERATOR_ERROR = 11000,                // 算子错误
    OPERATOR_NOT_SUPPORTED = 11001,        // 算子不支持
    OPERATOR_INVALID_INPUT = 11002,        // 算子输入无效
    OPERATOR_INVALID_OUTPUT = 11003,       // 算子输出无效
    OPERATOR_COMPUTATION_ERROR = 11004,    // 算子计算错误
    OPERATOR_GRADIENT_ERROR = 11005,       // 算子梯度错误
    OPERATOR_BACKWARD_ERROR = 11006,       // 算子反向传播错误
    
    // ==================== 兼容性错误码 (向后兼容) ====================
    SHAPE_MISMATCH = TENSOR_SHAPE_MISMATCH,           // 兼容旧版本
    UNSUPPORTED_TYPE = OPERATOR_NOT_SUPPORTED,         // 兼容旧版本
    UNKNOWN_ERROR = SYSTEM_ERROR,                      // 兼容旧版本
    TENSOR_DIMONSION_MISMATCH = TENSOR_DIMENSION_MISMATCH, // 兼容旧版本
    NOT_INITIALIZED = JIT_NOT_INITIALIZED,             // 兼容旧版本
    INITIALIZATION_ERROR = JIT_INITIALIZATION_ERROR,   // 兼容旧版本
    KERNEL_NOT_FOUND = EXECUTION_KERNEL_NOT_FOUND,    // 兼容旧版本
    NOT_COMPILED = JIT_NOT_COMPILED                   // 兼容旧版本
};

// 全局GetErrorCategory函数 - 直接实现避免循环依赖
inline ErrorCategory GetErrorCategory(StatusCode code) {
    switch (code) {
        case StatusCode::SUCCESS: return ErrorCategory::SUCCESS;
        case StatusCode::CUDA_ERROR:
        case StatusCode::CUDA_DRIVER_ERROR:
        case StatusCode::CUDA_RUNTIME_ERROR:
        case StatusCode::CUDA_MEMORY_ERROR:
        case StatusCode::CUDA_KERNEL_ERROR:
        case StatusCode::CUDA_LAUNCH_ERROR:
        case StatusCode::CUDA_SYNC_ERROR:
        case StatusCode::CUDA_DEVICE_ERROR:
        case StatusCode::CUDA_CONTEXT_ERROR:
        case StatusCode::CUDA_STREAM_ERROR:
        case StatusCode::CUDA_EVENT_ERROR:
        case StatusCode::CUDA_INVALID_DEVICE:
        case StatusCode::CUDA_INVALID_VALUE:
        case StatusCode::CUDA_OUT_OF_MEMORY:
        case StatusCode::CUDA_LAUNCH_FAILED:
            return ErrorCategory::CUDA;
        case StatusCode::MEMORY_ERROR:
        case StatusCode::OUT_OF_MEMORY:
        case StatusCode::MEMORY_ALLOCATION_FAILED:
        case StatusCode::MEMORY_DEALLOCATION_FAILED:
        case StatusCode::MEMORY_COPY_FAILED:
        case StatusCode::MEMORY_ALIGNMENT_ERROR:
        case StatusCode::MEMORY_POOL_ERROR:
        case StatusCode::MEMORY_FREE_FAILED:
            return ErrorCategory::MEMORY;
        case StatusCode::VALIDATION_ERROR:
        case StatusCode::INVALID_ARGUMENT:
        case StatusCode::INVALID_CONFIGURATION:
        case StatusCode::INVALID_STATE:
        case StatusCode::INVALID_FORMAT:
        case StatusCode::INVALID_DIMENSION:
        case StatusCode::INVALID_TYPE:
        case StatusCode::INVALID_OPERATION:
        case StatusCode::INVALID_INPUT:
        case StatusCode::INVALID_OUTPUT:
        case StatusCode::INVALID_DIMENSIONS:
        case StatusCode::INVALID_DATA_TYPE:
        case StatusCode::TENSOR_DIMENSION_MISMATCH:
        case StatusCode::TENSOR_SHAPE_MISMATCH:
        case StatusCode::TENSOR_TYPE_MISMATCH:
        case StatusCode::TENSOR_SIZE_MISMATCH:
        case StatusCode::TENSOR_INDEX_OUT_OF_BOUNDS:
        case StatusCode::TENSOR_STRIDE_ERROR:
        case StatusCode::TENSOR_LAYOUT_ERROR:
            return ErrorCategory::VALIDATION;
        case StatusCode::COMPILATION_ERROR:
        case StatusCode::COMPILATION_FAILED:
        case StatusCode::COMPILATION_TIMEOUT:
        case StatusCode::COMPILATION_INVALID_CODE:
        case StatusCode::COMPILATION_INVALID_OPTIONS:
        case StatusCode::COMPILATION_LINKING_ERROR:
        case StatusCode::COMPILATION_OPTIMIZATION_ERROR:
        case StatusCode::JIT_COMPILATION_FAILED:
        case StatusCode::JIT_ERROR:
        case StatusCode::JIT_NOT_INITIALIZED:
        case StatusCode::JIT_INITIALIZATION_ERROR:
        case StatusCode::JIT_NOT_COMPILED:
        case StatusCode::JIT_COMPILATION_ERROR:
        case StatusCode::JIT_EXECUTION_ERROR:
        case StatusCode::JIT_PLUGIN_ERROR:
        case StatusCode::JIT_TEMPLATE_ERROR:
        case StatusCode::JIT_CACHE_ERROR:
        case StatusCode::EXECUTION_ERROR:
        case StatusCode::EXECUTION_FAILED:
        case StatusCode::EXECUTION_TIMEOUT:
        case StatusCode::EXECUTION_INTERRUPTED:
        case StatusCode::EXECUTION_INVALID_KERNEL:
        case StatusCode::EXECUTION_KERNEL_NOT_FOUND:
        case StatusCode::KERNEL_LAUNCH_FAILED:
            return ErrorCategory::EXECUTION;
        case StatusCode::SYSTEM_ERROR:
        case StatusCode::NOT_IMPLEMENTED:
        case StatusCode::UNSUPPORTED_OPERATION:
        case StatusCode::TIMEOUT:
        case StatusCode::INTERRUPTED:
        case StatusCode::RESOURCE_UNAVAILABLE:
        case StatusCode::FILE_NOT_FOUND:
        case StatusCode::PERMISSION_DENIED:
            return ErrorCategory::SYSTEM;
        default: return ErrorCategory::UNKNOWN;
    }
}

// 错误信息结构
struct ErrorInfo {
    StatusCode code;                       // 错误码
    ErrorCategory category;                // 错误分类
    std::string message;                   // 错误消息
    std::string details;                   // 详细错误信息
    std::string function;                  // 发生错误的函数
    std::string file;                      // 发生错误的文件
    int line;                              // 发生错误的行号
    std::string timestamp;                 // 错误发生时间戳
    
    ErrorInfo() : code(StatusCode::SUCCESS), category(ErrorCategory::SUCCESS), line(0) {}
    
    ErrorInfo(StatusCode c, const std::string& msg, const std::string& func = "", 
              const std::string& f = "", int l = 0)
        : code(c), category(GetErrorCategory(c)), message(msg), function(func), file(f), line(l) {}
};

// 状态结果类
class Status {
public:
    Status() : code_(StatusCode::SUCCESS) {}
    
    Status(StatusCode code) : code_(code) {}
    
    Status(StatusCode code, const std::string& message, const std::string& function = "",
           const std::string& file = "", int line = 0)
        : code_(code), error_info_(code, message, function, file, line) {}
    
    // 检查是否成功
    bool IsSuccess() const { return code_ == StatusCode::SUCCESS; }
    bool IsError() const { return !IsSuccess(); }
    
    // 获取错误码
    StatusCode GetCode() const { return code_; }
    
    // 获取错误分类
    ErrorCategory GetCategory() const { return error_info_.category; }
    
    // 获取错误信息
    const std::string& GetMessage() const { return error_info_.message; }
    const std::string& GetDetails() const { return error_info_.details; }
    const std::string& GetFunction() const { return error_info_.function; }
    const std::string& GetFile() const { return error_info_.file; }
    int GetLine() const { return error_info_.line; }
    
    // 设置详细错误信息
    void SetDetails(const std::string& details) { error_info_.details = details; }
    
    // 转换为布尔值
    explicit operator bool() const { return IsSuccess(); }
    
    // 错误信息格式化
    std::string ToString() const;
    
    // 错误信息详细格式化
    std::string ToDetailedString() const;
    
private:
    StatusCode code_;
    ErrorInfo error_info_;
};

// 工具函数
namespace ErrorUtils {

// 获取错误分类
inline ErrorCategory GetErrorCategory(StatusCode code) {
    switch (code) {
        case StatusCode::SUCCESS:
            return ErrorCategory::SUCCESS;
        case StatusCode::CUDA_ERROR:
        case StatusCode::CUDA_DRIVER_ERROR:
        case StatusCode::CUDA_RUNTIME_ERROR:
        case StatusCode::CUDA_MEMORY_ERROR:
        case StatusCode::CUDA_KERNEL_ERROR:
        case StatusCode::CUDA_LAUNCH_ERROR:
        case StatusCode::CUDA_SYNC_ERROR:
        case StatusCode::CUDA_DEVICE_ERROR:
        case StatusCode::CUDA_CONTEXT_ERROR:
        case StatusCode::CUDA_STREAM_ERROR:
        case StatusCode::CUDA_EVENT_ERROR:
        case StatusCode::CUDA_INVALID_DEVICE:
        case StatusCode::CUDA_INVALID_VALUE:
        case StatusCode::CUDA_OUT_OF_MEMORY:
        case StatusCode::CUDA_LAUNCH_FAILED:
            return ErrorCategory::CUDA;
        case StatusCode::MEMORY_ERROR:
        case StatusCode::OUT_OF_MEMORY:
        case StatusCode::MEMORY_ALLOCATION_FAILED:
        case StatusCode::MEMORY_DEALLOCATION_FAILED:
        case StatusCode::MEMORY_COPY_FAILED:
        case StatusCode::MEMORY_ALIGNMENT_ERROR:
        case StatusCode::MEMORY_POOL_ERROR:
        case StatusCode::MEMORY_FREE_FAILED:
            return ErrorCategory::MEMORY;
        case StatusCode::VALIDATION_ERROR:
        case StatusCode::INVALID_ARGUMENT:
        case StatusCode::INVALID_CONFIGURATION:
        case StatusCode::INVALID_STATE:
        case StatusCode::INVALID_FORMAT:
        case StatusCode::INVALID_DIMENSION:
        case StatusCode::INVALID_TYPE:
        case StatusCode::INVALID_OPERATION:
        case StatusCode::INVALID_INPUT:
        case StatusCode::INVALID_OUTPUT:
        case StatusCode::INVALID_DIMENSIONS:
        case StatusCode::INVALID_DATA_TYPE:
            return ErrorCategory::VALIDATION;
        case StatusCode::COMPILATION_ERROR:
        case StatusCode::COMPILATION_FAILED:
        case StatusCode::COMPILATION_TIMEOUT:
        case StatusCode::COMPILATION_INVALID_CODE:
        case StatusCode::COMPILATION_INVALID_OPTIONS:
        case StatusCode::COMPILATION_LINKING_ERROR:
        case StatusCode::COMPILATION_OPTIMIZATION_ERROR:
        case StatusCode::JIT_COMPILATION_FAILED:
            return ErrorCategory::COMPILATION;
        case StatusCode::EXECUTION_ERROR:
        case StatusCode::EXECUTION_FAILED:
        case StatusCode::EXECUTION_TIMEOUT:
        case StatusCode::EXECUTION_INTERRUPTED:
        case StatusCode::EXECUTION_INVALID_KERNEL:
        case StatusCode::EXECUTION_KERNEL_NOT_FOUND:
        case StatusCode::KERNEL_LAUNCH_FAILED:
            return ErrorCategory::EXECUTION;
        case StatusCode::JIT_ERROR:
        case StatusCode::JIT_NOT_INITIALIZED:
        case StatusCode::JIT_INITIALIZATION_ERROR:
        case StatusCode::JIT_NOT_COMPILED:
        case StatusCode::JIT_COMPILATION_ERROR:
        case StatusCode::JIT_EXECUTION_ERROR:
        case StatusCode::JIT_PLUGIN_ERROR:
        case StatusCode::JIT_TEMPLATE_ERROR:
        case StatusCode::JIT_CACHE_ERROR:
            return ErrorCategory::JIT;
        case StatusCode::CACHE_ERROR:
        case StatusCode::CACHE_NOT_FOUND:
        case StatusCode::CACHE_CORRUPTED:
        case StatusCode::CACHE_FULL:
        case StatusCode::CACHE_ACCESS_ERROR:
        case StatusCode::CACHE_SERIALIZATION_ERROR:
        case StatusCode::CACHE_DESERIALIZATION_ERROR:
        case StatusCode::CACHE_VERSION_MISMATCH:
        case StatusCode::CACHE_MISS:
            return ErrorCategory::CACHE;
        case StatusCode::PLUGIN_ERROR:
        case StatusCode::PLUGIN_NOT_FOUND:
            return ErrorCategory::PLUGIN;
        case StatusCode::TENSOR_ERROR:
        case StatusCode::TENSOR_SHAPE_MISMATCH:
            return ErrorCategory::TENSOR;
        case StatusCode::OPERATOR_ERROR:
        case StatusCode::OPERATOR_NOT_SUPPORTED:
            return ErrorCategory::OPERATOR;
        case StatusCode::SYSTEM_ERROR:
        case StatusCode::FILE_NOT_FOUND:
        case StatusCode::PERMISSION_DENIED:
            return ErrorCategory::SYSTEM;
        default:
            return ErrorCategory::UNKNOWN;
    }
}

// 获取错误码字符串表示
std::string GetStatusCodeString(StatusCode code);

// 获取错误分类字符串表示
std::string GetErrorCategoryString(ErrorCategory category);

// 获取错误描述
std::string GetErrorDescription(StatusCode code);

// 检查是否为严重错误
bool IsCriticalError(StatusCode code);

// 检查是否为可恢复错误
bool IsRecoverableError(StatusCode code);

// 创建成功状态
inline Status Success() { return Status(StatusCode::SUCCESS); }

// 创建错误状态
Status Error(StatusCode code, const std::string& message, const std::string& function = "",
            const std::string& file = "", int line = 0);

// 创建CUDA错误状态
Status CudaError(cudaError_t cuda_error, const std::string& function = "",
                 const std::string& file = "", int line = 0);

// 创建系统错误状态
Status SystemError(const std::string& message, const std::string& function = "",
                   const std::string& file = "", int line = 0);

// 创建验证错误状态
Status ValidationError(const std::string& message, const std::string& function = "",
                       const std::string& file = "", int line = 0);

// 创建内存错误状态
Status MemoryError(const std::string& message, const std::string& function = "",
                   const std::string& file = "", int line = 0);

// 创建编译错误状态
Status CompilationError(const std::string& message, const std::string& function = "",
                        const std::string& file = "", int line = 0);

// 创建执行错误状态
Status ExecutionError(const std::string& message, const std::string& function = "",
                      const std::string& file = "", int line = 0);

} // namespace ErrorUtils

// 流操作符重载
std::ostream& operator<<(std::ostream& os, const StatusCode& code);
std::ostream& operator<<(std::ostream& os, const ErrorCategory& category);
std::ostream& operator<<(std::ostream& os, const Status& status);

// 向后兼容的函数
inline const char* StatusCodeToString(StatusCode code) {
    return ErrorUtils::GetStatusCodeString(code).c_str();
}


} // namespace cu_op_mem