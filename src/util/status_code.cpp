#include "util/status_code.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>

namespace cu_op_mem {

// ==================== Status 类实现 ====================

std::string Status::ToString() const {
    if (IsSuccess()) {
        return "SUCCESS";
    }
    
    std::stringstream ss;
    ss << "ERROR[" << static_cast<int>(code_) << "]: " << error_info_.message;
    
    if (!error_info_.function.empty()) {
        ss << " (in " << error_info_.function << ")";
    }
    
    return ss.str();
}

std::string Status::ToDetailedString() const {
    if (IsSuccess()) {
        return "SUCCESS";
    }
    
    std::stringstream ss;
    ss << "=== Error Details ===" << std::endl;
    ss << "Code: " << static_cast<int>(code_) << " (" << ErrorUtils::GetStatusCodeString(code_) << ")" << std::endl;
    ss << "Category: " << ErrorUtils::GetErrorCategoryString(error_info_.category) << std::endl;
    ss << "Message: " << error_info_.message << std::endl;
    
    if (!error_info_.details.empty()) {
        ss << "Details: " << error_info_.details << std::endl;
    }
    
    if (!error_info_.function.empty()) {
        ss << "Function: " << error_info_.function << std::endl;
    }
    
    if (!error_info_.file.empty()) {
        ss << "File: " << error_info_.file;
        if (error_info_.line > 0) {
            ss << ":" << error_info_.line;
        }
        ss << std::endl;
    }
    
    if (!error_info_.timestamp.empty()) {
        ss << "Timestamp: " << error_info_.timestamp << std::endl;
    }
    
    ss << "===================";
    
    return ss.str();
}

// ==================== ErrorUtils 命名空间实现 ====================

// GetErrorCategory is already defined inline in the header file

std::string ErrorUtils::GetStatusCodeString(StatusCode code) {
    switch (code) {
        // 成功状态
        case StatusCode::SUCCESS: return "SUCCESS";
        
        // 系统级错误
        case StatusCode::SYSTEM_ERROR: return "SYSTEM_ERROR";
        case StatusCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        case StatusCode::UNSUPPORTED_OPERATION: return "UNSUPPORTED_OPERATION";
        case StatusCode::TIMEOUT: return "TIMEOUT";
        case StatusCode::INTERRUPTED: return "INTERRUPTED";
        case StatusCode::RESOURCE_UNAVAILABLE: return "RESOURCE_UNAVAILABLE";
        
        // CUDA相关错误
        case StatusCode::CUDA_ERROR: return "CUDA_ERROR";
        case StatusCode::CUDA_DRIVER_ERROR: return "CUDA_DRIVER_ERROR";
        case StatusCode::CUDA_RUNTIME_ERROR: return "CUDA_RUNTIME_ERROR";
        case StatusCode::CUDA_MEMORY_ERROR: return "CUDA_MEMORY_ERROR";
        case StatusCode::CUDA_KERNEL_ERROR: return "CUDA_KERNEL_ERROR";
        case StatusCode::CUDA_LAUNCH_ERROR: return "CUDA_LAUNCH_ERROR";
        case StatusCode::CUDA_SYNC_ERROR: return "CUDA_SYNC_ERROR";
        case StatusCode::CUDA_DEVICE_ERROR: return "CUDA_DEVICE_ERROR";
        case StatusCode::CUDA_CONTEXT_ERROR: return "CUDA_CONTEXT_ERROR";
        case StatusCode::CUDA_STREAM_ERROR: return "CUDA_STREAM_ERROR";
        case StatusCode::CUDA_EVENT_ERROR: return "CUDA_EVENT_ERROR";
        
        // 内存相关错误
        case StatusCode::MEMORY_ERROR: return "MEMORY_ERROR";
        case StatusCode::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case StatusCode::MEMORY_ALLOCATION_FAILED: return "MEMORY_ALLOCATION_FAILED";
        case StatusCode::MEMORY_DEALLOCATION_FAILED: return "MEMORY_DEALLOCATION_FAILED";
        case StatusCode::MEMORY_COPY_FAILED: return "MEMORY_COPY_FAILED";
        case StatusCode::MEMORY_ALIGNMENT_ERROR: return "MEMORY_ALIGNMENT_ERROR";
        case StatusCode::MEMORY_POOL_ERROR: return "MEMORY_POOL_ERROR";
        
        // 验证错误
        case StatusCode::VALIDATION_ERROR: return "VALIDATION_ERROR";
        case StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case StatusCode::INVALID_CONFIGURATION: return "INVALID_CONFIGURATION";
        case StatusCode::INVALID_STATE: return "INVALID_STATE";
        case StatusCode::INVALID_FORMAT: return "INVALID_FORMAT";
        case StatusCode::INVALID_DIMENSION: return "INVALID_DIMENSION";
        case StatusCode::INVALID_TYPE: return "INVALID_TYPE";
        case StatusCode::INVALID_OPERATION: return "INVALID_OPERATION";
        
        // 编译错误
        case StatusCode::COMPILATION_ERROR: return "COMPILATION_ERROR";
        case StatusCode::COMPILATION_FAILED: return "COMPILATION_FAILED";
        case StatusCode::COMPILATION_TIMEOUT: return "COMPILATION_TIMEOUT";
        case StatusCode::COMPILATION_INVALID_CODE: return "COMPILATION_INVALID_CODE";
        case StatusCode::COMPILATION_INVALID_OPTIONS: return "COMPILATION_INVALID_OPTIONS";
        case StatusCode::COMPILATION_LINKING_ERROR: return "COMPILATION_LINKING_ERROR";
        case StatusCode::COMPILATION_OPTIMIZATION_ERROR: return "COMPILATION_OPTIMIZATION_ERROR";
        
        // 执行错误
        case StatusCode::EXECUTION_ERROR: return "EXECUTION_ERROR";
        case StatusCode::EXECUTION_FAILED: return "EXECUTION_FAILED";
        case StatusCode::EXECUTION_TIMEOUT: return "EXECUTION_TIMEOUT";
        case StatusCode::EXECUTION_INTERRUPTED: return "EXECUTION_INTERRUPTED";
        case StatusCode::EXECUTION_INVALID_KERNEL: return "EXECUTION_INVALID_KERNEL";
        case StatusCode::EXECUTION_KERNEL_NOT_FOUND: return "EXECUTION_KERNEL_NOT_FOUND";
        
        // JIT系统错误
        case StatusCode::JIT_ERROR: return "JIT_ERROR";
        case StatusCode::JIT_NOT_INITIALIZED: return "JIT_NOT_INITIALIZED";
        case StatusCode::JIT_INITIALIZATION_ERROR: return "JIT_INITIALIZATION_ERROR";
        case StatusCode::JIT_NOT_COMPILED: return "JIT_NOT_COMPILED";
        case StatusCode::JIT_COMPILATION_ERROR: return "JIT_COMPILATION_ERROR";
        case StatusCode::JIT_EXECUTION_ERROR: return "JIT_EXECUTION_ERROR";
        case StatusCode::JIT_PLUGIN_ERROR: return "JIT_PLUGIN_ERROR";
        case StatusCode::JIT_TEMPLATE_ERROR: return "JIT_TEMPLATE_ERROR";
        
        // 缓存相关错误
        case StatusCode::CACHE_ERROR: return "CACHE_ERROR";
        case StatusCode::CACHE_NOT_FOUND: return "CACHE_NOT_FOUND";
        case StatusCode::CACHE_CORRUPTED: return "CACHE_CORRUPTED";
        case StatusCode::CACHE_FULL: return "CACHE_FULL";
        case StatusCode::CACHE_ACCESS_ERROR: return "CACHE_ACCESS_ERROR";
        case StatusCode::CACHE_SERIALIZATION_ERROR: return "CACHE_SERIALIZATION_ERROR";
        case StatusCode::CACHE_DESERIALIZATION_ERROR: return "CACHE_DESERIALIZATION_ERROR";
        case StatusCode::CACHE_VERSION_MISMATCH: return "CACHE_VERSION_MISMATCH";
        
        // 插件相关错误
        case StatusCode::PLUGIN_ERROR: return "PLUGIN_ERROR";
        case StatusCode::PLUGIN_NOT_FOUND: return "PLUGIN_NOT_FOUND";
        case StatusCode::PLUGIN_LOAD_ERROR: return "PLUGIN_LOAD_ERROR";
        case StatusCode::PLUGIN_INITIALIZATION_ERROR: return "PLUGIN_INITIALIZATION_ERROR";
        case StatusCode::PLUGIN_EXECUTION_ERROR: return "PLUGIN_EXECUTION_ERROR";
        case StatusCode::PLUGIN_VERSION_MISMATCH: return "PLUGIN_VERSION_MISMATCH";
        case StatusCode::PLUGIN_INCOMPATIBLE: return "PLUGIN_INCOMPATIBLE";
        
        // 张量相关错误
        case StatusCode::TENSOR_ERROR: return "TENSOR_ERROR";
        case StatusCode::TENSOR_DIMENSION_MISMATCH: return "TENSOR_DIMENSION_MISMATCH";
        case StatusCode::TENSOR_SHAPE_MISMATCH: return "TENSOR_SHAPE_MISMATCH";
        case StatusCode::TENSOR_TYPE_MISMATCH: return "TENSOR_TYPE_MISMATCH";
        case StatusCode::TENSOR_SIZE_MISMATCH: return "TENSOR_SIZE_MISMATCH";
        case StatusCode::TENSOR_INDEX_OUT_OF_BOUNDS: return "TENSOR_INDEX_OUT_OF_BOUNDS";
        case StatusCode::TENSOR_STRIDE_ERROR: return "TENSOR_STRIDE_ERROR";
        case StatusCode::TENSOR_LAYOUT_ERROR: return "TENSOR_LAYOUT_ERROR";
        
        // 算子相关错误
        case StatusCode::OPERATOR_ERROR: return "OPERATOR_ERROR";
        case StatusCode::OPERATOR_NOT_SUPPORTED: return "OPERATOR_NOT_SUPPORTED";
        case StatusCode::OPERATOR_INVALID_INPUT: return "OPERATOR_INVALID_INPUT";
        case StatusCode::OPERATOR_INVALID_OUTPUT: return "OPERATOR_INVALID_OUTPUT";
        case StatusCode::OPERATOR_COMPUTATION_ERROR: return "OPERATOR_COMPUTATION_ERROR";
        case StatusCode::OPERATOR_GRADIENT_ERROR: return "OPERATOR_GRADIENT_ERROR";
        case StatusCode::OPERATOR_BACKWARD_ERROR: return "OPERATOR_BACKWARD_ERROR";
        
        // 兼容性错误码 - 这些已经在前面定义过了，删除重复
        
        default: return "UNKNOWN_ERROR_CODE";
    }
}

std::string ErrorUtils::GetErrorCategoryString(ErrorCategory category) {
    switch (category) {
        case ErrorCategory::SUCCESS: return "SUCCESS";
        case ErrorCategory::SYSTEM: return "SYSTEM";
        case ErrorCategory::CUDA: return "CUDA";
        case ErrorCategory::MEMORY: return "MEMORY";
        case ErrorCategory::VALIDATION: return "VALIDATION";
        case ErrorCategory::COMPILATION: return "COMPILATION";
        case ErrorCategory::EXECUTION: return "EXECUTION";
        case ErrorCategory::JIT: return "JIT";
        case ErrorCategory::CACHE: return "CACHE";
        case ErrorCategory::PLUGIN: return "PLUGIN";
        case ErrorCategory::TENSOR: return "TENSOR";
        case ErrorCategory::OPERATOR: return "OPERATOR";
        case ErrorCategory::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN_CATEGORY";
    }
}

std::string ErrorUtils::GetErrorDescription(StatusCode code) {
    switch (code) {
        // 成功状态
        case StatusCode::SUCCESS: return "Operation completed successfully";
        
        // 系统级错误
        case StatusCode::SYSTEM_ERROR: return "General system error occurred";
        case StatusCode::NOT_IMPLEMENTED: return "Requested functionality is not implemented";
        case StatusCode::UNSUPPORTED_OPERATION: return "Operation is not supported in current context";
        case StatusCode::TIMEOUT: return "Operation timed out";
        case StatusCode::INTERRUPTED: return "Operation was interrupted";
        case StatusCode::RESOURCE_UNAVAILABLE: return "Required resource is not available";
        
        // CUDA相关错误
        case StatusCode::CUDA_ERROR: return "General CUDA error occurred";
        case StatusCode::CUDA_DRIVER_ERROR: return "CUDA driver error occurred";
        case StatusCode::CUDA_RUNTIME_ERROR: return "CUDA runtime error occurred";
        case StatusCode::CUDA_MEMORY_ERROR: return "CUDA memory operation failed";
        case StatusCode::CUDA_KERNEL_ERROR: return "CUDA kernel execution failed";
        case StatusCode::CUDA_LAUNCH_ERROR: return "CUDA kernel launch failed";
        case StatusCode::CUDA_SYNC_ERROR: return "CUDA synchronization failed";
        case StatusCode::CUDA_DEVICE_ERROR: return "CUDA device operation failed";
        case StatusCode::CUDA_CONTEXT_ERROR: return "CUDA context operation failed";
        case StatusCode::CUDA_STREAM_ERROR: return "CUDA stream operation failed";
        case StatusCode::CUDA_EVENT_ERROR: return "CUDA event operation failed";
        
        // 内存相关错误
        case StatusCode::MEMORY_ERROR: return "General memory error occurred";
        case StatusCode::OUT_OF_MEMORY: return "Insufficient memory available";
        case StatusCode::MEMORY_ALLOCATION_FAILED: return "Memory allocation failed";
        case StatusCode::MEMORY_DEALLOCATION_FAILED: return "Memory deallocation failed";
        case StatusCode::MEMORY_COPY_FAILED: return "Memory copy operation failed";
        case StatusCode::MEMORY_ALIGNMENT_ERROR: return "Memory alignment requirement not met";
        case StatusCode::MEMORY_POOL_ERROR: return "Memory pool operation failed";
        
        // 验证错误
        case StatusCode::VALIDATION_ERROR: return "General validation error occurred";
        case StatusCode::INVALID_ARGUMENT: return "Invalid argument provided";
        case StatusCode::INVALID_CONFIGURATION: return "Invalid configuration specified";
        case StatusCode::INVALID_STATE: return "Operation not allowed in current state";
        case StatusCode::INVALID_FORMAT: return "Invalid format specified";
        case StatusCode::INVALID_DIMENSION: return "Invalid dimension specified";
        case StatusCode::INVALID_TYPE: return "Invalid type specified";
        case StatusCode::INVALID_OPERATION: return "Invalid operation requested";
        
        // 编译错误
        case StatusCode::COMPILATION_ERROR: return "General compilation error occurred";
        case StatusCode::COMPILATION_FAILED: return "Compilation process failed";
        case StatusCode::COMPILATION_TIMEOUT: return "Compilation timed out";
        case StatusCode::COMPILATION_INVALID_CODE: return "Invalid code provided for compilation";
        case StatusCode::COMPILATION_INVALID_OPTIONS: return "Invalid compilation options specified";
        case StatusCode::COMPILATION_LINKING_ERROR: return "Linking phase failed during compilation";
        case StatusCode::COMPILATION_OPTIMIZATION_ERROR: return "Optimization phase failed during compilation";
        
        // 执行错误
        case StatusCode::EXECUTION_ERROR: return "General execution error occurred";
        case StatusCode::EXECUTION_FAILED: return "Execution process failed";
        case StatusCode::EXECUTION_TIMEOUT: return "Execution timed out";
        case StatusCode::EXECUTION_INTERRUPTED: return "Execution was interrupted";
        case StatusCode::EXECUTION_INVALID_KERNEL: return "Invalid kernel for execution";
        case StatusCode::EXECUTION_KERNEL_NOT_FOUND: return "Required kernel not found for execution";
        
        // JIT系统错误
        case StatusCode::JIT_ERROR: return "General JIT system error occurred";
        case StatusCode::JIT_NOT_INITIALIZED: return "JIT system not initialized";
        case StatusCode::JIT_INITIALIZATION_ERROR: return "JIT system initialization failed";
        case StatusCode::JIT_NOT_COMPILED: return "JIT compilation not completed";
        case StatusCode::JIT_COMPILATION_ERROR: return "JIT compilation failed";
        case StatusCode::JIT_EXECUTION_ERROR: return "JIT execution failed";
        case StatusCode::JIT_PLUGIN_ERROR: return "JIT plugin error occurred";
        case StatusCode::JIT_TEMPLATE_ERROR: return "JIT template error occurred";
        
        // 缓存相关错误
        case StatusCode::CACHE_ERROR: return "General cache error occurred";
        case StatusCode::CACHE_NOT_FOUND: return "Requested cache entry not found";
        case StatusCode::CACHE_CORRUPTED: return "Cache data is corrupted";
        case StatusCode::CACHE_FULL: return "Cache is full and cannot accept new entries";
        case StatusCode::CACHE_ACCESS_ERROR: return "Cache access operation failed";
        case StatusCode::CACHE_SERIALIZATION_ERROR: return "Cache serialization failed";
        case StatusCode::CACHE_DESERIALIZATION_ERROR: return "Cache deserialization failed";
        case StatusCode::CACHE_VERSION_MISMATCH: return "Cache version mismatch detected";
        
        // 插件相关错误
        case StatusCode::PLUGIN_ERROR: return "General plugin error occurred";
        case StatusCode::PLUGIN_NOT_FOUND: return "Requested plugin not found";
        case StatusCode::PLUGIN_LOAD_ERROR: return "Plugin loading failed";
        case StatusCode::PLUGIN_INITIALIZATION_ERROR: return "Plugin initialization failed";
        case StatusCode::PLUGIN_EXECUTION_ERROR: return "Plugin execution failed";
        case StatusCode::PLUGIN_VERSION_MISMATCH: return "Plugin version mismatch detected";
        case StatusCode::PLUGIN_INCOMPATIBLE: return "Plugin is incompatible with current system";
        
        // 张量相关错误
        case StatusCode::TENSOR_ERROR: return "General tensor error occurred";
        case StatusCode::TENSOR_DIMENSION_MISMATCH: return "Tensor dimension mismatch";
        case StatusCode::TENSOR_SHAPE_MISMATCH: return "Tensor shape mismatch";
        case StatusCode::TENSOR_TYPE_MISMATCH: return "Tensor type mismatch";
        case StatusCode::TENSOR_SIZE_MISMATCH: return "Tensor size mismatch";
        case StatusCode::TENSOR_INDEX_OUT_OF_BOUNDS: return "Tensor index out of bounds";
        case StatusCode::TENSOR_STRIDE_ERROR: return "Tensor stride error";
        case StatusCode::TENSOR_LAYOUT_ERROR: return "Tensor layout error";
        
        // 算子相关错误
        case StatusCode::OPERATOR_ERROR: return "General operator error occurred";
        case StatusCode::OPERATOR_NOT_SUPPORTED: return "Operator not supported for given input";
        case StatusCode::OPERATOR_INVALID_INPUT: return "Operator received invalid input";
        case StatusCode::OPERATOR_INVALID_OUTPUT: return "Operator produced invalid output";
        case StatusCode::OPERATOR_COMPUTATION_ERROR: return "Operator computation failed";
        case StatusCode::OPERATOR_GRADIENT_ERROR: return "Operator gradient computation failed";
        case StatusCode::OPERATOR_BACKWARD_ERROR: return "Operator backward pass failed";
        
        default: return "Unknown error code";
    }
}

bool ErrorUtils::IsCriticalError(StatusCode code) {
    // 定义严重错误码
    switch (code) {
        case StatusCode::OUT_OF_MEMORY:
        case StatusCode::CUDA_DRIVER_ERROR:
        case StatusCode::CUDA_CONTEXT_ERROR:
        case StatusCode::CACHE_CORRUPTED:
        case StatusCode::PLUGIN_INCOMPATIBLE:
            return true;
        default:
            return false;
    }
}

bool ErrorUtils::IsRecoverableError(StatusCode code) {
    // 定义可恢复错误码
    switch (code) {
        case StatusCode::TIMEOUT:
        case StatusCode::INTERRUPTED:
        case StatusCode::RESOURCE_UNAVAILABLE:
        case StatusCode::CACHE_NOT_FOUND:
        case StatusCode::EXECUTION_TIMEOUT:
            return true;
        default:
            return false;
    }
}

Status ErrorUtils::Error(StatusCode code, const std::string& message, const std::string& function,
                        const std::string& file, int line) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    Status status(code, message, function, file, line);
    status.SetDetails("Error occurred at " + ss.str());
    
    return status;
}

Status ErrorUtils::CudaError(cudaError_t cuda_error, const std::string& function,
                            const std::string& file, int line) {
    std::string message = "CUDA error: " + std::string(cudaGetErrorString(cuda_error));
    
    // 根据CUDA错误类型映射到相应的状态码
    StatusCode status_code;
    switch (cuda_error) {
        case cudaErrorMemoryAllocation:
            status_code = StatusCode::OUT_OF_MEMORY;
            break;
        case cudaErrorInvalidValue:
            status_code = StatusCode::INVALID_ARGUMENT;
            break;
        case cudaErrorInvalidDevice:
            status_code = StatusCode::CUDA_DEVICE_ERROR;
            break;
        // cudaErrorInvalidContext is not a valid CUDA error code
        case cudaErrorInvalidMemcpyDirection:
            status_code = StatusCode::MEMORY_COPY_FAILED;
            break;
        case cudaErrorLaunchFailure:
            status_code = StatusCode::CUDA_LAUNCH_ERROR;
            break;
        case cudaErrorLaunchTimeout:
            status_code = StatusCode::EXECUTION_TIMEOUT;
            break;
        case cudaErrorLaunchOutOfResources:
            status_code = StatusCode::RESOURCE_UNAVAILABLE;
            break;
        default:
            status_code = StatusCode::CUDA_ERROR;
            break;
    }
    
    return Error(status_code, message, function, file, line);
}

Status ErrorUtils::SystemError(const std::string& message, const std::string& function,
                              const std::string& file, int line) {
    return Error(StatusCode::SYSTEM_ERROR, message, function, file, line);
}

Status ErrorUtils::ValidationError(const std::string& message, const std::string& function,
                                  const std::string& file, int line) {
    return Error(StatusCode::VALIDATION_ERROR, message, function, file, line);
}

Status ErrorUtils::MemoryError(const std::string& message, const std::string& function,
                              const std::string& file, int line) {
    return Error(StatusCode::MEMORY_ERROR, message, function, file, line);
}

Status ErrorUtils::CompilationError(const std::string& message, const std::string& function,
                                   const std::string& file, int line) {
    return Error(StatusCode::COMPILATION_ERROR, message, function, file, line);
}

Status ErrorUtils::ExecutionError(const std::string& message, const std::string& function,
                                 const std::string& file, int line) {
    return Error(StatusCode::EXECUTION_ERROR, message, function, file, line);
}

// ==================== 流操作符重载 ====================

std::ostream& operator<<(std::ostream& os, const StatusCode& code) {
    os << ErrorUtils::GetStatusCodeString(code);
    return os;
}

std::ostream& operator<<(std::ostream& os, const ErrorCategory& category) {
    os << ErrorUtils::GetErrorCategoryString(category);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Status& status) {
    os << status.ToString();
    return os;
}

} // namespace cu_op_mem
