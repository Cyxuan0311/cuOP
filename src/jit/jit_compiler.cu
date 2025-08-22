#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace cu_op_mem {

// JIT编译器实现
JITCompiler::JITCompiler() : nvrtc_initialized_(false) {
    // 初始化NVRTC
    nvrtcResult result = nvrtcCreateProgram(&program_, nullptr, nullptr, 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        LOG(ERROR) << "Failed to create NVRTC program: " << nvrtcGetErrorString(result);
        return;
    }
    nvrtc_initialized_ = true;
}

JITCompiler::~JITCompiler() {
    if (nvrtc_initialized_) {
        nvrtcDestroyProgram(&program_);
    }
    
    // 清理缓存的kernel
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& [key, kernel] : kernel_cache_) {
        if (kernel) {
            cuModuleUnload(kernel);
        }
    }
    kernel_cache_.clear();
}

JITCompileResult JITCompiler::CompileKernel(const std::string& kernel_code,
                                           const std::string& kernel_name,
                                           const std::vector<std::string>& options) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 生成缓存键
    std::string cache_key = GenerateKernelKey(kernel_code, options);
    
    // 检查缓存
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = kernel_cache_.find(cache_key);
        if (it != kernel_cache_.end()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            // 更新统计信息
            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                statistics_.cache_hits++;
            }
            
            return JITCompileResult(true, it->second, "", "", 
                                  duration.count() / 1000.0, options);
        }
    }
    
    // 验证kernel代码
    if (!ValidateKernelCode(kernel_code)) {
        std::string error_msg = "Invalid kernel code";
        LogCompileError(error_msg, kernel_code, options);
        return JITCompileResult(false, nullptr, "", error_msg, 0.0, options);
    }
    
    // 合并编译选项
    auto merged_options = MergeCompileOptions(options);
    
    // 使用NVRTC编译
    auto result = CompileWithNVRTC(kernel_code, kernel_name, merged_options);
    
    // 更新统计信息
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        if (result.success) {
            statistics_.cache_misses++;
            statistics_.total_compilations++;
            statistics_.total_compilation_time += result.compilation_time_ms / 1000.0;
        }
    }
    
    // 如果编译成功，缓存kernel
    if (result.success) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        kernel_cache_[cache_key] = result.kernel;
        cache_timestamps_[cache_key] = std::chrono::system_clock::now();
        statistics_.active_kernels = kernel_cache_.size();
    }
    
    return result;
}

JITCompileResult JITCompiler::CompileWithNVRTC(const std::string& kernel_code,
                                              const std::string& kernel_name,
                                              const std::vector<std::string>& options) {
    if (!nvrtc_initialized_) {
        return JITCompileResult(false, nullptr, "", "NVRTC not initialized", 0.0, options);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 准备编译选项
    std::vector<const char*> option_ptrs;
    for (const auto& option : options) {
        option_ptrs.push_back(option.c_str());
    }
    
    // 编译kernel
    nvrtcResult result = nvrtcCompileProgram(program_, 
                                           kernel_code.size(), 
                                           kernel_code.c_str(),
                                           option_ptrs.size(), 
                                           option_ptrs.data());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double compilation_time_ms = duration.count() / 1000.0;
    
    if (result != NVRTC_SUCCESS) {
        // 获取编译错误信息
        size_t log_size;
        nvrtcGetProgramLogSize(program_, &log_size);
        std::string error_log(log_size, '\0');
        nvrtcGetProgramLog(program_, &error_log[0]);
        
        LogCompileError(error_log, kernel_code, options);
        return JITCompileResult(false, nullptr, "", error_log, compilation_time_ms, options);
    }
    
    // 获取PTX代码
    size_t ptx_size;
    nvrtcGetPTXSize(program_, &ptx_size);
    std::string ptx_code(ptx_size, '\0');
    nvrtcGetPTX(program_, &ptx_code[0]);
    
    // 加载PTX到CUDA模块
    CUmodule module;
    CUresult cu_result = cuModuleLoadData(&module, ptx_code.c_str());
    if (cu_result != CUDA_SUCCESS) {
        std::string error_msg = "Failed to load PTX module: " + std::to_string(cu_result);
        LogCompileError(error_msg, kernel_code, options);
        return JITCompileResult(false, nullptr, ptx_code, error_msg, compilation_time_ms, options);
    }
    
    // 获取kernel函数
    CUfunction kernel;
    cu_result = cuModuleGetFunction(&kernel, module, kernel_name.c_str());
    if (cu_result != CUDA_SUCCESS) {
        std::string error_msg = "Failed to get kernel function: " + std::to_string(cu_result);
        cuModuleUnload(module);
        LogCompileError(error_msg, kernel_code, options);
        return JITCompileResult(false, nullptr, ptx_code, error_msg, compilation_time_ms, options);
    }
    
    return JITCompileResult(true, kernel, ptx_code, "", compilation_time_ms, options);
}

void JITCompiler::CacheKernel(const std::string& key, const CUfunction& kernel) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    kernel_cache_[key] = kernel;
    cache_timestamps_[key] = std::chrono::system_clock::now();
    statistics_.active_kernels = kernel_cache_.size();
}

CUfunction JITCompiler::GetCachedKernel(const std::string& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = kernel_cache_.find(key);
    if (it != kernel_cache_.end()) {
        return it->second;
    }
    return nullptr;
}

bool JITCompiler::IsKernelCached(const std::string& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return kernel_cache_.find(key) != kernel_cache_.end();
}

void JITCompiler::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& [key, kernel] : kernel_cache_) {
        if (kernel) {
            cuModuleUnload(kernel);
        }
    }
    kernel_cache_.clear();
    cache_timestamps_.clear();
    statistics_.active_kernels = 0;
}

void JITCompiler::ClearExpiredCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto now = std::chrono::system_clock::now();
    auto expire_time = std::chrono::hours(24); // 24小时过期
    
    auto it = cache_timestamps_.begin();
    while (it != cache_timestamps_.end()) {
        if (now - it->second > expire_time) {
            auto kernel_it = kernel_cache_.find(it->first);
            if (kernel_it != kernel_cache_.end()) {
                if (kernel_it->second) {
                    cuModuleUnload(kernel_it->second);
                }
                kernel_cache_.erase(kernel_it);
            }
            it = cache_timestamps_.erase(it);
        } else {
            ++it;
        }
    }
    statistics_.active_kernels = kernel_cache_.size();
}

void JITCompiler::SetCompilationTimeout(int timeout_seconds) {
    compilation_timeout_seconds_ = timeout_seconds;
}

void JITCompiler::SetMaxCacheSize(size_t max_size) {
    max_cache_size_ = max_size;
}

void JITCompiler::EnableDebug(bool enable) {
    debug_enabled_ = enable;
}

JITStatistics JITCompiler::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void JITCompiler::ResetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.Reset();
}

std::vector<JITCompileError> JITCompiler::GetCompileErrors() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return compile_errors_;
}

void JITCompiler::ClearCompileErrors() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    compile_errors_.clear();
}

std::vector<std::string> JITCompiler::GetDefaultCompileOptions() {
    return {
        "--device-c",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "--expt-deprecated-backend"
    };
}

std::vector<std::string> JITCompiler::GetOptimizationOptions(const std::string& level) {
    std::vector<std::string> options;
    
    if (level == "fast") {
        options = {
            "-O1",
            "--use_fast_math"
        };
    } else if (level == "best") {
        options = {
            "-O3",
            "--use_fast_math",
            "--maxrregcount=64"
        };
    } else { // "auto" or default
        options = {
            "-O2",
            "--use_fast_math"
        };
    }
    
    return options;
}

std::string JITCompiler::GenerateKernelKey(const std::string& kernel_code,
                                          const std::vector<std::string>& options) {
    // 简单的哈希算法生成缓存键
    std::hash<std::string> hasher;
    std::string combined = kernel_code;
    for (const auto& option : options) {
        combined += option;
    }
    return std::to_string(hasher(combined));
}

bool JITCompiler::ValidateKernelCode(const std::string& kernel_code) const {
    // 基本验证：检查是否包含必要的CUDA关键字
    if (kernel_code.find("__global__") == std::string::npos) {
        return false;
    }
    
    // 检查是否包含基本的CUDA头文件或必要的宏
    if (kernel_code.find("cuda_runtime.h") == std::string::npos && 
        kernel_code.find("__CUDA_ARCH__") == std::string::npos) {
        // 不是强制要求，但通常需要
    }
    
    return true;
}

std::vector<std::string> JITCompiler::MergeCompileOptions(const std::vector<std::string>& user_options) const {
    auto options = GetDefaultCompileOptions();
    
    // 添加用户选项
    options.insert(options.end(), user_options.begin(), user_options.end());
    
    // 添加调试选项
    if (debug_enabled_) {
        options.push_back("-G");
        options.push_back("-lineinfo");
    }
    
    return options;
}

void JITCompiler::LogCompileError(const std::string& error_message, 
                                 const std::string& kernel_code,
                                 const std::vector<std::string>& options) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    compile_errors_.emplace_back(error_message, kernel_code, options);
    
    if (debug_enabled_) {
        LOG(ERROR) << "JIT Compilation Error: " << error_message;
        LOG(ERROR) << "Kernel Code: " << kernel_code;
        LOG(ERROR) << "Compile Options: ";
        for (const auto& option : options) {
            LOG(ERROR) << "  " << option;
        }
    }
}

// 模板管理器实现
KernelTemplateManager& KernelTemplateManager::Instance() {
    static KernelTemplateManager instance;
    return instance;
}

void KernelTemplateManager::RegisterTemplate(const std::string& name, 
                                           std::unique_ptr<IKernelTemplate> template_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    templates_[name] = std::move(template_ptr);
}

IKernelTemplate* KernelTemplateManager::GetTemplate(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = templates_.find(name);
    if (it != templates_.end()) {
        return it->second.get();
    }
    return nullptr;
}

JITCompileResult KernelTemplateManager::CreateKernel(const std::string& template_name,
                                                   const JITConfig& config,
                                                   JITCompiler& compiler) {
    auto* template_ptr = GetTemplate(template_name);
    if (!template_ptr) {
        return JITCompileResult(false, nullptr, "", "Template not found: " + template_name, 0.0, {});
    }
    
    if (!template_ptr->ValidateConfig(config)) {
        return JITCompileResult(false, nullptr, "", "Invalid config for template: " + template_name, 0.0, {});
    }
    
    std::string kernel_code = template_ptr->GenerateKernelCode(config);
    std::string kernel_name = template_ptr->GetKernelName();
    std::vector<std::string> options = template_ptr->GetCompileOptions(config);
    
    return compiler.CompileKernel(kernel_code, kernel_name, options);
}

std::vector<std::string> KernelTemplateManager::GetSupportedTemplates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(templates_.size());
    for (const auto& [name, _] : templates_) {
        names.push_back(name);
    }
    return names;
}

} // namespace cu_op_mem 