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
    
    // 初始化持久化缓存
    if (persistent_cache_enabled_) {
        GlobalPersistentCacheManager::Instance().Initialize(persistent_cache_dir_);
        LOG(INFO) << "Persistent cache initialized at: " << persistent_cache_dir_;
    }
}

JITCompiler::~JITCompiler() {
    if (nvrtc_initialized_) {
        nvrtcDestroyProgram(&program_);
    }
    
    // 清理缓存的kernel
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& [key, kernel] : kernel_cache_) {
        if (kernel) {
            // CUfunction不需要显式卸载，它们会随着模块一起被卸载
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
    
    // 1. 检查内存缓存
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
    
    // 2. 尝试从持久化缓存加载
    if (persistent_cache_enabled_) {
        JITCompileResult cached_result;
        if (TryLoadFromPersistentCache(cache_key, cached_result)) {
            // 从持久化缓存成功加载，需要重新加载到CUDA模块
            auto result = LoadKernelFromPTX(cached_result.ptx_code, kernel_name);
            if (result.success) {
                // 缓存到内存
                std::lock_guard<std::mutex> lock(cache_mutex_);
                kernel_cache_[cache_key] = result.kernel;
                cache_timestamps_[cache_key] = std::chrono::system_clock::now();
                
                // 更新统计信息
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                    statistics_.cache_hits++;
                    statistics_.total_saved_compilation_time += static_cast<size_t>(cached_result.compilation_time_ms);
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                
                LOG(INFO) << "Kernel loaded from persistent cache: " << kernel_name 
                          << " (saved " << cached_result.compilation_time_ms << " ms)";
                
                return JITCompileResult(true, result.kernel, cached_result.ptx_code, "", 
                                      duration.count() / 1000.0, options);
            }
        }
    }
    
    // 3. 验证kernel代码
    if (!ValidateKernelCode(kernel_code)) {
        std::string error_msg = "Invalid kernel code";
        LogCompileError(error_msg, kernel_code, options);
        return JITCompileResult(false, nullptr, "", error_msg, 0.0, options);
    }
    
    // 4. 合并编译选项
    auto merged_options = MergeCompileOptions(options);
    
    // 5. 使用NVRTC编译
    auto result = CompileWithNVRTC(kernel_code, kernel_name, merged_options);
    
    // 6. 更新统计信息
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        if (result.success) {
            statistics_.cache_misses++;
            statistics_.total_compilations++;
            statistics_.total_compilation_time += result.compilation_time_ms / 1000.0;
        }
    }
    
    // 7. 如果编译成功，缓存kernel
    if (result.success) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        kernel_cache_[cache_key] = result.kernel;
        cache_timestamps_[cache_key] = std::chrono::system_clock::now();
        statistics_.active_kernels = kernel_cache_.size();
        
        // 8. 保存到持久化缓存
        if (persistent_cache_enabled_) {
            SaveToPersistentCache(cache_key, result);
        }
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
            // CUfunction不需要显式卸载
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
                    // CUfunction不需要显式卸载
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

void JITCompiler::EnablePersistentCache(bool enable) {
    persistent_cache_enabled_ = enable;
    
    if (enable && !persistent_cache_dir_.empty()) {
        GlobalPersistentCacheManager::Instance().Initialize(persistent_cache_dir_);
        LOG(INFO) << "Persistent cache enabled at: " << persistent_cache_dir_;
    } else if (!enable) {
        GlobalPersistentCacheManager::Instance().Cleanup();
        LOG(INFO) << "Persistent cache disabled";
    }
}

bool JITCompiler::IsPersistentCacheEnabled() const {
    return persistent_cache_enabled_;
}

void JITCompiler::SetPersistentCacheDirectory(const std::string& cache_dir) {
    persistent_cache_dir_ = cache_dir;
    
    if (persistent_cache_enabled_) {
        GlobalPersistentCacheManager::Instance().Initialize(cache_dir);
        LOG(INFO) << "Persistent cache directory updated to: " << cache_dir;
    }
}

std::string JITCompiler::GetPersistentCacheDirectory() const {
    return persistent_cache_dir_;
}

bool JITCompiler::TryLoadFromPersistentCache(const std::string& cache_key, JITCompileResult& result) {
    try {
        std::string kernel_name, ptx_code;
        std::vector<std::string> compile_options;
        
        if (GlobalPersistentCacheManager::Instance().LoadKernel(cache_key, kernel_name, compile_options, ptx_code)) {
            result.ptx_code = ptx_code;
            result.compile_options = compile_options;
            result.success = true;
            
            // 从元数据中恢复编译时间（如果可用）
            auto metadata = GlobalPersistentCacheManager::Instance().GetKernelMetadata(cache_key);
            if (metadata.compilation_time_ms > 0) {
                result.compilation_time_ms = static_cast<double>(metadata.compilation_time_ms);
            }
            
            VLOG(1) << "Successfully loaded kernel from persistent cache: " << cache_key;
            return true;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to load from persistent cache: " << e.what();
        return false;
    }
}

void JITCompiler::SaveToPersistentCache(const std::string& cache_key, const JITCompileResult& result) {
    try {
        // 从kernel代码生成缓存键（这里需要重构以获取原始kernel代码）
        std::string kernel_code = ""; // TODO: 需要从调用者传递
        std::string kernel_name = "unknown"; // TODO: 需要从调用者传递
        
        GlobalPersistentCacheManager::Instance().SaveKernel(
            cache_key, kernel_name, kernel_code, 
            result.compile_options, result.ptx_code, 
            result.compilation_time_ms
        );
        
        VLOG(1) << "Successfully saved kernel to persistent cache: " << cache_key;
        
    } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to save to persistent cache: " << e.what();
    }
}

JITCompileResult JITCompiler::LoadKernelFromPTX(const std::string& ptx_code, const std::string& kernel_name) {
    JITCompileResult result;
    
    try {
        // 加载PTX到CUDA模块
        CUmodule module;
        CUresult cu_result = cuModuleLoadData(&module, ptx_code.c_str());
        if (cu_result != CUDA_SUCCESS) {
            result.success = false;
            result.error_message = "Failed to load PTX module: " + std::to_string(cu_result);
            return result;
        }
        
        // 获取kernel函数
        CUfunction kernel;
        cu_result = cuModuleGetFunction(&kernel, module, kernel_name.c_str());
        if (cu_result != CUDA_SUCCESS) {
            cuModuleUnload(module);
            result.success = false;
            result.error_message = "Failed to get kernel function: " + std::to_string(cu_result);
            return result;
        }
        
        result.success = true;
        result.kernel = kernel;
        result.ptx_code = ptx_code;
        
        return result;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = "Exception while loading kernel from PTX: " + std::string(e.what());
        return result;
    }
}

} // namespace cu_op_mem 