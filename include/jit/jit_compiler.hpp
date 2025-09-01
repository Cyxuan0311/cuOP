#pragma once

#include "jit_config.hpp"
#include "jit_persistent_cache.hpp"
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace cu_op_mem {

// JIT编译错误信息
struct JITCompileError {
    std::string error_message;
    std::string kernel_code;
    std::vector<std::string> compile_options;
    std::chrono::system_clock::time_point timestamp;
    
    JITCompileError(const std::string& msg, const std::string& code, 
                   const std::vector<std::string>& options)
        : error_message(msg), kernel_code(code), compile_options(options),
          timestamp(std::chrono::system_clock::now()) {}
};

// JIT编译结果
struct JITCompileResult {
    bool success = false;
    CUfunction kernel = nullptr;
    std::string ptx_code;
    std::string error_message;
    double compilation_time_ms = 0.0;
    std::vector<std::string> compile_options;
    
    JITCompileResult() = default;
    JITCompileResult(bool s, CUfunction k, const std::string& ptx, 
                    const std::string& err, double time, 
                    const std::vector<std::string>& options)
        : success(s), kernel(k), ptx_code(ptx), error_message(err),
          compilation_time_ms(time), compile_options(options) {}
};

// JIT编译器核心类
class JITCompiler {
public:
    JITCompiler();
    ~JITCompiler();
    
    // 禁用拷贝
    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;
    
    // 编译接口
    JITCompileResult CompileKernel(const std::string& kernel_code,
                                  const std::string& kernel_name,
                                  const std::vector<std::string>& options = {});
    
    // 缓存管理
    void CacheKernel(const std::string& key, const CUfunction& kernel);
    CUfunction GetCachedKernel(const std::string& key);
    bool IsKernelCached(const std::string& key) const;
    void ClearCache();
    void ClearExpiredCache();
    
    // 持久化缓存管理
    void EnablePersistentCache(bool enable);
    bool IsPersistentCacheEnabled() const;
    void SetPersistentCacheDirectory(const std::string& cache_dir);
    std::string GetPersistentCacheDirectory() const;
    
    // 配置管理
    void SetCompilationTimeout(int timeout_seconds);
    void SetMaxCacheSize(size_t max_size);
    void EnableDebug(bool enable);
    
    // 统计信息
    JITStatistics GetStatistics() const;
    void ResetStatistics();
    
    // 错误处理
    std::vector<JITCompileError> GetCompileErrors() const;
    void ClearCompileErrors();
    
    // 工具函数
    static std::vector<std::string> GetDefaultCompileOptions();
    static std::vector<std::string> GetOptimizationOptions(const std::string& level);
    static std::string GenerateKernelKey(const std::string& kernel_code,
                                        const std::vector<std::string>& options);
    
private:
    // 内部实现
    JITCompileResult CompileWithNVRTC(const std::string& kernel_code,
                                     const std::string& kernel_name,
                                     const std::vector<std::string>& options);
    
    bool ValidateKernelCode(const std::string& kernel_code) const;
    std::vector<std::string> MergeCompileOptions(const std::vector<std::string>& user_options) const;
    void LogCompileError(const std::string& error_message, const std::string& kernel_code,
                        const std::vector<std::string>& options);
    
    // 持久化缓存相关
    bool TryLoadFromPersistentCache(const std::string& cache_key, JITCompileResult& result);
    void SaveToPersistentCache(const std::string& cache_key, const JITCompileResult& result);
    
    // 成员变量
    std::unordered_map<std::string, CUfunction> kernel_cache_;
    std::unordered_map<std::string, std::chrono::system_clock::time_point> cache_timestamps_;
    std::vector<JITCompileError> compile_errors_;
    JITStatistics statistics_;
    
    // 配置
    int compilation_timeout_seconds_ = 30;
    size_t max_cache_size_ = 1024 * 1024 * 1024; // 1GB
    bool debug_enabled_ = false;
    
    // 持久化缓存配置
    bool persistent_cache_enabled_ = false;
    std::string persistent_cache_dir_ = "./jit_cache";
    
    // 同步
    mutable std::mutex cache_mutex_;
    mutable std::mutex error_mutex_;
    mutable std::mutex stats_mutex_;
    
    // NVRTC相关
    nvrtcProgram program_;
    bool nvrtc_initialized_ = false;
};

// 内核模板基类
class IKernelTemplate {
public:
    virtual ~IKernelTemplate() = default;
    
    // 生成内核代码
    virtual std::string GenerateKernelCode(const JITConfig& config) = 0;
    
    // 获取内核名称
    virtual std::string GetKernelName() const = 0;
    
    // 获取编译选项
    virtual std::vector<std::string> GetCompileOptions(const JITConfig& config) const = 0;
    
    // 验证配置
    virtual bool ValidateConfig(const JITConfig& config) const = 0;
    
    // 获取模板类型
    virtual std::string GetTemplateType() const = 0;
};

// 模板管理器
class KernelTemplateManager {
public:
    static KernelTemplateManager& Instance();
    
    // 注册模板
    void RegisterTemplate(const std::string& name, std::unique_ptr<IKernelTemplate> template_ptr);
    
    // 获取模板
    IKernelTemplate* GetTemplate(const std::string& name);
    
    // 创建内核
    JITCompileResult CreateKernel(const std::string& template_name,
                                 const JITConfig& config,
                                 JITCompiler& compiler);
    
    // 获取支持的模板类型
    std::vector<std::string> GetSupportedTemplates() const;
    
private:
    std::unordered_map<std::string, std::unique_ptr<IKernelTemplate>> templates_;
    mutable std::mutex mutex_;
};

// 宏定义：简化模板注册
#define REGISTER_KERNEL_TEMPLATE(template_name, template_class) \
    static auto template_name##_registrar = []() { \
        auto template_ptr = std::make_unique<template_class>(); \
        KernelTemplateManager::Instance().RegisterTemplate(#template_name, std::move(template_ptr)); \
        return 0; \
    }()

} // namespace cu_op_mem 