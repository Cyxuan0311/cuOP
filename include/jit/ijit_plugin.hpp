#pragma once

#include "jit_config.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

namespace cu_op_mem {

// 前向声明
class JITCompiler;

// JIT插件接口
class IJITPlugin {
public:
    virtual ~IJITPlugin() = default;
    
    // 基础接口
    virtual StatusCode Initialize() = 0;
    virtual StatusCode Compile(const JITConfig& config) = 0;
    virtual StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                              std::vector<Tensor<float>>& outputs) = 0;
    virtual void Optimize(const PerformanceProfile& profile) = 0;
    virtual std::string GetPluginName() const = 0;
    
    // 配置接口
    virtual void SetConfig(const JITConfig& config) = 0;
    virtual JITConfig GetConfig() const = 0;
    
    // 性能接口
    virtual void EnableAutoTuning(bool enable) = 0;
    virtual bool IsAutoTuningEnabled() const = 0;
    virtual PerformanceProfile GetPerformanceProfile() const = 0;
    
    // 状态接口
    virtual bool IsInitialized() const = 0;
    virtual bool IsCompiled() const = 0;
    virtual std::string GetLastError() const = 0;
    
    // 资源管理
    virtual void Cleanup() = 0;
    virtual size_t GetMemoryUsage() const = 0;
    
protected:
    // 子类可以访问的辅助函数
    StatusCode CompileKernel(const std::string& kernel_code, 
                            const std::string& kernel_name,
                            const std::vector<std::string>& options);
    CUfunction GetCompiledKernel(const std::string& kernel_name);
    void CacheKernel(const std::string& key, const CUfunction& kernel);
    
    // 性能分析辅助函数
    PerformanceProfile MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                         const std::vector<Tensor<float>>& outputs);
    
    // 硬件检测辅助函数
    HardwareSpec GetHardwareSpec() const;
    bool SupportsTensorCore() const;
    bool SupportsTMA() const;
    
    // 配置验证辅助函数
    bool ValidateConfig(const JITConfig& config) const;
    JITConfig OptimizeConfig(const JITConfig& config) const;
    
protected:
    // 内部实现细节 - 子类可以访问
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_ = false;
    bool compiled_ = false;
    bool auto_tuning_enabled_ = false;
    std::string last_error_;
    PerformanceProfile last_profile_;
};

// 插件工厂接口
class IPluginFactory {
public:
    virtual ~IPluginFactory() = default;
    virtual std::unique_ptr<IJITPlugin> CreatePlugin() = 0;
    virtual std::string GetPluginType() const = 0;
    virtual bool SupportsOperator(const std::string& op_name) const = 0;
};

// 插件注册器
class PluginRegistrar {
public:
    static PluginRegistrar& Instance();
    
    // 注册插件工厂
    void RegisterPluginFactory(const std::string& plugin_type, 
                              std::unique_ptr<IPluginFactory> factory);
    
    // 创建插件
    std::unique_ptr<IJITPlugin> CreatePlugin(const std::string& plugin_type);
    
    // 获取支持的插件类型
    std::vector<std::string> GetSupportedPluginTypes() const;
    
    // 检查插件是否支持特定算子
    bool IsOperatorSupported(const std::string& op_name) const;
    
private:
    std::unordered_map<std::string, std::unique_ptr<IPluginFactory>> factories_;
    mutable std::mutex mutex_;
};

// 插件管理器
class PluginManager {
public:
    static PluginManager& Instance();
    
    // 插件管理
    void RegisterPlugin(const std::string& name, std::unique_ptr<IJITPlugin> plugin);
    IJITPlugin* GetPlugin(const std::string& name);
    void RemovePlugin(const std::string& name);
    
    // 批量操作
    void InitializeAllPlugins();
    void CompileAllPlugins(const JITConfig& config);
    void CleanupAllPlugins();
    
    // 统计信息
    std::vector<std::string> GetActivePlugins() const;
    size_t GetTotalMemoryUsage() const;
    JITStatistics GetTotalStatistics() const;
    
private:
    std::unordered_map<std::string, std::unique_ptr<IJITPlugin>> plugins_;
    mutable std::mutex mutex_;
};

// 宏定义：简化插件注册
#define REGISTER_JIT_PLUGIN(plugin_type, plugin_class) \
    static auto plugin_type##_registrar = []() { \
        auto factory = std::make_unique<plugin_class##Factory>(); \
        PluginRegistrar::Instance().RegisterPluginFactory(#plugin_type, std::move(factory)); \
        return 0; \
    }()

// 宏定义：创建插件工厂类
#define CREATE_PLUGIN_FACTORY(plugin_class) \
    class plugin_class##Factory : public IPluginFactory { \
    public: \
        std::unique_ptr<IJITPlugin> CreatePlugin() override { \
            return std::make_unique<plugin_class>(); \
        } \
        std::string GetPluginType() const override { \
            return #plugin_class; \
        } \
        bool SupportsOperator(const std::string& op_name) const override { \
            return plugin_class::SupportsOperator(op_name); \
        } \
    }

} // namespace cu_op_mem 