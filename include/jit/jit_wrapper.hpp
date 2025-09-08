#pragma once

#include "jit_config.hpp"
#include "ijit_plugin.hpp"
#include "jit_compiler.hpp"
#include "global_jit_manager.hpp"
#include "data/tensor.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <memory>
#include <type_traits>
#include <functional>
#include <unordered_map>
#include <mutex>

namespace cu_op_mem {

// 智能JIT包装器 - 核心组件
template<typename OperatorType>
class JITWrapper {
private:
    OperatorType original_operator_;                    // 原始算子
    std::unique_ptr<IJITPlugin> jit_plugin_;          // JIT插件
    JITConfig jit_config_;                            // JIT配置
    bool jit_enabled_;                                // JIT开关
    bool auto_tuning_enabled_;                        // 自动调优开关
    PerformanceProfile last_profile_;                 // 最后一次性能分析
    
public:
    // 构造函数 - 包装现有算子（移动语义）
    explicit JITWrapper(OperatorType&& op) 
        : original_operator_(std::move(op)), jit_enabled_(false), auto_tuning_enabled_(false) {
        InitializeJIT();
    }
    
    // 删除拷贝构造函数（因为OperatorType可能不支持拷贝）
    JITWrapper(const JITWrapper& other) = delete;
    
    // 移动构造函数
    JITWrapper(JITWrapper&& other) noexcept
        : original_operator_(std::move(other.original_operator_)),
          jit_plugin_(std::move(other.jit_plugin_)),
          jit_config_(std::move(other.jit_config_)),
          jit_enabled_(other.jit_enabled_),
          auto_tuning_enabled_(other.auto_tuning_enabled_),
          last_profile_(std::move(other.last_profile_)) {
        other.jit_enabled_ = false;
        other.auto_tuning_enabled_ = false;
    }
    
    // 删除拷贝赋值操作符（因为OperatorType可能不支持拷贝）
    JITWrapper& operator=(const JITWrapper& other) = delete;
    
    JITWrapper& operator=(JITWrapper&& other) noexcept {
        if (this != &other) {
            original_operator_ = std::move(other.original_operator_);
            jit_plugin_ = std::move(other.jit_plugin_);
            jit_config_ = std::move(other.jit_config_);
            jit_enabled_ = other.jit_enabled_;
            auto_tuning_enabled_ = other.auto_tuning_enabled_;
            last_profile_ = std::move(other.last_profile_);
            other.jit_enabled_ = false;
            other.auto_tuning_enabled_ = false;
        }
        return *this;
    }
    
    // 析构函数
    ~JITWrapper() = default;
    
    // 通用调用操作符 - 保持现有接口
    template<typename... Args>
    auto operator()(Args&&... args) -> decltype(original_operator_(std::forward<Args>(args)...)) {
        if (jit_enabled_ && jit_plugin_ && jit_plugin_->IsCompiled()) {
            return ExecuteJIT(std::forward<Args>(args)...);
        } else {
            return ExecuteOriginal(std::forward<Args>(args)...);
        }
    }
    
    // 显式JIT执行接口
    template<typename... Args>
    auto ExecuteJIT(Args&&... args) -> decltype(original_operator_(std::forward<Args>(args)...)) {
        if (!jit_plugin_ || !jit_plugin_->IsCompiled()) {
            // 如果JIT未编译，回退到原始执行
            return ExecuteOriginal(std::forward<Args>(args)...);
        }
        
        try {
            // 转换参数为Tensor格式
            auto inputs = ConvertToTensors(std::forward<Args>(args)...);
            std::vector<Tensor<float>> outputs;
            
            // 执行JIT kernel
            auto status = jit_plugin_->Execute(inputs, outputs);
            if (status != StatusCode::SUCCESS) {
                // JIT执行失败，回退到原始执行
                return ExecuteOriginal(std::forward<Args>(args)...);
            }
            
            // 转换输出
            return ConvertFromTensors(outputs, std::forward<Args>(args)...);
            
        } catch (const std::exception& e) {
            // 异常处理，回退到原始执行
            return ExecuteOriginal(std::forward<Args>(args)...);
        }
    }
    
    // 原始执行接口
    template<typename... Args>
    auto ExecuteOriginal(Args&&... args) -> decltype(original_operator_(std::forward<Args>(args)...)) {
        return original_operator_(std::forward<Args>(args)...);
    }
    
    // 配置接口
    void SetJITConfig(const JITConfig& config) { 
        jit_config_ = config; 
        if (jit_plugin_) {
            jit_plugin_->SetConfig(config);
        }
    }
    JITConfig GetJITConfig() const { return jit_config_; }
    
    // JIT开关控制
    void EnableJIT(bool enable) { 
        jit_enabled_ = enable; 
        if (enable && jit_plugin_ && !jit_plugin_->IsCompiled()) {
            CompileJIT();
        }
    }
    bool IsJITEnabled() const { return jit_enabled_; }
    
    // 自动调优控制
    void EnableAutoTuning(bool enable) { 
        auto_tuning_enabled_ = enable; 
        if (jit_plugin_) {
            jit_plugin_->EnableAutoTuning(enable);
        }
    }
    bool IsAutoTuningEnabled() const { return auto_tuning_enabled_; }
    
    // 性能分析接口
    PerformanceProfile GetPerformanceProfile() const { return last_profile_; }
    void UpdatePerformanceProfile(const PerformanceProfile& profile) { last_profile_ = profile; }
    
    // 状态查询接口
    bool IsJITCompiled() const { return jit_plugin_ && jit_plugin_->IsCompiled(); }
    bool IsJITInitialized() const { return jit_plugin_ && jit_plugin_->IsInitialized(); }
    std::string GetLastError() const { return jit_plugin_ ? jit_plugin_->GetLastError() : ""; }
    
    // 资源管理接口
    void Cleanup() { 
        if (jit_plugin_) {
            jit_plugin_->Cleanup();
        }
    }
    size_t GetMemoryUsage() const { 
        return jit_plugin_ ? jit_plugin_->GetMemoryUsage() : 0; 
    }
    
    // 访问原始算子
    OperatorType& GetOriginalOperator() { return original_operator_; }
    const OperatorType& GetOriginalOperator() const { return original_operator_; }
    
    // 强制编译JIT
    StatusCode CompileJIT() {
        if (!jit_plugin_) return StatusCode::NOT_INITIALIZED;
        
        // 初始化插件
        auto status = jit_plugin_->Initialize();
        if (status != StatusCode::SUCCESS) return status;
        
        // 编译插件
        return jit_plugin_->Compile(jit_config_);
    }
    
    // 优化JIT配置
    void OptimizeJIT() {
        if (jit_plugin_ && auto_tuning_enabled_) {
            jit_plugin_->Optimize(last_profile_);
        }
    }
    
private:
    // 初始化JIT系统
    void InitializeJIT() {
        // 检查全局JIT是否启用
        // 获取全局JIT管理器实例
        auto& global_manager = GlobalJITManager::Instance();
        if (!global_manager.GetGlobalConfig().enable_jit) {
            jit_enabled_ = false;
            return;
        }
        
        // 创建对应的JIT插件
        jit_plugin_ = CreateJITPlugin();
        if (jit_plugin_) {
            jit_plugin_->SetConfig(jit_config_);
            jit_plugin_->EnableAutoTuning(auto_tuning_enabled_);
        }
    }
    
    // 创建JIT插件
    std::unique_ptr<IJITPlugin> CreateJITPlugin() {
        // 根据算子类型创建对应的插件
        std::string plugin_type = GetPluginType<OperatorType>();
        return PluginRegistrar::Instance().CreatePlugin(plugin_type);
    }
    
    // 获取插件类型
    template<typename T>
    static std::string GetPluginType() {
        // 这里需要根据具体的算子类型返回对应的插件类型
        // 可以通过特化或类型特征来实现
        if constexpr (std::is_same_v<T, Gemm<float>>) return "gemm_jit";
        // 其他操作类型可以在这里添加
        return "unknown_jit";
    }
    
    // 参数转换辅助函数 (需要根据具体算子实现)
    template<typename... Args>
    std::vector<Tensor<float>> ConvertToTensors(Args&&... args) {
        // 这里需要根据具体的参数类型实现转换逻辑
        // 暂时返回空向量，具体实现需要根据算子接口定义
        return {};
    }
    
    template<typename... Args>
    auto ConvertFromTensors(const std::vector<Tensor<float>>& outputs, Args&&... args) 
        -> decltype(original_operator_(std::forward<Args>(args)...)) {
        // 这里需要根据具体的输出类型实现转换逻辑
        // 暂时调用原始算子，具体实现需要根据算子接口定义
        return original_operator_(std::forward<Args>(args)...);
    }
};


// 便捷函数：创建JIT包装器
template<typename OperatorType>
JITWrapper<OperatorType> MakeJITWrapper(const OperatorType& op) {
    return JITWrapper<OperatorType>(op);
}

// 便捷函数：启用JIT的包装器
template<typename OperatorType>
JITWrapper<OperatorType> MakeJITWrapper(const OperatorType& op, const JITConfig& config) {
    JITWrapper<OperatorType> wrapper(op);
    wrapper.SetJITConfig(config);
    wrapper.EnableJIT(true);
    return wrapper;
}

} // namespace cu_op_mem 