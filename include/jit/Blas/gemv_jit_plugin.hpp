#pragma once

#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include <memory>
#include <unordered_map>

namespace cu_op_mem {

// GEMV JIT插件 - 实现IJITPlugin接口
class GemvJITPlugin : public IJITPlugin {
public:
    GemvJITPlugin();
    ~GemvJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "gemv_jit"; }
    void SetConfig(const JITConfig& config) override;
    JITConfig GetConfig() const override;
    void EnableAutoTuning(bool enable) override;
    bool IsAutoTuningEnabled() const override;
    PerformanceProfile GetPerformanceProfile() const override;
    bool IsInitialized() const override;
    bool IsCompiled() const override;
    std::string GetLastError() const override;
    void Cleanup() override;
    size_t GetMemoryUsage() const override;
    
    // GEMV特定接口
    void SetGemvParams(bool transA, float alpha, float beta);
    void SetWeight(const Tensor<float>& weight);
    static bool SupportsOperator(const std::string& op_name);

private:
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateOptimizedKernel(const JITConfig& config);
    
    // 编译和执行方法
    CUfunction CompileKernel(const std::string& kernel_code, const std::string& kernel_name);
    void CacheKernel(const std::string& key, const CUfunction& kernel);
    CUfunction GetCachedKernel(const std::string& key);
    
    // 性能分析
    PerformanceProfile MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                         const std::vector<Tensor<float>>& outputs);
    
    // 配置和验证
    bool ValidateConfig(const JITConfig& config) const;
    JITConfig OptimizeConfig(const JITConfig& config) const;
    HardwareSpec GetHardwareSpec() const;
    
    // 内核选择
    std::string SelectKernelType(int m, int n) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // GEMV特定参数
    bool transA_;
    float alpha_;
    float beta_;
    Tensor<float> weight_;
    
    // 内核缓存
    std::unordered_map<std::string, CUfunction> kernel_cache_;
    std::unordered_map<std::string, std::string> kernel_names_;
    std::string current_kernel_type_;
    
    // 统计信息
    size_t total_executions_;
    double total_execution_time_;
    double total_compilation_time_;
    size_t memory_usage_;
    
    // 性能历史
    std::vector<PerformanceProfile> performance_history_;
    std::unordered_map<std::string, double> kernel_performance_;
};

// 使用宏注册插件
REGISTER_JIT_PLUGIN(gemv, GemvJITPlugin);

} // namespace cu_op_mem 