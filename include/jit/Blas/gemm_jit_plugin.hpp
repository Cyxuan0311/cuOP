#pragma once

#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <cuda.h>
#include <memory>
#include <unordered_map>

namespace cu_op_mem {

// GEMM JIT插件 - 实现IJITPlugin接口
class GemmJITPlugin : public IJITPlugin {
public:
    GemmJITPlugin();
    ~GemmJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "gemm_jit"; }
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
    
    // GEMM特定接口
    void SetGemmParams(bool transA, bool transB, float alpha, float beta);
    void SetWeight(const Tensor<float>& weight);
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法 (public for GemmKernelTemplate access)
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateTiledKernel(const JITConfig& config);
    std::string GenerateWarpOptimizedKernel(const JITConfig& config);
    std::string GenerateTensorCoreKernel(const JITConfig& config);
    std::string GenerateBlockedKernel(const JITConfig& config);

private:
    
    // 编译和执行方法
    CUfunction CompileKernel(const std::string& kernel_code, const std::string& kernel_name);
    void CacheKernel(const std::string& key, const CUfunction& kernel);
    CUfunction GetCachedKernel(const std::string& key);
    std::string GenerateKernelKey(const std::string& kernel_code, const JITConfig& config);
    
    // 性能分析
    PerformanceProfile MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                         const std::vector<Tensor<float>>& outputs);
    
    // 配置和验证
    bool ValidateConfig(const JITConfig& config) const;
    JITConfig OptimizeConfig(const JITConfig& config) const;
    HardwareSpec GetHardwareSpec() const;
    bool SupportsTensorCore() const;
    bool SupportsTMA() const;
    
    // 内核选择
    std::string SelectKernelType(int m, int n, int k) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // GEMM特定参数
    bool transA_;
    bool transB_;
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

// GEMM内核模板 - 实现IKernelTemplate接口
class GemmKernelTemplate : public IKernelTemplate {
public:
    std::string GenerateKernelCode(const JITConfig& config) override;
    std::string GetKernelName() const override { return "gemm_kernel"; }
    std::vector<std::string> GetCompileOptions(const JITConfig& config) const override;
    bool ValidateConfig(const JITConfig& config) const override;
    std::string GetTemplateType() const override { return "gemm"; }

private:
    // 辅助方法
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateTiledKernel(const JITConfig& config);
    std::string GenerateWarpOptimizedKernel(const JITConfig& config);
    std::string GenerateTensorCoreKernel(const JITConfig& config);
    std::string GenerateBlockedKernel(const JITConfig& config);
    std::string GenerateTMAKernel(const JITConfig& config);
    
    // 模板生成辅助
    std::string GetKernelTemplate(const std::string& kernel_type, const JITConfig& config);
    std::vector<std::string> GetOptimizationFlags(const JITConfig& config) const;
    std::string GenerateKernelSignature(const JITConfig& config) const;
    std::string GenerateKernelBody(const std::string& kernel_type, const JITConfig& config) const;
};

// 插件工厂类
class GemmJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<GemmJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "gemm" || op_name == "Gemm";
    }
    
    std::string GetPluginType() const override {
        return "gemm";
    }
};

// 使用宏注册插件
REGISTER_JIT_PLUGIN(gemm, GemmJITPlugin);

} // namespace cu_op_mem 