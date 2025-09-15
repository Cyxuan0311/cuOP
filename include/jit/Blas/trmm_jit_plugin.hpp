#pragma once

#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "jit/jit_config.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <cuda.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cu_op_mem {

// TRMM JIT插件 - 实现IJITPlugin接口
// TRMM: Triangular Matrix-Matrix Multiply
// 计算 B = α * op(A) * B 或 B = α * B * op(A) (A为三角矩阵)
class TrmmJITPlugin : public IJITPlugin {
public:
    TrmmJITPlugin();
    ~TrmmJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "trmm_jit"; }
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
    
    // TRMM特定接口
    void SetTrmmParams(int side, int uplo, int trans, int diag, float alpha);
    void SetMatrixA(const Tensor<float>& A);
    void SetMatrixB(const Tensor<float>& B);
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateTiledKernel(const JITConfig& config);
    std::string GenerateWarpOptimizedKernel(const JITConfig& config);
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
    
    // 配置验证和优化
    bool ValidateConfig(const JITConfig& config) const;
    JITConfig OptimizeConfig(const JITConfig& config) const;
    HardwareSpec GetHardwareSpec() const;
    
    // 内核选择
    std::string SelectKernelType(int side, int m, int n) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // TRMM特定参数
    int side_;      // 0: left side (B = α * op(A) * B), 1: right side (B = α * B * op(A))
    int uplo_;      // 0: upper triangle, 1: lower triangle
    int trans_;     // 0: no transpose, 1: transpose, 2: conjugate transpose
    int diag_;      // 0: non-unit diagonal, 1: unit diagonal
    float alpha_;
    const Tensor<float>* matrix_A_ref_;
    const Tensor<float>* matrix_B_ref_;
    
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

// 插件工厂类
class TrmmJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<TrmmJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "trmm" || op_name == "Trmm";
    }
    
    std::string GetPluginType() const override {
        return "trmm";
    }
};

// 插件工厂类已在上方定义

} // namespace cu_op_mem
