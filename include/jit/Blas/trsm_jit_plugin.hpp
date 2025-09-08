#pragma once

#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "cuda_op/detail/cuBlas/trsm.hpp"
#include <cuda.h>
#include <memory>
#include <unordered_map>

namespace cu_op_mem {

// TRSM JIT插件 - 实现IJITPlugin接口
class TrsmJITPlugin : public IJITPlugin {
public:
    TrsmJITPlugin();
    ~TrsmJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "trsm_jit"; }
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
    
    // TRSM特定接口
    void SetTrsmParams(int side, int uplo, int trans, int diag, float alpha);
    void SetMatrixA(const Tensor<float>& A);
    static bool SupportsOperator(const std::string& op_name);

private:
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateTiledKernel(const JITConfig& config);
    std::string GenerateWarpOptimizedKernel(const JITConfig& config);
    std::string GenerateBlockedKernel(const JITConfig& config);
    
    // 编译和执行方法
    CUfunction CompileKernel(const std::string& kernel_code, const std::string& kernel_name);
    void CacheKernel(const std::string& key, const CUfunction& kernel);
    CUfunction GetCachedKernel(const std::string& key);
    
    // 性能分析
    PerformanceProfile MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                         const std::vector<Tensor<float>>& outputs);
    
    // 配置验证和优化
    bool ValidateConfig(const JITConfig& config);
    JITConfig OptimizeConfig(const JITConfig& config);
    std::string GenerateKernelKey(const std::string& kernel_code, const JITConfig& config);
    
    // 内核选择
    std::string SelectOptimalKernel(int m, int n);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile performance_profile_;
    
    // TRSM特定参数
    int side_;      // 0: left, 1: right
    int uplo_;      // 0: upper, 1: lower
    int trans_;     // 0: no trans, 1: trans
    int diag_;      // 0: non-unit, 1: unit
    float alpha_;
    Tensor<float> matrix_A_;
    
    // 内核缓存
    std::unordered_map<std::string, CUfunction> kernel_cache_;
    std::unordered_map<std::string, std::string> kernel_names_;
    
    // 性能统计
    size_t total_executions_;
    double total_execution_time_;
    double total_compilation_time_;
    size_t memory_usage_;
    std::string current_kernel_type_;
};

} // namespace cu_op_mem 