#pragma once

#include "jit/ijit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "jit/jit_config.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <cuda.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cu_op_mem {

// Batched GEMM JIT插件 - 实现IJITPlugin接口
class GemmBatchedJITPlugin : public IJITPlugin {
public:
    GemmBatchedJITPlugin();
    ~GemmBatchedJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "gemm_batched_jit"; }
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
    
    // Batched GEMM特定接口
    void SetBatchSize(int batch_size);
    void SetMatrixDimensions(int m, int n, int k);
    void SetTransposeOptions(bool transA, bool transB);
    void SetAlphaBeta(float alpha, float beta);
    void SetBatchStride(int strideA, int strideB, int strideC);
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateStandardKernel(const JITConfig& config);
    std::string GenerateOptimizedKernel(const JITConfig& config);
    std::string GenerateTensorCoreKernel(const JITConfig& config);

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
    bool SupportsTensorCore() const;
    
    // 内核选择
    std::string SelectKernelType(int batch_size, int m, int n, int k) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // Batched GEMM特定参数
    int batch_size_;
    int m_, n_, k_;           // 矩阵维度
    bool transA_, transB_;    // 转置标志
    float alpha_, beta_;      // 标量系数
    int strideA_, strideB_, strideC_;  // 批量步长
    
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
class GemmBatchedJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<GemmBatchedJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "gemm_batched" || op_name == "GemmBatched" || 
               op_name == "batched_gemm" || op_name == "BatchedGemm";
    }
    
    std::string GetPluginType() const override {
        return "gemm_batched";
    }
};

// 插件工厂类已在上方定义

} // namespace cu_op_mem
