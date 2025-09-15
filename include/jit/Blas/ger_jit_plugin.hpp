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

// GER JIT插件 - 实现IJITPlugin接口
// GER: General Rank-1 Update
// 计算 A = α * x * y^T + A (A为矩阵，x、y为向量)
class GerJITPlugin : public IJITPlugin {
public:
    GerJITPlugin();
    ~GerJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "ger_jit"; }
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
    
    // GER特定接口
    void SetGerParams(float alpha);
    void SetMatrixDimensions(int m, int n);
    void SetVectorIncrements(int incx, int incy);
    void SetMatrixA(const Tensor<float>& A);
    void SetVectorX(const Tensor<float>& x);
    void SetVectorY(const Tensor<float>& y);
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateBasicKernel(const JITConfig& config);
    std::string GenerateTiledKernel(const JITConfig& config);
    std::string GenerateWarpOptimizedKernel(const JITConfig& config);
    std::string GenerateSharedMemoryKernel(const JITConfig& config);

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
    
    // GER特定参数
    float alpha_;
    int m_, n_;              // 矩阵维度
    int incx_, incy_;        // 向量增量
    const Tensor<float>* matrix_A_ref_;
    const Tensor<float>* vector_x_ref_;
    const Tensor<float>* vector_y_ref_;
    
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
class GerJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<GerJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "ger" || op_name == "Ger";
    }
    
    std::string GetPluginType() const override {
        return "ger";
    }
};

// 插件工厂类已在上方定义

} // namespace cu_op_mem
