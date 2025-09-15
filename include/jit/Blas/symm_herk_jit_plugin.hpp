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

// 对称矩阵运算类型枚举
enum class SymmetricOpType {
    SYMM,   // 对称矩阵乘法: C = α * A * B + β * C (A对称)
    HERK,   // Hermitian秩-k更新: C = α * A * A^H + β * C
    SYRK,   // 对称秩-k更新: C = α * A * A^T + β * C
    HER2K,  // Hermitian秩-2k更新: C = α * A * B^H + α * B * A^H + β * C
    SYR2K   // 对称秩-2k更新: C = α * A * B^T + α * B * A^T + β * C
};

// 对称矩阵运算JIT插件 - 实现IJITPlugin接口
class SymmHerkJITPlugin : public IJITPlugin {
public:
    SymmHerkJITPlugin();
    ~SymmHerkJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "symm_herk_jit"; }
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
    
    // 对称矩阵运算特定接口
    void SetOperationType(SymmetricOpType op_type);
    void SetMatrixDimensions(int m, int n);
    void SetSideMode(bool left_side);  // true: left side, false: right side
    void SetUploMode(bool upper);      // true: upper triangle, false: lower triangle
    void SetTranspose(bool trans);
    void SetAlphaBeta(float alpha, float beta);
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateSymmKernel(const JITConfig& config);
    std::string GenerateHerkKernel(const JITConfig& config);
    std::string GenerateSyrkKernel(const JITConfig& config);
    std::string GenerateHer2kKernel(const JITConfig& config);
    std::string GenerateSyr2kKernel(const JITConfig& config);

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
    std::string SelectKernelType(SymmetricOpType op_type, int m, int n) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // 对称矩阵运算特定参数
    SymmetricOpType op_type_;
    int m_, n_;              // 矩阵维度
    bool left_side_;         // 侧模式
    bool upper_;             // 三角模式
    bool trans_;             // 转置标志
    float alpha_, beta_;     // 标量系数
    
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
class SymmHerkJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<SymmHerkJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "symm" || op_name == "Symm" ||
               op_name == "herk" || op_name == "Herk" ||
               op_name == "syrk" || op_name == "Syrk" ||
               op_name == "her2k" || op_name == "Her2k" ||
               op_name == "syr2k" || op_name == "Syr2k";
    }
    
    std::string GetPluginType() const override {
        return "symm_herk";
    }
};

// 插件工厂类已在上方定义

} // namespace cu_op_mem
