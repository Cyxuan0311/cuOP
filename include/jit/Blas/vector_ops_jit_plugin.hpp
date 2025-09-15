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

// 向量运算类型枚举
enum class VectorOpType {
    DOT,    // 向量点积: result = x^T * y
    AXPY,   // 向量缩放加法: y = α * x + y
    SCAL,   // 向量缩放: x = α * x
    COPY,   // 向量复制: y = x
    SWAP,   // 向量交换: x ↔ y
    ROT,    // 向量旋转: x' = c*x + s*y, y' = -s*x + c*y
    NRM2,   // 欧几里得范数: result = ||x||_2
    ASUM,   // 绝对值之和: result = sum(|x_i|)
    IAMAX,  // 最大绝对值索引: result = argmax(|x_i|)
    IAMIN   // 最小绝对值索引: result = argmin(|x_i|)
};

// 向量运算JIT插件 - 实现IJITPlugin接口
class VectorOpsJITPlugin : public IJITPlugin {
public:
    VectorOpsJITPlugin();
    ~VectorOpsJITPlugin() override;
    
    // IJITPlugin接口实现
    StatusCode Initialize() override;
    StatusCode Compile(const JITConfig& config) override;
    StatusCode Execute(const std::vector<Tensor<float>>& inputs, 
                      std::vector<Tensor<float>>& outputs) override;
    void Optimize(const PerformanceProfile& profile) override;
    std::string GetPluginName() const override { return "vector_ops_jit"; }
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
    
    // 向量运算特定接口
    void SetOperationType(VectorOpType op_type);
    void SetVectorSize(int size);
    void SetAlpha(float alpha);
    void SetBeta(float beta);
    void SetIncrement(int incx, int incy = 1);
    void SetRotationParams(float c, float s);  // 用于ROT操作
    static bool SupportsOperator(const std::string& op_name);
    
    // 内核生成方法
    std::string GenerateKernelCode(const JITConfig& config);
    std::string GenerateDotKernel(const JITConfig& config);
    std::string GenerateAxpyKernel(const JITConfig& config);
    std::string GenerateScalKernel(const JITConfig& config);
    std::string GenerateCopyKernel(const JITConfig& config);
    std::string GenerateSwapKernel(const JITConfig& config);
    std::string GenerateRotKernel(const JITConfig& config);
    std::string GenerateNrm2Kernel(const JITConfig& config);
    std::string GenerateAsumKernel(const JITConfig& config);
    std::string GenerateIamaxKernel(const JITConfig& config);
    std::string GenerateIaminKernel(const JITConfig& config);

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
    std::string SelectKernelType(VectorOpType op_type, int vector_size) const;
    void UpdateKernelSelection(const PerformanceProfile& profile);
    
    // 成员变量
    std::unique_ptr<JITCompiler> compiler_;
    JITConfig config_;
    bool initialized_;
    bool compiled_;
    bool auto_tuning_enabled_;
    std::string last_error_;
    PerformanceProfile last_profile_;
    
    // 向量运算特定参数
    VectorOpType op_type_;
    int vector_size_;
    float alpha_, beta_;
    int incx_, incy_;        // 向量增量
    float c_, s_;            // 旋转参数 (用于ROT)
    
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
class VectorOpsJITPluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IJITPlugin> CreatePlugin() override {
        return std::make_unique<VectorOpsJITPlugin>();
    }
    
    bool SupportsOperator(const std::string& op_name) const override {
        return op_name == "dot" || op_name == "Dot" ||
               op_name == "axpy" || op_name == "Axpy" ||
               op_name == "scal" || op_name == "Scal" ||
               op_name == "copy" || op_name == "Copy" ||
               op_name == "swap" || op_name == "Swap" ||
               op_name == "rot" || op_name == "Rot" ||
               op_name == "nrm2" || op_name == "Nrm2" ||
               op_name == "asum" || op_name == "Asum" ||
               op_name == "iamax" || op_name == "Iamax" ||
               op_name == "iamin" || op_name == "Iamin";
    }
    
    std::string GetPluginType() const override {
        return "vector_ops";
    }
};

// 插件工厂类已在上方定义

} // namespace cu_op_mem
