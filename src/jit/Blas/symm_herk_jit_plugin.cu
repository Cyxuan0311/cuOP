#include "jit/Blas/symm_herk_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

SymmHerkJITPlugin::SymmHerkJITPlugin() 
    : compiler_(std::make_unique<JITCompiler>())
    , initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , op_type_(SymmetricOpType::SYMM)
    , m_(0), n_(0)
    , left_side_(true)
    , upper_(true)
    , trans_(false)
    , alpha_(1.0f), beta_(0.0f)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0) {
    LOG(INFO) << "SymmHerkJITPlugin created";
}

SymmHerkJITPlugin::~SymmHerkJITPlugin() {
    Cleanup();
}

StatusCode SymmHerkJITPlugin::Initialize() {
    if (initialized_) {
        return StatusCode::SUCCESS;
    }
    
    try {
        // 初始化JIT编译器
        if (!compiler_) {
            compiler_ = std::make_unique<JITCompiler>();
        }
        
        // 设置默认配置
        config_.enable_jit = true;
        config_.use_shared_memory = true;
        config_.optimization_level = "O3";
        config_.block_size = 256;
        
        initialized_ = true;
        LOG(INFO) << "SymmHerkJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_INITIALIZATION_ERROR;
    }
}

StatusCode SymmHerkJITPlugin::Compile(const JITConfig& config) {
    if (!initialized_) {
        return StatusCode::JIT_NOT_INITIALIZED;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 更新配置
        config_ = config;
        
        // 验证配置
        if (!ValidateConfig(config)) {
            last_error_ = "Invalid JIT configuration";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 选择最优内核类型
        std::string kernel_type = SelectKernelType(op_type_, m_, n_);
        current_kernel_type_ = kernel_type;
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "symm_herk_kernel_" + kernel_type;
        CUfunction kernel = CompileKernel(kernel_code, kernel_name);
        if (!kernel) {
            last_error_ = "Failed to compile kernel: " + kernel_name;
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 缓存内核
        std::string cache_key = GenerateKernelKey(kernel_code, config);
        CacheKernel(cache_key, kernel);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        total_compilation_time_ += duration.count();
        
        compiled_ = true;
        LOG(INFO) << "SymmHerkJITPlugin compiled successfully in " << duration.count() << "ms";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_COMPILATION_ERROR;
    }
}

StatusCode SymmHerkJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
                                     std::vector<Tensor<float>>& outputs) {
    if (!compiled_) {
        return StatusCode::JIT_NOT_COMPILED;
    }
    
    if (inputs.empty() || outputs.empty()) {
        last_error_ = "Invalid input/output tensor count";
        return StatusCode::INVALID_ARGUMENT;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 获取编译好的内核
        std::string cache_key = GenerateKernelKey(GenerateKernelCode(config_), config_);
        CUfunction kernel = GetCachedKernel(cache_key);
        if (!kernel) {
            last_error_ = "Kernel not found in cache";
            return StatusCode::JIT_EXECUTION_ERROR;
        }
        
        // 设置内核参数
        const Tensor<float>& A = inputs[0];
        Tensor<float>& C = outputs[0];
        
        // 计算网格和块大小
        int block_size_x = config_.block_size_x > 0 ? config_.block_size_x : 16;
        int block_size_y = config_.block_size_y > 0 ? config_.block_size_y : 16;
        
        dim3 block(block_size_x, block_size_y);
        dim3 grid((m_ + block_size_x - 1) / block_size_x, 
                  (n_ + block_size_y - 1) / block_size_y);
        
        // 设置内核参数
        void* args[] = {
            A.data(), C.data(),
            &m_, &n_,
            &alpha_, &beta_,
            &left_side_, &upper_, &trans_
        };
        
        // 启动内核
        CUresult result = cuLaunchKernel(kernel, 
                                        grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z,
                                        0, 0, args, 0);
        
        if (result != CUDA_SUCCESS) {
            last_error_ = "Kernel launch failed";
            return StatusCode::CUDA_LAUNCH_ERROR;
        }
        
        // 同步等待完成
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_execution_time_ += duration.count() / 1000.0;
        total_executions_++;
        
        // 更新性能配置
        if (auto_tuning_enabled_) {
            PerformanceProfile profile = MeasurePerformance(inputs, outputs);
            UpdateKernelSelection(profile);
        }
        
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Execution failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_EXECUTION_ERROR;
    }
}

void SymmHerkJITPlugin::Optimize(const PerformanceProfile& profile) {
    if (!auto_tuning_enabled_) {
        return;
    }
    
    // 基于性能配置优化参数
    if (profile.execution_time > last_profile_.execution_time * 1.1) {
        // 性能下降，尝试不同的配置
        if (config_.block_size_x < 32) {
            config_.block_size_x *= 2;
        } else if (config_.block_size_y < 32) {
            config_.block_size_y *= 2;
        }
        
        // 重新编译
        Compile(config_);
    }
    
    last_profile_ = profile;
}

void SymmHerkJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig SymmHerkJITPlugin::GetConfig() const {
    return config_;
}

void SymmHerkJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool SymmHerkJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile SymmHerkJITPlugin::GetPerformanceProfile() const {
    PerformanceProfile profile = last_profile_;
    profile.execution_time = total_executions_ > 0 ? total_execution_time_ / total_executions_ : 0.0;
    profile.compilation_time_ms = total_compilation_time_;
    profile.memory_usage = memory_usage_;
    return profile;
}

bool SymmHerkJITPlugin::IsInitialized() const {
    return initialized_;
}

bool SymmHerkJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string SymmHerkJITPlugin::GetLastError() const {
    return last_error_;
}

void SymmHerkJITPlugin::Cleanup() {
    kernel_cache_.clear();
    kernel_names_.clear();
    compiled_ = false;
    LOG(INFO) << "SymmHerkJITPlugin cleaned up";
}

size_t SymmHerkJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

// 对称矩阵运算特定接口实现
void SymmHerkJITPlugin::SetOperationType(SymmetricOpType op_type) {
    op_type_ = op_type;
}

void SymmHerkJITPlugin::SetMatrixDimensions(int m, int n) {
    m_ = m;
    n_ = n;
}

void SymmHerkJITPlugin::SetSideMode(bool left_side) {
    left_side_ = left_side;
}

void SymmHerkJITPlugin::SetUploMode(bool upper) {
    upper_ = upper;
}

void SymmHerkJITPlugin::SetTranspose(bool trans) {
    trans_ = trans;
}

void SymmHerkJITPlugin::SetAlphaBeta(float alpha, float beta) {
    alpha_ = alpha;
    beta_ = beta;
}

bool SymmHerkJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "symm" || op_name == "Symm" ||
           op_name == "herk" || op_name == "Herk" ||
           op_name == "syrk" || op_name == "Syrk" ||
           op_name == "her2k" || op_name == "Her2k" ||
           op_name == "syr2k" || op_name == "Syr2k";
}

// 内核生成方法实现
std::string SymmHerkJITPlugin::GenerateKernelCode(const JITConfig& config) {
    switch (op_type_) {
        case SymmetricOpType::SYMM:
            return GenerateSymmKernel(config);
        case SymmetricOpType::HERK:
            return GenerateHerkKernel(config);
        case SymmetricOpType::SYRK:
            return GenerateSyrkKernel(config);
        case SymmetricOpType::HER2K:
            return GenerateHer2kKernel(config);
        case SymmetricOpType::SYR2K:
            return GenerateSyr2kKernel(config);
        default:
            return GenerateSymmKernel(config);
    }
}

std::string SymmHerkJITPlugin::GenerateSymmKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void symm_herk_kernel_symm(
    const float* __restrict__ A,
    float* __restrict__ C,
    int m, int n,
    float alpha, float beta,
    bool left_side, bool upper, bool trans
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    if (left_side) {
        // C = α * A * B + β * C (A是m×m对称矩阵)
        for (int k = 0; k < m; ++k) {
            float a_val;
            if (upper) {
                a_val = (k <= row) ? A[row * m + k] : A[k * m + row];
            } else {
                a_val = (k >= row) ? A[row * m + k] : A[k * m + row];
            }
            sum += a_val * B[k * n + col];
        }
    } else {
        // C = α * B * A + β * C (A是n×n对称矩阵)
        for (int k = 0; k < n; ++k) {
            float a_val;
            if (upper) {
                a_val = (k <= col) ? A[col * n + k] : A[k * n + col];
            } else {
                a_val = (k >= col) ? A[col * n + k] : A[k * n + col];
            }
            sum += B[row * n + k] * a_val;
        }
    }
    
    C[row * n + col] = alpha * sum + beta * C[row * n + col];
}
)";
    
    return kernel.str();
}

std::string SymmHerkJITPlugin::GenerateHerkKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void symm_herk_kernel_herk(
    const float* __restrict__ A,
    float* __restrict__ C,
    int m, int n,
    float alpha, float beta,
    bool left_side, bool upper, bool trans
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;
    
    // C = α * A * A^H + β * C
    for (int k = 0; k < m; ++k) {
        sum += A[row * m + k] * A[col * m + k];
    }
    
    // 只更新上三角或下三角部分
    if (upper && row <= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    } else if (!upper && row >= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return kernel.str();
}

std::string SymmHerkJITPlugin::GenerateSyrkKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void symm_herk_kernel_syrk(
    const float* __restrict__ A,
    float* __restrict__ C,
    int m, int n,
    float alpha, float beta,
    bool left_side, bool upper, bool trans
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;
    
    // C = α * A * A^T + β * C
    for (int k = 0; k < m; ++k) {
        sum += A[row * m + k] * A[col * m + k];
    }
    
    // 只更新上三角或下三角部分
    if (upper && row <= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    } else if (!upper && row >= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return kernel.str();
}

std::string SymmHerkJITPlugin::GenerateHer2kKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void symm_herk_kernel_her2k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n,
    float alpha, float beta,
    bool left_side, bool upper, bool trans
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;
    
    // C = α * A * B^H + α * B * A^H + β * C
    for (int k = 0; k < m; ++k) {
        sum += A[row * m + k] * B[col * m + k] + B[row * m + k] * A[col * m + k];
    }
    
    // 只更新上三角或下三角部分
    if (upper && row <= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    } else if (!upper && row >= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return kernel.str();
}

std::string SymmHerkJITPlugin::GenerateSyr2kKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void symm_herk_kernel_syr2k(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n,
    float alpha, float beta,
    bool left_side, bool upper, bool trans
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;
    
    // C = α * A * B^T + α * B * A^T + β * C
    for (int k = 0; k < m; ++k) {
        sum += A[row * m + k] * B[col * m + k] + B[row * m + k] * A[col * m + k];
    }
    
    // 只更新上三角或下三角部分
    if (upper && row <= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    } else if (!upper && row >= col) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return kernel.str();
}

// 私有方法实现
CUfunction SymmHerkJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
    if (!compiler_) {
        return nullptr;
    }
    
    std::vector<std::string> options = {
        "-std=c++17",
        "-O3",
        "--use_fast_math"
    };
    
    auto result = compiler_->CompileKernel(kernel_code, kernel_name, options);
    if (result.success) {
        return result.kernel;
    }
    
    last_error_ = result.error_message;
    return nullptr;
}

void SymmHerkJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
    kernel_names_[key] = "symm_herk_kernel_" + current_kernel_type_;
}

CUfunction SymmHerkJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

std::string SymmHerkJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::ostringstream key;
    key << "symm_herk_" << static_cast<int>(op_type_) << "_" 
        << m_ << "x" << n_ << "_"
        << config.block_size_x << "x" << config.block_size_y << "_"
        << (left_side_ ? "left" : "right") << "_"
        << (upper_ ? "upper" : "lower");
    return key.str();
}

PerformanceProfile SymmHerkJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                        const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    
    // 计算GFLOPS (简化计算)
    double flops = 2.0 * m_ * n_ * m_;  // 简化估算
    profile.gflops = flops / (total_execution_time_ / total_executions_) / 1e6;
    
    // 计算内存带宽
    double bytes = (m_ * m_ + m_ * n_ + m_ * n_) * sizeof(float);
    profile.bandwidth_gb_s = bytes / (total_execution_time_ / total_executions_) / 1e6;
    
    profile.kernel_time_ms = total_execution_time_ / total_executions_;
    profile.execution_time = total_execution_time_ / total_executions_;
    profile.kernel_name = kernel_names_[GenerateKernelKey(GenerateKernelCode(config_), config_)];
    profile.kernel_type = current_kernel_type_;
    
    return profile;
}

bool SymmHerkJITPlugin::ValidateConfig(const JITConfig& config) const {
    return config.block_size_x > 0 && config.block_size_y > 0 &&
           config.block_size_x <= 32 && config.block_size_y <= 32;
}

JITConfig SymmHerkJITPlugin::OptimizeConfig(const JITConfig& config) const {
    JITConfig optimized = config;
    
    // 基于矩阵大小优化块大小
    if (m_ <= 64 && n_ <= 64) {
        optimized.block_size_x = 16;
        optimized.block_size_y = 16;
    } else if (m_ <= 128 && n_ <= 128) {
        optimized.block_size_x = 32;
        optimized.block_size_y = 32;
    } else {
        optimized.block_size_x = 16;
        optimized.block_size_y = 16;
    }
    
    return optimized;
}

HardwareSpec SymmHerkJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    // 这里应该从CUDA运行时获取硬件信息
    // 简化实现
    spec.compute_capability_major = 7;
    spec.compute_capability_minor = 5;
    return spec;
}

std::string SymmHerkJITPlugin::SelectKernelType(SymmetricOpType op_type, int m, int n) const {
    // 根据运算类型和矩阵大小选择内核
    if (m >= 32 && n >= 32) {
        return "optimized";
    } else {
        return "basic";
    }
}

void SymmHerkJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于性能配置更新内核选择策略
    kernel_performance_[current_kernel_type_] = profile.execution_time;
}

} // namespace cu_op_mem
