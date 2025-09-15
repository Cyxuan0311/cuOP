#include "jit/Blas/ger_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

GerJITPlugin::GerJITPlugin() 
    : compiler_(std::make_unique<JITCompiler>())
    , initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , alpha_(1.0f)
    , m_(0), n_(0)
    , incx_(1), incy_(1)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0) {
    LOG(INFO) << "GerJITPlugin created";
}

GerJITPlugin::~GerJITPlugin() {
    Cleanup();
}

StatusCode GerJITPlugin::Initialize() {
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
        LOG(INFO) << "GerJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_INITIALIZATION_ERROR;
    }
}

StatusCode GerJITPlugin::Compile(const JITConfig& config) {
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
        std::string kernel_type = SelectKernelType(m_, n_);
        current_kernel_type_ = kernel_type;
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "ger_kernel_" + kernel_type;
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
        LOG(INFO) << "GerJITPlugin compiled successfully in " << duration.count() << "ms";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_COMPILATION_ERROR;
    }
}

StatusCode GerJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
                                std::vector<Tensor<float>>& outputs) {
    if (!compiled_) {
        return StatusCode::JIT_NOT_COMPILED;
    }
    
    if (inputs.size() < 3 || outputs.empty()) {
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
        const Tensor<float>& x = inputs[0];
        const Tensor<float>& y = inputs[1];
        const Tensor<float>& A = inputs[2];
        Tensor<float>& result = outputs[0];
        
        // 计算网格和块大小
        int block_size_x = config_.block_size_x > 0 ? config_.block_size_x : 16;
        int block_size_y = config_.block_size_y > 0 ? config_.block_size_y : 16;
        
        dim3 block(block_size_x, block_size_y);
        dim3 grid((m_ + block_size_x - 1) / block_size_x, 
                  (n_ + block_size_y - 1) / block_size_y);
        
        // 设置内核参数
        void* args[] = {
            x.data(), y.data(), A.data(), result.data(),
            &m_, &n_, &alpha_, &incx_, &incy_
        };
        
        // 启动内核
        CUresult result_code = cuLaunchKernel(kernel, 
                                            grid.x, grid.y, grid.z,
                                            block.x, block.y, block.z,
                                            0, 0, args, 0);
        
        if (result_code != CUDA_SUCCESS) {
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

void GerJITPlugin::Optimize(const PerformanceProfile& profile) {
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

void GerJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig GerJITPlugin::GetConfig() const {
    return config_;
}

void GerJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool GerJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile GerJITPlugin::GetPerformanceProfile() const {
    PerformanceProfile profile = last_profile_;
    profile.execution_time = total_executions_ > 0 ? total_execution_time_ / total_executions_ : 0.0;
    profile.compilation_time_ms = total_compilation_time_;
    profile.memory_usage = memory_usage_;
    return profile;
}

bool GerJITPlugin::IsInitialized() const {
    return initialized_;
}

bool GerJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string GerJITPlugin::GetLastError() const {
    return last_error_;
}

void GerJITPlugin::Cleanup() {
    kernel_cache_.clear();
    kernel_names_.clear();
    compiled_ = false;
    LOG(INFO) << "GerJITPlugin cleaned up";
}

size_t GerJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

// GER特定接口实现
void GerJITPlugin::SetGerParams(float alpha) {
    alpha_ = alpha;
}

void GerJITPlugin::SetMatrixDimensions(int m, int n) {
    m_ = m;
    n_ = n;
}

void GerJITPlugin::SetVectorIncrements(int incx, int incy) {
    incx_ = incx;
    incy_ = incy;
}

void GerJITPlugin::SetMatrixA(const Tensor<float>& A) {
    // 由于Tensor不支持拷贝，我们只保存引用
    matrix_A_ref_ = &A;
}

void GerJITPlugin::SetVectorX(const Tensor<float>& x) {
    // 由于Tensor不支持拷贝，我们只保存引用
    vector_x_ref_ = &x;
}

void GerJITPlugin::SetVectorY(const Tensor<float>& y) {
    // 由于Tensor不支持拷贝，我们只保存引用
    vector_y_ref_ = &y;
}

bool GerJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "ger" || op_name == "Ger";
}

// 内核生成方法实现
std::string GerJITPlugin::GenerateKernelCode(const JITConfig& config) {
    std::string kernel_type = SelectKernelType(m_, n_);
    
    if (kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else if (kernel_type == "tiled") {
        return GenerateTiledKernel(config);
    } else if (kernel_type == "warp_optimized") {
        return GenerateWarpOptimizedKernel(config);
    } else if (kernel_type == "shared_memory") {
        return GenerateSharedMemoryKernel(config);
    }
    
    return GenerateBasicKernel(config);
}

std::string GerJITPlugin::GenerateBasicKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void ger_kernel_basic(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ A,
    float* __restrict__ result,
    int m, int n, float alpha, int incx, int incy
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    // A = α * x * y^T + A
    float x_val = x[row * incx];
    float y_val = y[col * incy];
    float a_val = A[row * n + col];
    
    result[row * n + col] = alpha * x_val * y_val + a_val;
}
)";
    
    return kernel.str();
}

std::string GerJITPlugin::GenerateTiledKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void ger_kernel_tiled(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ A,
    float* __restrict__ result,
    int m, int n, float alpha, int incx, int incy
) {
    __shared__ float xs[16];
    __shared__ float ys[16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载共享内存
    if (threadIdx.x == 0 && row < m) {
        xs[threadIdx.y] = x[row * incx];
    }
    if (threadIdx.y == 0 && col < n) {
        ys[threadIdx.x] = y[col * incy];
    }
    
    __syncthreads();
    
    if (row >= m || col >= n) return;
    
    // A = α * x * y^T + A
    float x_val = xs[threadIdx.y];
    float y_val = ys[threadIdx.x];
    float a_val = A[row * n + col];
    
    result[row * n + col] = alpha * x_val * y_val + a_val;
}
)";
    
    return kernel.str();
}

std::string GerJITPlugin::GenerateWarpOptimizedKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void ger_kernel_warp_optimized(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ A,
    float* __restrict__ result,
    int m, int n, float alpha, int incx, int incy
) {
    int warp_id = blockIdx.x;
    int lane_id = threadIdx.x;
    
    int row = warp_id;
    int col = lane_id;
    
    if (row >= m || col >= n) return;
    
    // A = α * x * y^T + A
    float x_val = x[row * incx];
    float y_val = y[col * incy];
    float a_val = A[row * n + col];
    
    result[row * n + col] = alpha * x_val * y_val + a_val;
}
)";
    
    return kernel.str();
}

std::string GerJITPlugin::GenerateSharedMemoryKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void ger_kernel_shared_memory(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ A,
    float* __restrict__ result,
    int m, int n, float alpha, int incx, int incy
) {
    __shared__ float xs[32];
    __shared__ float ys[32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载共享内存
    if (threadIdx.x == 0 && row < m) {
        xs[threadIdx.y] = x[row * incx];
    }
    if (threadIdx.y == 0 && col < n) {
        ys[threadIdx.x] = y[col * incy];
    }
    
    __syncthreads();
    
    if (row >= m || col >= n) return;
    
    // A = α * x * y^T + A
    float x_val = xs[threadIdx.y];
    float y_val = ys[threadIdx.x];
    float a_val = A[row * n + col];
    
    result[row * n + col] = alpha * x_val * y_val + a_val;
}
)";
    
    return kernel.str();
}

// 私有方法实现
CUfunction GerJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
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

void GerJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
    kernel_names_[key] = "ger_kernel_" + current_kernel_type_;
}

CUfunction GerJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

std::string GerJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::ostringstream key;
    key << "ger_" << current_kernel_type_ << "_" 
        << m_ << "x" << n_ << "_"
        << config.block_size_x << "x" << config.block_size_y;
    return key.str();
}

PerformanceProfile GerJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                   const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    
    // 计算GFLOPS
    double flops = 2.0 * m_ * n_;  // 每个元素需要一次乘法和一次加法
    profile.gflops = flops / (total_execution_time_ / total_executions_) / 1e6;
    
    // 计算内存带宽
    double bytes = (m_ + n_ + m_ * n_ + m_ * n_) * sizeof(float);  // x, y, A, result
    profile.bandwidth_gb_s = bytes / (total_execution_time_ / total_executions_) / 1e6;
    
    profile.kernel_time_ms = total_execution_time_ / total_executions_;
    profile.execution_time = total_execution_time_ / total_executions_;
    profile.kernel_name = kernel_names_[GenerateKernelKey(GenerateKernelCode(config_), config_)];
    profile.kernel_type = current_kernel_type_;
    
    return profile;
}

bool GerJITPlugin::ValidateConfig(const JITConfig& config) const {
    return config.block_size_x > 0 && config.block_size_y > 0 &&
           config.block_size_x <= 32 && config.block_size_y <= 32;
}

JITConfig GerJITPlugin::OptimizeConfig(const JITConfig& config) const {
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

HardwareSpec GerJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    // 这里应该从CUDA运行时获取硬件信息
    // 简化实现
    spec.compute_capability_major = 7;
    spec.compute_capability_minor = 5;
    return spec;
}

std::string GerJITPlugin::SelectKernelType(int m, int n) const {
    // 根据矩阵大小选择内核
    if (m >= 128 && n >= 128) {
        return "shared_memory";
    } else if (m >= 64 && n >= 64) {
        return "tiled";
    } else if (m >= 32 && n >= 32) {
        return "warp_optimized";
    } else {
        return "basic";
    }
}

void GerJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于性能配置更新内核选择策略
    kernel_performance_[current_kernel_type_] = profile.execution_time;
}

} // namespace cu_op_mem
