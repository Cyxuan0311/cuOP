#include "jit/Blas/vector_ops_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace cu_op_mem {

VectorOpsJITPlugin::VectorOpsJITPlugin() 
    : compiler_(std::make_unique<JITCompiler>())
    , initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , op_type_(VectorOpType::DOT)
    , vector_size_(0)
    , alpha_(1.0f), beta_(0.0f)
    , incx_(1), incy_(1)
    , c_(1.0f), s_(0.0f)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0) {
    LOG(INFO) << "VectorOpsJITPlugin created";
}

VectorOpsJITPlugin::~VectorOpsJITPlugin() {
    Cleanup();
}

StatusCode VectorOpsJITPlugin::Initialize() {
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
        LOG(INFO) << "VectorOpsJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_INITIALIZATION_ERROR;
    }
}

StatusCode VectorOpsJITPlugin::Compile(const JITConfig& config) {
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
        std::string kernel_type = SelectKernelType(op_type_, vector_size_);
        current_kernel_type_ = kernel_type;
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "vector_ops_kernel_" + kernel_type;
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
        LOG(INFO) << "VectorOpsJITPlugin compiled successfully in " << duration.count() << "ms";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_COMPILATION_ERROR;
    }
}

StatusCode VectorOpsJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
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
        const Tensor<float>& x = inputs[0];
        Tensor<float>& result = outputs[0];
        
        // 计算网格和块大小
        int block_size = config_.block_size > 0 ? config_.block_size : 256;
        int grid_size = (vector_size_ + block_size - 1) / block_size;
        
        // 设置内核参数
        void* args[] = {
            x.data(), result.data(),
            &vector_size_, &alpha_, &beta_,
            &incx_, &incy_, &c_, &s_
        };
        
        // 启动内核
        CUresult result_code = cuLaunchKernel(kernel, 
                                            grid_size, 1, 1,
                                            block_size, 1, 1,
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

void VectorOpsJITPlugin::Optimize(const PerformanceProfile& profile) {
    if (!auto_tuning_enabled_) {
        return;
    }
    
    // 基于性能配置优化参数
    if (profile.execution_time > last_profile_.execution_time * 1.1) {
        // 性能下降，尝试不同的配置
        if (config_.block_size < 512) {
            config_.block_size *= 2;
        }
        
        // 重新编译
        Compile(config_);
    }
    
    last_profile_ = profile;
}

void VectorOpsJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig VectorOpsJITPlugin::GetConfig() const {
    return config_;
}

void VectorOpsJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool VectorOpsJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile VectorOpsJITPlugin::GetPerformanceProfile() const {
    PerformanceProfile profile = last_profile_;
    profile.execution_time = total_executions_ > 0 ? total_execution_time_ / total_executions_ : 0.0;
    profile.compilation_time_ms = total_compilation_time_;
    profile.memory_usage = memory_usage_;
    return profile;
}

bool VectorOpsJITPlugin::IsInitialized() const {
    return initialized_;
}

bool VectorOpsJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string VectorOpsJITPlugin::GetLastError() const {
    return last_error_;
}

void VectorOpsJITPlugin::Cleanup() {
    kernel_cache_.clear();
    kernel_names_.clear();
    compiled_ = false;
    LOG(INFO) << "VectorOpsJITPlugin cleaned up";
}

size_t VectorOpsJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

// 向量运算特定接口实现
void VectorOpsJITPlugin::SetOperationType(VectorOpType op_type) {
    op_type_ = op_type;
}

void VectorOpsJITPlugin::SetVectorSize(int size) {
    vector_size_ = size;
}

void VectorOpsJITPlugin::SetAlpha(float alpha) {
    alpha_ = alpha;
}

void VectorOpsJITPlugin::SetBeta(float beta) {
    beta_ = beta;
}

void VectorOpsJITPlugin::SetIncrement(int incx, int incy) {
    incx_ = incx;
    incy_ = incy;
}

void VectorOpsJITPlugin::SetRotationParams(float c, float s) {
    c_ = c;
    s_ = s;
}

bool VectorOpsJITPlugin::SupportsOperator(const std::string& op_name) {
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

// 内核生成方法实现
std::string VectorOpsJITPlugin::GenerateKernelCode(const JITConfig& config) {
    switch (op_type_) {
        case VectorOpType::DOT:
            return GenerateDotKernel(config);
        case VectorOpType::AXPY:
            return GenerateAxpyKernel(config);
        case VectorOpType::SCAL:
            return GenerateScalKernel(config);
        case VectorOpType::COPY:
            return GenerateCopyKernel(config);
        case VectorOpType::SWAP:
            return GenerateSwapKernel(config);
        case VectorOpType::ROT:
            return GenerateRotKernel(config);
        case VectorOpType::NRM2:
            return GenerateNrm2Kernel(config);
        case VectorOpType::ASUM:
            return GenerateAsumKernel(config);
        case VectorOpType::IAMAX:
            return GenerateIamaxKernel(config);
        case VectorOpType::IAMIN:
            return GenerateIaminKernel(config);
        default:
            return GenerateDotKernel(config);
    }
}

std::string VectorOpsJITPlugin::GenerateDotKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_dot(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        sum += x[i * incx] * y[i * incy];
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateAxpyKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_axpy(
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        y[i * incy] = alpha * x[i * incx] + y[i * incy];
        i += blockDim.x * gridDim.x;
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateScalKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_scal(
    float* __restrict__ x,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        x[i * incx] = alpha * x[i * incx];
        i += blockDim.x * gridDim.x;
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateCopyKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_copy(
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        y[i * incy] = x[i * incx];
        i += blockDim.x * gridDim.x;
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateSwapKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_swap(
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        float temp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = temp;
        i += blockDim.x * gridDim.x;
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateRotKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_rot(
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        float x_val = x[i * incx];
        float y_val = y[i * incy];
        
        x[i * incx] = c * x_val + s * y_val;
        y[i * incy] = -s * x_val + c * y_val;
        
        i += blockDim.x * gridDim.x;
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateNrm2Kernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_nrm2(
    const float* __restrict__ x,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        float val = x[i * incx];
        sum += val * val;
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sqrtf(sdata[0]));
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateAsumKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_asum(
    const float* __restrict__ x,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        sum += fabsf(x[i * incx]);
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateIamaxKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_iamax(
    const float* __restrict__ x,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    __shared__ float sdata[256];
    __shared__ int sindex[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = 0.0f;
    int max_idx = 0;
    
    while (i < n) {
        float val = fabsf(x[i * incx]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = max_val;
    sindex[tid] = max_idx;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)result, sindex[0]);
    }
}
)";
    
    return kernel.str();
}

std::string VectorOpsJITPlugin::GenerateIaminKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void vector_ops_kernel_iamin(
    const float* __restrict__ x,
    float* __restrict__ result,
    int n, float alpha, float beta,
    int incx, int incy, float c, float s
) {
    __shared__ float sdata[256];
    __shared__ int sindex[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float min_val = INFINITY;
    int min_idx = 0;
    
    while (i < n) {
        float val = fabsf(x[i * incx]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = min_val;
    sindex[tid] = min_idx;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin((int*)result, sindex[0]);
    }
}
)";
    
    return kernel.str();
}

// 私有方法实现
CUfunction VectorOpsJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
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

void VectorOpsJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
    kernel_names_[key] = "vector_ops_kernel_" + current_kernel_type_;
}

CUfunction VectorOpsJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

std::string VectorOpsJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::ostringstream key;
    key << "vector_ops_" << static_cast<int>(op_type_) << "_" 
        << vector_size_ << "_"
        << config.block_size;
    return key.str();
}

PerformanceProfile VectorOpsJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                         const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    
    // 计算GFLOPS (简化计算)
    double flops = vector_size_;  // 简化估算
    profile.gflops = flops / (total_execution_time_ / total_executions_) / 1e6;
    
    // 计算内存带宽
    double bytes = vector_size_ * sizeof(float);
    profile.bandwidth_gb_s = bytes / (total_execution_time_ / total_executions_) / 1e6;
    
    profile.kernel_time_ms = total_execution_time_ / total_executions_;
    profile.execution_time = total_execution_time_ / total_executions_;
    profile.kernel_name = kernel_names_[GenerateKernelKey(GenerateKernelCode(config_), config_)];
    profile.kernel_type = current_kernel_type_;
    
    return profile;
}

bool VectorOpsJITPlugin::ValidateConfig(const JITConfig& config) const {
    return config.block_size > 0 && config.block_size <= 1024;
}

JITConfig VectorOpsJITPlugin::OptimizeConfig(const JITConfig& config) const {
    JITConfig optimized = config;
    
    // 基于向量大小优化块大小
    if (vector_size_ <= 1024) {
        optimized.block_size = 256;
    } else if (vector_size_ <= 4096) {
        optimized.block_size = 512;
    } else {
        optimized.block_size = 1024;
    }
    
    return optimized;
}

HardwareSpec VectorOpsJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    // 这里应该从CUDA运行时获取硬件信息
    // 简化实现
    spec.compute_capability_major = 7;
    spec.compute_capability_minor = 5;
    return spec;
}

std::string VectorOpsJITPlugin::SelectKernelType(VectorOpType op_type, int vector_size) const {
    // 根据运算类型和向量大小选择内核
    if (vector_size >= 1024) {
        return "optimized";
    } else {
        return "basic";
    }
}

void VectorOpsJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于性能配置更新内核选择策略
    kernel_performance_[current_kernel_type_] = profile.execution_time;
}

} // namespace cu_op_mem
