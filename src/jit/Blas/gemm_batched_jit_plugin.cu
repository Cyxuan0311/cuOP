#include "jit/Blas/gemm_batched_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

GemmBatchedJITPlugin::GemmBatchedJITPlugin() 
    : compiler_(std::make_unique<JITCompiler>())
    , initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , batch_size_(1)
    , m_(0), n_(0), k_(0)
    , transA_(false), transB_(false)
    , alpha_(1.0f), beta_(0.0f)
    , strideA_(0), strideB_(0), strideC_(0)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0) {
    LOG(INFO) << "GemmBatchedJITPlugin created";
}

GemmBatchedJITPlugin::~GemmBatchedJITPlugin() {
    Cleanup();
}

StatusCode GemmBatchedJITPlugin::Initialize() {
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
        config_.use_tensor_core = true;
        config_.use_shared_memory = true;
        config_.optimization_level = "O3";
        config_.block_size = 256;
        
        initialized_ = true;
        LOG(INFO) << "GemmBatchedJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_INITIALIZATION_ERROR;
    }
}

StatusCode GemmBatchedJITPlugin::Compile(const JITConfig& config) {
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
        std::string kernel_type = SelectKernelType(batch_size_, m_, n_, k_);
        current_kernel_type_ = kernel_type;
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "gemm_batched_kernel_" + kernel_type;
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
        LOG(INFO) << "GemmBatchedJITPlugin compiled successfully in " << duration.count() << "ms";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_COMPILATION_ERROR;
    }
}

StatusCode GemmBatchedJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
                                        std::vector<Tensor<float>>& outputs) {
    if (!compiled_) {
        return StatusCode::JIT_NOT_COMPILED;
    }
    
    if (inputs.size() < 2 || outputs.empty()) {
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
        const Tensor<float>& B = inputs[1];
        Tensor<float>& C = outputs[0];
        
        // 计算网格和块大小
        int block_size_x = config_.block_size_x > 0 ? config_.block_size_x : 16;
        int block_size_y = config_.block_size_y > 0 ? config_.block_size_y : 16;
        
        dim3 block(block_size_x, block_size_y);
        dim3 grid((m_ + block_size_x - 1) / block_size_x, 
                  (n_ + block_size_y - 1) / block_size_y, 
                  batch_size_);
        
        // 设置内核参数
        void* args[] = {
            A.data(), B.data(), C.data(),
            &m_, &n_, &k_,
            &alpha_, &beta_,
            &strideA_, &strideB_, &strideC_
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

void GemmBatchedJITPlugin::Optimize(const PerformanceProfile& profile) {
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

void GemmBatchedJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig GemmBatchedJITPlugin::GetConfig() const {
    return config_;
}

void GemmBatchedJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool GemmBatchedJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile GemmBatchedJITPlugin::GetPerformanceProfile() const {
    PerformanceProfile profile = last_profile_;
    profile.execution_time = total_executions_ > 0 ? total_execution_time_ / total_executions_ : 0.0;
    profile.compilation_time_ms = total_compilation_time_;
    profile.memory_usage = memory_usage_;
    return profile;
}

bool GemmBatchedJITPlugin::IsInitialized() const {
    return initialized_;
}

bool GemmBatchedJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string GemmBatchedJITPlugin::GetLastError() const {
    return last_error_;
}

void GemmBatchedJITPlugin::Cleanup() {
    kernel_cache_.clear();
    kernel_names_.clear();
    compiled_ = false;
    LOG(INFO) << "GemmBatchedJITPlugin cleaned up";
}

size_t GemmBatchedJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

// Batched GEMM特定接口实现
void GemmBatchedJITPlugin::SetBatchSize(int batch_size) {
    batch_size_ = batch_size;
}

void GemmBatchedJITPlugin::SetMatrixDimensions(int m, int n, int k) {
    m_ = m;
    n_ = n;
    k_ = k;
}

void GemmBatchedJITPlugin::SetTransposeOptions(bool transA, bool transB) {
    transA_ = transA;
    transB_ = transB;
}

void GemmBatchedJITPlugin::SetAlphaBeta(float alpha, float beta) {
    alpha_ = alpha;
    beta_ = beta;
}

void GemmBatchedJITPlugin::SetBatchStride(int strideA, int strideB, int strideC) {
    strideA_ = strideA;
    strideB_ = strideB;
    strideC_ = strideC;
}

bool GemmBatchedJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "gemm_batched" || op_name == "GemmBatched" || 
           op_name == "batched_gemm" || op_name == "BatchedGemm";
}

// 内核生成方法实现
std::string GemmBatchedJITPlugin::GenerateKernelCode(const JITConfig& config) {
    std::string kernel_type = SelectKernelType(batch_size_, m_, n_, k_);
    
    if (kernel_type == "standard") {
        return GenerateStandardKernel(config);
    } else if (kernel_type == "optimized") {
        return GenerateOptimizedKernel(config);
    } else if (kernel_type == "tensor_core") {
        return GenerateTensorCoreKernel(config);
    }
    
    return GenerateStandardKernel(config);
}

std::string GemmBatchedJITPlugin::GenerateStandardKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void gemm_batched_kernel_standard(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k,
    float alpha, float beta,
    int strideA, int strideB, int strideC
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += A[batch_idx * strideA + row * k + i] * 
               B[batch_idx * strideB + i * n + col];
    }
    
    int idx = batch_idx * strideC + row * n + col;
    C[idx] = alpha * sum + beta * C[idx];
}
)";
    
    return kernel.str();
}

std::string GemmBatchedJITPlugin::GenerateOptimizedKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void gemm_batched_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k,
    float alpha, float beta,
    int strideA, int strideB, int strideC
) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (k + 15) / 16; ++tile) {
        // 加载共享内存
        int A_row = row;
        int A_col = tile * 16 + threadIdx.x;
        int B_row = tile * 16 + threadIdx.y;
        int B_col = col;
        
        if (A_row < m && A_col < k) {
            As[threadIdx.y][threadIdx.x] = A[batch_idx * strideA + A_row * k + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (B_row < k && B_col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[batch_idx * strideB + B_row * n + B_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算
        for (int i = 0; i < 16; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        int idx = batch_idx * strideC + row * n + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}
)";
    
    return kernel.str();
}

std::string GemmBatchedJITPlugin::GenerateTensorCoreKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
#include <mma.h>

extern "C" __global__ void gemm_batched_kernel_tensor_core(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k,
    float alpha, float beta,
    int strideA, int strideB, int strideC
) {
    using namespace nvcuda::wmma;
    
    // Tensor Core配置
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int batch_idx = blockIdx.z;
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // 声明片段
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // 初始化累加器
    fill_fragment(acc_frag, 0.0f);
    
    // 计算
    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // 加载片段
        load_matrix_sync(a_frag, A + batch_idx * strideA + aRow * k + aCol, k);
        load_matrix_sync(b_frag, B + batch_idx * strideB + bRow * n + bCol, n);
        
        // 矩阵乘法
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // 存储结果
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    store_matrix_sync(C + batch_idx * strideC + cRow * n + cCol, acc_frag, n, mem_row_major);
}
)";
    
    return kernel.str();
}

// 私有方法实现
CUfunction GemmBatchedJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
    if (!compiler_) {
        return nullptr;
    }
    
    std::vector<std::string> options = {
        "-std=c++17",
        "-O3",
        "--use_fast_math"
    };
    
    if (config_.use_tensor_core) {
        options.push_back("-lcublas");
    }
    
    auto result = compiler_->CompileKernel(kernel_code, kernel_name, options);
    if (result.success) {
        return result.kernel;
    }
    
    last_error_ = result.error_message;
    return nullptr;
}

void GemmBatchedJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
    kernel_names_[key] = "gemm_batched_kernel_" + current_kernel_type_;
}

CUfunction GemmBatchedJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

std::string GemmBatchedJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::ostringstream key;
    key << "gemm_batched_" << current_kernel_type_ << "_" 
        << batch_size_ << "x" << m_ << "x" << n_ << "x" << k_ << "_"
        << config.block_size_x << "x" << config.block_size_y << "_"
        << (config.use_tensor_core ? "tc" : "no_tc");
    return key.str();
}

PerformanceProfile GemmBatchedJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                           const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    
    // 计算GFLOPS
    double flops = 2.0 * batch_size_ * m_ * n_ * k_;
    profile.gflops = flops / (total_execution_time_ / total_executions_) / 1e6;
    
    // 计算内存带宽
    double bytes = batch_size_ * (m_ * k_ + k_ * n_ + m_ * n_) * sizeof(float);
    profile.bandwidth_gb_s = bytes / (total_execution_time_ / total_executions_) / 1e6;
    
    profile.kernel_time_ms = total_execution_time_ / total_executions_;
    profile.execution_time = total_execution_time_ / total_executions_;
    profile.kernel_name = kernel_names_[GenerateKernelKey(GenerateKernelCode(config_), config_)];
    profile.kernel_type = current_kernel_type_;
    
    return profile;
}

bool GemmBatchedJITPlugin::ValidateConfig(const JITConfig& config) const {
    return config.block_size_x > 0 && config.block_size_y > 0 &&
           config.block_size_x <= 32 && config.block_size_y <= 32;
}

JITConfig GemmBatchedJITPlugin::OptimizeConfig(const JITConfig& config) const {
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

HardwareSpec GemmBatchedJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    // 这里应该从CUDA运行时获取硬件信息
    // 简化实现
    spec.compute_capability_major = 7;
    spec.compute_capability_minor = 5;
    spec.supports_tensor_core = true;
    return spec;
}

bool GemmBatchedJITPlugin::SupportsTensorCore() const {
    return GetHardwareSpec().supports_tensor_core;
}

std::string GemmBatchedJITPlugin::SelectKernelType(int batch_size, int m, int n, int k) const {
    if (SupportsTensorCore() && m >= 16 && n >= 16 && k >= 16) {
        return "tensor_core";
    } else if (m >= 32 && n >= 32 && k >= 32) {
        return "optimized";
    } else {
        return "standard";
    }
}

void GemmBatchedJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于性能配置更新内核选择策略
    kernel_performance_[current_kernel_type_] = profile.execution_time;
}

} // namespace cu_op_mem
