#include "jit/Blas/trmm_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

TrmmJITPlugin::TrmmJITPlugin() 
    : compiler_(std::make_unique<JITCompiler>())
    , initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , side_(0)      // 0: left, 1: right
    , uplo_(0)      // 0: upper, 1: lower
    , trans_(0)     // 0: no trans, 1: trans, 2: conjugate transpose
    , diag_(0)      // 0: non-unit, 1: unit
    , alpha_(1.0f)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0) {
    LOG(INFO) << "TrmmJITPlugin created";
}

TrmmJITPlugin::~TrmmJITPlugin() {
    Cleanup();
}

StatusCode TrmmJITPlugin::Initialize() {
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
        LOG(INFO) << "TrmmJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_INITIALIZATION_ERROR;
    }
}

StatusCode TrmmJITPlugin::Compile(const JITConfig& config) {
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
        std::string kernel_type = SelectKernelType(side_, 
            matrix_A_ref_ ? matrix_A_ref_->shape()[0] : 0, 
            matrix_B_ref_ ? matrix_B_ref_->shape()[1] : 0);
        current_kernel_type_ = kernel_type;
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::JIT_COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "trmm_kernel_" + kernel_type;
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
        LOG(INFO) << "TrmmJITPlugin compiled successfully in " << duration.count() << "ms";
        return StatusCode::SUCCESS;
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::JIT_COMPILATION_ERROR;
    }
}

StatusCode TrmmJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
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
        dim3 grid((matrix_B_ref_->shape()[0] + block_size_x - 1) / block_size_x, 
                  (matrix_B_ref_->shape()[1] + block_size_y - 1) / block_size_y);
        
        // 设置内核参数
        void* args[] = {
            A.data(), B.data(), C.data(),
            &alpha_, &side_, &uplo_, &trans_, &diag_
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

void TrmmJITPlugin::Optimize(const PerformanceProfile& profile) {
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

void TrmmJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig TrmmJITPlugin::GetConfig() const {
    return config_;
}

void TrmmJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool TrmmJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile TrmmJITPlugin::GetPerformanceProfile() const {
    PerformanceProfile profile = last_profile_;
    profile.execution_time = total_executions_ > 0 ? total_execution_time_ / total_executions_ : 0.0;
    profile.compilation_time_ms = total_compilation_time_;
    profile.memory_usage = memory_usage_;
    return profile;
}

bool TrmmJITPlugin::IsInitialized() const {
    return initialized_;
}

bool TrmmJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string TrmmJITPlugin::GetLastError() const {
    return last_error_;
}

void TrmmJITPlugin::Cleanup() {
    kernel_cache_.clear();
    kernel_names_.clear();
    compiled_ = false;
    LOG(INFO) << "TrmmJITPlugin cleaned up";
}

size_t TrmmJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

// TRMM特定接口实现
void TrmmJITPlugin::SetTrmmParams(int side, int uplo, int trans, int diag, float alpha) {
    side_ = side;
    uplo_ = uplo;
    trans_ = trans;
    diag_ = diag;
    alpha_ = alpha;
}

void TrmmJITPlugin::SetMatrixA(const Tensor<float>& A) {
    // 由于Tensor不支持拷贝，我们只保存引用
    matrix_A_ref_ = &A;
}

void TrmmJITPlugin::SetMatrixB(const Tensor<float>& B) {
    // 由于Tensor不支持拷贝，我们只保存引用
    matrix_B_ref_ = &B;
}

bool TrmmJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "trmm" || op_name == "Trmm";
}

// 内核生成方法实现
std::string TrmmJITPlugin::GenerateKernelCode(const JITConfig& config) {
    std::string kernel_type = SelectKernelType(side_, 
        matrix_A_ref_ ? matrix_A_ref_->shape()[0] : 0, 
        matrix_B_ref_ ? matrix_B_ref_->shape()[1] : 0);
    
    if (kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else if (kernel_type == "tiled") {
        return GenerateTiledKernel(config);
    } else if (kernel_type == "warp_optimized") {
        return GenerateWarpOptimizedKernel(config);
    } else if (kernel_type == "blocked") {
        return GenerateBlockedKernel(config);
    }
    
    return GenerateBasicKernel(config);
}

std::string TrmmJITPlugin::GenerateBasicKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void trmm_kernel_basic(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float alpha, int side, int uplo, int trans, int diag
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = gridDim.y * blockDim.y;
    int n = gridDim.x * blockDim.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    if (side == 0) {  // left side: C = α * A * B
        for (int k = 0; k < m; ++k) {
            float a_val;
            if (uplo == 0) {  // upper
                a_val = (k <= row) ? A[row * m + k] : 0.0f;
            } else {  // lower
                a_val = (k >= row) ? A[row * m + k] : 0.0f;
            }
            
            if (diag == 1 && k == row) {  // unit diagonal
                a_val = 1.0f;
            }
            
            sum += a_val * B[k * n + col];
        }
    } else {  // right side: C = α * B * A
        for (int k = 0; k < n; ++k) {
            float a_val;
            if (uplo == 0) {  // upper
                a_val = (k <= col) ? A[col * n + k] : 0.0f;
            } else {  // lower
                a_val = (k >= col) ? A[col * n + k] : 0.0f;
            }
            
            if (diag == 1 && k == col) {  // unit diagonal
                a_val = 1.0f;
            }
            
            sum += B[row * n + k] * a_val;
        }
    }
    
    C[row * n + col] = alpha * sum;
}
)";
    
    return kernel.str();
}

std::string TrmmJITPlugin::GenerateTiledKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void trmm_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float alpha, int side, int uplo, int trans, int diag
) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = gridDim.y * blockDim.y;
    int n = gridDim.x * blockDim.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    if (side == 0) {  // left side: C = α * A * B
        for (int tile = 0; tile < (m + 15) / 16; ++tile) {
            // 加载共享内存
            int A_row = row;
            int A_col = tile * 16 + threadIdx.x;
            int B_row = tile * 16 + threadIdx.y;
            int B_col = col;
            
            if (A_row < m && A_col < m) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (A_col <= A_row) ? A[A_row * m + A_col] : 0.0f;
                } else {  // lower
                    a_val = (A_col >= A_row) ? A[A_row * m + A_col] : 0.0f;
                }
                
                if (diag == 1 && A_col == A_row) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                As[threadIdx.y][threadIdx.x] = a_val;
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (B_row < m && B_col < n) {
                Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
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
    } else {  // right side: C = α * B * A
        for (int tile = 0; tile < (n + 15) / 16; ++tile) {
            // 加载共享内存
            int B_row = row;
            int B_col = tile * 16 + threadIdx.x;
            int A_row = tile * 16 + threadIdx.y;
            int A_col = col;
            
            if (B_row < m && B_col < n) {
                Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (A_row < n && A_col < n) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (A_col <= A_row) ? A[A_row * n + A_col] : 0.0f;
                } else {  // lower
                    a_val = (A_col >= A_row) ? A[A_row * n + A_col] : 0.0f;
                }
                
                if (diag == 1 && A_col == A_row) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                As[threadIdx.y][threadIdx.x] = a_val;
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // 计算
            for (int i = 0; i < 16; ++i) {
                sum += Bs[threadIdx.y][i] * As[i][threadIdx.x];
            }
            
            __syncthreads();
        }
    }
    
    C[row * n + col] = alpha * sum;
}
)";
    
    return kernel.str();
}

std::string TrmmJITPlugin::GenerateWarpOptimizedKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void trmm_kernel_warp_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float alpha, int side, int uplo, int trans, int diag
) {
    int warp_id = blockIdx.x;
    int lane_id = threadIdx.x;
    
    int m = gridDim.y * blockDim.y;
    int n = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    if (side == 0) {  // left side: C = α * A * B
        for (int k = 0; k < m; ++k) {
            int row = warp_id;
            int col = lane_id;
            
            if (row < m && col < n) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (k <= row) ? A[row * m + k] : 0.0f;
                } else {  // lower
                    a_val = (k >= row) ? A[row * m + k] : 0.0f;
                }
                
                if (diag == 1 && k == row) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                sum += a_val * B[k * n + col];
            }
        }
    } else {  // right side: C = α * B * A
        for (int k = 0; k < n; ++k) {
            int row = warp_id;
            int col = lane_id;
            
            if (row < m && col < n) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (k <= col) ? A[col * n + k] : 0.0f;
                } else {  // lower
                    a_val = (k >= col) ? A[col * n + k] : 0.0f;
                }
                
                if (diag == 1 && k == col) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                sum += B[row * n + k] * a_val;
            }
        }
    }
    
    if (warp_id < m && lane_id < n) {
        C[warp_id * n + lane_id] = alpha * sum;
    }
}
)";
    
    return kernel.str();
}

std::string TrmmJITPlugin::GenerateBlockedKernel(const JITConfig& config) {
    std::ostringstream kernel;
    
    kernel << R"(
extern "C" __global__ void trmm_kernel_blocked(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    float alpha, int side, int uplo, int trans, int diag
) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = gridDim.y * blockDim.y;
    int n = gridDim.x * blockDim.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    if (side == 0) {  // left side: C = α * A * B
        for (int tile = 0; tile < (m + 31) / 32; ++tile) {
            // 加载共享内存
            int A_row = row;
            int A_col = tile * 32 + threadIdx.x;
            int B_row = tile * 32 + threadIdx.y;
            int B_col = col;
            
            if (A_row < m && A_col < m) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (A_col <= A_row) ? A[A_row * m + A_col] : 0.0f;
                } else {  // lower
                    a_val = (A_col >= A_row) ? A[A_row * m + A_col] : 0.0f;
                }
                
                if (diag == 1 && A_col == A_row) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                As[threadIdx.y][threadIdx.x] = a_val;
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (B_row < m && B_col < n) {
                Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // 计算
            for (int i = 0; i < 32; ++i) {
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
            
            __syncthreads();
        }
    } else {  // right side: C = α * B * A
        for (int tile = 0; tile < (n + 31) / 32; ++tile) {
            // 加载共享内存
            int B_row = row;
            int B_col = tile * 32 + threadIdx.x;
            int A_row = tile * 32 + threadIdx.y;
            int A_col = col;
            
            if (B_row < m && B_col < n) {
                Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (A_row < n && A_col < n) {
                float a_val;
                if (uplo == 0) {  // upper
                    a_val = (A_col <= A_row) ? A[A_row * n + A_col] : 0.0f;
                } else {  // lower
                    a_val = (A_col >= A_row) ? A[A_row * n + A_col] : 0.0f;
                }
                
                if (diag == 1 && A_col == A_row) {  // unit diagonal
                    a_val = 1.0f;
                }
                
                As[threadIdx.y][threadIdx.x] = a_val;
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // 计算
            for (int i = 0; i < 32; ++i) {
                sum += Bs[threadIdx.y][i] * As[i][threadIdx.x];
            }
            
            __syncthreads();
        }
    }
    
    C[row * n + col] = alpha * sum;
}
)";
    
    return kernel.str();
}

// 私有方法实现
CUfunction TrmmJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
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

void TrmmJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
    kernel_names_[key] = "trmm_kernel_" + current_kernel_type_;
}

CUfunction TrmmJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

std::string TrmmJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::ostringstream key;
    key << "trmm_" << current_kernel_type_ << "_" 
        << side_ << "_" << uplo_ << "_" << trans_ << "_" << diag_ << "_"
        << config.block_size_x << "x" << config.block_size_y;
    return key.str();
}

PerformanceProfile TrmmJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                    const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    
    // 计算GFLOPS
    double flops = 2.0 * (matrix_A_ref_ ? matrix_A_ref_->shape()[0] : 0) * 
                   (matrix_B_ref_ ? matrix_B_ref_->shape()[1] : 0) * 
                   (matrix_A_ref_ ? matrix_A_ref_->shape()[0] : 0);
    profile.gflops = flops / (total_execution_time_ / total_executions_) / 1e6;
    
    // 计算内存带宽
    double bytes = ((matrix_A_ref_ ? matrix_A_ref_->shape()[0] * matrix_A_ref_->shape()[0] : 0) + 
                   (matrix_B_ref_ ? matrix_B_ref_->shape()[0] * matrix_B_ref_->shape()[1] : 0) + 
                   (matrix_B_ref_ ? matrix_B_ref_->shape()[0] * matrix_B_ref_->shape()[1] : 0)) * sizeof(float);
    profile.bandwidth_gb_s = bytes / (total_execution_time_ / total_executions_) / 1e6;
    
    profile.kernel_time_ms = total_execution_time_ / total_executions_;
    profile.execution_time = total_execution_time_ / total_executions_;
    profile.kernel_name = kernel_names_[GenerateKernelKey(GenerateKernelCode(config_), config_)];
    profile.kernel_type = current_kernel_type_;
    
    return profile;
}

bool TrmmJITPlugin::ValidateConfig(const JITConfig& config) const {
    return config.block_size_x > 0 && config.block_size_y > 0 &&
           config.block_size_x <= 32 && config.block_size_y <= 32;
}

JITConfig TrmmJITPlugin::OptimizeConfig(const JITConfig& config) const {
    JITConfig optimized = config;
    
    // 基于矩阵大小优化块大小
    int m = matrix_A_ref_ ? matrix_A_ref_->shape()[0] : 0;
    int n = matrix_B_ref_ ? matrix_B_ref_->shape()[1] : 0;
    
    if (m <= 64 && n <= 64) {
        optimized.block_size_x = 16;
        optimized.block_size_y = 16;
    } else if (m <= 128 && n <= 128) {
        optimized.block_size_x = 32;
        optimized.block_size_y = 32;
    } else {
        optimized.block_size_x = 16;
        optimized.block_size_y = 16;
    }
    
    return optimized;
}

HardwareSpec TrmmJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    // 这里应该从CUDA运行时获取硬件信息
    // 简化实现
    spec.compute_capability_major = 7;
    spec.compute_capability_minor = 5;
    return spec;
}

std::string TrmmJITPlugin::SelectKernelType(int side, int m, int n) const {
    // 根据矩阵大小选择内核
    if (m >= 128 && n >= 128) {
        return "blocked";
    } else if (m >= 64 && n >= 64) {
        return "tiled";
    } else if (m >= 32 && n >= 32) {
        return "warp_optimized";
    } else {
        return "basic";
    }
}

void TrmmJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于性能配置更新内核选择策略
    kernel_performance_[current_kernel_type_] = profile.execution_time;
}

} // namespace cu_op_mem
