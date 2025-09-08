#include "jit/Blas/trsm_jit_plugin.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

// ==================== TrsmJITPlugin 实现 ====================

TrsmJITPlugin::TrsmJITPlugin()
    : initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , side_(0)      // left
    , uplo_(1)      // lower
    , trans_(0)     // no trans
    , diag_(0)      // non-unit
    , alpha_(1.0f)
    , total_executions_(0)
    , total_execution_time_(0.0)
    , total_compilation_time_(0.0)
    , memory_usage_(0)
    , current_kernel_type_("basic") {
    
    compiler_ = std::make_unique<JITCompiler>();
    config_.kernel_type = "basic";
    config_.tile_size = 16;
    config_.block_size = 256;
    config_.optimization_level = "O2";
    config_.enable_tensor_core = false;
    config_.enable_tma = false;
}

TrsmJITPlugin::~TrsmJITPlugin() {
    Cleanup();
}

StatusCode TrsmJITPlugin::Initialize() {
    if (initialized_) {
        return StatusCode::SUCCESS;
    }
    
    try {
        if (!compiler_) {
            compiler_ = std::make_unique<JITCompiler>();
        }
        
        // 验证TRSM参数
        if (side_ != 0) {
            last_error_ = "Only left-side TRSM is currently supported";
            return StatusCode::UNSUPPORTED_TYPE;
        }
        
        initialized_ = true;
        VLOG(1) << "TrsmJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::INITIALIZATION_ERROR;
    }
}

StatusCode TrsmJITPlugin::Compile(const JITConfig& config) {
    if (!initialized_) {
        last_error_ = "Plugin not initialized";
        return StatusCode::NOT_INITIALIZED;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 更新配置
        config_ = config;
        
        // 验证配置
        if (!ValidateConfig(config_)) {
            last_error_ = "Invalid configuration";
            return StatusCode::INVALID_ARGUMENT;
        }
        
        // 优化配置
        config_ = OptimizeConfig(config_);
        
        // 生成内核代码
        std::string kernel_code = GenerateKernelCode(config_);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::COMPILATION_ERROR;
        }
        
        // 编译内核
        std::string kernel_name = "trsm_kernel_" + config_.kernel_type;
        CUfunction kernel = CompileKernel(kernel_code, kernel_name);
        if (!kernel) {
            last_error_ = "Failed to compile kernel";
            return StatusCode::COMPILATION_ERROR;
        }
        
        // 缓存内核
        std::string cache_key = GenerateKernelKey(kernel_code, config_);
        CacheKernel(cache_key, kernel);
        kernel_names_[cache_key] = kernel_name;
        current_kernel_type_ = config_.kernel_type;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        total_compilation_time_ += std::chrono::duration<double>(end_time - start_time).count();
        
        compiled_ = true;
        VLOG(1) << "TrsmJITPlugin compiled successfully with kernel type: " << config_.kernel_type;
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::COMPILATION_ERROR;
    }
}

StatusCode TrsmJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
                                 std::vector<Tensor<float>>& outputs) {
    if (!compiled_) {
        last_error_ = "Plugin not compiled";
        return StatusCode::NOT_COMPILED;
    }
    
    if (inputs.size() < 1 || outputs.size() < 1) {
        last_error_ = "Invalid input/output count";
        return StatusCode::INVALID_ARGUMENT;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        const Tensor<float>& input = inputs[0];  // B矩阵
        Tensor<float>& output = outputs[0];      // 结果X矩阵
        
        // 获取矩阵维度
        int m = matrix_A_.shape()[0];  // A是m x m
        int n = input.shape()[1];      // B是m x n
        
        // 验证维度
        if (matrix_A_.shape()[0] != matrix_A_.shape()[1] || input.shape()[0] != m) {
            last_error_ = "Matrix dimension mismatch";
            return StatusCode::SHAPE_MISMATCH;
        }
        
        // 获取设备指针
        const float* d_A = matrix_A_.data();
        const float* d_B = input.data();
        float* d_X = output.data();
        
        // 获取编译的内核
        std::string cache_key = GenerateKernelKey("", config_);
        CUfunction kernel = GetCachedKernel(cache_key);
        if (!kernel) {
            last_error_ = "Kernel not found in cache";
            return StatusCode::KERNEL_NOT_FOUND;
        }
        
        // 设置内核参数
        void* args[] = {
            &m, &n, &alpha_, 
            const_cast<float**>(&d_A), 
            const_cast<float**>(&d_B), 
            &d_X,
            &side_, &uplo_, &trans_, &diag_
        };
        
        // 计算网格和块大小
        dim3 threads, blocks;
        if (config_.kernel_type == "basic") {
            threads = dim3(256);
            blocks = dim3((n + 255) / 256);
        } else if (config_.kernel_type == "tiled") {
            threads = dim3(256);
            blocks = dim3((n + 255) / 256);
        } else if (config_.kernel_type == "warp_optimized") {
            threads = dim3(256);
            blocks = dim3((n + 255) / 256);
        } else if (config_.kernel_type == "blocked") {
            threads = dim3(256);
            blocks = dim3((n + 255) / 256);
        } else {
            threads = dim3(256);
            blocks = dim3((n + 255) / 256);
        }
        
        // 启动内核
        CUresult result = cuLaunchKernel(kernel, 
                                        blocks.x, blocks.y, blocks.z,
                                        threads.x, threads.y, threads.z,
                                        0, nullptr, args, nullptr);
        
        if (result != CUDA_SUCCESS) {
            last_error_ = "Kernel launch failed";
            return StatusCode::CUDA_ERROR;
        }
        
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end_time - start_time).count();
        
        total_executions_++;
        total_execution_time_ += execution_time;
        
        // 更新性能统计
        performance_profile_.execution_time = execution_time;
        performance_profile_.kernel_type = current_kernel_type_;
        performance_profile_.throughput = (2.0 * m * m * n) / (execution_time * 1e9); // GFLOPS
        
        VLOG(1) << "TrsmJITPlugin executed successfully: m=" << m << ", n=" << n 
                << ", time=" << execution_time * 1000 << "ms";
        
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Execution failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::EXECUTION_ERROR;
    }
}

void TrsmJITPlugin::Optimize(const PerformanceProfile& profile) {
    if (!auto_tuning_enabled_) {
        return;
    }
    
    // 基于性能分析结果优化配置
    if (profile.execution_time > 0) {
        // 如果执行时间过长，尝试更激进的优化
        if (profile.execution_time > 1.0) {  // 超过1秒
            if (config_.kernel_type == "basic") {
                config_.kernel_type = "tiled";
            } else if (config_.kernel_type == "tiled") {
                config_.kernel_type = "warp_optimized";
            } else if (config_.kernel_type == "warp_optimized") {
                config_.kernel_type = "blocked";
            }
        }
        
        // 根据矩阵大小调整tile_size
        if (profile.throughput < 100.0) {  // 低于100 GFLOPS
            config_.tile_size = std::min(config_.tile_size * 2, 64);
        }
    }
}

void TrsmJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig TrsmJITPlugin::GetConfig() const {
    return config_;
}

void TrsmJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool TrsmJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile TrsmJITPlugin::GetPerformanceProfile() const {
    return performance_profile_;
}

bool TrsmJITPlugin::IsInitialized() const {
    return initialized_;
}

bool TrsmJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string TrsmJITPlugin::GetLastError() const {
    return last_error_;
}

void TrsmJITPlugin::Cleanup() {
    if (compiled_) {
        // 清理内核缓存
        // CUfunction不需要手动unload，它们会随着其父模块自动释放
        kernel_cache_.clear();
        kernel_names_.clear();
        compiled_ = false;
    }
    
    if (initialized_) {
        compiler_.reset();
        initialized_ = false;
    }
}

size_t TrsmJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

void TrsmJITPlugin::SetTrsmParams(int side, int uplo, int trans, int diag, float alpha) {
    side_ = side;
    uplo_ = uplo;
    trans_ = trans;
    diag_ = diag;
    alpha_ = alpha;
}

void TrsmJITPlugin::SetMatrixA(const Tensor<float>& A) {
    matrix_A_ = Tensor<float>(A.shape());
    cudaMemcpy(matrix_A_.data(), A.data(), A.bytes(), cudaMemcpyDeviceToDevice);
}

bool TrsmJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "trsm" || op_name == "Trsm";
}

// ==================== 内核生成方法 ====================

std::string TrsmJITPlugin::GenerateKernelCode(const JITConfig& config) {
    if (config.kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else if (config.kernel_type == "tiled") {
        return GenerateTiledKernel(config);
    } else if (config.kernel_type == "warp_optimized") {
        return GenerateWarpOptimizedKernel(config);
    } else if (config.kernel_type == "blocked") {
        return GenerateBlockedKernel(config);
    } else {
        return GenerateBasicKernel(config);
    }
}

std::string TrsmJITPlugin::GenerateBasicKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void trsm_kernel_basic(
    int m, int n, float alpha, 
    const float* A, const float* B, float* X,
    int side, int uplo, int trans, int diag) {
    
    // 当前只支持 left, lower, no-trans, non-unit
    if (side != 0 || uplo != 1 || trans != 0 || diag != 0) {
        return;
    }
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    // 前向替换: 求解 AX = alpha*B，覆盖X
    for (int row = 0; row < m; ++row) {
        float sum = alpha * B[row * n + col];
        
        // 减去已知项
        for (int k = 0; k < row; ++k) {
            sum -= A[row * m + k] * X[k * n + col];
        }
        
        // 除以对角元素
        X[row * n + col] = sum / A[row * m + row];
    }
}
)";
    
    return code.str();
}

std::string TrsmJITPlugin::GenerateTiledKernel(const JITConfig& config) {
    std::stringstream code;
    int tile_size = config.tile_size;
    
    code << R"(
extern "C" __global__ void trsm_kernel_tiled(
    int m, int n, float alpha, 
    const float* A, const float* B, float* X,
    int side, int uplo, int trans, int diag) {
    
    // 当前只支持 left, lower, no-trans, non-unit
    if (side != 0 || uplo != 1 || trans != 0 || diag != 0) {
        return;
    }
    
    __shared__ float tile_A[)" << tile_size << "][" << tile_size << R"(];
    __shared__ float tile_X[)" << tile_size << R"(];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int tx = threadIdx.x;
    
    // 分块处理
    for (int tile_start = 0; tile_start < m; tile_start += )" << tile_size << R"() {
        int tile_end = min(tile_start + )" << tile_size << R"(, m);
        int tile_size_actual = tile_end - tile_start;
        
        // 加载A的tile
        if (tx < tile_size_actual) {
            for (int j = 0; j < tile_size_actual; ++j) {
                tile_A[tx][j] = A[(tile_start + tx) * m + (tile_start + j)];
            }
        }
        
        // 加载X的tile
        if (tx < tile_size_actual) {
            tile_X[tx] = X[(tile_start + tx) * n + col];
        }
        
        __syncthreads();
        
        // 在tile内求解
        for (int i = 0; i < tile_size_actual; ++i) {
            if (tx == 0) {
                float sum = alpha * B[(tile_start + i) * n + col];
                
                // 减去tile内的已知项
                for (int k = 0; k < i; ++k) {
                    sum -= tile_A[i][k] * tile_X[k];
                }
                
                // 减去tile外的已知项
                for (int k = 0; k < tile_start; ++k) {
                    sum -= A[(tile_start + i) * m + k] * X[k * n + col];
                }
                
                tile_X[i] = sum / tile_A[i][i];
            }
            __syncthreads();
        }
        
        // 写回结果
        if (tx < tile_size_actual) {
            X[(tile_start + tx) * n + col] = tile_X[tx];
        }
        
        __syncthreads();
    }
}
)";
    
    return code.str();
}

std::string TrsmJITPlugin::GenerateWarpOptimizedKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void trsm_kernel_warp_optimized(
    int m, int n, float alpha, 
    const float* A, const float* B, float* X,
    int side, int uplo, int trans, int diag) {
    
    // 当前只支持 left, lower, no-trans, non-unit
    if (side != 0 || uplo != 1 || trans != 0 || diag != 0) {
        return;
    }
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // 使用warp级原语优化
    for (int row = 0; row < m; ++row) {
        float sum = alpha * B[row * n + col];
        
        // 使用warp shuffle进行部分求和
        for (int k = 0; k < row; k += 32) {
            float local_sum = 0.0f;
            for (int kk = k; kk < min(k + 32, row); ++kk) {
                local_sum += A[row * m + kk] * X[kk * n + col];
            }
            
            // Warp内归约
            for (int offset = 16; offset > 0; offset /= 2) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            }
            
            if (lane_id == 0) {
                sum -= local_sum;
            }
        }
        
        // 广播结果
        sum = __shfl_sync(0xffffffff, sum, 0);
        
        if (lane_id == 0) {
            X[row * n + col] = sum / A[row * m + row];
        }
    }
}
)";
    
    return code.str();
}

std::string TrsmJITPlugin::GenerateBlockedKernel(const JITConfig& config) {
    std::stringstream code;
    int block_size = config.tile_size;
    
    code << R"(
extern "C" __global__ void trsm_kernel_blocked(
    int m, int n, float alpha, 
    const float* A, const float* B, float* X,
    int side, int uplo, int trans, int diag) {
    
    // 当前只支持 left, lower, no-trans, non-unit
    if (side != 0 || uplo != 1 || trans != 0 || diag != 0) {
        return;
    }
    
    __shared__ float block_A[)" << block_size << "][" << block_size << R"(];
    __shared__ float block_X[)" << block_size << R"(];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    
    int tx = threadIdx.x;
    
    // 大块分块策略
    for (int block_start = 0; block_start < m; block_start += )" << block_size << R"() {
        int block_end = min(block_start + )" << block_size << R"(, m);
        int block_size_actual = block_end - block_start;
        
        // 协作加载A块
        if (tx < block_size_actual) {
            for (int j = 0; j < block_size_actual; ++j) {
                block_A[tx][j] = A[(block_start + tx) * m + (block_start + j)];
            }
        }
        
        // 协作加载X块
        if (tx < block_size_actual) {
            block_X[tx] = X[(block_start + tx) * n + col];
        }
        
        __syncthreads();
        
        // 块内求解
        for (int i = 0; i < block_size_actual; ++i) {
            if (tx == 0) {
                float sum = alpha * B[(block_start + i) * n + col];
                
                // 减去块内已知项
                for (int k = 0; k < i; ++k) {
                    sum -= block_A[i][k] * block_X[k];
                }
                
                // 减去块外已知项（使用协作加载）
                for (int k = 0; k < block_start; k += )" << block_size << R"() {
                    float local_sum = 0.0f;
                    for (int kk = k; kk < min(k + )" << block_size << R"(, block_start); ++kk) {
                        local_sum += A[(block_start + i) * m + kk] * X[kk * n + col];
                    }
                    sum -= local_sum;
                }
                
                block_X[i] = sum / block_A[i][i];
            }
            __syncthreads();
        }
        
        // 协作写回
        if (tx < block_size_actual) {
            X[(block_start + tx) * n + col] = block_X[tx];
        }
        
        __syncthreads();
    }
}
)";
    
    return code.str();
}

// ==================== 辅助方法 ====================

CUfunction TrsmJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
    if (!compiler_) {
        return nullptr;
    }
    
    std::vector<std::string> options = {
        "-O2",
        "-arch=sm_70",
        "-lcublas"
    };
    
    JITCompileResult result = compiler_->CompileKernel(kernel_code, kernel_name, options);
    return result.kernel;
}

void TrsmJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
}

CUfunction TrsmJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

PerformanceProfile TrsmJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                    const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    profile.execution_time = 0.0;
    profile.kernel_type = current_kernel_type_;
    profile.throughput = 0.0;
    return profile;
}

bool TrsmJITPlugin::ValidateConfig(const JITConfig& config) {
    return !config.kernel_type.empty() && config.tile_size > 0 && config.block_size > 0;
}

JITConfig TrsmJITPlugin::OptimizeConfig(const JITConfig& config) {
    JITConfig optimized = config;
    
    // 根据矩阵大小优化tile_size
    if (matrix_A_.shape()[0] > 1024) {
        optimized.tile_size = std::max(optimized.tile_size, 32);
    }
    
    // 根据矩阵大小选择内核类型
    if (optimized.kernel_type == "auto") {
        int m = matrix_A_.shape()[0];
        optimized.kernel_type = SelectOptimalKernel(m, 1);
    }
    
    return optimized;
}

std::string TrsmJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::stringstream key;
    key << "trsm_" << config.kernel_type << "_" << config.tile_size << "_" << config.block_size;
    return key.str();
}

std::string TrsmJITPlugin::SelectOptimalKernel(int m, int n) {
    if (m >= 1024) {
        return "blocked";
    } else if (m >= 256) {
        return "warp_optimized";
    } else if (m >= 64) {
        return "tiled";
    } else {
        return "basic";
    }
}

} // namespace cu_op_mem 