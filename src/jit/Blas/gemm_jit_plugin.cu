#include "jit/Blas/gemm_jit_plugin.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
#include <sstream>
#include <algorithm>

namespace cu_op_mem {

// 辅助函数：生成内核缓存键
std::string GemmJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::string key = kernel_code;
    key += "_" + std::to_string(config.block_size);
    key += "_" + std::to_string(config.tile_size);
    key += "_" + std::to_string(config.max_registers);
    key += "_" + (config.enable_shared_memory_opt ? std::string("1") : std::string("0"));
    key += "_" + config.optimization_level;
    return key;
}

// ==================== GemmJITPlugin 实现 ====================

GemmJITPlugin::GemmJITPlugin()
    : initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , transA_(false)
    , transB_(false)
    , alpha_(1.0f)
    , beta_(0.0f)
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

GemmJITPlugin::~GemmJITPlugin() {
    Cleanup();
}

StatusCode GemmJITPlugin::Initialize() {
    if (initialized_) {
        return StatusCode::SUCCESS;
    }
    
    try {
        // 初始化编译器
        if (!compiler_) {
            compiler_ = std::make_unique<JITCompiler>();
        }
        
        // 获取硬件规格
        HardwareSpec hw_spec = GetHardwareSpec();
        config_.hardware_spec = std::to_string(hw_spec.compute_capability_major) + "." + 
                                std::to_string(hw_spec.compute_capability_minor);
        
        // 根据硬件调整配置
        if (SupportsTensorCore()) {
            config_.enable_tensor_core = true;
            LOG(INFO) << "Tensor Core supported, enabling in config";
        }
        
        if (SupportsTMA()) {
            config_.enable_tma = true;
            LOG(INFO) << "TMA supported, enabling in config";
        }
        
        initialized_ = true;
        LOG(INFO) << "GemmJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::INITIALIZATION_ERROR;
    }
}

StatusCode GemmJITPlugin::Compile(const JITConfig& config) {
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
        std::string kernel_name = "gemm_kernel_" + config_.kernel_type;
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
        LOG(INFO) << "GemmJITPlugin compiled successfully with kernel type: " << config_.kernel_type;
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::COMPILATION_ERROR;
    }
}

StatusCode GemmJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
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
        
        const Tensor<float>& input = inputs[0];
        Tensor<float>& output = outputs[0];
        
        // 获取矩阵维度
        int m = input.shape()[0];
        int k = input.shape()[1];
        int n = weight_.shape()[1];
        
        // 验证维度
        if (input.shape()[1] != weight_.shape()[0]) {
            last_error_ = "Matrix dimension mismatch";
            return StatusCode::SHAPE_MISMATCH;
        }
        
        // 获取设备指针
        const float* d_A = input.data();
        const float* d_B = weight_.data();
        float* d_C = output.data();
        
        // 获取编译的内核
        std::string cache_key = GenerateKernelKey("", config_);
        CUfunction kernel = GetCachedKernel(cache_key);
        if (!kernel) {
            last_error_ = "Kernel not found in cache";
            return StatusCode::KERNEL_NOT_FOUND;
        }
        
        // 设置内核参数
        void* args[] = {
            &m, &n, &k, &alpha_, 
            const_cast<float**>(&d_A), 
            const_cast<float**>(&d_B), 
            &beta_, &d_C,
            &transA_, &transB_
        };
        
        // 计算网格和块大小
        dim3 threads, blocks;
        if (config_.kernel_type == "basic") {
            threads = dim3(16, 16);
            blocks = dim3((n + 15) / 16, (m + 15) / 16);
        } else if (config_.kernel_type == "tiled") {
            threads = dim3(config_.tile_size, config_.tile_size);
            blocks = dim3((n + config_.tile_size - 1) / config_.tile_size, 
                         (m + config_.tile_size - 1) / config_.tile_size);
        } else if (config_.kernel_type == "warp_optimized") {
            threads = dim3(32, 32);
            blocks = dim3((n + 31) / 32, (m + 31) / 32);
        } else if (config_.kernel_type == "blocked") {
            threads = dim3(32, 32);
            blocks = dim3((n + 31) / 32, (m + 31) / 32);
        } else {
            threads = dim3(16, 16);
            blocks = dim3((n + 15) / 16, (m + 15) / 16);
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
        
        // 同步设备
        cudaDeviceSynchronize();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // 更新统计信息
        total_executions_++;
        total_execution_time_ += execution_time;
        
        // 更新性能配置
        last_profile_.execution_time = execution_time;
        last_profile_.kernel_type = config_.kernel_type;
        last_profile_.matrix_size = {m, n, k};
        last_profile_.throughput = (2.0 * m * n * k) / (execution_time * 1e9); // GFLOPS
        
        // 性能历史记录
        performance_history_.push_back(last_profile_);
        if (performance_history_.size() > 100) {
            performance_history_.erase(performance_history_.begin());
        }
        
        // 更新内核性能记录
        kernel_performance_[config_.kernel_type] = execution_time;
        
        VLOG(2) << "GemmJITPlugin executed successfully: " 
                << "time=" << execution_time << "s, "
                << "throughput=" << last_profile_.throughput << " GFLOPS";
        
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Execution failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::EXECUTION_ERROR;
    }
}

void GemmJITPlugin::Optimize(const PerformanceProfile& profile) {
    if (!auto_tuning_enabled_) {
        return;
    }
    
    // 基于性能配置优化内核选择
    UpdateKernelSelection(profile);
    
    // 如果性能不理想，尝试重新编译
    if (profile.execution_time > 0.001) { // 如果执行时间超过1ms
        std::string optimal_kernel = SelectKernelType(
            profile.matrix_size[0], 
            profile.matrix_size[1], 
            profile.matrix_size[2]
        );
        
        if (optimal_kernel != config_.kernel_type) {
            config_.kernel_type = optimal_kernel;
            LOG(INFO) << "Auto-tuning: switching to kernel type: " << optimal_kernel;
            
            // 重新编译
            Compile(config_);
        }
    }
}

void GemmJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig GemmJITPlugin::GetConfig() const {
    return config_;
}

void GemmJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool GemmJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile GemmJITPlugin::GetPerformanceProfile() const {
    return last_profile_;
}

bool GemmJITPlugin::IsInitialized() const {
    return initialized_;
}

bool GemmJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string GemmJITPlugin::GetLastError() const {
    return last_error_;
}

void GemmJITPlugin::Cleanup() {
    if (compiler_) {
        compiler_->ClearCache();
    }
    
    kernel_cache_.clear();
    kernel_names_.clear();
    performance_history_.clear();
    kernel_performance_.clear();
    
    initialized_ = false;
    compiled_ = false;
    memory_usage_ = 0;
}

size_t GemmJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

void GemmJITPlugin::SetGemmParams(bool transA, bool transB, float alpha, float beta) {
    transA_ = transA;
    transB_ = transB;
    alpha_ = alpha;
    beta_ = beta;
}

void GemmJITPlugin::SetWeight(const Tensor<float>& weight) {
    // 由于Tensor不支持拷贝，我们需要重新设计这个接口
    // 暂时使用移动语义，但需要修改接口
    weight_ = std::move(const_cast<Tensor<float>&>(weight));
}

bool GemmJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "gemm" || op_name == "Gemm";
}

// ==================== 私有方法实现 ====================

std::string GemmJITPlugin::GenerateKernelCode(const JITConfig& config) {
    if (config.kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else if (config.kernel_type == "tiled") {
        return GenerateTiledKernel(config);
    } else if (config.kernel_type == "warp_optimized") {
        return GenerateWarpOptimizedKernel(config);
    } else if (config.kernel_type == "tensor_core") {
        return GenerateTensorCoreKernel(config);
    } else if (config.kernel_type == "blocked") {
        return GenerateBlockedKernel(config);
    } else {
        return GenerateBasicKernel(config);
    }
}

std::string GemmJITPlugin::GenerateBasicKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void gemm_kernel_basic(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < k; ++i) {
        float a_val, b_val;
        
        if (transA) {
            a_val = A[i * k + row];
        } else {
            a_val = A[row * k + i];
        }
        
        if (transB) {
            b_val = B[col * k + i];
        } else {
            b_val = B[i * n + col];
        }
        
        sum += a_val * b_val;
    }
    
    C[row * n + col] = alpha * sum + beta * C[row * n + col];
}
)";
    
    return code.str();
}

std::string GemmJITPlugin::GenerateTiledKernel(const JITConfig& config) {
    std::stringstream code;
    int tile_size = config.tile_size;
    
    code << R"(
extern "C" __global__ void gemm_kernel_tiled(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    __shared__ float As[)" << tile_size << "][" << tile_size << R"(];
    __shared__ float Bs[)" << tile_size << "][" << tile_size << R"(];
    
    int row = blockIdx.y * )" << tile_size << R"( + threadIdx.y;
    int col = blockIdx.x * )" << tile_size << R"( + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + )" << tile_size << R"( - 1) / )" << tile_size << R"(; ++t) {
        int tiled_col = t * )" << tile_size << R"( + threadIdx.x;
        int tiled_row = t * )" << tile_size << R"( + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < n && tiled_row < k) ? B[tiled_row * n + col] : 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < )" << tile_size << R"(; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return code.str();
}

std::string GemmJITPlugin::GenerateWarpOptimizedKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void gemm_kernel_warp_optimized(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int row = blockIdx.y * 32 + warp_id;
    int col = blockIdx.x * 32 + lane_id;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + 31) / 32; ++t) {
        int tiled_col = t * 32 + lane_id;
        int tiled_row = t * 32 + warp_id;
        
        As[warp_id][lane_id] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0.0f;
        Bs[warp_id][lane_id] = (col < n && tiled_row < k) ? B[tiled_row * n + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum += As[warp_id][i] * Bs[i][lane_id];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return code.str();
}

std::string GemmJITPlugin::GenerateTensorCoreKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
#include <mma.h>

extern "C" __global__ void gemm_kernel_tensor_core(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    using namespace nvcuda::wmma;
    
    // 声明fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化accumulator
    fill_fragment(c_frag, 0.0f);
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    
    for (int i = 0; i < k; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;
        
        if (aRow < m && aCol < k) {
            load_matrix_sync(a_frag, (half*)(A + aRow * k + aCol), k);
        } else {
            fill_fragment(a_frag, __float2half(0.0f));
        }
        
        if (bRow < k && bCol < n) {
            load_matrix_sync(b_frag, (half*)(B + bRow * n + bCol), n);
        } else {
            fill_fragment(b_frag, __float2half(0.0f));
        }
        
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    
    if (cRow < m && cCol < n) {
        store_matrix_sync(C + cRow * n + cCol, c_frag, n, mem_row_major);
    }
}
)";
    
    return code.str();
}

std::string GemmJITPlugin::GenerateBlockedKernel(const JITConfig& config) {
    std::stringstream code;
    int block_size = config.block_size;
    
    code << R"(
extern "C" __global__ void gemm_kernel_blocked(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    __shared__ float As[)" << block_size << "][" << block_size << R"(];
    __shared__ float Bs[)" << block_size << "][" << block_size << R"(];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * )" << block_size << R"( + ty;
    int col = bx * )" << block_size << R"( + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + )" << block_size << R"( - 1) / )" << block_size << R"(; ++t) {
        if (row < m && t * )" << block_size << R"( + tx < k) {
            As[ty][tx] = A[row * k + t * )" << block_size << R"( + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < n && t * )" << block_size << R"( + ty < k) {
            Bs[ty][tx] = B[(t * )" << block_size << R"( + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < )" << block_size << R"(; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}
)";
    
    return code.str();
}

CUfunction GemmJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
    if (!compiler_) {
        return nullptr;
    }
    
    std::vector<std::string> options = {
        "-O2",
        "-arch=sm_70",
        "-lcublas",
        "-lcudnn"
    };
    
    if (config_.enable_tensor_core) {
        options.push_back("-DCUDA_TENSOR_CORE");
    }
    
    JITCompileResult result = compiler_->CompileKernel(kernel_code, kernel_name, options);
    return result.kernel;
}

void GemmJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
}

CUfunction GemmJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

PerformanceProfile GemmJITPlugin::MeasurePerformance(const std::vector<Tensor<float>>& inputs,
                                                    const std::vector<Tensor<float>>& outputs) {
    PerformanceProfile profile;
    profile.execution_time = last_profile_.execution_time;
    profile.kernel_type = config_.kernel_type;
    profile.matrix_size = {static_cast<int>(inputs[0].shape()[0]), 
                          static_cast<int>(outputs[0].shape()[1]), 
                          static_cast<int>(inputs[0].shape()[1])};
    profile.throughput = last_profile_.throughput;
    return profile;
}

bool GemmJITPlugin::ValidateConfig(const JITConfig& config) const {
    return !config.kernel_type.empty() && 
           config.tile_size > 0 && 
           config.block_size > 0;
}

JITConfig GemmJITPlugin::OptimizeConfig(const JITConfig& config) const {
    JITConfig optimized = config;
    
    // 根据矩阵大小优化tile_size
    if (config.kernel_type == "tiled") {
        if (config.matrix_size[0] < 512 || config.matrix_size[1] < 512) {
            optimized.tile_size = 16;
        } else {
            optimized.tile_size = 32;
        }
    }
    
    return optimized;
}

HardwareSpec GemmJITPlugin::GetHardwareSpec() const {
    HardwareSpec spec;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    spec.compute_capability_major = prop.major;
    spec.compute_capability_minor = prop.minor;
    spec.multi_processor_count = prop.multiProcessorCount;
    spec.max_threads_per_block = prop.maxThreadsPerBlock;
    spec.shared_memory_per_block = prop.sharedMemPerBlock;
    spec.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    
    return spec;
}

bool GemmJITPlugin::SupportsTensorCore() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major >= 7;
}

bool GemmJITPlugin::SupportsTMA() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major >= 9;
}

std::string GemmJITPlugin::SelectKernelType(int m, int n, int k) const {
    // 简单的内核选择策略
    if (m < 64 || n < 64 || k < 64) {
        return "basic";
    } else if (m >= 1024 && n >= 1024 && k >= 1024) {
        return "blocked";
    } else if (SupportsTensorCore() && m >= 256 && n >= 256 && k >= 256) {
        return "tensor_core";
    } else {
        return "tiled";
    }
}

void GemmJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    // 基于历史性能数据更新内核选择策略
    if (performance_history_.size() > 10) {
        // 分析最近10次的性能数据
        double avg_time = 0.0;
        for (const auto& hist : performance_history_) {
            avg_time += hist.execution_time;
        }
        avg_time /= performance_history_.size();
        
        // 如果平均执行时间过长，考虑切换到更优化的内核
        if (avg_time > 0.001 && config_.kernel_type == "basic") {
            config_.kernel_type = "tiled";
        }
    }
}

// ==================== GemmKernelTemplate 实现 ====================

std::string GemmKernelTemplate::GenerateKernelCode(const JITConfig& config) {
    if (config.kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else if (config.kernel_type == "tiled") {
        return GenerateTiledKernel(config);
    } else if (config.kernel_type == "warp_optimized") {
        return GenerateWarpOptimizedKernel(config);
    } else if (config.kernel_type == "tensor_core") {
        return GenerateTensorCoreKernel(config);
    } else if (config.kernel_type == "blocked") {
        return GenerateBlockedKernel(config);
    } else {
        return GenerateBasicKernel(config);
    }
}

std::vector<std::string> GemmKernelTemplate::GetCompileOptions(const JITConfig& config) const {
    std::vector<std::string> options = {
        "-O2",
        "-arch=sm_70"
    };
    
    if (config.enable_tensor_core) {
        options.push_back("-DCUDA_TENSOR_CORE");
    }
    
    return options;
}

bool GemmKernelTemplate::ValidateConfig(const JITConfig& config) const {
    return !config.kernel_type.empty() && 
           config.tile_size > 0 && 
           config.block_size > 0;
}

// 其他模板方法的实现与GemmJITPlugin中的相同
std::string GemmKernelTemplate::GenerateBasicKernel(const JITConfig& config) {
    // 与GemmJITPlugin::GenerateBasicKernel相同
      GemmJITPlugin plugin;
      return plugin.GenerateBasicKernel(config);
}

std::string GemmKernelTemplate::GenerateTiledKernel(const JITConfig& config) {
    // 与GemmJITPlugin::GenerateTiledKernel相同
      GemmJITPlugin plugin;
      return plugin.GenerateTiledKernel(config);
}

std::string GemmKernelTemplate::GenerateWarpOptimizedKernel(const JITConfig& config) {
    // 与GemmJITPlugin::GenerateWarpOptimizedKernel相同
      GemmJITPlugin plugin;
      return plugin.GenerateWarpOptimizedKernel(config);
}

std::string GemmKernelTemplate::GenerateTensorCoreKernel(const JITConfig& config) {
    // 与GemmJITPlugin::GenerateTensorCoreKernel相同
      GemmJITPlugin plugin;
      return plugin.GenerateTensorCoreKernel(config);
}

std::string GemmKernelTemplate::GenerateBlockedKernel(const JITConfig& config) {
    // 与GemmJITPlugin::GenerateBlockedKernel相同
      GemmJITPlugin plugin;
      return plugin.GenerateBlockedKernel(config);
}

std::string GemmKernelTemplate::GenerateTMAKernel(const JITConfig& config) {
    // TMA内核实现（高级特性）
    std::stringstream code;
    code << R"(
// TMA内核实现 - 需要H100+ GPU
extern "C" __global__ void gemm_kernel_tma(
    int m, int n, int k, float alpha, 
    const float* A, const float* B, float beta, float* C,
    bool transA, bool transB) {
    
    // TMA实现将在后续版本中添加
    // 当前版本暂不支持TMA
}
)";
    return code.str();
}

} // namespace cu_op_mem 