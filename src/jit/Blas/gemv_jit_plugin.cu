#include "jit/Blas/gemv_jit_plugin.hpp"
#include "jit/jit_compiler.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <chrono>
#include <sstream>

namespace cu_op_mem {

// 辅助函数：生成内核缓存键
std::string GemvJITPlugin::GenerateKernelKey(const std::string& kernel_code, const JITConfig& config) {
    std::string key = kernel_code;
    key += "_" + std::to_string(config.block_size);
    key += "_" + std::to_string(config.tile_size);
    key += "_" + std::to_string(config.max_registers);
    key += "_" + (config.enable_shared_memory_opt ? std::string("1") : std::string("0"));
    key += "_" + config.optimization_level;
    return key;
}

// ==================== GemvJITPlugin 实现 ====================

GemvJITPlugin::GemvJITPlugin()
    : initialized_(false)
    , compiled_(false)
    , auto_tuning_enabled_(false)
    , transA_(false)
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

GemvJITPlugin::~GemvJITPlugin() {
    Cleanup();
}

StatusCode GemvJITPlugin::Initialize() {
    if (initialized_) {
        return StatusCode::SUCCESS;
    }
    
    try {
        if (!compiler_) {
            compiler_ = std::make_unique<JITCompiler>();
        }
        
        HardwareSpec hw_spec = GetGemvHardwareSpec();
        config_.hardware_spec = std::to_string(hw_spec.compute_capability_major) + "." + 
                                std::to_string(hw_spec.compute_capability_minor);
        
        initialized_ = true;
        LOG(INFO) << "GemvJITPlugin initialized successfully";
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Initialization failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::INITIALIZATION_ERROR;
    }
}

StatusCode GemvJITPlugin::Compile(const JITConfig& config) {
    if (!initialized_) {
        last_error_ = "Plugin not initialized";
        return StatusCode::NOT_INITIALIZED;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        config_ = config;
        
        if (!ValidateGemvConfig(config_)) {
            last_error_ = "Invalid configuration";
            return StatusCode::INVALID_ARGUMENT;
        }
        
        config_ = OptimizeGemvConfig(config_);
        
        std::string kernel_code = GenerateKernelCode(config_);
        if (kernel_code.empty()) {
            last_error_ = "Failed to generate kernel code";
            return StatusCode::COMPILATION_ERROR;
        }
        
        std::string kernel_name = "gemv_kernel_" + config_.kernel_type;
        CUfunction kernel = CompileKernel(kernel_code, kernel_name);
        if (!kernel) {
            last_error_ = "Failed to compile kernel";
            return StatusCode::COMPILATION_ERROR;
        }
        
        std::string cache_key = GenerateKernelKey(kernel_code, config_);
        CacheKernel(cache_key, kernel);
        kernel_names_[cache_key] = kernel_name;
        current_kernel_type_ = config_.kernel_type;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        total_compilation_time_ += std::chrono::duration<double>(end_time - start_time).count();
        
        compiled_ = true;
        LOG(INFO) << "GemvJITPlugin compiled successfully with kernel type: " << config_.kernel_type;
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Compilation failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::COMPILATION_ERROR;
    }
}

StatusCode GemvJITPlugin::Execute(const std::vector<Tensor<float>>& inputs, 
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
        
        int m = weight_.shape()[0];
        int n = weight_.shape()[1];
        
        if (input.shape()[0] != n) {
            last_error_ = "Matrix dimension mismatch";
            return StatusCode::SHAPE_MISMATCH;
        }
        
        const float* d_A = weight_.data();
        const float* d_x = input.data();
        float* d_y = output.data();
        
        std::string cache_key = GenerateKernelKey("", config_);
        CUfunction kernel = GetCachedKernel(cache_key);
        if (!kernel) {
            last_error_ = "Kernel not found in cache";
            return StatusCode::KERNEL_NOT_FOUND;
        }
        
        void* args[] = {
            &m, &n, &alpha_, 
            const_cast<float**>(&d_A), 
            const_cast<float**>(&d_x), 
            &beta_, &d_y,
            &transA_
        };
        
        dim3 threads, blocks;
        if (config_.kernel_type == "basic") {
            threads = dim3(256);
            blocks = dim3((m + 255) / 256);
        } else {
            threads = dim3(256);
            blocks = dim3((m + 255) / 256);
        }
        
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
        
        last_profile_.execution_time = execution_time;
        last_profile_.kernel_type = config_.kernel_type;
        last_profile_.matrix_size = {m, n, 1};
        last_profile_.throughput = (2.0 * m * n) / (execution_time * 1e9); // GFLOPS
        
        performance_history_.push_back(last_profile_);
        if (performance_history_.size() > 100) {
            performance_history_.erase(performance_history_.begin());
        }
        
        kernel_performance_[config_.kernel_type] = execution_time;
        
        LOG(INFO) << "GemvJITPlugin executed successfully: " 
                  << "time=" << execution_time << "s, "
                  << "throughput=" << last_profile_.throughput << " GFLOPS";
        
        return StatusCode::SUCCESS;
        
    } catch (const std::exception& e) {
        last_error_ = "Execution failed: " + std::string(e.what());
        LOG(ERROR) << last_error_;
        return StatusCode::EXECUTION_ERROR;
    }
}

void GemvJITPlugin::Optimize(const PerformanceProfile& profile) {
    if (!auto_tuning_enabled_) {
        return;
    }
    
    UpdateKernelSelection(profile);
    
    if (profile.execution_time > 0.001) {
        std::string optimal_kernel = SelectKernelType(
            profile.matrix_size[0], 
            profile.matrix_size[1]
        );
        
        if (optimal_kernel != config_.kernel_type) {
            config_.kernel_type = optimal_kernel;
            LOG(INFO) << "Auto-tuning: switching to kernel type: " << optimal_kernel;
            Compile(config_);
        }
    }
}

void GemvJITPlugin::SetConfig(const JITConfig& config) {
    config_ = config;
}

JITConfig GemvJITPlugin::GetConfig() const {
    return config_;
}

void GemvJITPlugin::EnableAutoTuning(bool enable) {
    auto_tuning_enabled_ = enable;
}

bool GemvJITPlugin::IsAutoTuningEnabled() const {
    return auto_tuning_enabled_;
}

PerformanceProfile GemvJITPlugin::GetPerformanceProfile() const {
    return last_profile_;
}

bool GemvJITPlugin::IsInitialized() const {
    return initialized_;
}

bool GemvJITPlugin::IsCompiled() const {
    return compiled_;
}

std::string GemvJITPlugin::GetLastError() const {
    return last_error_;
}

void GemvJITPlugin::Cleanup() {
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

size_t GemvJITPlugin::GetMemoryUsage() const {
    return memory_usage_;
}

void GemvJITPlugin::SetGemvParams(bool transA, float alpha, float beta) {
    transA_ = transA;
    alpha_ = alpha;
    beta_ = beta;
}

void GemvJITPlugin::SetWeight(const Tensor<float>& weight) {
    // 由于Tensor不支持拷贝，我们需要重新设计这个接口
    // 暂时使用移动语义，但需要修改接口
    weight_ = std::move(const_cast<Tensor<float>&>(weight));
}

bool GemvJITPlugin::SupportsOperator(const std::string& op_name) {
    return op_name == "gemv" || op_name == "Gemv";
}

// ==================== 私有方法实现 ====================

std::string GemvJITPlugin::GenerateKernelCode(const JITConfig& config) {
    if (config.kernel_type == "basic") {
        return GenerateBasicKernel(config);
    } else {
        return GenerateOptimizedKernel(config);
    }
}

std::string GemvJITPlugin::GenerateBasicKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void gemv_kernel_basic(
    int m, int n, float alpha, 
    const float* A, const float* x, float beta, float* y,
    bool transA) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m) return;
    
    float sum = 0.0f;
    
    if (transA) {
        for (int i = 0; i < n; ++i) {
            sum += A[i * m + row] * x[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * x[i];
        }
    }
    
    y[row] = alpha * sum + beta * y[row];
}
)";
    
    return code.str();
}

std::string GemvJITPlugin::GenerateOptimizedKernel(const JITConfig& config) {
    std::stringstream code;
    
    code << R"(
extern "C" __global__ void gemv_kernel_optimized(
    int m, int n, float alpha, 
    const float* A, const float* x, float beta, float* y,
    bool transA) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m) return;
    
    float sum = 0.0f;
    
    // 使用循环展开优化
    if (transA) {
        for (int i = 0; i < n; i += 4) {
            if (i + 3 < n) {
                sum += A[i * m + row] * x[i] + 
                       A[(i+1) * m + row] * x[i+1] + 
                       A[(i+2) * m + row] * x[i+2] + 
                       A[(i+3) * m + row] * x[i+3];
            } else {
                for (int j = i; j < n; ++j) {
                    sum += A[j * m + row] * x[j];
                }
            }
        }
    } else {
        for (int i = 0; i < n; i += 4) {
            if (i + 3 < n) {
                sum += A[row * n + i] * x[i] + 
                       A[row * n + i + 1] * x[i+1] + 
                       A[row * n + i + 2] * x[i+2] + 
                       A[row * n + i + 3] * x[i+3];
            } else {
                for (int j = i; j < n; ++j) {
                    sum += A[row * n + j] * x[j];
                }
            }
        }
    }
    
    y[row] = alpha * sum + beta * y[row];
}
)";
    
    return code.str();
}

CUfunction GemvJITPlugin::CompileKernel(const std::string& kernel_code, const std::string& kernel_name) {
    if (!compiler_) {
        return nullptr;
    }
    
    std::vector<std::string> options = {
        "-O2",
        "-arch=sm_70"
    };
    
    JITCompileResult result = compiler_->CompileKernel(kernel_code, kernel_name, options);
    return result.kernel;
}

void GemvJITPlugin::CacheKernel(const std::string& key, const CUfunction& kernel) {
    kernel_cache_[key] = kernel;
}

CUfunction GemvJITPlugin::GetCachedKernel(const std::string& key) {
    auto it = kernel_cache_.find(key);
    return (it != kernel_cache_.end()) ? it->second : nullptr;
}

PerformanceProfile GemvJITPlugin::MeasurePerformance(const CUfunction& kernel, const JITConfig& config) {
    PerformanceProfile profile;
    profile.execution_time = last_profile_.execution_time;
    profile.kernel_type = config_.kernel_type;
    profile.matrix_size = {static_cast<int>(weight_.shape()[0]), 
                          static_cast<int>(weight_.shape()[1]), 
                          1};
    profile.throughput = last_profile_.throughput;
    return profile;
}

bool GemvJITPlugin::ValidateGemvConfig(const JITConfig& config) const {
    return !config.kernel_type.empty() && 
           config.tile_size > 0 && 
           config.block_size > 0;
}

JITConfig GemvJITPlugin::OptimizeGemvConfig(const JITConfig& config) const {
    JITConfig optimized = config;
    
    if (config.kernel_type == "optimized") {
        optimized.block_size = 256;
    }
    
    return optimized;
}

HardwareSpec GemvJITPlugin::GetGemvHardwareSpec() const {
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

std::string GemvJITPlugin::SelectKernelType(int m, int n) const {
    if (m < 1024 || n < 1024) {
        return "basic";
    } else {
        return "optimized";
    }
}

void GemvJITPlugin::UpdateKernelSelection(const PerformanceProfile& profile) {
    if (performance_history_.size() > 10) {
        double avg_time = 0.0;
        for (const auto& hist : performance_history_) {
            avg_time += hist.execution_time;
        }
        avg_time /= performance_history_.size();
        
        if (avg_time > 0.001 && config_.kernel_type == "basic") {
            config_.kernel_type = "optimized";
        }
    }
}

} // namespace cu_op_mem 