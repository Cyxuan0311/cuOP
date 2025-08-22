#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cu_op_mem {

// 全局JIT配置
struct GlobalJITConfig {
    bool enable_jit = true;                    // 全局JIT开关
    bool enable_auto_tuning = true;            // 自动调优开关
    bool enable_caching = true;                // 缓存开关
    std::string cache_dir = "./jit_cache";     // 缓存目录
    int max_cache_size = 1024 * 1024 * 1024;  // 最大缓存大小 (1GB)
    int compilation_timeout = 30;              // 编译超时时间 (秒)
    bool enable_tensor_core = true;            // Tensor Core开关
    bool enable_tma = true;                    // TMA开关
    int max_compilation_threads = 4;           // 最大编译线程数
    bool enable_debug = false;                 // 调试模式
};

// 算子JIT配置
struct JITConfig {
    bool enable_jit = true;                    // 算子JIT开关
    std::vector<int> block_sizes = {16, 32, 64, 128};  // 块大小选项
    std::vector<int> tile_sizes = {16, 32, 64};        // 瓦片大小选项
    int num_stages = 2;                        // 流水线阶段数
    bool use_tensor_core = true;               // 使用Tensor Core
    bool use_tma = true;                       // 使用TMA
    std::string optimization_level = "auto";   // 优化级别: "auto", "fast", "best"
    int max_registers = 255;                   // 最大寄存器数
    bool enable_shared_memory_opt = true;      // 共享内存优化
    bool enable_loop_unroll = true;            // 循环展开
    bool enable_memory_coalescing = true;      // 内存合并优化
};

// JIT统计信息
struct JITStatistics {
    int total_compilations = 0;                // 总编译次数
    int cache_hits = 0;                        // 缓存命中次数
    int cache_misses = 0;                      // 缓存未命中次数
    double total_compilation_time = 0.0;       // 总编译时间 (秒)
    double total_execution_time = 0.0;         // 总执行时间 (秒)
    size_t cache_size = 0;                     // 当前缓存大小 (字节)
    int active_kernels = 0;                    // 活跃kernel数量
    
    // 计算缓存命中率
    double GetCacheHitRate() const {
        int total_requests = cache_hits + cache_misses;
        return total_requests > 0 ? static_cast<double>(cache_hits) / total_requests : 0.0;
    }
    
    // 计算平均编译时间
    double GetAverageCompilationTime() const {
        return total_compilations > 0 ? total_compilation_time / total_compilations : 0.0;
    }
    
    // 重置统计信息
    void Reset() {
        total_compilations = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_compilation_time = 0.0;
        total_execution_time = 0.0;
        cache_size = 0;
        active_kernels = 0;
    }
};

// 性能分析信息
struct PerformanceProfile {
    double gflops = 0.0;                       // GFLOPS
    double bandwidth_gb_s = 0.0;               // 内存带宽 (GB/s)
    double kernel_time_ms = 0.0;               // kernel执行时间 (ms)
    double launch_overhead_ms = 0.0;           // 启动开销 (ms)
    double compilation_time_ms = 0.0;          // 编译时间 (ms)
    int block_size_x = 0;                      // 实际使用的block大小
    int block_size_y = 0;
    int tile_size = 0;                         // 实际使用的tile大小
    bool used_tensor_core = false;             // 是否使用了Tensor Core
    bool used_tma = false;                     // 是否使用了TMA
    std::string kernel_name;                   // kernel名称
    std::string optimization_level;            // 使用的优化级别
    
    // 计算总时间
    double GetTotalTime() const {
        return kernel_time_ms + launch_overhead_ms + compilation_time_ms;
    }
    
    // 计算效率 (GFLOPS/理论峰值)
    double GetEfficiency(double theoretical_peak_gflops) const {
        return theoretical_peak_gflops > 0 ? gflops / theoretical_peak_gflops : 0.0;
    }
};

// 硬件规格信息
struct HardwareSpec {
    int compute_capability_major = 0;          // 计算能力主版本
    int compute_capability_minor = 0;          // 计算能力次版本
    int num_sms = 0;                           // SM数量
    int max_threads_per_sm = 0;                // 每SM最大线程数
    int max_shared_memory_per_sm = 0;          // 每SM最大共享内存 (字节)
    int max_registers_per_sm = 0;              // 每SM最大寄存器数
    int max_threads_per_block = 0;             // 每块最大线程数
    int max_blocks_per_sm = 0;                 // 每SM最大块数
    size_t total_global_memory = 0;            // 总全局内存 (字节)
    int memory_clock_rate = 0;                 // 内存时钟频率 (MHz)
    int memory_bus_width = 0;                  // 内存总线宽度 (位)
    bool supports_tensor_core = false;         // 是否支持Tensor Core
    bool supports_tma = false;                 // 是否支持TMA
    std::string gpu_name;                      // GPU名称
    
    // 获取计算能力字符串
    std::string GetComputeCapabilityString() const {
        return std::to_string(compute_capability_major) + "." + 
               std::to_string(compute_capability_minor);
    }
    
    // 计算理论峰值GFLOPS (FP32)
    double GetTheoreticalPeakGFLOPS() const {
        // 简化计算，实际应该根据具体架构调整
        return num_sms * 2.0 * 1024.0; // 假设每SM 2K FP32 FLOPS/cycle
    }
    
    // 计算理论内存带宽 (GB/s)
    double GetTheoreticalMemoryBandwidth() const {
        return (memory_clock_rate * 2.0 * memory_bus_width / 8.0) / 1000.0;
    }
};

} // namespace cu_op_mem 