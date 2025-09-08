#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>
#include <fstream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cu_op_mem {

// 性能事件类型
enum class PerformanceEventType {
    KERNEL_LAUNCH,      // 内核启动
    MEMORY_ALLOC,       // 内存分配
    MEMORY_COPY,        // 内存拷贝
    JIT_COMPILE,        // JIT编译
    CACHE_ACCESS,       // 缓存访问
    OPERATOR_EXEC,      // 算子执行
    CUSTOM_EVENT        // 自定义事件
};

// 性能事件
struct PerformanceEvent {
    std::string name;                    // 事件名称
    PerformanceEventType type;           // 事件类型
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    cudaEvent_t cuda_start;              // CUDA事件开始
    cudaEvent_t cuda_end;                // CUDA事件结束
    size_t memory_used;                  // 使用的内存
    size_t memory_allocated;             // 分配的内存
    std::string device_info;             // 设备信息
    std::string kernel_info;             // 内核信息
    std::unordered_map<std::string, std::string> metadata; // 元数据
    
    PerformanceEvent() : type(PerformanceEventType::CUSTOM_EVENT), 
                        memory_used(0), memory_allocated(0) {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_end);
    }
    
    ~PerformanceEvent() {
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_end);
    }
    
    double GetDuration() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
    
    double GetCudaDuration() const {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, cuda_start, cuda_end);
        return milliseconds;
    }
};

// 内存使用统计
struct MemoryStats {
    size_t total_allocated = 0;          // 总分配内存
    size_t total_freed = 0;              // 总释放内存
    size_t peak_usage = 0;               // 峰值使用
    size_t current_usage = 0;            // 当前使用
    size_t allocation_count = 0;         // 分配次数
    size_t free_count = 0;               // 释放次数
    std::vector<size_t> allocation_sizes; // 分配大小历史
    std::chrono::high_resolution_clock::time_point last_update;
    
    void Update(size_t allocated, size_t freed) {
        total_allocated += allocated;
        total_freed += freed;
        current_usage = total_allocated - total_freed;
        peak_usage = std::max(peak_usage, current_usage);
        allocation_count++;
        free_count++;
        if (allocated > 0) allocation_sizes.push_back(allocated);
        last_update = std::chrono::high_resolution_clock::now();
    }
};

// 热点分析结果
struct HotspotAnalysis {
    std::string function_name;           // 函数名
    double total_time = 0.0;             // 总时间
    double percentage = 0.0;             // 时间百分比
    size_t call_count = 0;               // 调用次数
    double avg_time = 0.0;               // 平均时间
    double min_time = 0.0;               // 最小时间
    double max_time = 0.0;               // 最大时间
    std::vector<double> time_history;    // 时间历史
    std::string optimization_suggestion; // 优化建议
};

// 性能分析结果
struct PerformanceAnalysis {
    std::vector<HotspotAnalysis> hotspots;           // 热点分析
    MemoryStats memory_stats;                        // 内存统计
    std::vector<PerformanceEvent> events;            // 性能事件
    double total_execution_time = 0.0;               // 总执行时间
    double gpu_utilization = 0.0;                    // GPU利用率
    double memory_bandwidth = 0.0;                   // 内存带宽
    std::string report_summary;                      // 报告摘要
    std::vector<std::string> recommendations;        // 优化建议
    std::chrono::high_resolution_clock::time_point analysis_time;
};

// 自动调优配置
struct AutoTuneConfig {
    bool enabled = true;                              // 是否启用
    size_t max_iterations = 100;                      // 最大迭代次数
    double improvement_threshold = 0.05;              // 改进阈值
    std::vector<std::string> tunable_params;          // 可调参数
    std::string optimization_target = "throughput";   // 优化目标
    bool save_best_config = true;                     // 保存最佳配置
    std::string config_save_path = "./best_configs/"; // 配置保存路径
};

// 自动调优结果
struct AutoTuneResult {
    std::unordered_map<std::string, std::string> best_params; // 最佳参数
    double best_performance = 0.0;                    // 最佳性能
    size_t iterations_used = 0;                       // 使用的迭代次数
    std::vector<double> performance_history;          // 性能历史
    std::string optimization_summary;                 // 优化摘要
    bool converged = false;                           // 是否收敛
};

// 性能监控器主类
class PerformanceMonitor {
public:
    static PerformanceMonitor& Instance();
    
    // 基本控制
    void StartMonitoring();
    void StopMonitoring();
    void Reset();
    bool IsMonitoring() const { return monitoring_enabled_; }
    
    // 事件记录
    void StartEvent(const std::string& name, PerformanceEventType type = PerformanceEventType::CUSTOM_EVENT);
    void EndEvent(const std::string& name);
    void RecordEvent(const std::string& name, double duration, PerformanceEventType type = PerformanceEventType::CUSTOM_EVENT);
    
    // 内存监控
    void RecordMemoryAllocation(size_t size);
    void RecordMemoryFree(size_t size);
    MemoryStats GetMemoryStats() const;
    
    // 性能分析
    PerformanceAnalysis AnalyzePerformance();
    std::vector<HotspotAnalysis> AnalyzeHotspots();
    std::string GenerateReport();
    
    // 自动调优
    AutoTuneResult AutoTune(const std::string& operator_name, 
                           const std::unordered_map<std::string, std::vector<std::string>>& param_ranges,
                           std::function<double(const std::unordered_map<std::string, std::string>&)> benchmark_func);
    
    // 配置管理
    void SetAutoTuneConfig(const AutoTuneConfig& config);
    AutoTuneConfig GetAutoTuneConfig() const;
    
    // 数据导出
    void ExportToCSV(const std::string& filename);
    void ExportToJSON(const std::string& filename);
    void SaveAnalysis(const std::string& filename);
    
    // 实时监控
    void EnableRealTimeMonitoring(bool enable);
    void SetMonitoringInterval(int interval_ms);
    void RegisterCallback(std::function<void(const PerformanceAnalysis&)> callback);
    
private:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // 禁用拷贝构造和赋值
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;
    
    // 内部方法
    void BackgroundMonitoring();
    void ProcessEvents();
    void UpdateMemoryStats();
    void GenerateRecommendations(PerformanceAnalysis& analysis);
    std::string GetOptimizationSuggestion(const HotspotAnalysis& hotspot);
    
    // 成员变量
    std::atomic<bool> monitoring_enabled_;
    std::atomic<bool> real_time_monitoring_;
    mutable std::mutex events_mutex_;
    mutable std::mutex memory_mutex_;
    mutable std::mutex analysis_mutex_;
    
    std::unordered_map<std::string, std::unique_ptr<PerformanceEvent>> active_events_;
    std::vector<std::unique_ptr<PerformanceEvent>> completed_events_;
    MemoryStats memory_stats_;
    AutoTuneConfig auto_tune_config_;
    
    std::thread monitoring_thread_;
    std::atomic<int> monitoring_interval_ms_;
    std::vector<std::function<void(const PerformanceAnalysis&)>> callbacks_;
    
    // 性能历史
    std::vector<PerformanceAnalysis> analysis_history_;
    std::unordered_map<std::string, std::vector<double>> performance_history_;
};

// 性能监控宏
#define CUOP_PERF_START(name) \
    PerformanceMonitor::Instance().StartEvent(name, PerformanceEventType::CUSTOM_EVENT)

#define CUOP_PERF_END(name) \
    PerformanceMonitor::Instance().EndEvent(name)

#define CUOP_PERF_SCOPE(name) \
    PerformanceScope _perf_scope(name)

#define CUOP_PERF_RECORD(name, duration) \
    PerformanceMonitor::Instance().RecordEvent(name, duration, PerformanceEventType::CUSTOM_EVENT)

// 性能作用域类
class PerformanceScope {
public:
    PerformanceScope(const std::string& name, PerformanceEventType type = PerformanceEventType::CUSTOM_EVENT)
        : name_(name), type_(type) {
        PerformanceMonitor::Instance().StartEvent(name_, type_);
    }
    
    ~PerformanceScope() {
        PerformanceMonitor::Instance().EndEvent(name_);
    }
    
private:
    std::string name_;
    PerformanceEventType type_;
};

// 内存监控宏
#define CUOP_MEM_ALLOC(size) \
    PerformanceMonitor::Instance().RecordMemoryAllocation(size)

#define CUOP_MEM_FREE(size) \
    PerformanceMonitor::Instance().RecordMemoryFree(size)

// 自动调优宏 - 使用AutoTuner类
#define CUOP_AUTO_TUNE_PERF(op_name, param_ranges, benchmark_func) \
    PerformanceMonitor::Instance().AutoTune(op_name, param_ranges, benchmark_func)

} // namespace cu_op_mem
