#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <cuda_runtime.h>

namespace cu_op_mem {

// 内存分配信息
struct MemoryAllocation {
    void* ptr;                          // 内存指针
    size_t size;                        // 分配大小
    std::string file;                   // 分配文件
    int line;                          // 分配行号
    std::string function;               // 分配函数
    std::chrono::high_resolution_clock::time_point timestamp; // 分配时间
    bool is_freed;                     // 是否已释放
    std::string stack_trace;           // 调用栈
    size_t allocation_id;              // 分配ID
};

// 内存泄漏检测结果
struct MemoryLeakInfo {
    std::string file;                   // 泄漏文件
    int line;                          // 泄漏行号
    std::string function;               // 泄漏函数
    size_t size;                       // 泄漏大小
    std::chrono::high_resolution_clock::time_point allocation_time; // 分配时间
    std::string stack_trace;           // 调用栈
    size_t allocation_id;              // 分配ID
};

// 内存使用模式分析
struct MemoryPattern {
    std::string pattern_name;           // 模式名称
    size_t total_allocations;           // 总分配次数
    size_t total_size;                  // 总分配大小
    double avg_allocation_size;         // 平均分配大小
    double allocation_frequency;        // 分配频率
    std::vector<size_t> size_distribution; // 大小分布
    std::chrono::high_resolution_clock::time_point first_allocation;
    std::chrono::high_resolution_clock::time_point last_allocation;
};

// 内存分析报告
struct MemoryAnalysisReport {
    std::vector<MemoryLeakInfo> leaks;           // 内存泄漏
    std::vector<MemoryPattern> patterns;         // 内存使用模式
    size_t total_allocated;                      // 总分配内存
    size_t total_freed;                          // 总释放内存
    size_t current_usage;                        // 当前使用内存
    size_t peak_usage;                           // 峰值使用内存
    size_t allocation_count;                     // 分配次数
    size_t free_count;                           // 释放次数
    double fragmentation_ratio;                  // 碎片化比率
    std::vector<std::string> recommendations;    // 优化建议
    std::chrono::high_resolution_clock::time_point analysis_time;
};

// 内存分析器类
class MemoryAnalyzer {
public:
    static MemoryAnalyzer& Instance();
    
    // 基本控制
    void StartTracking();
    void StopTracking();
    void Reset();
    bool IsTracking() const { return tracking_enabled_; }
    
    // 内存跟踪
    void TrackAllocation(void* ptr, size_t size, const std::string& file = "", 
                        int line = 0, const std::string& function = "");
    void TrackFree(void* ptr);
    void TrackReallocation(void* old_ptr, void* new_ptr, size_t new_size);
    
    // 内存分析
    MemoryAnalysisReport AnalyzeMemory();
    std::vector<MemoryLeakInfo> DetectLeaks();
    std::vector<MemoryPattern> AnalyzePatterns();
    
    // 内存统计
    size_t GetTotalAllocated() const;
    size_t GetTotalFreed() const;
    size_t GetCurrentUsage() const;
    size_t GetPeakUsage() const;
    size_t GetAllocationCount() const;
    size_t GetFreeCount() const;
    MemoryAnalysisReport GetMemoryStats() const;
    
    // 内存优化建议
    std::vector<std::string> GetOptimizationSuggestions();
    double CalculateFragmentationRatio();
    
    // 报告生成
    std::string GenerateReport();
    void ExportToCSV(const std::string& filename);
    void ExportToJSON(const std::string& filename);
    
    // 实时监控
    void EnableRealTimeMonitoring(bool enable);
    void SetMonitoringInterval(int interval_ms);
    void RegisterCallback(std::function<void(const MemoryAnalysisReport&)> callback);
    
    // 内存池集成
    void SetMemoryPoolIntegration(bool enable);
    void TrackMemoryPoolAllocation(void* ptr, size_t size, const std::string& pool_name);
    void TrackMemoryPoolFree(void* ptr, const std::string& pool_name);
    
private:
    MemoryAnalyzer();
    ~MemoryAnalyzer();
    
    // 禁用拷贝构造和赋值
    MemoryAnalyzer(const MemoryAnalyzer&) = delete;
    MemoryAnalyzer& operator=(const MemoryAnalyzer&) = delete;
    
    // 内部方法
    void BackgroundMonitoring();
    void UpdateStatistics();
    std::string GetStackTrace();
    void ProcessAllocations();
    void DetectPatterns();
    void GenerateRecommendations(MemoryAnalysisReport& report);
    
    // 成员变量
    std::atomic<bool> tracking_enabled_;
    std::atomic<bool> real_time_monitoring_;
    std::atomic<bool> memory_pool_integration_;
    std::mutex allocations_mutex_;
    mutable std::mutex statistics_mutex_;
    
    std::unordered_map<void*, std::unique_ptr<MemoryAllocation>> active_allocations_;
    std::vector<std::unique_ptr<MemoryAllocation>> allocation_history_;
    std::atomic<size_t> next_allocation_id_;
    
    // 统计信息
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> total_freed_;
    std::atomic<size_t> current_usage_;
    std::atomic<size_t> peak_usage_;
    std::atomic<size_t> allocation_count_;
    std::atomic<size_t> free_count_;
    
    // 实时监控
    std::thread monitoring_thread_;
    std::atomic<int> monitoring_interval_ms_;
    std::vector<std::function<void(const MemoryAnalysisReport&)>> callbacks_;
    
    // 内存池集成
    std::unordered_map<std::string, size_t> pool_allocations_;
    std::unordered_map<std::string, size_t> pool_frees_;
    std::mutex pool_mutex_;
};

// 内存跟踪宏
#define CUOP_MEM_TRACK_ALLOC(ptr, size) \
    MemoryAnalyzer::Instance().TrackAllocation(ptr, size, __FILE__, __LINE__, __FUNCTION__)

#define CUOP_MEM_TRACK_FREE(ptr) \
    MemoryAnalyzer::Instance().TrackFree(ptr)

#define CUOP_MEM_TRACK_REALLOC(old_ptr, new_ptr, new_size) \
    MemoryAnalyzer::Instance().TrackReallocation(old_ptr, new_ptr, new_size)

// 内存分析宏
#define CUOP_MEM_ANALYZE() \
    MemoryAnalyzer::Instance().AnalyzeMemory()

#define CUOP_MEM_DETECT_LEAKS() \
    MemoryAnalyzer::Instance().DetectLeaks()

#define CUOP_MEM_GENERATE_REPORT() \
    MemoryAnalyzer::Instance().GenerateReport()

// 内存统计宏
#define CUOP_MEM_GET_STATS() \
    MemoryAnalyzer::Instance().GetTotalAllocated(), \
    MemoryAnalyzer::Instance().GetTotalFreed(), \
    MemoryAnalyzer::Instance().GetCurrentUsage(), \
    MemoryAnalyzer::Instance().GetPeakUsage()

} // namespace cu_op_mem
