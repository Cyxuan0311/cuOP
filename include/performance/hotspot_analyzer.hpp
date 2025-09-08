#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <thread>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cu_op_mem {

// 函数调用信息
struct FunctionCall {
    std::string function_name;           // 函数名
    std::string file_name;              // 文件名
    int line_number;                    // 行号
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    cudaEvent_t cuda_start;             // CUDA事件开始
    cudaEvent_t cuda_end;               // CUDA事件结束
    size_t memory_allocated;            // 分配的内存
    size_t memory_freed;                // 释放的内存
    std::string call_stack;             // 调用栈
    std::unordered_map<std::string, std::string> metadata; // 元数据
    
    FunctionCall() : line_number(0), memory_allocated(0), memory_freed(0) {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_end);
    }
    
    ~FunctionCall() {
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

// 热点分析结果
struct HotspotResult {
    std::string function_name;           // 函数名
    std::string file_name;              // 文件名
    int line_number;                    // 行号
    double total_time;                  // 总时间
    double percentage;                  // 时间百分比
    size_t call_count;                  // 调用次数
    double avg_time;                    // 平均时间
    double min_time;                    // 最小时间
    double max_time;                    // 最大时间
    double std_deviation;               // 标准差
    std::vector<double> time_history;   // 时间历史
    std::vector<size_t> memory_history; // 内存使用历史
    std::string bottleneck_type;        // 瓶颈类型
    std::string optimization_suggestion; // 优化建议
    double priority_score;              // 优先级分数
};

// 瓶颈类型
enum class BottleneckType {
    CPU_BOUND,          // CPU瓶颈
    GPU_BOUND,          // GPU瓶颈
    MEMORY_BOUND,       // 内存瓶颈
    I_O_BOUND,          // I/O瓶颈
    SYNCHRONIZATION,    // 同步瓶颈
    CACHE_MISS,         // 缓存未命中
    BRANCH_DIVERGENCE,  // 分支发散
    MEMORY_COALESCING,  // 内存合并
    UNKNOWN             // 未知
};

// 热点分析配置
struct HotspotAnalysisConfig {
    bool enable_cpu_profiling = true;        // 启用CPU性能分析
    bool enable_gpu_profiling = true;        // 启用GPU性能分析
    bool enable_memory_profiling = true;     // 启用内存性能分析
    bool enable_call_stack = true;           // 启用调用栈分析
    double min_time_threshold = 1.0;         // 最小时间阈值(ms)
    double min_percentage_threshold = 1.0;   // 最小百分比阈值
    size_t max_hotspots = 50;                // 最大热点数量
    bool enable_real_time_analysis = false;  // 启用实时分析
    int analysis_interval_ms = 1000;         // 分析间隔(ms)
    std::string output_format = "text";      // 输出格式
    std::string output_file = "";            // 输出文件
};

// 热点分析器类
class HotspotAnalyzer {
public:
    static HotspotAnalyzer& Instance();
    
    // 基本控制
    void StartAnalysis();
    void StopAnalysis();
    void Reset();
    bool IsAnalyzing() const { return analysis_enabled_; }
    
    // 函数跟踪
    void StartFunction(const std::string& function_name, const std::string& file_name = "", 
                      int line_number = 0);
    void EndFunction(const std::string& function_name);
    void RecordFunctionCall(const std::string& function_name, double duration, 
                           const std::string& file_name = "", int line_number = 0);
    
    // 热点分析
    std::vector<HotspotResult> AnalyzeHotspots();
    std::vector<HotspotResult> GetTopHotspots(size_t count = 10);
    HotspotResult AnalyzeFunction(const std::string& function_name);
    
    // 瓶颈识别
    BottleneckType IdentifyBottleneck(const std::string& function_name);
    std::vector<std::string> GetBottleneckFunctions(BottleneckType type);
    
    // 性能分析
    double CalculateFunctionEfficiency(const std::string& function_name);
    std::vector<std::string> GetOptimizationCandidates();
    std::string GenerateOptimizationReport();
    
    // 配置管理
    void SetConfig(const HotspotAnalysisConfig& config);
    HotspotAnalysisConfig GetConfig() const;
    
    // 数据导出
    void ExportToCSV(const std::string& filename);
    void ExportToJSON(const std::string& filename);
    void ExportToChromeTrace(const std::string& filename);
    
    // 实时监控
    void EnableRealTimeAnalysis(bool enable);
    void SetAnalysisInterval(int interval_ms);
    void RegisterCallback(std::function<void(const std::vector<HotspotResult>&)> callback);
    
    // 高级分析
    void AnalyzeCallGraph();
    void AnalyzeMemoryPatterns();
    void AnalyzePerformanceTrends();
    
private:
    HotspotAnalyzer();
    ~HotspotAnalyzer();
    
    // 禁用拷贝构造和赋值
    HotspotAnalyzer(const HotspotAnalyzer&) = delete;
    HotspotAnalyzer& operator=(const HotspotAnalyzer&) = delete;
    
    // 内部方法
    void BackgroundAnalysis();
    void ProcessFunctionCalls();
    void UpdateStatistics();
    std::string GetCallStack();
    BottleneckType AnalyzeBottleneckType(const FunctionCall& call);
    std::string GenerateOptimizationSuggestion(const HotspotResult& result);
    double CalculatePriorityScore(const HotspotResult& result);
    
    // 成员变量
    std::atomic<bool> analysis_enabled_;
    std::atomic<bool> real_time_analysis_;
    std::mutex calls_mutex_;
    std::mutex results_mutex_;
    
    std::unordered_map<std::string, std::unique_ptr<FunctionCall>> active_calls_;
    std::vector<std::unique_ptr<FunctionCall>> completed_calls_;
    std::vector<HotspotResult> hotspot_results_;
    
    HotspotAnalysisConfig config_;
    
    // 实时监控
    std::thread analysis_thread_;
    std::atomic<int> analysis_interval_ms_;
    std::vector<std::function<void(const std::vector<HotspotResult>&)>> callbacks_;
    
    // 统计信息
    std::atomic<double> total_analysis_time_;
    std::atomic<size_t> total_function_calls_;
    std::chrono::high_resolution_clock::time_point analysis_start_time_;
};

// 热点分析宏
#define CUOP_HOTSPOT_START(func_name) \
    HotspotAnalyzer::Instance().StartFunction(func_name, __FILE__, __LINE__)

#define CUOP_HOTSPOT_END(func_name) \
    HotspotAnalyzer::Instance().EndFunction(func_name)

#define CUOP_HOTSPOT_SCOPE(func_name) \
    HotspotScope _hotspot_scope(func_name, __FILE__, __LINE__)

#define CUOP_HOTSPOT_RECORD(func_name, duration) \
    HotspotAnalyzer::Instance().RecordFunctionCall(func_name, duration, __FILE__, __LINE__)

// 热点作用域类
class HotspotScope {
public:
    HotspotScope(const std::string& function_name, const std::string& file_name = "", int line_number = 0)
        : function_name_(function_name), file_name_(file_name), line_number_(line_number) {
        HotspotAnalyzer::Instance().StartFunction(function_name_, file_name_, line_number_);
    }
    
    ~HotspotScope() {
        HotspotAnalyzer::Instance().EndFunction(function_name_);
    }
    
private:
    std::string function_name_;
    std::string file_name_;
    int line_number_;
};

// 瓶颈分析宏
#define CUOP_BOTTLENECK_ANALYZE(func_name) \
    HotspotAnalyzer::Instance().IdentifyBottleneck(func_name)

#define CUOP_OPTIMIZATION_CANDIDATES() \
    HotspotAnalyzer::Instance().GetOptimizationCandidates()

// 性能分析宏
#define CUOP_PERFORMANCE_ANALYZE() \
    HotspotAnalyzer::Instance().AnalyzeHotspots()

#define CUOP_TOP_HOTSPOTS(count) \
    HotspotAnalyzer::Instance().GetTopHotspots(count)

} // namespace cu_op_mem
