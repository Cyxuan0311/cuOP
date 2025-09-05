#include "performance/hotspot_analyzer.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <execinfo.h>
#include <json/json.h>

namespace cu_op_mem {

HotspotAnalyzer& HotspotAnalyzer::Instance() {
    static HotspotAnalyzer instance;
    return instance;
}

HotspotAnalyzer::HotspotAnalyzer() 
    : analysis_enabled_(false), real_time_analysis_(false), 
      total_analysis_time_(0.0), total_function_calls_(0), analysis_interval_ms_(1000) {
}

HotspotAnalyzer::~HotspotAnalyzer() {
    StopAnalysis();
}

void HotspotAnalyzer::StartAnalysis() {
    if (analysis_enabled_) {
        LOG(WARNING) << "Hotspot analysis is already enabled";
        return;
    }
    
    analysis_enabled_ = true;
    analysis_start_time_ = std::chrono::high_resolution_clock::now();
    
    // 启动后台分析线程
    if (real_time_analysis_) {
        analysis_thread_ = std::thread(&HotspotAnalyzer::BackgroundAnalysis, this);
    }
    
    LOG(INFO) << "Hotspot analysis started";
}

void HotspotAnalyzer::StopAnalysis() {
    if (!analysis_enabled_) {
        return;
    }
    
    analysis_enabled_ = false;
    
    // 停止后台分析线程
    if (analysis_thread_.joinable()) {
        analysis_thread_.join();
    }
    
    // 处理剩余函数调用
    ProcessFunctionCalls();
    
    LOG(INFO) << "Hotspot analysis stopped";
}

void HotspotAnalyzer::Reset() {
    std::lock_guard<std::mutex> calls_lock(calls_mutex_);
    std::lock_guard<std::mutex> results_lock(results_mutex_);
    
    active_calls_.clear();
    completed_calls_.clear();
    hotspot_results_.clear();
    
    total_analysis_time_ = 0.0;
    total_function_calls_ = 0;
    
    LOG(INFO) << "Hotspot analyzer reset";
}

void HotspotAnalyzer::StartFunction(const std::string& function_name, const std::string& file_name, int line_number) {
    if (!analysis_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    auto call = std::make_unique<FunctionCall>();
    call->function_name = function_name;
    call->file_name = file_name;
    call->line_number = line_number;
    call->start_time = std::chrono::high_resolution_clock::now();
    call->call_stack = GetCallStack();
    
    // 记录CUDA事件
    cudaEventRecord(call->cuda_start);
    
    // 记录内存状态
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    call->memory_allocated = total - free;
    
    active_calls_[function_name] = std::move(call);
}

void HotspotAnalyzer::EndFunction(const std::string& function_name) {
    if (!analysis_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    auto it = active_calls_.find(function_name);
    if (it == active_calls_.end()) {
        LOG(WARNING) << "Function '" << function_name << "' not found in active calls";
        return;
    }
    
    auto& call = it->second;
    call->end_time = std::chrono::high_resolution_clock::now();
    
    // 记录CUDA事件
    cudaEventRecord(call->cuda_end);
    cudaEventSynchronize(call->cuda_end);
    
    // 记录内存状态
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    call->memory_freed = call->memory_allocated - (total - free);
    
    // 移动到完成调用列表
    completed_calls_.push_back(std::move(call));
    active_calls_.erase(it);
    
    total_function_calls_++;
}

void HotspotAnalyzer::RecordFunctionCall(const std::string& function_name, double duration, 
                                        const std::string& file_name, int line_number) {
    if (!analysis_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    auto call = std::make_unique<FunctionCall>();
    call->function_name = function_name;
    call->file_name = file_name;
    call->line_number = line_number;
    call->start_time = std::chrono::high_resolution_clock::now();
    call->end_time = call->start_time + std::chrono::microseconds(static_cast<long long>(duration * 1000));
    call->call_stack = GetCallStack();
    
    completed_calls_.push_back(std::move(call));
    total_function_calls_++;
}

std::vector<HotspotResult> HotspotAnalyzer::AnalyzeHotspots() {
    std::lock_guard<std::mutex> calls_lock(calls_mutex_);
    std::lock_guard<std::mutex> results_lock(results_mutex_);
    
    std::unordered_map<std::string, std::vector<double>> function_times;
    std::unordered_map<std::string, std::vector<size_t>> function_memory;
    std::unordered_map<std::string, std::string> function_files;
    std::unordered_map<std::string, int> function_lines;
    
    // 收集函数数据
    for (const auto& call : completed_calls_) {
        function_times[call->function_name].push_back(call->GetDuration());
        function_memory[call->function_name].push_back(call->memory_allocated);
        function_files[call->function_name] = call->file_name;
        function_lines[call->function_name] = call->line_number;
    }
    
    // 计算总时间
    double total_time = 0.0;
    for (const auto& pair : function_times) {
        total_time += std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
    }
    
    hotspot_results_.clear();
    
    // 分析每个函数
    for (const auto& pair : function_times) {
        const std::string& function_name = pair.first;
        const std::vector<double>& times = pair.second;
        const std::vector<size_t>& memory = function_memory[function_name];
        
        HotspotResult result;
        result.function_name = function_name;
        result.file_name = function_files[function_name];
        result.line_number = function_lines[function_name];
        result.call_count = times.size();
        result.total_time = std::accumulate(times.begin(), times.end(), 0.0);
        result.percentage = total_time > 0 ? (result.total_time / total_time) * 100.0 : 0.0;
        result.avg_time = result.total_time / result.call_count;
        result.min_time = *std::min_element(times.begin(), times.end());
        result.max_time = *std::max_element(times.begin(), times.end());
        result.time_history = times;
        result.memory_history = memory;
        
        // 计算标准差
        double variance = 0.0;
        for (double time : times) {
            variance += (time - result.avg_time) * (time - result.avg_time);
        }
        result.std_deviation = std::sqrt(variance / times.size());
        
        // 识别瓶颈类型
        result.bottleneck_type = AnalyzeBottleneckType(*completed_calls_[0]); // 简化实现
        
        // 生成优化建议
        result.optimization_suggestion = GenerateOptimizationSuggestion(result);
        
        // 计算优先级分数
        result.priority_score = CalculatePriorityScore(result);
        
        hotspot_results_.push_back(result);
    }
    
    // 按优先级分数排序
    std::sort(hotspot_results_.begin(), hotspot_results_.end(),
        [](const HotspotResult& a, const HotspotResult& b) {
            return a.priority_score > b.priority_score;
        });
    
    // 限制热点数量
    if (hotspot_results_.size() > config_.max_hotspots) {
        hotspot_results_.resize(config_.max_hotspots);
    }
    
    return hotspot_results_;
}

std::vector<HotspotResult> HotspotAnalyzer::GetTopHotspots(size_t count) {
    auto hotspots = AnalyzeHotspots();
    if (hotspots.size() > count) {
        hotspots.resize(count);
    }
    return hotspots;
}

HotspotResult HotspotAnalyzer::AnalyzeFunction(const std::string& function_name) {
    auto hotspots = AnalyzeHotspots();
    
    for (const auto& hotspot : hotspots) {
        if (hotspot.function_name == function_name) {
            return hotspot;
        }
    }
    
    // 返回空结果
    return HotspotResult{};
}

BottleneckType HotspotAnalyzer::IdentifyBottleneck(const std::string& function_name) {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    // 查找函数调用
    for (const auto& call : completed_calls_) {
        if (call->function_name == function_name) {
            return AnalyzeBottleneckType(*call);
        }
    }
    
    return BottleneckType::UNKNOWN;
}

std::vector<std::string> HotspotAnalyzer::GetBottleneckFunctions(BottleneckType type) {
    auto hotspots = AnalyzeHotspots();
    std::vector<std::string> functions;
    
    for (const auto& hotspot : hotspots) {
        if (hotspot.bottleneck_type == type) {
            functions.push_back(hotspot.function_name);
        }
    }
    
    return functions;
}

double HotspotAnalyzer::CalculateFunctionEfficiency(const std::string& function_name) {
    auto hotspot = AnalyzeFunction(function_name);
    if (hotspot.function_name.empty()) {
        return 0.0;
    }
    
    // 简化的效率计算：基于时间稳定性和内存使用
    double time_efficiency = 1.0 - (hotspot.std_deviation / hotspot.avg_time);
    double memory_efficiency = 1.0 - (hotspot.memory_history.empty() ? 0.0 : 
        static_cast<double>(*std::max_element(hotspot.memory_history.begin(), hotspot.memory_history.end())) / 
        (1024 * 1024 * 1024)); // 1GB基准
    
    return (time_efficiency + memory_efficiency) / 2.0;
}

std::vector<std::string> HotspotAnalyzer::GetOptimizationCandidates() {
    auto hotspots = AnalyzeHotspots();
    std::vector<std::string> candidates;
    
    for (const auto& hotspot : hotspots) {
        if (hotspot.priority_score > 0.7) { // 高优先级
            candidates.push_back(hotspot.function_name);
        }
    }
    
    return candidates;
}

std::string HotspotAnalyzer::GenerateOptimizationReport() {
    auto hotspots = AnalyzeHotspots();
    
    std::ostringstream report;
    report << "=== cuOP Hotspot Analysis Report ===\n";
    report << "Analysis Time: " << std::put_time(std::localtime(&analysis_start_time_), "%Y-%m-%d %H:%M:%S") << "\n";
    report << "Total Function Calls: " << total_function_calls_ << "\n";
    report << "Total Analysis Time: " << std::fixed << std::setprecision(2) << total_analysis_time_ << " ms\n\n";
    
    // 热点分析
    report << "=== Top Hotspots ===\n";
    for (size_t i = 0; i < std::min(size_t(10), hotspots.size()); ++i) {
        const auto& hotspot = hotspots[i];
        report << i + 1 << ". " << hotspot.function_name << "\n";
        report << "   File: " << hotspot.file_name << ":" << hotspot.line_number << "\n";
        report << "   Total Time: " << std::fixed << std::setprecision(2) << hotspot.total_time << " ms\n";
        report << "   Percentage: " << std::fixed << std::setprecision(1) << hotspot.percentage << "%\n";
        report << "   Call Count: " << hotspot.call_count << "\n";
        report << "   Avg Time: " << std::fixed << std::setprecision(2) << hotspot.avg_time << " ms\n";
        report << "   Priority Score: " << std::fixed << std::setprecision(2) << hotspot.priority_score << "\n";
        report << "   Bottleneck Type: " << static_cast<int>(hotspot.bottleneck_type) << "\n";
        report << "   Suggestion: " << hotspot.optimization_suggestion << "\n\n";
    }
    
    // 瓶颈分析
    report << "=== Bottleneck Analysis ===\n";
    std::unordered_map<BottleneckType, size_t> bottleneck_counts;
    for (const auto& hotspot : hotspots) {
        bottleneck_counts[hotspot.bottleneck_type]++;
    }
    
    for (const auto& pair : bottleneck_counts) {
        report << "Bottleneck Type " << static_cast<int>(pair.first) << ": " << pair.second << " functions\n";
    }
    
    // 优化建议
    report << "\n=== Optimization Recommendations ===\n";
    auto candidates = GetOptimizationCandidates();
    for (size_t i = 0; i < candidates.size(); ++i) {
        report << i + 1 << ". Optimize function: " << candidates[i] << "\n";
    }
    
    return report.str();
}

void HotspotAnalyzer::SetConfig(const HotspotAnalysisConfig& config) {
    config_ = config;
}

HotspotAnalysisConfig HotspotAnalyzer::GetConfig() const {
    return config_;
}

void HotspotAnalyzer::ExportToCSV(const std::string& filename) {
    auto hotspots = AnalyzeHotspots();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for CSV export: " << filename;
        return;
    }
    
    // 写入头部
    file << "Function Name,File,Line,Total Time,Percentage,Call Count,Avg Time,Min Time,Max Time,Std Deviation,Priority Score,Bottleneck Type,Optimization Suggestion\n";
    
    // 写入数据
    for (const auto& hotspot : hotspots) {
        file << hotspot.function_name << ","
             << hotspot.file_name << ","
             << hotspot.line_number << ","
             << hotspot.total_time << ","
             << hotspot.percentage << ","
             << hotspot.call_count << ","
             << hotspot.avg_time << ","
             << hotspot.min_time << ","
             << hotspot.max_time << ","
             << hotspot.std_deviation << ","
             << hotspot.priority_score << ","
             << static_cast<int>(hotspot.bottleneck_type) << ","
             << "\"" << hotspot.optimization_suggestion << "\"\n";
    }
    
    file.close();
    LOG(INFO) << "Hotspot analysis data exported to CSV: " << filename;
}

void HotspotAnalyzer::ExportToJSON(const std::string& filename) {
    auto hotspots = AnalyzeHotspots();
    
    Json::Value root;
    root["analysis_time"] = std::chrono::duration_cast<std::chrono::seconds>(
        analysis_start_time_.time_since_epoch()).count();
    root["total_function_calls"] = total_function_calls_;
    root["total_analysis_time"] = total_analysis_time_;
    
    // 热点数据
    Json::Value hotspots_json(Json::arrayValue);
    for (const auto& hotspot : hotspots) {
        Json::Value hotspot_json;
        hotspot_json["function_name"] = hotspot.function_name;
        hotspot_json["file_name"] = hotspot.file_name;
        hotspot_json["line_number"] = hotspot.line_number;
        hotspot_json["total_time"] = hotspot.total_time;
        hotspot_json["percentage"] = hotspot.percentage;
        hotspot_json["call_count"] = hotspot.call_count;
        hotspot_json["avg_time"] = hotspot.avg_time;
        hotspot_json["min_time"] = hotspot.min_time;
        hotspot_json["max_time"] = hotspot.max_time;
        hotspot_json["std_deviation"] = hotspot.std_deviation;
        hotspot_json["priority_score"] = hotspot.priority_score;
        hotspot_json["bottleneck_type"] = static_cast<int>(hotspot.bottleneck_type);
        hotspot_json["optimization_suggestion"] = hotspot.optimization_suggestion;
        hotspots_json.append(hotspot_json);
    }
    root["hotspots"] = hotspots_json;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for JSON export: " << filename;
        return;
    }
    
    file << root;
    file.close();
    LOG(INFO) << "Hotspot analysis data exported to JSON: " << filename;
}

void HotspotAnalyzer::ExportToChromeTrace(const std::string& filename) {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    Json::Value root(Json::arrayValue);
    
    for (const auto& call : completed_calls_) {
        Json::Value event;
        event["name"] = call->function_name;
        event["cat"] = "function";
        event["ph"] = "X"; // Complete event
        event["ts"] = std::chrono::duration_cast<std::chrono::microseconds>(
            call->start_time.time_since_epoch()).count();
        event["dur"] = std::chrono::duration_cast<std::chrono::microseconds>(
            call->end_time - call->start_time).count();
        event["pid"] = 1;
        event["tid"] = 1;
        
        Json::Value args;
        args["file"] = call->file_name;
        args["line"] = call->line_number;
        args["memory_allocated"] = call->memory_allocated;
        args["memory_freed"] = call->memory_freed;
        event["args"] = args;
        
        root.append(event);
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for Chrome trace export: " << filename;
        return;
    }
    
    file << root;
    file.close();
    LOG(INFO) << "Hotspot analysis data exported to Chrome trace: " << filename;
}

void HotspotAnalyzer::EnableRealTimeAnalysis(bool enable) {
    real_time_analysis_ = enable;
    if (enable && analysis_enabled_) {
        analysis_thread_ = std::thread(&HotspotAnalyzer::BackgroundAnalysis, this);
    }
}

void HotspotAnalyzer::SetAnalysisInterval(int interval_ms) {
    analysis_interval_ms_ = interval_ms;
}

void HotspotAnalyzer::RegisterCallback(std::function<void(const std::vector<HotspotResult>&)> callback) {
    callbacks_.push_back(callback);
}

void HotspotAnalyzer::AnalyzeCallGraph() {
    // 调用图分析实现
    LOG(INFO) << "Call graph analysis not yet implemented";
}

void HotspotAnalyzer::AnalyzeMemoryPatterns() {
    // 内存模式分析实现
    LOG(INFO) << "Memory pattern analysis not yet implemented";
}

void HotspotAnalyzer::AnalyzePerformanceTrends() {
    // 性能趋势分析实现
    LOG(INFO) << "Performance trend analysis not yet implemented";
}

void HotspotAnalyzer::BackgroundAnalysis() {
    while (analysis_enabled_ && real_time_analysis_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(analysis_interval_ms_));
        
        if (analysis_enabled_) {
            ProcessFunctionCalls();
            UpdateStatistics();
            
            // 调用回调函数
            auto hotspots = AnalyzeHotspots();
            for (const auto& callback : callbacks_) {
                try {
                    callback(hotspots);
                } catch (const std::exception& e) {
                    LOG(WARNING) << "Hotspot analysis callback execution failed: " << e.what();
                }
            }
        }
    }
}

void HotspotAnalyzer::ProcessFunctionCalls() {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    // 处理超时调用
    auto now = std::chrono::high_resolution_clock::now();
    auto it = active_calls_.begin();
    while (it != active_calls_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - it->second->start_time);
        if (duration.count() > 300) { // 5分钟超时
            LOG(WARNING) << "Function call '" << it->first << "' timed out";
            it->second->end_time = now;
            completed_calls_.push_back(std::move(it->second));
            it = active_calls_.erase(it);
        } else {
            ++it;
        }
    }
}

void HotspotAnalyzer::UpdateStatistics() {
    std::lock_guard<std::mutex> lock(calls_mutex_);
    
    // 更新总分析时间
    auto now = std::chrono::high_resolution_clock::now();
    total_analysis_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - analysis_start_time_).count();
}

std::string HotspotAnalyzer::GetCallStack() {
    const int max_frames = 10;
    void* frames[max_frames];
    int frame_count = backtrace(frames, max_frames);
    
    char** symbols = backtrace_symbols(frames, frame_count);
    if (!symbols) {
        return "Failed to get call stack";
    }
    
    std::ostringstream oss;
    for (int i = 0; i < frame_count; ++i) {
        oss << symbols[i] << "\n";
    }
    
    free(symbols);
    return oss.str();
}

BottleneckType HotspotAnalyzer::AnalyzeBottleneckType(const FunctionCall& call) {
    // 简化的瓶颈类型分析
    double duration = call.GetDuration();
    double cuda_duration = call.GetCudaDuration();
    size_t memory_used = call.memory_allocated;
    
    if (cuda_duration > duration * 0.8) {
        return BottleneckType::GPU_BOUND;
    } else if (memory_used > 1024 * 1024 * 1024) { // 1GB
        return BottleneckType::MEMORY_BOUND;
    } else if (duration > 100.0) { // 100ms
        return BottleneckType::CPU_BOUND;
    } else {
        return BottleneckType::UNKNOWN;
    }
}

std::string HotspotAnalyzer::GenerateOptimizationSuggestion(const HotspotResult& result) {
    if (result.percentage > 50.0) {
        return "Critical bottleneck - requires immediate optimization";
    } else if (result.percentage > 20.0) {
        return "Significant bottleneck - consider optimization";
    } else if (result.avg_time > 100.0) {
        return "Slow execution - consider parallelization or algorithm optimization";
    } else if (result.call_count > 1000) {
        return "Frequent calls - consider caching or batching";
    } else if (result.std_deviation > result.avg_time * 0.5) {
        return "High variance - consider stabilizing performance";
    } else {
        return "Performance acceptable";
    }
}

double HotspotAnalyzer::CalculatePriorityScore(const HotspotResult& result) {
    // 基于多个因素计算优先级分数
    double time_score = result.percentage / 100.0; // 时间百分比
    double frequency_score = std::min(1.0, result.call_count / 1000.0); // 调用频率
    double variance_score = 1.0 - (result.std_deviation / result.avg_time); // 时间稳定性
    double memory_score = result.memory_history.empty() ? 0.0 : 
        std::min(1.0, static_cast<double>(*std::max_element(result.memory_history.begin(), result.memory_history.end())) / (1024 * 1024 * 1024)); // 内存使用
    
    // 加权平均
    return (time_score * 0.4 + frequency_score * 0.3 + variance_score * 0.2 + memory_score * 0.1);
}

} // namespace cu_op_mem
