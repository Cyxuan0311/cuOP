#include "performance/performance_monitor.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <json/json.h>

namespace cu_op_mem {

PerformanceMonitor& PerformanceMonitor::Instance() {
    static PerformanceMonitor instance;
    return instance;
}

PerformanceMonitor::PerformanceMonitor() 
    : monitoring_enabled_(false), real_time_monitoring_(false), monitoring_interval_ms_(1000) {
    // 初始化CUDA事件
    cudaSetDevice(0);
}

PerformanceMonitor::~PerformanceMonitor() {
    StopMonitoring();
}

void PerformanceMonitor::StartMonitoring() {
    if (monitoring_enabled_) {
        LOG(WARNING) << "Performance monitoring is already enabled";
        return;
    }
    
    monitoring_enabled_ = true;
    
    // 启动后台监控线程
    if (real_time_monitoring_) {
        monitoring_thread_ = std::thread(&PerformanceMonitor::BackgroundMonitoring, this);
    }
    
    LOG(INFO) << "Performance monitoring started";
}

void PerformanceMonitor::StopMonitoring() {
    if (!monitoring_enabled_) {
        return;
    }
    
    monitoring_enabled_ = false;
    
    // 停止后台监控线程
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    // 处理剩余事件
    ProcessEvents();
    
    LOG(INFO) << "Performance monitoring stopped";
}

void PerformanceMonitor::Reset() {
    std::lock_guard<std::mutex> events_lock(events_mutex_);
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    
    active_events_.clear();
    completed_events_.clear();
    memory_stats_ = MemoryStats{};
    analysis_history_.clear();
    performance_history_.clear();
    
    LOG(INFO) << "Performance monitor reset";
}

void PerformanceMonitor::StartEvent(const std::string& name, PerformanceEventType type) {
    if (!monitoring_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    auto event = std::make_unique<PerformanceEvent>();
    event->name = name;
    event->type = type;
    event->start_time = std::chrono::high_resolution_clock::now();
    
    // 记录CUDA事件
    cudaEventRecord(event->cuda_start);
    
    // 记录内存状态
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    event->memory_allocated = total - free;
    
    active_events_[name] = std::move(event);
}

void PerformanceMonitor::EndEvent(const std::string& name) {
    if (!monitoring_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    auto it = active_events_.find(name);
    if (it == active_events_.end()) {
        LOG(WARNING) << "Event '" << name << "' not found in active events";
        return;
    }
    
    auto& event = it->second;
    event->end_time = std::chrono::high_resolution_clock::now();
    
    // 记录CUDA事件
    cudaEventRecord(event->cuda_end);
    cudaEventSynchronize(event->cuda_end);
    
    // 记录内存状态
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    event->memory_used = (total - free) - event->memory_allocated;
    
    // 移动到完成事件列表
    completed_events_.push_back(std::move(event));
    active_events_.erase(it);
}

void PerformanceMonitor::RecordEvent(const std::string& name, double duration, PerformanceEventType type) {
    if (!monitoring_enabled_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    auto event = std::make_unique<PerformanceEvent>();
    event->name = name;
    event->type = type;
    event->start_time = std::chrono::high_resolution_clock::now();
    event->end_time = event->start_time + std::chrono::microseconds(static_cast<long long>(duration * 1000));
    
    completed_events_.push_back(std::move(event));
}

void PerformanceMonitor::RecordMemoryAllocation(size_t size) {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    memory_stats_.Update(size, 0);
}

void PerformanceMonitor::RecordMemoryFree(size_t size) {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    memory_stats_.Update(0, size);
}

MemoryStats PerformanceMonitor::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return memory_stats_;
}

PerformanceAnalysis PerformanceMonitor::AnalyzePerformance() {
    std::lock_guard<std::mutex> events_lock(events_mutex_);
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    std::lock_guard<std::mutex> analysis_lock(analysis_mutex_);
    
    PerformanceAnalysis analysis;
    analysis.memory_stats = memory_stats_;
    analysis.analysis_time = std::chrono::high_resolution_clock::now();
    
    // 复制事件数据
    for (const auto& event : completed_events_) {
        analysis.events.push_back(*event);
    }
    
    // 计算总执行时间
    if (!analysis.events.empty()) {
        auto min_time = std::min_element(analysis.events.begin(), analysis.events.end(),
            [](const PerformanceEvent& a, const PerformanceEvent& b) {
                return a.start_time < b.start_time;
            });
        auto max_time = std::max_element(analysis.events.begin(), analysis.events.end(),
            [](const PerformanceEvent& a, const PerformanceEvent& b) {
                return a.end_time < b.end_time;
            });
        
        analysis.total_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
            max_time->end_time - min_time->start_time).count() / 1000.0;
    }
    
    // 分析热点
    analysis.hotspots = AnalyzeHotspots();
    
    // 生成建议
    GenerateRecommendations(analysis);
    
    // 保存到历史
    analysis_history_.push_back(analysis);
    
    return analysis;
}

std::vector<HotspotAnalysis> PerformanceMonitor::AnalyzeHotspots() {
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    std::unordered_map<std::string, std::vector<double>> function_times;
    
    // 收集函数时间数据
    for (const auto& event : completed_events_) {
        function_times[event->name].push_back(event->GetDuration());
    }
    
    std::vector<HotspotAnalysis> hotspots;
    
    // 计算总时间
    double total_time = 0.0;
    for (const auto& pair : function_times) {
        total_time += std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
    }
    
    // 分析每个函数
    for (const auto& pair : function_times) {
        HotspotAnalysis hotspot;
        hotspot.function_name = pair.first;
        hotspot.call_count = pair.second.size();
        hotspot.total_time = std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
        hotspot.percentage = total_time > 0 ? (hotspot.total_time / total_time) * 100.0 : 0.0;
        hotspot.avg_time = hotspot.total_time / hotspot.call_count;
        hotspot.min_time = *std::min_element(pair.second.begin(), pair.second.end());
        hotspot.max_time = *std::max_element(pair.second.begin(), pair.second.end());
        hotspot.time_history = pair.second;
        hotspot.optimization_suggestion = GetOptimizationSuggestion(hotspot);
        
        hotspots.push_back(hotspot);
    }
    
    // 按总时间排序
    std::sort(hotspots.begin(), hotspots.end(),
        [](const HotspotAnalysis& a, const HotspotAnalysis& b) {
            return a.total_time > b.total_time;
        });
    
    return hotspots;
}

std::string PerformanceMonitor::GenerateReport() {
    auto analysis = AnalyzePerformance();
    
    std::ostringstream report;
    report << "=== cuOP Performance Analysis Report ===\n";
    auto time_t_val = std::chrono::system_clock::to_time_t(analysis.analysis_time);
    report << "Analysis Time: " << std::put_time(std::localtime(&time_t_val), "%Y-%m-%d %H:%M:%S") << "\n";
    report << "Total Execution Time: " << std::fixed << std::setprecision(2) << analysis.total_execution_time << " ms\n";
    report << "Total Events: " << analysis.events.size() << "\n\n";
    
    // 内存统计
    report << "=== Memory Statistics ===\n";
    report << "Total Allocated: " << analysis.memory_stats.total_allocated << " bytes\n";
    report << "Total Freed: " << analysis.memory_stats.total_freed << " bytes\n";
    report << "Current Usage: " << analysis.memory_stats.current_usage << " bytes\n";
    report << "Peak Usage: " << analysis.memory_stats.peak_usage << " bytes\n";
    report << "Allocation Count: " << analysis.memory_stats.allocation_count << "\n\n";
    
    // 热点分析
    report << "=== Hotspot Analysis ===\n";
    for (size_t i = 0; i < std::min(size_t(10), analysis.hotspots.size()); ++i) {
        const auto& hotspot = analysis.hotspots[i];
        report << i + 1 << ". " << hotspot.function_name << "\n";
        report << "   Total Time: " << std::fixed << std::setprecision(2) << hotspot.total_time << " ms\n";
        report << "   Percentage: " << std::fixed << std::setprecision(1) << hotspot.percentage << "%\n";
        report << "   Call Count: " << hotspot.call_count << "\n";
        report << "   Avg Time: " << std::fixed << std::setprecision(2) << hotspot.avg_time << " ms\n";
        report << "   Suggestion: " << hotspot.optimization_suggestion << "\n\n";
    }
    
    // 优化建议
    report << "=== Optimization Recommendations ===\n";
    for (size_t i = 0; i < analysis.recommendations.size(); ++i) {
        report << i + 1 << ". " << analysis.recommendations[i] << "\n";
    }
    
    return report.str();
}

AutoTuneResult PerformanceMonitor::AutoTune(const std::string& operator_name,
                                           const std::unordered_map<std::string, std::vector<std::string>>& param_ranges,
                                           std::function<double(const std::unordered_map<std::string, std::string>&)> benchmark_func) {
    AutoTuneResult result;
    result.best_performance = 0.0;
    
    if (!auto_tune_config_.enabled) {
        LOG(WARNING) << "Auto-tuning is disabled";
        return result;
    }
    
    LOG(INFO) << "Starting auto-tuning for operator: " << operator_name;
    
    // 生成参数组合
    std::vector<std::unordered_map<std::string, std::string>> param_combinations;
    
    // 简单的网格搜索实现
    std::function<void(std::unordered_map<std::string, std::string>, 
                      const std::vector<std::string>&, 
                      const std::vector<std::string>&)> generate_combinations;
    
    generate_combinations = [&](std::unordered_map<std::string, std::string> current_params,
                               const std::vector<std::string>& param_names,
                               const std::vector<std::string>& param_values) {
        if (param_names.empty()) {
            param_combinations.push_back(current_params);
            return;
        }
        
        std::string param_name = param_names[0];
        std::vector<std::string> remaining_names(param_names.begin() + 1, param_names.end());
        
        for (const auto& value : param_ranges.at(param_name)) {
            current_params[param_name] = value;
            generate_combinations(current_params, remaining_names, param_values);
        }
    };
    
    std::vector<std::string> param_names;
    for (const auto& pair : param_ranges) {
        param_names.push_back(pair.first);
    }
    
    generate_combinations({}, param_names, {});
    
    // 限制组合数量
    if (param_combinations.size() > auto_tune_config_.max_iterations) {
        param_combinations.resize(auto_tune_config_.max_iterations);
    }
    
    // 测试每个参数组合
    for (size_t i = 0; i < param_combinations.size(); ++i) {
        const auto& params = param_combinations[i];
        
        try {
            double performance = benchmark_func(params);
            result.performance_history.push_back(performance);
            
            if (performance > result.best_performance) {
                result.best_performance = performance;
                result.best_params = params;
            }
            
            LOG(INFO) << "Iteration " << i + 1 << "/" << param_combinations.size() 
                     << " - Performance: " << performance 
                     << " - Best: " << result.best_performance;
            
        } catch (const std::exception& e) {
            LOG(WARNING) << "Benchmark failed for iteration " << i + 1 << ": " << e.what();
            result.performance_history.push_back(0.0);
        }
        
        result.iterations_used = i + 1;
    }
    
    // 检查收敛
    if (result.performance_history.size() >= 10) {
        auto recent_performance = std::vector<double>(
            result.performance_history.end() - 10, result.performance_history.end());
        double avg_recent = std::accumulate(recent_performance.begin(), recent_performance.end(), 0.0) / 10.0;
        double improvement = (result.best_performance - avg_recent) / result.best_performance;
        result.converged = improvement < auto_tune_config_.improvement_threshold;
    }
    
    // 生成优化摘要
    std::ostringstream summary;
    summary << "Auto-tuning completed for " << operator_name << "\n";
    summary << "Best performance: " << result.best_performance << "\n";
    summary << "Iterations used: " << result.iterations_used << "\n";
    summary << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
    summary << "Best parameters:\n";
    for (const auto& pair : result.best_params) {
        summary << "  " << pair.first << ": " << pair.second << "\n";
    }
    
    result.optimization_summary = summary.str();
    
    // 保存最佳配置
    if (auto_tune_config_.save_best_config && !result.best_params.empty()) {
        std::string config_file = auto_tune_config_.config_save_path + operator_name + "_best_config.json";
        std::ofstream file(config_file);
        if (file.is_open()) {
            Json::Value root;
            for (const auto& pair : result.best_params) {
                root[pair.first] = pair.second;
            }
            file << root;
            file.close();
            LOG(INFO) << "Best configuration saved to: " << config_file;
        }
    }
    
    LOG(INFO) << "Auto-tuning completed for " << operator_name;
    return result;
}

void PerformanceMonitor::SetAutoTuneConfig(const AutoTuneConfig& config) {
    auto_tune_config_ = config;
}

AutoTuneConfig PerformanceMonitor::GetAutoTuneConfig() const {
    return auto_tune_config_;
}

void PerformanceMonitor::ExportToCSV(const std::string& filename) {
    auto analysis = AnalyzePerformance();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for CSV export: " << filename;
        return;
    }
    
    // 写入头部
    file << "Event Name,Type,Duration (ms),Memory Used,Memory Allocated,Device Info\n";
    
    // 写入事件数据
    for (const auto& event : analysis.events) {
        file << event.name << "," 
             << static_cast<int>(event.type) << ","
             << event.GetDuration() << ","
             << event.memory_used << ","
             << event.memory_allocated << ","
             << event.device_info << "\n";
    }
    
    file.close();
    LOG(INFO) << "Performance data exported to CSV: " << filename;
}

void PerformanceMonitor::ExportToJSON(const std::string& filename) {
    auto analysis = AnalyzePerformance();
    
    Json::Value root;
    root["analysis_time"] = std::chrono::duration_cast<std::chrono::seconds>(
        analysis.analysis_time.time_since_epoch()).count();
    root["total_execution_time"] = analysis.total_execution_time;
    root["total_events"] = analysis.events.size();
    
    // 内存统计
    Json::Value memory_stats;
    memory_stats["total_allocated"] = analysis.memory_stats.total_allocated;
    memory_stats["total_freed"] = analysis.memory_stats.total_freed;
    memory_stats["current_usage"] = analysis.memory_stats.current_usage;
    memory_stats["peak_usage"] = analysis.memory_stats.peak_usage;
    memory_stats["allocation_count"] = analysis.memory_stats.allocation_count;
    root["memory_stats"] = memory_stats;
    
    // 热点分析
    Json::Value hotspots(Json::arrayValue);
    for (const auto& hotspot : analysis.hotspots) {
        Json::Value hotspot_json;
        hotspot_json["function_name"] = hotspot.function_name;
        hotspot_json["total_time"] = hotspot.total_time;
        hotspot_json["percentage"] = hotspot.percentage;
        hotspot_json["call_count"] = hotspot.call_count;
        hotspot_json["avg_time"] = hotspot.avg_time;
        hotspot_json["optimization_suggestion"] = hotspot.optimization_suggestion;
        hotspots.append(hotspot_json);
    }
    root["hotspots"] = hotspots;
    
    // 优化建议
    Json::Value recommendations(Json::arrayValue);
    for (const auto& rec : analysis.recommendations) {
        recommendations.append(rec);
    }
    root["recommendations"] = recommendations;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for JSON export: " << filename;
        return;
    }
    
    file << root;
    file.close();
    LOG(INFO) << "Performance data exported to JSON: " << filename;
}

void PerformanceMonitor::SaveAnalysis(const std::string& filename) {
    std::string report = GenerateReport();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for analysis save: " << filename;
        return;
    }
    
    file << report;
    file.close();
    LOG(INFO) << "Performance analysis saved to: " << filename;
}

void PerformanceMonitor::EnableRealTimeMonitoring(bool enable) {
    real_time_monitoring_ = enable;
    if (enable && monitoring_enabled_) {
        monitoring_thread_ = std::thread(&PerformanceMonitor::BackgroundMonitoring, this);
    }
}

void PerformanceMonitor::SetMonitoringInterval(int interval_ms) {
    monitoring_interval_ms_ = interval_ms;
}

void PerformanceMonitor::RegisterCallback(std::function<void(const PerformanceAnalysis&)> callback) {
    callbacks_.push_back(callback);
}

void PerformanceMonitor::BackgroundMonitoring() {
    while (monitoring_enabled_ && real_time_monitoring_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms_));
        
        if (monitoring_enabled_) {
            ProcessEvents();
            UpdateMemoryStats();
            
            // 调用回调函数
            auto analysis = AnalyzePerformance();
            for (const auto& callback : callbacks_) {
                try {
                    callback(analysis);
                } catch (const std::exception& e) {
                    LOG(WARNING) << "Callback execution failed: " << e.what();
                }
            }
        }
    }
}

void PerformanceMonitor::ProcessEvents() {
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    // 处理超时事件
    auto now = std::chrono::high_resolution_clock::now();
    auto it = active_events_.begin();
    while (it != active_events_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - it->second->start_time);
        if (duration.count() > 300) { // 5分钟超时
            LOG(WARNING) << "Event '" << it->first << "' timed out";
            it->second->end_time = now;
            completed_events_.push_back(std::move(it->second));
            it = active_events_.erase(it);
        } else {
            ++it;
        }
    }
}

void PerformanceMonitor::UpdateMemoryStats() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    size_t current_usage = total - free;
    
    if (current_usage > memory_stats_.peak_usage) {
        memory_stats_.peak_usage = current_usage;
    }
    
    memory_stats_.current_usage = current_usage;
    memory_stats_.last_update = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::GenerateRecommendations(PerformanceAnalysis& analysis) {
    analysis.recommendations.clear();
    
    // 基于热点分析生成建议
    for (const auto& hotspot : analysis.hotspots) {
        if (hotspot.percentage > 20.0) {
            analysis.recommendations.push_back(
                "High CPU usage in " + hotspot.function_name + 
                " (" + std::to_string(hotspot.percentage) + "%). Consider optimization.");
        }
        
        if (hotspot.avg_time > 100.0) {
            analysis.recommendations.push_back(
                "Slow execution in " + hotspot.function_name + 
                " (avg: " + std::to_string(hotspot.avg_time) + "ms). Consider parallelization.");
        }
    }
    
    // 基于内存使用生成建议
    if (analysis.memory_stats.peak_usage > 1024 * 1024 * 1024) { // 1GB
        analysis.recommendations.push_back("High memory usage detected. Consider memory optimization.");
    }
    
    if (analysis.memory_stats.allocation_count > 1000) {
        analysis.recommendations.push_back("Frequent memory allocations detected. Consider memory pooling.");
    }
    
    // 基于执行时间生成建议
    if (analysis.total_execution_time > 1000.0) {
        analysis.recommendations.push_back("Long execution time detected. Consider performance optimization.");
    }
}

std::string PerformanceMonitor::GetOptimizationSuggestion(const HotspotAnalysis& hotspot) {
    if (hotspot.percentage > 50.0) {
        return "Critical bottleneck - requires immediate optimization";
    } else if (hotspot.percentage > 20.0) {
        return "Significant bottleneck - consider optimization";
    } else if (hotspot.avg_time > 100.0) {
        return "Slow execution - consider parallelization or algorithm optimization";
    } else if (hotspot.call_count > 1000) {
        return "Frequent calls - consider caching or batching";
    } else {
        return "Performance acceptable";
    }
}

} // namespace cu_op_mem
