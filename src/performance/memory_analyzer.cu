#include "performance/memory_analyzer.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <execinfo.h>
#include <cxxabi.h>
#include <json/json.h>

namespace cu_op_mem {

MemoryAnalyzer& MemoryAnalyzer::Instance() {
    static MemoryAnalyzer instance;
    return instance;
}

MemoryAnalyzer::MemoryAnalyzer() 
    : tracking_enabled_(false), real_time_monitoring_(false), memory_pool_integration_(false),
      next_allocation_id_(1), total_allocated_(0), total_freed_(0), current_usage_(0),
      peak_usage_(0), allocation_count_(0), free_count_(0), monitoring_interval_ms_(1000) {
}

MemoryAnalyzer::~MemoryAnalyzer() {
    StopTracking();
}

void MemoryAnalyzer::StartTracking() {
    if (tracking_enabled_) {
        LOG(WARNING) << "Memory tracking is already enabled";
        return;
    }
    
    tracking_enabled_ = true;
    
    // 启动后台监控线程
    if (real_time_monitoring_) {
        monitoring_thread_ = std::thread(&MemoryAnalyzer::BackgroundMonitoring, this);
    }
    
    LOG(INFO) << "Memory tracking started";
}

void MemoryAnalyzer::StopTracking() {
    if (!tracking_enabled_) {
        return;
    }
    
    tracking_enabled_ = false;
    
    // 停止后台监控线程
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    // 处理剩余分配
    ProcessAllocations();
    
    LOG(INFO) << "Memory tracking stopped";
}

void MemoryAnalyzer::Reset() {
    std::lock_guard<std::mutex> allocations_lock(allocations_mutex_);
    std::lock_guard<std::mutex> statistics_lock(statistics_mutex_);
    
    active_allocations_.clear();
    allocation_history_.clear();
    next_allocation_id_ = 1;
    
    total_allocated_ = 0;
    total_freed_ = 0;
    current_usage_ = 0;
    peak_usage_ = 0;
    allocation_count_ = 0;
    free_count_ = 0;
    
    pool_allocations_.clear();
    pool_frees_.clear();
    
    LOG(INFO) << "Memory analyzer reset";
}

void MemoryAnalyzer::TrackAllocation(void* ptr, size_t size, const std::string& file, 
                                   int line, const std::string& function) {
    if (!tracking_enabled_ || ptr == nullptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    auto allocation = std::make_unique<MemoryAllocation>();
    allocation->ptr = ptr;
    allocation->size = size;
    allocation->file = file;
    allocation->line = line;
    allocation->function = function;
    allocation->timestamp = std::chrono::high_resolution_clock::now();
    allocation->is_freed = false;
    allocation->allocation_id = next_allocation_id_++;
    allocation->stack_trace = GetStackTrace();
    
    active_allocations_[ptr] = std::move(allocation);
    
    // 更新统计信息
    total_allocated_ += size;
    current_usage_ += size;
    allocation_count_++;
    
    if (current_usage_.load() > peak_usage_.load()) {
        peak_usage_.store(current_usage_.load());
    }
    
    // 添加到历史记录
    allocation_history_.push_back(std::make_unique<MemoryAllocation>(*active_allocations_[ptr]));
}

void MemoryAnalyzer::TrackFree(void* ptr) {
    if (!tracking_enabled_ || ptr == nullptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    auto it = active_allocations_.find(ptr);
    if (it != active_allocations_.end()) {
        auto& allocation = it->second;
        allocation->is_freed = true;
        
        // 更新统计信息
        total_freed_ += allocation->size;
        current_usage_ -= allocation->size;
        free_count_++;
        
        // 从活跃分配中移除
        active_allocations_.erase(it);
    } else {
        LOG(WARNING) << "Attempted to free untracked memory: " << ptr;
    }
}

void MemoryAnalyzer::TrackReallocation(void* old_ptr, void* new_ptr, size_t new_size) {
    if (!tracking_enabled_) {
        return;
    }
    
    // 先释放旧指针
    if (old_ptr != nullptr) {
        TrackFree(old_ptr);
    }
    
    // 跟踪新分配
    if (new_ptr != nullptr) {
        TrackAllocation(new_ptr, new_size, __FILE__, __LINE__, __FUNCTION__);
    }
}

MemoryAnalysisReport MemoryAnalyzer::AnalyzeMemory() {
    std::lock_guard<std::mutex> allocations_lock(allocations_mutex_);
    std::lock_guard<std::mutex> statistics_lock(statistics_mutex_);
    
    MemoryAnalysisReport report;
    report.analysis_time = std::chrono::high_resolution_clock::now();
    
    // 基本统计信息
    report.total_allocated = total_allocated_;
    report.total_freed = total_freed_;
    report.current_usage = current_usage_;
    report.peak_usage = peak_usage_;
    report.allocation_count = allocation_count_;
    report.free_count = free_count_;
    report.fragmentation_ratio = CalculateFragmentationRatio();
    
    // 检测内存泄漏
    report.leaks = DetectLeaks();
    
    // 分析内存使用模式
    report.patterns = AnalyzePatterns();
    
    // 生成优化建议
    GenerateRecommendations(report);
    
    return report;
}

std::vector<MemoryLeakInfo> MemoryAnalyzer::DetectLeaks() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    std::vector<MemoryLeakInfo> leaks;
    
    for (const auto& pair : active_allocations_) {
        const auto& allocation = pair.second;
        if (!allocation->is_freed) {
            MemoryLeakInfo leak;
            leak.file = allocation->file;
            leak.line = allocation->line;
            leak.function = allocation->function;
            leak.size = allocation->size;
            leak.allocation_time = allocation->timestamp;
            leak.stack_trace = allocation->stack_trace;
            leak.allocation_id = allocation->allocation_id;
            
            leaks.push_back(leak);
        }
    }
    
    // 按大小排序
    std::sort(leaks.begin(), leaks.end(),
        [](const MemoryLeakInfo& a, const MemoryLeakInfo& b) {
            return a.size > b.size;
        });
    
    return leaks;
}

std::vector<MemoryPattern> MemoryAnalyzer::AnalyzePatterns() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    std::unordered_map<std::string, std::vector<size_t>> function_allocations;
    std::unordered_map<std::string, std::vector<std::chrono::high_resolution_clock::time_point>> function_timestamps;
    
    // 按函数分组分配
    for (const auto& allocation : allocation_history_) {
        function_allocations[allocation->function].push_back(allocation->size);
        function_timestamps[allocation->function].push_back(allocation->timestamp);
    }
    
    std::vector<MemoryPattern> patterns;
    
    for (const auto& pair : function_allocations) {
        const std::string& function_name = pair.first;
        const std::vector<size_t>& sizes = pair.second;
        const std::vector<std::chrono::high_resolution_clock::time_point>& timestamps = function_timestamps[function_name];
        
        MemoryPattern pattern;
        pattern.pattern_name = function_name;
        pattern.total_allocations = sizes.size();
        pattern.total_size = std::accumulate(sizes.begin(), sizes.end(), 0ULL);
        pattern.avg_allocation_size = static_cast<double>(pattern.total_size) / pattern.total_allocations;
        
        // 计算分配频率
        if (timestamps.size() > 1) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                timestamps.back() - timestamps.front());
            pattern.allocation_frequency = static_cast<double>(pattern.total_allocations) / duration.count();
        }
        
        // 大小分布
        pattern.size_distribution = sizes;
        std::sort(pattern.size_distribution.begin(), pattern.size_distribution.end());
        
        pattern.first_allocation = timestamps.front();
        pattern.last_allocation = timestamps.back();
        
        patterns.push_back(pattern);
    }
    
    // 按总分配大小排序
    std::sort(patterns.begin(), patterns.end(),
        [](const MemoryPattern& a, const MemoryPattern& b) {
            return a.total_size > b.total_size;
        });
    
    return patterns;
}

size_t MemoryAnalyzer::GetTotalAllocated() const {
    return total_allocated_;
}

size_t MemoryAnalyzer::GetTotalFreed() const {
    return total_freed_;
}

size_t MemoryAnalyzer::GetCurrentUsage() const {
    return current_usage_;
}

size_t MemoryAnalyzer::GetPeakUsage() const {
    return peak_usage_;
}

size_t MemoryAnalyzer::GetAllocationCount() const {
    return allocation_count_;
}

size_t MemoryAnalyzer::GetFreeCount() const {
    return free_count_;
}

std::vector<std::string> MemoryAnalyzer::GetOptimizationSuggestions() {
    auto report = AnalyzeMemory();
    return report.recommendations;
}

double MemoryAnalyzer::CalculateFragmentationRatio() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    if (active_allocations_.empty()) {
        return 0.0;
    }
    
    // 计算碎片化比率
    size_t total_size = 0;
    size_t max_size = 0;
    
    for (const auto& pair : active_allocations_) {
        size_t size = pair.second->size;
        total_size += size;
        max_size = std::max(max_size, size);
    }
    
    if (max_size == 0) {
        return 0.0;
    }
    
    // 碎片化比率 = (总大小 - 最大连续块) / 总大小
    return static_cast<double>(total_size - max_size) / total_size;
}

std::string MemoryAnalyzer::GenerateReport() {
    auto report = AnalyzeMemory();
    
    std::ostringstream oss;
    oss << "=== cuOP Memory Analysis Report ===\n";
    auto time_t_val = std::chrono::system_clock::to_time_t(report.analysis_time);
    oss << "Analysis Time: " << std::put_time(std::localtime(&time_t_val), "%Y-%m-%d %H:%M:%S") << "\n\n";
    
    // 基本统计
    oss << "=== Memory Statistics ===\n";
    oss << "Total Allocated: " << report.total_allocated << " bytes (" 
        << report.total_allocated / (1024.0 * 1024.0) << " MB)\n";
    oss << "Total Freed: " << report.total_freed << " bytes (" 
        << report.total_freed / (1024.0 * 1024.0) << " MB)\n";
    oss << "Current Usage: " << report.current_usage << " bytes (" 
        << report.current_usage / (1024.0 * 1024.0) << " MB)\n";
    oss << "Peak Usage: " << report.peak_usage << " bytes (" 
        << report.peak_usage / (1024.0 * 1024.0) << " MB)\n";
    oss << "Allocation Count: " << report.allocation_count << "\n";
    oss << "Free Count: " << report.free_count << "\n";
    oss << "Fragmentation Ratio: " << std::fixed << std::setprecision(2) 
        << report.fragmentation_ratio * 100.0 << "%\n\n";
    
    // 内存泄漏
    oss << "=== Memory Leaks ===\n";
    if (report.leaks.empty()) {
        oss << "No memory leaks detected.\n\n";
    } else {
        oss << "Found " << report.leaks.size() << " memory leaks:\n";
        for (size_t i = 0; i < std::min(size_t(10), report.leaks.size()); ++i) {
            const auto& leak = report.leaks[i];
            oss << i + 1 << ". " << leak.function << " (" << leak.file << ":" << leak.line << ")\n";
            oss << "   Size: " << leak.size << " bytes (" << leak.size / (1024.0 * 1024.0) << " MB)\n";
            oss << "   Allocation ID: " << leak.allocation_id << "\n\n";
        }
    }
    
    // 内存使用模式
    oss << "=== Memory Usage Patterns ===\n";
    for (size_t i = 0; i < std::min(size_t(5), report.patterns.size()); ++i) {
        const auto& pattern = report.patterns[i];
        oss << i + 1 << ". " << pattern.pattern_name << "\n";
        oss << "   Total Allocations: " << pattern.total_allocations << "\n";
        oss << "   Total Size: " << pattern.total_size << " bytes (" 
            << pattern.total_size / (1024.0 * 1024.0) << " MB)\n";
        oss << "   Average Size: " << std::fixed << std::setprecision(2) 
            << pattern.avg_allocation_size << " bytes\n";
        oss << "   Allocation Frequency: " << std::fixed << std::setprecision(2) 
            << pattern.allocation_frequency << " allocs/sec\n\n";
    }
    
    // 优化建议
    oss << "=== Optimization Recommendations ===\n";
    for (size_t i = 0; i < report.recommendations.size(); ++i) {
        oss << i + 1 << ". " << report.recommendations[i] << "\n";
    }
    
    return oss.str();
}

void MemoryAnalyzer::ExportToCSV(const std::string& filename) {
    auto report = AnalyzeMemory();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file for CSV export: " << filename;
        return;
    }
    
    // 写入头部
    file << "Allocation ID,Function,File,Line,Size,Allocation Time,Is Freed\n";
    
    // 写入分配数据
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    for (const auto& allocation : allocation_history_) {
        file << allocation->allocation_id << ","
             << allocation->function << ","
             << allocation->file << ","
             << allocation->line << ","
             << allocation->size << ","
             << std::chrono::duration_cast<std::chrono::seconds>(
                 allocation->timestamp.time_since_epoch()).count() << ","
             << (allocation->is_freed ? "Yes" : "No") << "\n";
    }
    
    file.close();
    LOG(INFO) << "Memory analysis data exported to CSV: " << filename;
}

void MemoryAnalyzer::ExportToJSON(const std::string& filename) {
    auto report = AnalyzeMemory();
    
    Json::Value root;
    root["analysis_time"] = std::chrono::duration_cast<std::chrono::seconds>(
        report.analysis_time.time_since_epoch()).count();
    root["total_allocated"] = report.total_allocated;
    root["total_freed"] = report.total_freed;
    root["current_usage"] = report.current_usage;
    root["peak_usage"] = report.peak_usage;
    root["allocation_count"] = report.allocation_count;
    root["free_count"] = report.free_count;
    root["fragmentation_ratio"] = report.fragmentation_ratio;
    
    // 内存泄漏
    Json::Value leaks(Json::arrayValue);
    for (const auto& leak : report.leaks) {
        Json::Value leak_json;
        leak_json["function"] = leak.function;
        leak_json["file"] = leak.file;
        leak_json["line"] = leak.line;
        leak_json["size"] = leak.size;
        leak_json["allocation_id"] = leak.allocation_id;
        leaks.append(leak_json);
    }
    root["leaks"] = leaks;
    
    // 内存使用模式
    Json::Value patterns(Json::arrayValue);
    for (const auto& pattern : report.patterns) {
        Json::Value pattern_json;
        pattern_json["pattern_name"] = pattern.pattern_name;
        pattern_json["total_allocations"] = pattern.total_allocations;
        pattern_json["total_size"] = pattern.total_size;
        pattern_json["avg_allocation_size"] = pattern.avg_allocation_size;
        pattern_json["allocation_frequency"] = pattern.allocation_frequency;
        patterns.append(pattern_json);
    }
    root["patterns"] = patterns;
    
    // 优化建议
    Json::Value recommendations(Json::arrayValue);
    for (const auto& rec : report.recommendations) {
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
    LOG(INFO) << "Memory analysis data exported to JSON: " << filename;
}

void MemoryAnalyzer::EnableRealTimeMonitoring(bool enable) {
    real_time_monitoring_ = enable;
    if (enable && tracking_enabled_) {
        monitoring_thread_ = std::thread(&MemoryAnalyzer::BackgroundMonitoring, this);
    }
}

void MemoryAnalyzer::SetMonitoringInterval(int interval_ms) {
    monitoring_interval_ms_ = interval_ms;
}

void MemoryAnalyzer::RegisterCallback(std::function<void(const MemoryAnalysisReport&)> callback) {
    callbacks_.push_back(callback);
}

void MemoryAnalyzer::SetMemoryPoolIntegration(bool enable) {
    memory_pool_integration_ = enable;
}

void MemoryAnalyzer::TrackMemoryPoolAllocation(void* ptr, size_t size, const std::string& pool_name) {
    if (!memory_pool_integration_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    pool_allocations_[pool_name] += size;
    
    // 同时进行常规跟踪
    TrackAllocation(ptr, size, __FILE__, __LINE__, "MemoryPool::" + pool_name);
}

void MemoryAnalyzer::TrackMemoryPoolFree(void* ptr, const std::string& pool_name) {
    if (!memory_pool_integration_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    // 这里需要根据指针找到对应的分配大小
    // 简化实现，实际应该维护更详细的映射关系
    
    // 同时进行常规跟踪
    TrackFree(ptr);
}

void MemoryAnalyzer::BackgroundMonitoring() {
    while (tracking_enabled_ && real_time_monitoring_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms_));
        
        if (tracking_enabled_) {
            ProcessAllocations();
            UpdateStatistics();
            
            // 调用回调函数
            auto report = AnalyzeMemory();
            for (const auto& callback : callbacks_) {
                try {
                    callback(report);
                } catch (const std::exception& e) {
                    LOG(WARNING) << "Memory analysis callback execution failed: " << e.what();
                }
            }
        }
    }
}

void MemoryAnalyzer::UpdateStatistics() {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    
    // 更新当前使用量
    size_t current_usage = 0;
    for (const auto& pair : active_allocations_) {
        current_usage += pair.second->size;
    }
    
    current_usage_ = current_usage;
    
    if (current_usage > peak_usage_) {
        peak_usage_ = current_usage;
    }
}

std::string MemoryAnalyzer::GetStackTrace() {
    const int max_frames = 10;
    void* frames[max_frames];
    int frame_count = backtrace(frames, max_frames);
    
    char** symbols = backtrace_symbols(frames, frame_count);
    if (!symbols) {
        return "Failed to get stack trace";
    }
    
    std::ostringstream oss;
    for (int i = 0; i < frame_count; ++i) {
        oss << symbols[i] << "\n";
    }
    
    free(symbols);
    return oss.str();
}

void MemoryAnalyzer::ProcessAllocations() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    
    // 处理超时分配
    auto now = std::chrono::high_resolution_clock::now();
    auto it = active_allocations_.begin();
    while (it != active_allocations_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - it->second->timestamp);
        if (duration.count() > 60) { // 1小时超时
            LOG(WARNING) << "Long-lived allocation detected: " << it->second->function 
                        << " (" << it->second->size << " bytes)";
            ++it;
        } else {
            ++it;
        }
    }
}

void MemoryAnalyzer::DetectPatterns() {
    // 模式检测逻辑已在AnalyzePatterns中实现
}

void MemoryAnalyzer::GenerateRecommendations(MemoryAnalysisReport& report) {
    report.recommendations.clear();
    
    // 基于内存泄漏生成建议
    if (!report.leaks.empty()) {
        report.recommendations.push_back(
            "Found " + std::to_string(report.leaks.size()) + " memory leaks. Review allocation/deallocation patterns.");
    }
    
    // 基于碎片化生成建议
    if (report.fragmentation_ratio > 0.3) {
        report.recommendations.push_back("High memory fragmentation detected. Consider using memory pools.");
    }
    
    // 基于分配频率生成建议
    for (const auto& pattern : report.patterns) {
        if (pattern.allocation_frequency > 1000.0) {
            report.recommendations.push_back(
                "High allocation frequency in " + pattern.pattern_name + ". Consider object pooling.");
        }
        
        if (pattern.avg_allocation_size < 1024 && pattern.total_allocations > 10000) {
            report.recommendations.push_back(
                "Many small allocations in " + pattern.pattern_name + ". Consider batch allocation.");
        }
    }
    
    // 基于峰值使用生成建议
    if (report.peak_usage > 1024 * 1024 * 1024) { // 1GB
        report.recommendations.push_back("High peak memory usage detected. Consider memory optimization.");
    }
    
    // 基于分配/释放不平衡生成建议
    if (report.allocation_count > report.free_count * 1.1) {
        report.recommendations.push_back("Allocation/free imbalance detected. Check for memory leaks.");
    }
}

MemoryAnalysisReport MemoryAnalyzer::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    
    MemoryAnalysisReport report;
    report.total_allocated = total_allocated_.load();
    report.total_freed = total_freed_.load();
    report.current_usage = current_usage_.load();
    report.peak_usage = peak_usage_.load();
    report.allocation_count = allocation_count_.load();
    report.free_count = free_count_.load();
    report.fragmentation_ratio = 0.0; // 简化实现，避免const问题
    report.analysis_time = std::chrono::high_resolution_clock::now();
    
    return report;
}

} // namespace cu_op_mem
