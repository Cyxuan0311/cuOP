#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <memory>
#include "performance/performance_monitor.hpp"
#include "performance/memory_analyzer.hpp"
#include "performance/hotspot_analyzer.hpp"
#include "performance/auto_tuner.hpp"

using namespace cu_op_mem;

class PerformanceMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset all monitors before each test
        PerformanceMonitor::Instance().Reset();
        MemoryAnalyzer::Instance().Reset();
        HotspotAnalyzer::Instance().Reset();
        AutoTuner::Instance().Reset();
    }
    
    void TearDown() override {
        // Stop monitoring after each test
        PerformanceMonitor::Instance().StopMonitoring();
        MemoryAnalyzer::Instance().StopTracking();
        HotspotAnalyzer::Instance().StopAnalysis();
        AutoTuner::Instance().StopTuning();
    }
};

// Performance Monitor Tests
TEST_F(PerformanceMonitorTest, BasicFunctionality) {
    auto& monitor = PerformanceMonitor::Instance();
    
    // Test start/stop monitoring
    EXPECT_FALSE(monitor.IsMonitoring());
    monitor.StartMonitoring();
    EXPECT_TRUE(monitor.IsMonitoring());
    monitor.StopMonitoring();
    EXPECT_FALSE(monitor.IsMonitoring());
}

TEST_F(PerformanceMonitorTest, EventRecording) {
    auto& monitor = PerformanceMonitor::Instance();
    monitor.StartMonitoring();
    
    // Record some events
    monitor.StartEvent("test_function", PerformanceEventType::CUSTOM_EVENT);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    monitor.EndEvent("test_function");
    
    monitor.RecordEvent("test_event", 5.0, PerformanceEventType::CUSTOM_EVENT);
    
    // Analyze performance
    auto analysis = monitor.AnalyzePerformance();
    
    EXPECT_GT(analysis.events.size(), 0);
    EXPECT_GT(analysis.total_execution_time, 0);
    
    monitor.StopMonitoring();
}

TEST_F(PerformanceMonitorTest, MemoryTracking) {
    auto& monitor = PerformanceMonitor::Instance();
    monitor.StartMonitoring();
    
    // Record memory allocations
    monitor.RecordMemoryAllocation(1024);
    monitor.RecordMemoryAllocation(2048);
    monitor.RecordMemoryFree(1024);
    
    auto memory_stats = monitor.GetMemoryStats();
    EXPECT_EQ(memory_stats.total_allocated, 3072);
    EXPECT_EQ(memory_stats.total_freed, 1024);
    EXPECT_EQ(memory_stats.current_usage, 2048);
    EXPECT_EQ(memory_stats.allocation_count, 2);
    EXPECT_EQ(memory_stats.free_count, 1);
    
    monitor.StopMonitoring();
}

TEST_F(PerformanceMonitorTest, HotspotAnalysis) {
    auto& monitor = PerformanceMonitor::Instance();
    monitor.StartMonitoring();
    
    // Record multiple events for the same function
    for (int i = 0; i < 5; ++i) {
        monitor.StartEvent("slow_function", PerformanceEventType::CUSTOM_EVENT);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        monitor.EndEvent("slow_function");
    }
    
    for (int i = 0; i < 10; ++i) {
        monitor.StartEvent("fast_function", PerformanceEventType::CUSTOM_EVENT);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        monitor.EndEvent("fast_function");
    }
    
    auto hotspots = monitor.AnalyzeHotspots();
    EXPECT_GT(hotspots.size(), 0);
    
    // Find the slow function hotspot
    bool found_slow_function = false;
    for (const auto& hotspot : hotspots) {
        if (hotspot.function_name == "slow_function") {
            found_slow_function = true;
            EXPECT_GT(hotspot.total_time, 0);
            EXPECT_GT(hotspot.call_count, 0);
            break;
        }
    }
    EXPECT_TRUE(found_slow_function);
    
    monitor.StopMonitoring();
}

TEST_F(PerformanceMonitorTest, ReportGeneration) {
    auto& monitor = PerformanceMonitor::Instance();
    monitor.StartMonitoring();
    
    // Record some events
    monitor.StartEvent("test_function", PerformanceEventType::CUSTOM_EVENT);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    monitor.EndEvent("test_function");
    
    std::string report = monitor.GenerateReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("cuOP Performance Analysis Report"), std::string::npos);
    
    monitor.StopMonitoring();
}

// Memory Analyzer Tests
TEST_F(PerformanceMonitorTest, MemoryAnalyzerBasic) {
    auto& analyzer = MemoryAnalyzer::Instance();
    
    EXPECT_FALSE(analyzer.IsTracking());
    analyzer.StartTracking();
    EXPECT_TRUE(analyzer.IsTracking());
    analyzer.StopTracking();
    EXPECT_FALSE(analyzer.IsTracking());
}

TEST_F(PerformanceMonitorTest, MemoryAllocationTracking) {
    auto& analyzer = MemoryAnalyzer::Instance();
    analyzer.StartTracking();
    
    // Simulate memory allocations
    void* ptr1 = malloc(1024);
    void* ptr2 = malloc(2048);
    
    analyzer.TrackAllocation(ptr1, 1024, __FILE__, __LINE__, __FUNCTION__);
    analyzer.TrackAllocation(ptr2, 2048, __FILE__, __LINE__, __FUNCTION__);
    
    auto stats = analyzer.GetMemoryStats();
    EXPECT_EQ(stats.total_allocated, 3072);
    EXPECT_EQ(stats.current_usage, 3072);
    EXPECT_EQ(stats.allocation_count, 2);
    
    analyzer.TrackFree(ptr1);
    analyzer.TrackFree(ptr2);
    
    stats = analyzer.GetMemoryStats();
    EXPECT_EQ(stats.total_freed, 3072);
    EXPECT_EQ(stats.current_usage, 0);
    EXPECT_EQ(stats.free_count, 2);
    
    free(ptr1);
    free(ptr2);
    
    analyzer.StopTracking();
}

TEST_F(PerformanceMonitorTest, MemoryLeakDetection) {
    auto& analyzer = MemoryAnalyzer::Instance();
    analyzer.StartTracking();
    
    // Simulate memory leak
    void* ptr1 = malloc(1024);
    void* ptr2 = malloc(2048);
    
    analyzer.TrackAllocation(ptr1, 1024, __FILE__, __LINE__, __FUNCTION__);
    analyzer.TrackAllocation(ptr2, 2048, __FILE__, __LINE__, __FUNCTION__);
    
    // Only free one pointer
    analyzer.TrackFree(ptr1);
    
    auto leaks = analyzer.DetectLeaks();
    EXPECT_EQ(leaks.size(), 1);
    EXPECT_EQ(leaks[0].size, 2048);
    
    analyzer.TrackFree(ptr2);
    free(ptr1);
    free(ptr2);
    
    analyzer.StopTracking();
}

TEST_F(PerformanceMonitorTest, MemoryAnalysisReport) {
    auto& analyzer = MemoryAnalyzer::Instance();
    analyzer.StartTracking();
    
    // Simulate some memory activity
    void* ptr = malloc(1024);
    analyzer.TrackAllocation(ptr, 1024, __FILE__, __LINE__, __FUNCTION__);
    analyzer.TrackFree(ptr);
    free(ptr);
    
    std::string report = analyzer.GenerateReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("cuOP Memory Analysis Report"), std::string::npos);
    
    analyzer.StopTracking();
}

// Hotspot Analyzer Tests
TEST_F(PerformanceMonitorTest, HotspotAnalyzerBasic) {
    auto& analyzer = HotspotAnalyzer::Instance();
    
    EXPECT_FALSE(analyzer.IsAnalyzing());
    analyzer.StartAnalysis();
    EXPECT_TRUE(analyzer.IsAnalyzing());
    analyzer.StopAnalysis();
    EXPECT_FALSE(analyzer.IsAnalyzing());
}

TEST_F(PerformanceMonitorTest, FunctionCallTracking) {
    auto& analyzer = HotspotAnalyzer::Instance();
    analyzer.StartAnalysis();
    
    // Record function calls
    for (int i = 0; i < 3; ++i) {
        analyzer.StartFunction("test_function", __FILE__, __LINE__);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        analyzer.EndFunction("test_function");
    }
    
    auto hotspots = analyzer.AnalyzeHotspots();
    EXPECT_GT(hotspots.size(), 0);
    
    // Find the test function
    bool found_test_function = false;
    for (const auto& hotspot : hotspots) {
        if (hotspot.function_name == "test_function") {
            found_test_function = true;
            EXPECT_EQ(hotspot.call_count, 3);
            EXPECT_GT(hotspot.total_time, 0);
            EXPECT_GT(hotspot.avg_time, 0);
            break;
        }
    }
    EXPECT_TRUE(found_test_function);
    
    analyzer.StopAnalysis();
}

TEST_F(PerformanceMonitorTest, BottleneckIdentification) {
    auto& analyzer = HotspotAnalyzer::Instance();
    analyzer.StartAnalysis();
    
    // Record a slow function call
    analyzer.StartFunction("slow_function", __FILE__, __LINE__);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    analyzer.EndFunction("slow_function");
    
    auto bottleneck_type = analyzer.IdentifyBottleneck("slow_function");
    EXPECT_NE(bottleneck_type, BottleneckType::UNKNOWN);
    
    analyzer.StopAnalysis();
}

TEST_F(PerformanceMonitorTest, OptimizationCandidates) {
    auto& analyzer = HotspotAnalyzer::Instance();
    analyzer.StartAnalysis();
    
    // Record multiple function calls with different performance characteristics
    for (int i = 0; i < 5; ++i) {
        analyzer.StartFunction("slow_function", __FILE__, __LINE__);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        analyzer.EndFunction("slow_function");
    }
    
    for (int i = 0; i < 10; ++i) {
        analyzer.StartFunction("fast_function", __FILE__, __LINE__);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        analyzer.EndFunction("fast_function");
    }
    
    auto candidates = analyzer.GetOptimizationCandidates();
    EXPECT_GT(candidates.size(), 0);
    
    analyzer.StopAnalysis();
}

// Auto Tuner Tests
TEST_F(PerformanceMonitorTest, AutoTunerBasic) {
    auto& tuner = AutoTuner::Instance();
    
    EXPECT_FALSE(tuner.IsTuning());
    tuner.StartTuning();
    EXPECT_TRUE(tuner.IsTuning());
    tuner.StopTuning();
    EXPECT_FALSE(tuner.IsTuning());
}

TEST_F(PerformanceMonitorTest, ParameterDefinition) {
    auto& tuner = AutoTuner::Instance();
    
    // Add parameters
    ParameterDefinition param1;
    param1.name = "block_size";
    param1.type = ParameterType::INTEGER;
    param1.values = {"32", "64", "128", "256"};
    param1.default_value = "128";
    param1.is_tunable = true;
    
    ParameterDefinition param2;
    param2.name = "optimization_level";
    param2.type = ParameterType::STRING;
    param2.values = {"O0", "O1", "O2", "O3"};
    param2.default_value = "O2";
    param2.is_tunable = true;
    
    tuner.AddParameter(param1);
    tuner.AddParameter(param2);
    
    auto parameters = tuner.GetParameters();
    EXPECT_EQ(parameters.size(), 2);
    
    // Remove a parameter
    tuner.RemoveParameter("block_size");
    parameters = tuner.GetParameters();
    EXPECT_EQ(parameters.size(), 1);
    EXPECT_EQ(parameters[0].name, "optimization_level");
}

TEST_F(PerformanceMonitorTest, TuningStrategy) {
    auto& tuner = AutoTuner::Instance();
    
    // Test available strategies
    auto strategies = tuner.GetAvailableStrategies();
    EXPECT_GT(strategies.size(), 0);
    
    // Test setting strategy
    tuner.SetTuningStrategy("RandomSearch");
    // Note: We can't easily test the internal strategy without exposing it
    
    tuner.SetTuningStrategy("GridSearch");
    tuner.SetTuningStrategy("BayesianOptimization");
    tuner.SetTuningStrategy("GeneticAlgorithm");
}

TEST_F(PerformanceMonitorTest, SimpleTuning) {
    auto& tuner = AutoTuner::Instance();
    
    // Add a simple parameter
    ParameterDefinition param;
    param.name = "test_param";
    param.type = ParameterType::INTEGER;
    param.values = {"1", "2", "3"};
    param.default_value = "2";
    param.is_tunable = true;
    tuner.AddParameter(param);
    
    // Set up tuning config
    TuningConfig config;
    config.operator_name = "test_operator";
    config.max_iterations = 5;
    config.convergence_threshold = 0.1;
    config.improvement_threshold = 0.05;
    tuner.SetTuningConfig(config);
    
    // Simple benchmark function
    auto benchmark_func = [](const std::unordered_map<std::string, std::string>& params) -> double {
        auto it = params.find("test_param");
        if (it != params.end()) {
            return std::stod(it->second);
        }
        return 0.0;
    };
    
    // Run tuning
    auto result = tuner.Tune("test_operator", benchmark_func);
    
    EXPECT_GT(result.iterations_used, 0);
    EXPECT_GT(result.best_performance, 0);
    EXPECT_FALSE(result.best_parameters.empty());
}

TEST_F(PerformanceMonitorTest, TuningHistory) {
    auto& tuner = AutoTuner::Instance();
    
    // Add a parameter
    ParameterDefinition param;
    param.name = "test_param";
    param.type = ParameterType::INTEGER;
    param.values = {"1", "2"};
    param.default_value = "1";
    param.is_tunable = true;
    tuner.AddParameter(param);
    
    // Set up tuning config
    TuningConfig config;
    config.operator_name = "test_operator";
    config.max_iterations = 3;
    tuner.SetTuningConfig(config);
    
    // Simple benchmark function
    auto benchmark_func = [](const std::unordered_map<std::string, std::string>& params) -> double {
        auto it = params.find("test_param");
        if (it != params.end()) {
            return std::stod(it->second);
        }
        return 0.0;
    };
    
    // Run multiple tuning sessions
    tuner.Tune("test_operator", benchmark_func);
    tuner.Tune("test_operator", benchmark_func);
    
    auto history = tuner.GetTuningHistory();
    EXPECT_EQ(history.size(), 2);
}

// Integration Tests
TEST_F(PerformanceMonitorTest, IntegrationTest) {
    auto& monitor = PerformanceMonitor::Instance();
    auto& memory_analyzer = MemoryAnalyzer::Instance();
    auto& hotspot_analyzer = HotspotAnalyzer::Instance();
    
    // Start all monitoring
    monitor.StartMonitoring();
    memory_analyzer.StartTracking();
    hotspot_analyzer.StartAnalysis();
    
    // Simulate some work
    monitor.StartEvent("integration_test", PerformanceEventType::CUSTOM_EVENT);
    hotspot_analyzer.StartFunction("integration_function", __FILE__, __LINE__);
    
    // Simulate memory allocation
    void* ptr = malloc(1024);
    memory_analyzer.TrackAllocation(ptr, 1024, __FILE__, __LINE__, __FUNCTION__);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    // End monitoring
    monitor.EndEvent("integration_test");
    hotspot_analyzer.EndFunction("integration_function");
    memory_analyzer.TrackFree(ptr);
    free(ptr);
    
    // Analyze results
    auto performance_analysis = monitor.AnalyzePerformance();
    auto memory_analysis = memory_analyzer.AnalyzeMemory();
    auto hotspots = hotspot_analyzer.AnalyzeHotspots();
    
    EXPECT_GT(performance_analysis.events.size(), 0);
    EXPECT_GT(memory_analysis.total_allocated, 0);
    EXPECT_GT(hotspots.size(), 0);
    
    // Stop all monitoring
    monitor.StopMonitoring();
    memory_analyzer.StopTracking();
    hotspot_analyzer.StopAnalysis();
}

// Performance Tests
TEST_F(PerformanceMonitorTest, PerformanceOverhead) {
    auto& monitor = PerformanceMonitor::Instance();
    
    // Test without monitoring
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        // Simulate some work
        volatile int sum = 0;
        for (int j = 0; j < 100; ++j) {
            sum += j;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto without_monitoring = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Test with monitoring
    monitor.StartMonitoring();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        monitor.StartEvent("test_event", PerformanceEventType::CUSTOM_EVENT);
        // Simulate some work
        volatile int sum = 0;
        for (int j = 0; j < 100; ++j) {
            sum += j;
        }
        monitor.EndEvent("test_event");
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_monitoring = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    monitor.StopMonitoring();
    
    // Monitoring overhead should be reasonable (less than 50% overhead)
    double overhead = static_cast<double>(with_monitoring - without_monitoring) / without_monitoring;
    EXPECT_LT(overhead, 0.5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
