# cuOP 性能监控系统使用指南

## 概述

cuOP 性能监控系统是一个全面的性能分析和优化工具集，提供实时性能监控、内存分析、热点识别和自动调优功能。本指南将详细介绍如何使用这些功能来优化您的 CUDA 应用程序。

## 系统架构

```
cuOP 性能监控系统
├── 性能分析器 (PerformanceMonitor)
│   ├── 事件记录和跟踪
│   ├── 性能分析和报告生成
│   └── 实时监控
├── 内存分析器 (MemoryAnalyzer)
│   ├── 内存分配跟踪
│   ├── 内存泄漏检测
│   └── 内存使用模式分析
├── 热点分析器 (HotspotAnalyzer)
│   ├── 函数调用跟踪
│   ├── 瓶颈识别
│   └── 优化建议生成
├── 自动调优器 (AutoTuner)
│   ├── 参数优化
│   ├── 性能预测
│   └── 历史数据分析
└── 性能仪表板 (PerformanceDashboard)
    ├── Web 界面
    ├── 实时数据可视化
    └── 报告生成和导出
```

## 快速开始

### 1. 基本性能监控

```cpp
#include "performance/performance_monitor.hpp"

int main() {
    // 启动性能监控
    auto& monitor = PerformanceMonitor::Instance();
    monitor.StartMonitoring();
    
    // 记录性能事件
    CUOP_PERF_START("my_function");
    // ... 您的代码 ...
    CUOP_PERF_END("my_function");
    
    // 分析性能
    auto analysis = monitor.AnalyzePerformance();
    std::cout << "总执行时间: " << analysis.total_execution_time << " ms" << std::endl;
    
    // 生成报告
    std::string report = monitor.GenerateReport();
    std::cout << report << std::endl;
    
    monitor.StopMonitoring();
    return 0;
}
```

### 2. 内存分析

```cpp
#include "performance/memory_analyzer.hpp"

int main() {
    // 启动内存跟踪
    auto& analyzer = MemoryAnalyzer::Instance();
    analyzer.StartTracking();
    
    // 跟踪内存分配
    void* ptr = malloc(1024);
    CUOP_MEM_TRACK_ALLOC(ptr, 1024);
    
    // ... 使用内存 ...
    
    // 跟踪内存释放
    CUOP_MEM_TRACK_FREE(ptr);
    free(ptr);
    
    // 检测内存泄漏
    auto leaks = analyzer.DetectLeaks();
    if (!leaks.empty()) {
        std::cout << "检测到 " << leaks.size() << " 个内存泄漏" << std::endl;
    }
    
    // 生成内存分析报告
    std::string report = analyzer.GenerateReport();
    std::cout << report << std::endl;
    
    analyzer.StopTracking();
    return 0;
}
```

### 3. 热点分析

```cpp
#include "performance/hotspot_analyzer.hpp"

int main() {
    // 启动热点分析
    auto& analyzer = HotspotAnalyzer::Instance();
    analyzer.StartAnalysis();
    
    // 跟踪函数调用
    CUOP_HOTSPOT_START("slow_function");
    // ... 慢函数代码 ...
    CUOP_HOTSPOT_END("slow_function");
    
    // 分析热点
    auto hotspots = analyzer.AnalyzeHotspots();
    for (const auto& hotspot : hotspots) {
        std::cout << "函数: " << hotspot.function_name 
                  << ", 时间: " << hotspot.total_time << " ms"
                  << ", 百分比: " << hotspot.percentage << "%" << std::endl;
    }
    
    // 获取优化建议
    auto candidates = analyzer.GetOptimizationCandidates();
    std::cout << "优化候选函数: " << candidates.size() << " 个" << std::endl;
    
    analyzer.StopAnalysis();
    return 0;
}
```

### 4. 自动调优

```cpp
#include "performance/auto_tuner.hpp"

int main() {
    auto& tuner = AutoTuner::Instance();
    
    // 定义可调参数
    auto block_size_param = CUOP_PARAM_INT("block_size", {"32", "64", "128", "256"}, "128", 1.0);
    auto optimization_param = CUOP_PARAM_STRING("optimization", {"O0", "O1", "O2", "O3"}, "O2", 1.0);
    
    tuner.AddParameter(block_size_param);
    tuner.AddParameter(optimization_param);
    
    // 设置调优配置
    TuningConfig config;
    config.operator_name = "my_operator";
    config.max_iterations = 50;
    config.convergence_threshold = 0.05;
    config.tuning_strategy = "RandomSearch";
    tuner.SetTuningConfig(config);
    
    // 定义基准测试函数
    auto benchmark_func = [](const std::unordered_map<std::string, std::string>& params) -> double {
        // 根据参数运行您的算子
        // 返回性能指标（如 GFLOPS、延迟等）
        return run_operator_with_params(params);
    };
    
    // 运行自动调优
    auto result = CUOP_AUTO_TUNE("my_operator", benchmark_func);
    
    std::cout << "最佳性能: " << result.best_performance << std::endl;
    std::cout << "最佳参数:" << std::endl;
    for (const auto& param : result.best_parameters) {
        std::cout << "  " << param.first << ": " << param.second << std::endl;
    }
    
    return 0;
}
```

## 高级功能

### 1. 实时监控

```cpp
// 启用实时监控
monitor.EnableRealTimeMonitoring(true);
monitor.SetMonitoringInterval(1000); // 1秒间隔

// 注册回调函数
monitor.RegisterCallback([](const PerformanceAnalysis& analysis) {
    std::cout << "实时性能更新: " << analysis.total_execution_time << " ms" << std::endl;
});
```

### 2. 自定义性能事件

```cpp
// 创建自定义性能事件
PerformanceEvent custom_event;
custom_event.name = "custom_operation";
custom_event.type = PerformanceEventType::CUSTOM_EVENT;
custom_event.start_time = std::chrono::high_resolution_clock::now();

// ... 执行操作 ...

custom_event.end_time = std::chrono::high_resolution_clock::now();
monitor.RecordEvent("custom_operation", custom_event.GetDuration(), PerformanceEventType::CUSTOM_EVENT);
```

### 3. 内存池集成

```cpp
// 启用内存池集成
analyzer.SetMemoryPoolIntegration(true);

// 跟踪内存池分配
analyzer.TrackMemoryPoolAllocation(ptr, size, "my_pool");
analyzer.TrackMemoryPoolFree(ptr, "my_pool");
```

### 4. 瓶颈类型识别

```cpp
// 识别瓶颈类型
auto bottleneck_type = analyzer.IdentifyBottleneck("slow_function");
switch (bottleneck_type) {
    case BottleneckType::CPU_BOUND:
        std::cout << "CPU 瓶颈" << std::endl;
        break;
    case BottleneckType::GPU_BOUND:
        std::cout << "GPU 瓶颈" << std::endl;
        break;
    case BottleneckType::MEMORY_BOUND:
        std::cout << "内存瓶颈" << std::endl;
        break;
    default:
        std::cout << "未知瓶颈类型" << std::endl;
}
```

## 性能仪表板

### 启动仪表板

```bash
# 安装依赖
pip install flask flask-socketio plotly pandas numpy

# 启动仪表板
python python/performance_dashboard.py --host 0.0.0.0 --port 5000
```

### 访问仪表板

打开浏览器访问 `http://localhost:5000` 即可使用 Web 界面。

### 仪表板功能

- **实时监控**: 实时显示性能指标
- **历史分析**: 查看历史性能数据
- **内存分析**: 内存使用情况和泄漏检测
- **热点分析**: 性能瓶颈识别
- **自动调优**: 参数优化结果
- **报告生成**: 导出性能报告
- **数据可视化**: 图表和趋势分析

## 最佳实践

### 1. 性能监控最佳实践

```cpp
// 使用性能作用域自动管理
{
    CUOP_PERF_SCOPE("my_function");
    // 函数代码
    // 作用域结束时自动记录性能
}

// 避免过度监控
if (enable_detailed_monitoring) {
    CUOP_PERF_START("detailed_operation");
    // 详细操作
    CUOP_PERF_END("detailed_operation");
}
```

### 2. 内存分析最佳实践

```cpp
// 在程序开始时启动内存跟踪
int main() {
    MemoryAnalyzer::Instance().StartTracking();
    
    // 程序结束时检查内存泄漏
    atexit([]() {
        auto leaks = MemoryAnalyzer::Instance().DetectLeaks();
        if (!leaks.empty()) {
            std::cerr << "检测到内存泄漏!" << std::endl;
        }
    });
    
    // ... 程序逻辑 ...
}
```

### 3. 自动调优最佳实践

```cpp
// 定义合理的参数范围
ParameterDefinition param;
param.name = "block_size";
param.values = {"32", "64", "128", "256", "512"}; // 合理的范围
param.default_value = "128";
param.is_tunable = true;

// 使用合适的调优策略
tuner.SetTuningStrategy("BayesianOptimization"); // 对于复杂参数空间
// 或
tuner.SetTuningStrategy("GridSearch"); // 对于小参数空间
```

### 4. 性能优化建议

1. **定期分析**: 定期运行性能分析，识别性能回归
2. **内存管理**: 使用内存分析器检测内存泄漏和低效使用
3. **热点优化**: 专注于优化占用时间最多的函数
4. **自动调优**: 使用自动调优器找到最佳参数配置
5. **监控集成**: 将性能监控集成到 CI/CD 流程中

## 故障排除

### 常见问题

1. **监控开销过高**
   - 减少监控频率
   - 使用采样监控
   - 避免在关键路径上监控

2. **内存泄漏误报**
   - 检查内存分配/释放配对
   - 使用内存池减少分配次数
   - 验证第三方库的内存管理

3. **自动调优不收敛**
   - 调整收敛阈值
   - 增加最大迭代次数
   - 尝试不同的调优策略

4. **仪表板无法访问**
   - 检查端口是否被占用
   - 验证防火墙设置
   - 确认依赖包已正确安装

### 调试技巧

```cpp
// 启用详细日志
FLAGS_v = 2; // 设置 glog 详细级别

// 导出调试数据
monitor.ExportToCSV("debug_performance.csv");
analyzer.ExportToJSON("debug_memory.json");

// 检查监控状态
if (!monitor.IsMonitoring()) {
    std::cerr << "性能监控未启动!" << std::endl;
}
```

## 性能基准

### 监控开销

- **事件记录**: < 1μs 开销
- **内存跟踪**: < 0.5μs 开销
- **热点分析**: < 2μs 开销
- **自动调优**: 取决于参数空间大小

### 内存使用

- **性能监控器**: ~1MB 基础内存
- **内存分析器**: ~2MB 基础内存
- **热点分析器**: ~1MB 基础内存
- **自动调优器**: ~500KB 基础内存

## 总结

cuOP 性能监控系统提供了全面的性能分析和优化工具，帮助您：

1. **识别性能瓶颈**: 通过热点分析找到优化重点
2. **检测内存问题**: 及时发现内存泄漏和低效使用
3. **自动优化参数**: 使用机器学习方法找到最佳配置
4. **实时监控**: 通过 Web 仪表板实时查看性能状态
5. **生成报告**: 导出详细的性能分析报告

通过合理使用这些工具，您可以显著提升 CUDA 应用程序的性能和稳定性。
