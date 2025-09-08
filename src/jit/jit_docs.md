# cuOP JIT系统使用指南

## 概述

cuOP JIT（Just-In-Time）系统是一个智能的CUDA内核实时编译和优化框架，旨在为cuOP算子库提供高性能的运行时优化能力。该系统采用包装器模式设计，确保对现有代码的零侵入性，同时提供强大的JIT优化功能。

## 核心特性

### 🚀 **高性能优化**
- **多级内核架构**: 基础、瓦片优化、Warp优化、块优化、Tensor Core内核
- **智能内核选择**: 根据矩阵维度、数据类型、硬件能力动态选择最优内核
- **自动调优**: 运行时自动寻找最佳配置参数
- **硬件特性利用**: 充分利用Tensor Core、TMA等最新硬件特性

### 🔧 **零侵入性设计**
- **包装器模式**: 现有算子代码完全不变
- **渐进式集成**: 可选择启用或禁用JIT功能
- **向后兼容**: 100%兼容现有接口

### 📊 **智能缓存系统**
- **多级缓存**: 线程本地、全局、预分配缓存
- **智能缓存管理**: 自动清理过期缓存，内存碎片整理
- **持久化缓存**: 支持缓存保存和加载

### 🎛️ **灵活配置**
- **全局配置**: 系统级JIT开关和参数
- **算子配置**: 针对特定算子的优化参数
- **运行时配置**: 支持动态调整配置

## 系统架构

```
cuOP算子库 + JIT系统
├── 现有算子层 (Existing Layer) - 保持不变
│   ├── Gemm, Conv, Pool等现有算子
│   └── 现有接口和实现完全不变
├── JIT包装器层 (JIT Wrapper Layer) - 新增
│   ├── JITWrapper<T> (通用包装器)
│   ├── JITCompiler (实时编译器)
│   └── JITCache (智能缓存)
├── 插件管理层 (Plugin Management Layer) - 新增
│   ├── OperatorPluginManager (插件管理器)
│   ├── AutoTuner (自动调优器)
│   └── ConfigOptimizer (配置优化器)
└── 用户接口层 (User Interface Layer) - 扩展
    ├── 现有API (100%兼容)
    ├── JIT API (新增，可选)
    └── 配置API (新增，可选)
```

## 快速开始

### 1. 基本使用

```cpp
#include "jit/jit_wrapper.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"

// 创建原始算子
Gemm<float> gemm;
gemm.SetWeight(weight);

// 创建JIT包装器
JITWrapper<Gemm<float>> jit_gemm(gemm);

// 启用JIT
jit_gemm.EnableJIT(true);

// 使用方式与原始算子完全相同
jit_gemm.Forward(input, output);
```

### 2. 配置使用

```cpp
// 全局配置
GlobalJITConfig global_config;
global_config.enable_jit = true;
global_config.enable_auto_tuning = true;
global_config.cache_dir = "./jit_cache";
GlobalJITManager::Instance().SetGlobalConfig(global_config);

// 算子配置
JITConfig gemm_config;
gemm_config.block_sizes = {32, 64, 128};
gemm_config.tile_sizes = {32, 64};
gemm_config.use_tensor_core = true;
gemm_config.optimization_level = "auto";

JITWrapper<Gemm<float>> jit_gemm(gemm);
jit_gemm.SetJITConfig(gemm_config);
jit_gemm.EnableJIT(true);
jit_gemm.EnableAutoTuning(true);
```

### 3. 高级使用

```cpp
// 批量配置
std::vector<JITWrapper<Gemm<float>>> jit_gemms;
for (auto& gemm : gemms) {
    jit_gemms.emplace_back(gemm);
    jit_gemms.back().EnableAutoTuning(true);
}

// 并行执行
#pragma omp parallel for
for (int i = 0; i < jit_gemms.size(); ++i) {
    jit_gemms[i].Forward(inputs[i], outputs[i]);
}

// 性能分析
for (auto& jit_gemm : jit_gemms) {
    auto profile = jit_gemm.GetPerformanceProfile();
    std::cout << "Performance: " << profile.gflops << " GFLOPS" << std::endl;
}
```

## 配置参数详解

### 全局配置 (GlobalJITConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_jit` | bool | true | 全局JIT开关 |
| `enable_auto_tuning` | bool | true | 自动调优开关 |
| `enable_caching` | bool | true | 缓存开关 |
| `cache_dir` | string | "./jit_cache" | 缓存目录 |
| `max_cache_size` | int | 1GB | 最大缓存大小 |
| `compilation_timeout` | int | 30 | 编译超时时间(秒) |
| `enable_tensor_core` | bool | true | Tensor Core开关 |
| `enable_tma` | bool | true | TMA开关 |
| `max_compilation_threads` | int | 4 | 最大编译线程数 |
| `enable_debug` | bool | false | 调试模式 |

### 算子配置 (JITConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_jit` | bool | true | 算子JIT开关 |
| `block_sizes` | vector<int> | {16,32,64,128} | 块大小选项 |
| `tile_sizes` | vector<int> | {16,32,64} | 瓦片大小选项 |
| `num_stages` | int | 2 | 流水线阶段数 |
| `use_tensor_core` | bool | true | 使用Tensor Core |
| `use_tma` | bool | true | 使用TMA |
| `optimization_level` | string | "auto" | 优化级别 |
| `max_registers` | int | 255 | 最大寄存器数 |
| `enable_shared_memory_opt` | bool | true | 共享内存优化 |
| `enable_loop_unroll` | bool | true | 循环展开 |
| `enable_memory_coalescing` | bool | true | 内存合并优化 |

## 性能优化策略

### 1. 内核选择策略

系统会根据以下因素自动选择最优内核：

- **矩阵维度**: M、N、K的大小
- **数据类型**: FP32、FP16、INT8等
- **硬件能力**: 计算能力、SM数量、内存带宽
- **特殊要求**: 转置、批量处理等

### 2. 自动调优

自动调优系统会：

1. **参数空间探索**: 测试不同的block大小、tile大小等
2. **性能测量**: 测量每种配置的执行时间
3. **最优选择**: 选择性能最佳的配置
4. **缓存结果**: 将最优配置缓存起来

### 3. 内存优化

- **共享内存优化**: 减少全局内存访问
- **内存合并**: 优化内存访问模式
- **流水线优化**: 重叠计算和内存访问

## 监控和调试

### 1. 性能分析

```cpp
// 获取性能分析信息
auto profile = jit_gemm.GetPerformanceProfile();
std::cout << "GFLOPS: " << profile.gflops << std::endl;
std::cout << "内存带宽: " << profile.bandwidth_gb_s << " GB/s" << std::endl;
std::cout << "Kernel时间: " << profile.kernel_time_ms << " ms" << std::endl;
std::cout << "使用的Block大小: " << profile.block_size_x << "x" << profile.block_size_y << std::endl;
std::cout << "使用Tensor Core: " << (profile.used_tensor_core ? "是" : "否") << std::endl;
```

### 2. 统计信息

```cpp
// 获取全局统计信息
auto stats = GlobalJITManager::Instance().GetStatistics();
std::cout << "总编译次数: " << stats.total_compilations << std::endl;
std::cout << "缓存命中率: " << (stats.GetCacheHitRate() * 100) << "%" << std::endl;
std::cout << "平均编译时间: " << stats.GetAverageCompilationTime() << " s" << std::endl;
std::cout << "活跃kernel数量: " << stats.active_kernels << std::endl;
```

### 3. 调试模式

```cpp
// 启用调试模式
GlobalJITConfig config;
config.enable_debug = true;
GlobalJITManager::Instance().SetGlobalConfig(config);

// 查看编译错误
auto errors = compiler.GetCompileErrors();
for (const auto& error : errors) {
    std::cout << "编译错误: " << error.error_message << std::endl;
}
```

## 最佳实践

### 1. 初始化建议

```cpp
// 在程序开始时初始化
auto& global_manager = GlobalJITManager::Instance();
global_manager.Initialize();

// 设置全局配置
GlobalJITConfig config;
config.enable_jit = true;
config.enable_auto_tuning = true;
config.cache_dir = "./jit_cache";
global_manager.SetGlobalConfig(config);
```

### 2. 算子配置建议

```cpp
// 根据具体应用场景配置
JITConfig config;

// 小矩阵优化
if (matrix_size < 512) {
    config.block_sizes = {16, 32};
    config.tile_sizes = {16};
    config.optimization_level = "fast";
}

// 大矩阵优化
else {
    config.block_sizes = {64, 128, 256};
    config.tile_sizes = {32, 64};
    config.optimization_level = "best";
    config.use_tensor_core = true;
}
```

### 3. 内存管理建议

```cpp
// 定期清理缓存
GlobalJITManager::Instance().ClearAllCaches();

// 保存缓存到文件
GlobalJITManager::Instance().SaveCacheToFile("./jit_cache.bin");

// 从文件加载缓存
GlobalJITManager::Instance().LoadCacheFromFile("./jit_cache.bin");
```

### 4. 性能监控建议

```cpp
// 定期检查性能
auto profile = jit_gemm.GetPerformanceProfile();
if (profile.gflops < expected_gflops) {
    // 重新调优
    jit_gemm.EnableAutoTuning(true);
    jit_gemm.OptimizeJIT();
}
```

## 故障排除

### 1. 常见问题

**Q: JIT编译失败怎么办？**
A: 系统会自动回退到原始算子执行，检查编译错误日志。

**Q: 性能没有提升怎么办？**
A: 检查配置参数，启用自动调优，分析性能瓶颈。

**Q: 内存使用过多怎么办？**
A: 调整缓存大小，定期清理缓存，使用缓存持久化。

### 2. 调试技巧

```cpp
// 启用详细日志
google::SetStderrLogging(google::INFO);

// 检查JIT状态
if (!jit_gemm.IsJITCompiled()) {
    std::cout << "JIT未编译，错误: " << jit_gemm.GetLastError() << std::endl;
}

// 检查硬件支持
auto hw_spec = GetHardwareSpec();
std::cout << "计算能力: " << hw_spec.GetComputeCapabilityString() << std::endl;
std::cout << "支持Tensor Core: " << (hw_spec.supports_tensor_core ? "是" : "否") << std::endl;
```

## 扩展开发

### 1. 添加新算子支持

```cpp
// 1. 创建算子插件
class MyOperatorJITPlugin : public IJITPlugin {
    // 实现接口方法
};

// 2. 注册插件
REGISTER_JIT_PLUGIN("my_operator_jit", MyOperatorJITPlugin);

// 3. 创建包装器
JITWrapper<MyOperator<float>> jit_my_op(my_op);
```

### 2. 自定义内核模板

```cpp
// 1. 创建内核模板
class MyKernelTemplate : public IKernelTemplate {
    std::string GenerateKernelCode(const JITConfig& config) override {
        // 生成内核代码
    }
};

// 2. 注册模板
REGISTER_KERNEL_TEMPLATE("my_kernel", MyKernelTemplate);
```

## 总结

cuOP JIT系统提供了一个强大而灵活的运行时优化框架，通过智能的包装器设计，既保持了现有代码的稳定性，又提供了显著的性能提升。通过合理配置和最佳实践，可以充分发挥JIT系统的优化潜力，为您的CUDA算子库带来更好的性能表现。 