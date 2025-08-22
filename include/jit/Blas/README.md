# BLAS JIT插件目录

本目录包含所有BLAS（Basic Linear Algebra Subprograms）相关的JIT插件实现。

## 📁 目录结构

```
include/jit/Blas/
├── blas_jit_plugins.hpp      # BLAS插件统一入口
├── gemm_jit_plugin.hpp       # GEMM JIT插件头文件
├── gemv_jit_plugin.hpp       # GEMV JIT插件头文件
└── README.md                 # 本文件

src/jit/Blas/
├── gemm_jit_plugin.cu        # GEMM JIT插件实现
├── gemv_jit_plugin.cu        # GEMV JIT插件实现
└── blas_jit_plugin_manager.cu # BLAS插件管理器
```

## 🔧 支持的算子

### 1. GEMM (General Matrix Multiply)
- **功能**: 计算 `C = α * A * B + β * C`
- **支持的内核类型**:
  - `basic`: 基础实现，支持转置
  - `tiled`: 分块优化版本
  - `warp_optimized`: Warp级优化
  - `tensor_core`: Tensor Core加速
  - `blocked`: 大矩阵分块优化

### 2. GEMV (General Matrix-Vector Multiply)
- **功能**: 计算 `y = α * A * x + β * y`
- **支持的内核类型**:
  - `basic`: 基础实现
  - `optimized`: 循环展开优化

## 🚀 使用方法

### 1. 注册插件
```cpp
#include "jit/Blas/blas_jit_plugins.hpp"

// 注册所有BLAS插件
RegisterBlasJITPlugins();
```

### 2. 创建JIT包装器
```cpp
#include "jit/jit_wrapper.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"

// 创建原始算子
Gemm<float> gemm;
gemm.SetWeight(weight);

// 创建JIT包装器
JITWrapper<Gemm<float>> jit_gemm(gemm);
jit_gemm.EnableJIT(true);
```

### 3. 配置和编译
```cpp
// 配置JIT参数
JITConfig config;
config.kernel_type = "tiled";
config.tile_size = 32;
config.optimization_level = "O2";
config.enable_tensor_core = true;

jit_gemm.SetJITConfig(config);

// 编译JIT内核
jit_gemm.CompileJIT();
```

### 4. 执行
```cpp
// 使用方式与原始算子完全相同
jit_gemm.Forward(input, output);
```

## ⚙️ 配置参数

### 通用配置
- `kernel_type`: 内核类型选择
- `tile_size`: 分块大小
- `block_size`: 线程块大小
- `optimization_level`: 编译优化级别
- `enable_tensor_core`: 是否启用Tensor Core
- `enable_tma`: 是否启用TMA（需要H100+）

### GEMM特定配置
- 支持多种内核类型，根据矩阵大小自动选择
- Tensor Core支持FP16和混合精度
- 自动调优功能

### GEMV特定配置
- 基础内核和优化内核
- 循环展开优化
- 内存访问优化

## 📊 性能特性

### 自动调优
- 基于执行历史自动选择最优内核
- 动态调整配置参数
- 性能监控和统计

### 缓存机制
- 编译结果缓存
- 内核函数缓存
- 配置优化缓存

### 硬件感知
- 自动检测硬件能力
- Tensor Core支持检测
- TMA支持检测

## 🔍 调试和监控

### 性能分析
```cpp
auto profile = jit_gemm.GetPerformanceProfile();
std::cout << "执行时间: " << profile.execution_time << " s" << std::endl;
std::cout << "吞吐量: " << profile.throughput << " GFLOPS" << std::endl;
std::cout << "内核类型: " << profile.kernel_type << std::endl;
```

### 统计信息
```cpp
auto stats = global_manager.GetStatistics();
std::cout << "缓存命中率: " << stats.GetCacheHitRate() * 100 << "%" << std::endl;
std::cout << "平均编译时间: " << stats.GetAverageCompilationTime() << " s" << std::endl;
```

## 🛠️ 扩展开发

### 添加新的BLAS算子

1. **创建插件头文件**
```cpp
// include/jit/Blas/new_op_jit_plugin.hpp
class NewOpJITPlugin : public IJITPlugin {
    // 实现IJITPlugin接口
};
```

2. **实现插件功能**
```cpp
// src/jit/Blas/new_op_jit_plugin.cu
// 实现所有虚函数
```

3. **注册到管理器**
```cpp
// 在blas_jit_plugin_manager.cu中添加
plugin_factories_["new_op"] = []() -> std::unique_ptr<IJITPlugin> {
    return std::make_unique<NewOpJITPlugin>();
};
```

4. **更新索引文件**
```cpp
// 在blas_jit_plugins.hpp中添加
#include "jit/Blas/new_op_jit_plugin.hpp"
```

## 📝 注意事项

1. **内存管理**: 插件会自动管理内核缓存和内存使用
2. **错误处理**: 所有操作都有完整的错误检查和异常处理
3. **线程安全**: 插件管理器是线程安全的
4. **资源清理**: 使用完毕后调用`Cleanup()`释放资源

## 🔗 相关文档

- [JIT系统总览](../jit_docs.md)
- [GEMM算法详解](../../../src/cuda_op/detail/cuBlas/Introduce/gemm.md)
- [性能基准测试](../../../../bench/) 