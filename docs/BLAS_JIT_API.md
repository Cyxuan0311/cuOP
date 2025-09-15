# BLAS JIT API 文档

## 概述

cuOP的BLAS JIT系统提供了完整的BLAS算子支持，包括矩阵运算、向量运算和批量运算。所有算子都支持JIT实时编译优化，能够根据硬件特性和数据规模自动选择最优的内核实现。

## 支持的算子

### 1. 矩阵运算算子

#### GEMM (General Matrix Multiply)
- **功能**: 计算 `C = α * op(A) * op(B) + β * C`
- **支持的内核类型**: `basic`, `tiled`, `tensor_core`
- **特性**: 支持转置、标量缩放、Tensor Core加速

#### Batched GEMM
- **功能**: 批量矩阵乘法运算
- **支持的内核类型**: `standard`, `optimized`, `tensor_core`
- **特性**: 支持不同矩阵尺寸的批量处理

#### SYMM (Symmetric Matrix Multiply)
- **功能**: 对称矩阵乘法 `C = α * A * B + β * C` (A为对称矩阵)
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持上/下三角模式

#### HERK (Hermitian Rank-k Update)
- **功能**: Hermitian秩-k更新 `C = α * A * A^H + β * C`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持复数运算

#### TRMM (Triangular Matrix-Matrix Multiply)
- **功能**: 三角矩阵乘法 `B = α * op(A) * B` 或 `B = α * B * op(A)`
- **支持的内核类型**: `basic`, `tiled`, `warp_optimized`, `blocked`
- **特性**: 支持左/右侧模式，上/下三角模式

#### GER (General Rank-1 Update)
- **功能**: 外积运算 `A = α * x * y^T + A`
- **支持的内核类型**: `basic`, `tiled`, `warp_optimized`, `shared_memory`
- **特性**: 支持向量增量参数

### 2. 向量运算算子

#### DOT (Dot Product)
- **功能**: 向量点积 `result = x^T * y`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量，共享内存归约

#### AXPY (Scaled Vector Addition)
- **功能**: 向量缩放加法 `y = α * x + y`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量

#### SCAL (Vector Scale)
- **功能**: 向量缩放 `x = α * x`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量

#### COPY (Vector Copy)
- **功能**: 向量复制 `y = x`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量

#### SWAP (Vector Swap)
- **功能**: 向量交换 `x ↔ y`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量

#### ROT (Givens Rotation)
- **功能**: Givens旋转变换
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 支持向量增量

#### NRM2 (Vector 2-Norm)
- **功能**: 向量2范数 `result = ||x||_2`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 共享内存归约

#### ASUM (Vector 1-Norm)
- **功能**: 向量1范数 `result = ||x||_1`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 共享内存归约

#### IAMAX (Index of Maximum)
- **功能**: 最大元素索引 `result = argmax_i |x_i|`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 共享内存归约

#### IAMIN (Index of Minimum)
- **功能**: 最小元素索引 `result = argmin_i |x_i|`
- **支持的内核类型**: `basic`, `optimized`
- **特性**: 共享内存归约

## API 使用指南

### 1. 基本使用流程

```cpp
#include "jit/Blas/blas_jit_plugins.hpp"

// 1. 注册所有BLAS插件
RegisterBlasJITPlugins();

// 2. 创建插件实例
auto plugin = CreateBlasJITPlugin("gemm");

// 3. 初始化插件
plugin->Initialize();

// 4. 配置JIT参数
JITConfig config;
config.kernel_type = "tiled";
config.use_tensor_core = true;
plugin->SetConfig(config);

// 5. 编译内核
plugin->Compile(config);

// 6. 准备数据
std::vector<Tensor<float>> inputs = {A, B};
std::vector<Tensor<float>> outputs = {C};

// 7. 执行计算
StatusCode status = plugin->Execute(inputs, outputs);
```

### 2. 高级配置

```cpp
// 启用自动调优
plugin->EnableAutoTuning(true);

// 获取性能配置
PerformanceProfile profile = plugin->GetPerformanceProfile();
std::cout << "GFLOPS: " << profile.gflops << std::endl;
std::cout << "内存带宽: " << profile.bandwidth_gb_s << " GB/s" << std::endl;

// 优化配置
plugin->Optimize(profile);
```

### 3. 批量运算示例

```cpp
// 创建批量GEMM插件
auto batched_plugin = CreateBlasJITPlugin("gemm_batched");
auto* bg_plugin = dynamic_cast<GemmBatchedJITPlugin*>(batched_plugin.get());

// 配置批量参数
bg_plugin->SetBatchSize(4);
bg_plugin->SetMatrixDimensions(64, 64, 32);
bg_plugin->SetTransposeOptions(false, false);
bg_plugin->SetAlphaBeta(1.0f, 0.0f);

// 准备批量数据
std::vector<Tensor<float>> inputs, outputs;
for (int i = 0; i < 4; ++i) {
    Tensor<float> A({64, 32}); A.fill(1.0f + i);
    Tensor<float> B({32, 64}); B.fill(0.5f + i);
    Tensor<float> C({64, 64}); C.fill(0.0f);
    inputs.push_back(std::move(A));
    inputs.push_back(std::move(B));
    outputs.push_back(std::move(C));
}

// 执行批量计算
StatusCode status = bg_plugin->Execute(inputs, outputs);
```

### 4. 向量运算示例

```cpp
// 创建向量运算插件
auto vector_plugin = CreateBlasJITPlugin("dot");
auto* vo_plugin = dynamic_cast<VectorOpsJITPlugin*>(vector_plugin.get());

// 配置向量参数
vo_plugin->SetOperationType(VectorOpType::DOT);
vo_plugin->SetVectorSize(1024);
vo_plugin->SetAlpha(1.0f);
vo_plugin->SetIncrement(1, 1);

// 准备数据
Tensor<float> X({1024}); X.fill(1.0f);
Tensor<float> Y({1024}); Y.fill(2.0f);
Tensor<float> Result({1}); // 标量结果

std::vector<Tensor<float>> inputs = {X, Y};
std::vector<Tensor<float>> outputs = {Result};

// 执行计算
StatusCode status = vo_plugin->Execute(inputs, outputs);
```

## 性能优化建议

### 1. 内核类型选择

- **小矩阵 (< 64x64)**: 使用 `basic` 内核
- **中等矩阵 (64x64 - 256x256)**: 使用 `tiled` 内核
- **大矩阵 (> 256x256)**: 使用 `tensor_core` 内核（如果支持）

### 2. 内存访问优化

- 确保数据在GPU内存中连续存储
- 使用适当的数据类型（float32推荐）
- 避免频繁的内存分配和释放

### 3. 批量处理

- 对于多个相同大小的矩阵运算，使用批量算子
- 批量大小建议为2的幂次（2, 4, 8, 16等）

### 4. 自动调优

- 启用自动调优以获得最佳性能
- 让系统自动选择最优的内核配置
- 定期检查性能配置并优化

## 错误处理

```cpp
StatusCode status = plugin->Execute(inputs, outputs);
if (status != StatusCode::SUCCESS) {
    std::cerr << "执行失败: " << plugin->GetLastError() << std::endl;
    
    // 根据错误类型进行处理
    switch (status) {
        case StatusCode::JIT_NOT_INITIALIZED:
            plugin->Initialize();
            break;
        case StatusCode::JIT_NOT_COMPILED:
            plugin->Compile(config);
            break;
        case StatusCode::INVALID_ARGUMENT:
            // 检查输入参数
            break;
        default:
            // 其他错误处理
            break;
    }
}
```

## 最佳实践

1. **初始化顺序**: 先注册插件，再创建实例，最后初始化
2. **配置管理**: 在编译前设置所有必要的配置参数
3. **错误处理**: 始终检查返回的状态码
4. **性能监控**: 定期检查性能配置并优化
5. **内存管理**: 使用cuOP的内存池管理GPU内存
6. **批量处理**: 尽可能使用批量算子提高效率

## 相关文档

- [JIT系统概述](../jit_system_overview.md)
- [性能监控指南](../performance_monitoring_guide.md)
- [内存管理指南](../memory_management_guide.md)
- [错误码系统](../error_handling_guide.md)
