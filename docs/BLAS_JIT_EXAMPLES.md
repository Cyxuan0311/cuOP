# BLAS JIT 使用示例

本文档提供了cuOP BLAS JIT系统的详细使用示例，涵盖各种算子的基本用法和高级特性。

## 目录

- [基础示例](#基础示例)
- [批量运算示例](#批量运算示例)
- [向量运算示例](#向量运算示例)
- [高级配置示例](#高级配置示例)
- [性能优化示例](#性能优化示例)
- [错误处理示例](#错误处理示例)

## 基础示例

### 1. GEMM 矩阵乘法

```cpp
#include "jit/Blas/blas_jit_plugins.hpp"
#include "data/tensor.hpp"
#include <iostream>

void gemm_example() {
    std::cout << "=== GEMM 示例 ===" << std::endl;
    
    // 1. 注册插件
    RegisterBlasJITPlugins();
    
    // 2. 创建GEMM插件
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    if (!gemm_plugin) {
        std::cerr << "创建GEMM插件失败" << std::endl;
        return;
    }
    
    // 3. 初始化
    StatusCode status = gemm_plugin->Initialize();
    if (status != StatusCode::SUCCESS) {
        std::cerr << "初始化失败: " << gemm_plugin->GetLastError() << std::endl;
        return;
    }
    
    // 4. 配置JIT参数
    JITConfig config;
    config.kernel_type = "tiled";
    config.use_tensor_core = true;
    config.block_size_x = 16;
    config.block_size_y = 16;
    gemm_plugin->SetConfig(config);
    
    // 5. 编译内核
    status = gemm_plugin->Compile(config);
    if (status != StatusCode::SUCCESS) {
        std::cerr << "编译失败: " << gemm_plugin->GetLastError() << std::endl;
        return;
    }
    
    // 6. 准备数据
    int m = 512, n = 512, k = 512;
    Tensor<float> A({m, k}); A.fill(1.0f);
    Tensor<float> B({k, n}); B.fill(2.0f);
    Tensor<float> C({m, n}); C.fill(0.0f);
    
    std::vector<Tensor<float>> inputs = {A, B};
    std::vector<Tensor<float>> outputs = {C};
    
    // 7. 执行计算
    status = gemm_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "GEMM计算成功完成" << std::endl;
        
        // 获取性能信息
        PerformanceProfile profile = gemm_plugin->GetPerformanceProfile();
        std::cout << "执行时间: " << profile.execution_time << " ms" << std::endl;
        std::cout << "GFLOPS: " << profile.gflops << std::endl;
        std::cout << "内存带宽: " << profile.bandwidth_gb_s << " GB/s" << std::endl;
    } else {
        std::cerr << "执行失败: " << gemm_plugin->GetLastError() << std::endl;
    }
}
```

### 2. GEMV 矩阵向量乘法

```cpp
void gemv_example() {
    std::cout << "=== GEMV 示例 ===" << std::endl;
    
    // 创建GEMV插件
    auto gemv_plugin = CreateBlasJITPlugin("gemv");
    gemv_plugin->Initialize();
    
    // 配置参数
    JITConfig config;
    config.kernel_type = "optimized";
    gemv_plugin->SetConfig(config);
    gemv_plugin->Compile(config);
    
    // 准备数据
    int m = 1024, n = 1024;
    Tensor<float> A({m, n}); A.fill(1.0f);
    Tensor<float> x({n}); x.fill(2.0f);
    Tensor<float> y({m}); y.fill(0.0f);
    
    std::vector<Tensor<float>> inputs = {A, x};
    std::vector<Tensor<float>> outputs = {y};
    
    // 执行计算
    StatusCode status = gemv_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "GEMV计算成功完成" << std::endl;
    }
}
```

## 批量运算示例

### 1. Batched GEMM

```cpp
void batched_gemm_example() {
    std::cout << "=== Batched GEMM 示例 ===" << std::endl;
    
    // 创建批量GEMM插件
    auto batched_plugin = CreateBlasJITPlugin("gemm_batched");
    auto* bg_plugin = dynamic_cast<GemmBatchedJITPlugin*>(batched_plugin.get());
    
    if (!bg_plugin) {
        std::cerr << "类型转换失败" << std::endl;
        return;
    }
    
    // 初始化
    batched_plugin->Initialize();
    
    // 配置批量参数
    int batch_size = 4;
    int m = 64, n = 64, k = 32;
    bg_plugin->SetBatchSize(batch_size);
    bg_plugin->SetMatrixDimensions(m, n, k);
    bg_plugin->SetTransposeOptions(false, false);
    bg_plugin->SetAlphaBeta(1.0f, 0.0f);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "optimized";
    batched_plugin->SetConfig(config);
    batched_plugin->Compile(config);
    
    // 准备批量数据
    std::vector<Tensor<float>> inputs, outputs;
    for (int i = 0; i < batch_size; ++i) {
        Tensor<float> A({m, k}); A.fill(1.0f + i * 0.1f);
        Tensor<float> B({k, n}); B.fill(0.5f + i * 0.1f);
        Tensor<float> C({m, n}); C.fill(0.0f);
        
        inputs.push_back(std::move(A));
        inputs.push_back(std::move(B));
        outputs.push_back(std::move(C));
    }
    
    // 执行批量计算
    StatusCode status = bg_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "批量GEMM计算成功完成" << std::endl;
        std::cout << "处理了 " << batch_size << " 个矩阵对" << std::endl;
    }
}
```

### 2. 批量向量运算

```cpp
void batch_vector_ops_example() {
    std::cout << "=== 批量向量运算示例 ===" << std::endl;
    
    // 创建向量运算插件
    auto vector_plugin = CreateBlasJITPlugin("axpy");
    auto* vo_plugin = dynamic_cast<VectorOpsJITPlugin*>(vector_plugin.get());
    
    vector_plugin->Initialize();
    
    // 配置向量参数
    int vector_size = 1024;
    vo_plugin->SetOperationType(VectorOpType::AXPY);
    vo_plugin->SetVectorSize(vector_size);
    vo_plugin->SetAlpha(2.0f);
    vo_plugin->SetIncrement(1, 1);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "optimized";
    vector_plugin->SetConfig(config);
    vector_plugin->Compile(config);
    
    // 准备数据
    Tensor<float> X({vector_size}); X.fill(1.0f);
    Tensor<float> Y({vector_size}); Y.fill(2.0f);
    
    std::vector<Tensor<float>> inputs = {X, Y};
    std::vector<Tensor<float>> outputs = {Y}; // Y被更新
    
    // 执行计算
    StatusCode status = vo_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "AXPY计算成功完成" << std::endl;
    }
}
```

## 向量运算示例

### 1. 向量点积 (DOT)

```cpp
void dot_product_example() {
    std::cout << "=== 向量点积示例 ===" << std::endl;
    
    auto vector_plugin = CreateBlasJITPlugin("dot");
    auto* vo_plugin = dynamic_cast<VectorOpsJITPlugin*>(vector_plugin.get());
    
    vector_plugin->Initialize();
    
    // 配置参数
    int vector_size = 1024;
    vo_plugin->SetOperationType(VectorOpType::DOT);
    vo_plugin->SetVectorSize(vector_size);
    vo_plugin->SetAlpha(1.0f);
    vo_plugin->SetIncrement(1, 1);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "optimized";
    vector_plugin->SetConfig(config);
    vector_plugin->Compile(config);
    
    // 准备数据
    Tensor<float> X({vector_size}); X.fill(1.0f);
    Tensor<float> Y({vector_size}); Y.fill(2.0f);
    Tensor<float> Result({1}); // 标量结果
    
    std::vector<Tensor<float>> inputs = {X, Y};
    std::vector<Tensor<float>> outputs = {Result};
    
    // 执行计算
    StatusCode status = vo_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "点积计算成功完成" << std::endl;
        // 注意：这里需要从GPU内存读取结果
        // float dot_result = Result.to_host()[0];
        // std::cout << "点积结果: " << dot_result << std::endl;
    }
}
```

### 2. 向量范数 (NRM2)

```cpp
void vector_norm_example() {
    std::cout << "=== 向量范数示例 ===" << std::endl;
    
    auto vector_plugin = CreateBlasJITPlugin("nrm2");
    auto* vo_plugin = dynamic_cast<VectorOpsJITPlugin*>(vector_plugin.get());
    
    vector_plugin->Initialize();
    
    // 配置参数
    int vector_size = 1024;
    vo_plugin->SetOperationType(VectorOpType::NRM2);
    vo_plugin->SetVectorSize(vector_size);
    vo_plugin->SetIncrement(1);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "optimized";
    vector_plugin->SetConfig(config);
    vector_plugin->Compile(config);
    
    // 准备数据
    Tensor<float> X({vector_size}); X.fill(3.0f); // 所有元素为3
    Tensor<float> Result({1}); // 标量结果
    
    std::vector<Tensor<float>> inputs = {X};
    std::vector<Tensor<float>> outputs = {Result};
    
    // 执行计算
    StatusCode status = vo_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "向量范数计算成功完成" << std::endl;
        // 预期结果: sqrt(1024 * 3^2) = sqrt(9216) ≈ 96
    }
}
```

### 3. 最大元素索引 (IAMAX)

```cpp
void max_element_index_example() {
    std::cout << "=== 最大元素索引示例 ===" << std::endl;
    
    auto vector_plugin = CreateBlasJITPlugin("iamax");
    auto* vo_plugin = dynamic_cast<VectorOpsJITPlugin*>(vector_plugin.get());
    
    vector_plugin->Initialize();
    
    // 配置参数
    int vector_size = 1024;
    vo_plugin->SetOperationType(VectorOpType::IAMAX);
    vo_plugin->SetVectorSize(vector_size);
    vo_plugin->SetIncrement(1);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "optimized";
    vector_plugin->SetConfig(config);
    vector_plugin->Compile(config);
    
    // 准备数据
    Tensor<float> X({vector_size}); 
    // 设置第500个元素为最大值
    for (int i = 0; i < vector_size; ++i) {
        X.fill(i == 500 ? 100.0f : 1.0f);
    }
    Tensor<float> Result({1}); // 索引结果
    
    std::vector<Tensor<float>> inputs = {X};
    std::vector<Tensor<float>> outputs = {Result};
    
    // 执行计算
    StatusCode status = vo_plugin->Execute(inputs, outputs);
    if (status == StatusCode::SUCCESS) {
        std::cout << "最大元素索引计算成功完成" << std::endl;
        // 预期结果: 500
    }
}
```

## 高级配置示例

### 1. 自动调优

```cpp
void auto_tuning_example() {
    std::cout << "=== 自动调优示例 ===" << std::endl;
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    gemm_plugin->Initialize();
    
    // 启用自动调优
    gemm_plugin->EnableAutoTuning(true);
    
    // 配置JIT参数
    JITConfig config;
    config.kernel_type = "auto"; // 让系统自动选择
    config.use_tensor_core = true;
    gemm_plugin->SetConfig(config);
    gemm_plugin->Compile(config);
    
    // 准备数据
    int m = 1024, n = 1024, k = 1024;
    Tensor<float> A({m, k}); A.fill(1.0f);
    Tensor<float> B({k, n}); B.fill(2.0f);
    Tensor<float> C({m, n}); C.fill(0.0f);
    
    std::vector<Tensor<float>> inputs = {A, B};
    std::vector<Tensor<float>> outputs = {C};
    
    // 多次执行以触发自动调优
    for (int i = 0; i < 10; ++i) {
        StatusCode status = gemm_plugin->Execute(inputs, outputs);
        if (status == StatusCode::SUCCESS) {
            PerformanceProfile profile = gemm_plugin->GetPerformanceProfile();
            std::cout << "第 " << (i+1) << " 次执行: " 
                      << profile.execution_time << " ms, "
                      << profile.gflops << " GFLOPS" << std::endl;
        }
    }
    
    // 获取最终优化后的性能配置
    PerformanceProfile final_profile = gemm_plugin->GetPerformanceProfile();
    std::cout << "最终性能: " << final_profile.gflops << " GFLOPS" << std::endl;
}
```

### 2. 自定义内核配置

```cpp
void custom_kernel_config_example() {
    std::cout << "=== 自定义内核配置示例 ===" << std::endl;
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    gemm_plugin->Initialize();
    
    // 自定义JIT配置
    JITConfig config;
    config.kernel_type = "tiled";
    config.block_size_x = 32;
    config.block_size_y = 32;
    config.tile_size = 16;
    config.use_tensor_core = true;
    config.optimization_level = "O3";
    config.enable_shared_memory_opt = true;
    config.enable_loop_unroll = true;
    config.enable_memory_coalescing = true;
    
    gemm_plugin->SetConfig(config);
    
    // 编译内核
    StatusCode status = gemm_plugin->Compile(config);
    if (status != StatusCode::SUCCESS) {
        std::cerr << "编译失败: " << gemm_plugin->GetLastError() << std::endl;
        return;
    }
    
    std::cout << "自定义内核配置编译成功" << std::endl;
}
```

## 性能优化示例

### 1. 批量处理优化

```cpp
void batch_processing_optimization() {
    std::cout << "=== 批量处理优化示例 ===" << std::endl;
    
    // 比较单个处理和批量处理的性能
    int batch_size = 8;
    int matrix_size = 256;
    
    // 单个处理
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    gemm_plugin->Initialize();
    
    JITConfig config;
    config.kernel_type = "tiled";
    gemm_plugin->SetConfig(config);
    gemm_plugin->Compile(config);
    
    for (int i = 0; i < batch_size; ++i) {
        Tensor<float> A({matrix_size, matrix_size}); A.fill(1.0f);
        Tensor<float> B({matrix_size, matrix_size}); B.fill(2.0f);
        Tensor<float> C({matrix_size, matrix_size}); C.fill(0.0f);
        
        std::vector<Tensor<float>> inputs = {A, B};
        std::vector<Tensor<float>> outputs = {C};
        
        gemm_plugin->Execute(inputs, outputs);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto single_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 批量处理
    start_time = std::chrono::high_resolution_clock::now();
    
    auto batched_plugin = CreateBlasJITPlugin("gemm_batched");
    auto* bg_plugin = dynamic_cast<GemmBatchedJITPlugin*>(batched_plugin.get());
    
    batched_plugin->Initialize();
    bg_plugin->SetBatchSize(batch_size);
    bg_plugin->SetMatrixDimensions(matrix_size, matrix_size, matrix_size);
    
    batched_plugin->SetConfig(config);
    batched_plugin->Compile(config);
    
    std::vector<Tensor<float>> inputs, outputs;
    for (int i = 0; i < batch_size; ++i) {
        Tensor<float> A({matrix_size, matrix_size}); A.fill(1.0f);
        Tensor<float> B({matrix_size, matrix_size}); B.fill(2.0f);
        Tensor<float> C({matrix_size, matrix_size}); C.fill(0.0f);
        
        inputs.push_back(std::move(A));
        inputs.push_back(std::move(B));
        outputs.push_back(std::move(C));
    }
    
    bg_plugin->Execute(inputs, outputs);
    
    end_time = std::chrono::high_resolution_clock::now();
    auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "单个处理时间: " << single_time.count() << " ms" << std::endl;
    std::cout << "批量处理时间: " << batch_time.count() << " ms" << std::endl;
    std::cout << "加速比: " << (double)single_time.count() / batch_time.count() << "x" << std::endl;
}
```

### 2. 内存访问优化

```cpp
void memory_access_optimization() {
    std::cout << "=== 内存访问优化示例 ===" << std::endl;
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    gemm_plugin->Initialize();
    
    // 测试不同矩阵大小的性能
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    
    for (int size : sizes) {
        // 准备数据
        Tensor<float> A({size, size}); A.fill(1.0f);
        Tensor<float> B({size, size}); B.fill(2.0f);
        Tensor<float> C({size, size}); C.fill(0.0f);
        
        std::vector<Tensor<float>> inputs = {A, B};
        std::vector<Tensor<float>> outputs = {C};
        
        // 配置适合的内核类型
        JITConfig config;
        if (size < 256) {
            config.kernel_type = "basic";
        } else if (size < 1024) {
            config.kernel_type = "tiled";
        } else {
            config.kernel_type = "tensor_core";
        }
        
        gemm_plugin->SetConfig(config);
        gemm_plugin->Compile(config);
        
        // 执行计算
        auto start_time = std::chrono::high_resolution_clock::now();
        StatusCode status = gemm_plugin->Execute(inputs, outputs);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            PerformanceProfile profile = gemm_plugin->GetPerformanceProfile();
            
            std::cout << "矩阵大小: " << size << "x" << size 
                      << ", 执行时间: " << duration.count() << " μs"
                      << ", GFLOPS: " << profile.gflops << std::endl;
        }
    }
}
```

## 错误处理示例

### 1. 完整的错误处理

```cpp
void comprehensive_error_handling() {
    std::cout << "=== 完整错误处理示例 ===" << std::endl;
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    
    // 检查插件创建
    if (!gemm_plugin) {
        std::cerr << "插件创建失败" << std::endl;
        return;
    }
    
    // 初始化错误处理
    StatusCode status = gemm_plugin->Initialize();
    if (status != StatusCode::SUCCESS) {
        std::cerr << "初始化失败: " << gemm_plugin->GetLastError() << std::endl;
        return;
    }
    
    // 配置错误处理
    JITConfig config;
    config.kernel_type = "invalid_kernel_type"; // 故意设置无效值
    
    status = gemm_plugin->Compile(config);
    if (status != StatusCode::SUCCESS) {
        std::cerr << "编译失败: " << gemm_plugin->GetLastError() << std::endl;
        
        // 尝试使用默认配置
        config.kernel_type = "basic";
        status = gemm_plugin->Compile(config);
        if (status != StatusCode::SUCCESS) {
            std::cerr << "默认配置编译也失败: " << gemm_plugin->GetLastError() << std::endl;
            return;
        }
    }
    
    // 数据准备错误处理
    try {
        Tensor<float> A({512, 512}); A.fill(1.0f);
        Tensor<float> B({512, 512}); B.fill(2.0f);
        Tensor<float> C({512, 512}); C.fill(0.0f);
        
        std::vector<Tensor<float>> inputs = {A, B};
        std::vector<Tensor<float>> outputs = {C};
        
        // 执行错误处理
        status = gemm_plugin->Execute(inputs, outputs);
        if (status != StatusCode::SUCCESS) {
            std::cerr << "执行失败: " << gemm_plugin->GetLastError() << std::endl;
            
            // 根据错误类型进行不同处理
            switch (status) {
                case StatusCode::JIT_NOT_COMPILED:
                    std::cerr << "内核未编译，尝试重新编译" << std::endl;
                    gemm_plugin->Compile(config);
                    break;
                case StatusCode::INVALID_ARGUMENT:
                    std::cerr << "参数无效，检查输入数据" << std::endl;
                    break;
                case StatusCode::CUDA_LAUNCH_ERROR:
                    std::cerr << "CUDA内核启动失败" << std::endl;
                    break;
                default:
                    std::cerr << "未知错误" << std::endl;
                    break;
            }
        } else {
            std::cout << "计算成功完成" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    }
}
```

### 2. 资源清理

```cpp
void resource_cleanup_example() {
    std::cout << "=== 资源清理示例 ===" << std::endl;
    
    auto gemm_plugin = CreateBlasJITPlugin("gemm");
    gemm_plugin->Initialize();
    
    // 使用RAII确保资源正确清理
    {
        JITConfig config;
        config.kernel_type = "tiled";
        gemm_plugin->SetConfig(config);
        gemm_plugin->Compile(config);
        
        // 执行一些计算
        Tensor<float> A({256, 256}); A.fill(1.0f);
        Tensor<float> B({256, 256}); B.fill(2.0f);
        Tensor<float> C({256, 256}); C.fill(0.0f);
        
        std::vector<Tensor<float>> inputs = {A, B};
        std::vector<Tensor<float>> outputs = {C};
        
        gemm_plugin->Execute(inputs, outputs);
    } // 这里Tensor会自动清理
    
    // 手动清理插件资源
    gemm_plugin->Cleanup();
    
    std::cout << "资源清理完成" << std::endl;
}
```

## 总结

这些示例展示了cuOP BLAS JIT系统的各种使用方式：

1. **基础用法**: 简单的矩阵和向量运算
2. **批量处理**: 高效的批量运算
3. **高级配置**: 自定义内核和自动调优
4. **性能优化**: 批量处理和内存访问优化
5. **错误处理**: 完整的错误处理和资源管理

通过这些示例，您可以快速上手cuOP BLAS JIT系统，并根据具体需求进行定制和优化。
