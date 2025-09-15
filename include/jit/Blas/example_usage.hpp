#pragma once

/**
 * @file example_usage.hpp
 * @brief BLAS JIT插件使用示例
 * 
 * 本文件展示了如何使用新增的BLAS JIT插件，包括：
 * - Batched GEMM
 * - 对称矩阵运算 (SYMM/HERK/SYRK等)
 * - 向量运算 (DOT/AXPY/SCAL等)
 * - TRMM
 * - GER
 */

#include "blas_jit_plugins.hpp"
#include "data/tensor.hpp"
#include <iostream>
#include <vector>
#include <chrono>

namespace cu_op_mem {
namespace examples {

/**
 * @brief Batched GEMM使用示例
 */
void BatchedGemmExample() {
    std::cout << "=== Batched GEMM 示例 ===" << std::endl;
    
    // 创建Batched GEMM插件
    auto plugin = CreateBlasJITPlugin("gemm_batched");
    if (!plugin) {
        std::cerr << "无法创建Batched GEMM插件" << std::endl;
        return;
    }
    
    // 初始化插件
    if (plugin->Initialize() != StatusCode::SUCCESS) {
        std::cerr << "插件初始化失败" << std::endl;
        return;
    }
    
    // 配置插件
    auto* gemm_plugin = dynamic_cast<GemmBatchedJITPlugin*>(plugin.get());
    if (gemm_plugin) {
        gemm_plugin->SetBatchSize(4);
        gemm_plugin->SetMatrixDimensions(64, 64, 32);
        gemm_plugin->SetTransposeOptions(false, false);
        gemm_plugin->SetAlphaBeta(1.0f, 0.0f);
    }
    
    // 配置JIT参数
    JITConfig config;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.use_shared_memory = true;
    config.use_tensor_cores = true;
    config.optimization_level = "O3";
    
    // 编译插件
    if (plugin->Compile(config) != StatusCode::SUCCESS) {
        std::cerr << "插件编译失败" << std::endl;
        return;
    }
    
    // 准备数据
    std::vector<Tensor<float>> inputs, outputs;
    
    // 输入矩阵A (batch_size x m x k)
    Tensor<float> A({4, 64, 32});
    A.fill(1.0f);
    inputs.push_back(A);
    
    // 输入矩阵B (batch_size x k x n)
    Tensor<float> B({4, 32, 64});
    B.fill(2.0f);
    inputs.push_back(B);
    
    // 输出矩阵C (batch_size x m x n)
    Tensor<float> C({4, 64, 64});
    C.fill(0.0f);
    outputs.push_back(C);
    
    // 执行运算
    auto start = std::chrono::high_resolution_clock::now();
    StatusCode status = plugin->Execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == StatusCode::SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Batched GEMM执行成功！耗时: " << duration.count() << " μs" << std::endl;
        
        // 获取性能信息
        auto profile = plugin->GetPerformanceProfile();
        std::cout << "GFLOPS: " << profile.gflops << std::endl;
        std::cout << "内存带宽: " << profile.bandwidth_gb_s << " GB/s" << std::endl;
    } else {
        std::cerr << "Batched GEMM执行失败" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief 对称矩阵运算使用示例
 */
void SymmetricOpsExample() {
    std::cout << "=== 对称矩阵运算示例 ===" << std::endl;
    
    // 创建SYMM插件
    auto plugin = CreateBlasJITPlugin("symm");
    if (!plugin) {
        std::cerr << "无法创建SYMM插件" << std::endl;
        return;
    }
    
    // 初始化插件
    if (plugin->Initialize() != StatusCode::SUCCESS) {
        std::cerr << "插件初始化失败" << std::endl;
        return;
    }
    
    // 配置插件
    auto* symm_plugin = dynamic_cast<SymmHerkJITPlugin*>(plugin.get());
    if (symm_plugin) {
        symm_plugin->SetOperationType(SymmetricOpType::SYMM);
        symm_plugin->SetMatrixDimensions(128, 128);
        symm_plugin->SetSideMode(true);  // left side
        symm_plugin->SetUploMode(true);  // upper triangle
        symm_plugin->SetAlphaBeta(1.0f, 0.0f);
    }
    
    // 配置JIT参数
    JITConfig config;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.use_shared_memory = true;
    config.optimization_level = "O2";
    
    // 编译插件
    if (plugin->Compile(config) != StatusCode::SUCCESS) {
        std::cerr << "插件编译失败" << std::endl;
        return;
    }
    
    // 准备数据
    std::vector<Tensor<float>> inputs, outputs;
    
    // 对称矩阵A
    Tensor<float> A({128, 128});
    A.fill(1.0f);
    inputs.push_back(A);
    
    // 矩阵B
    Tensor<float> B({128, 128});
    B.fill(2.0f);
    inputs.push_back(B);
    
    // 输出矩阵C
    Tensor<float> C({128, 128});
    C.fill(0.0f);
    outputs.push_back(C);
    
    // 执行运算
    auto start = std::chrono::high_resolution_clock::now();
    StatusCode status = plugin->Execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == StatusCode::SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "SYMM执行成功！耗时: " << duration.count() << " μs" << std::endl;
    } else {
        std::cerr << "SYMM执行失败" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief 向量运算使用示例
 */
void VectorOpsExample() {
    std::cout << "=== 向量运算示例 ===" << std::endl;
    
    // 创建DOT插件
    auto plugin = CreateBlasJITPlugin("dot");
    if (!plugin) {
        std::cerr << "无法创建DOT插件" << std::endl;
        return;
    }
    
    // 初始化插件
    if (plugin->Initialize() != StatusCode::SUCCESS) {
        std::cerr << "插件初始化失败" << std::endl;
        return;
    }
    
    // 配置插件
    auto* vector_plugin = dynamic_cast<VectorOpsJITPlugin*>(plugin.get());
    if (vector_plugin) {
        vector_plugin->SetOperationType(VectorOpType::DOT);
        vector_plugin->SetVectorSize(1024);
        vector_plugin->SetAlpha(1.0f);
    }
    
    // 配置JIT参数
    JITConfig config;
    config.block_size_x = 256;
    config.use_shared_memory = true;
    config.optimization_level = "O3";
    
    // 编译插件
    if (plugin->Compile(config) != StatusCode::SUCCESS) {
        std::cerr << "插件编译失败" << std::endl;
        return;
    }
    
    // 准备数据
    std::vector<Tensor<float>> inputs, outputs;
    
    // 向量x
    Tensor<float> x({1024});
    x.fill(1.0f);
    inputs.push_back(x);
    
    // 向量y
    Tensor<float> y({1024});
    y.fill(2.0f);
    inputs.push_back(y);
    
    // 输出标量
    Tensor<float> result({1});
    result.fill(0.0f);
    outputs.push_back(result);
    
    // 执行运算
    auto start = std::chrono::high_resolution_clock::now();
    StatusCode status = plugin->Execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == StatusCode::SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DOT执行成功！耗时: " << duration.count() << " μs" << std::endl;
        std::cout << "结果: " << result.data()[0] << std::endl;
    } else {
        std::cerr << "DOT执行失败" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief TRMM使用示例
 */
void TrmmExample() {
    std::cout << "=== TRMM 示例 ===" << std::endl;
    
    // 创建TRMM插件
    auto plugin = CreateBlasJITPlugin("trmm");
    if (!plugin) {
        std::cerr << "无法创建TRMM插件" << std::endl;
        return;
    }
    
    // 初始化插件
    if (plugin->Initialize() != StatusCode::SUCCESS) {
        std::cerr << "插件初始化失败" << std::endl;
        return;
    }
    
    // 配置插件
    auto* trmm_plugin = dynamic_cast<TrmmJITPlugin*>(plugin.get());
    if (trmm_plugin) {
        trmm_plugin->SetTrmmParams(0, 0, 0, 0, 1.0f);  // left, upper, no trans, non-unit, alpha=1.0
    }
    
    // 配置JIT参数
    JITConfig config;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.use_shared_memory = true;
    config.optimization_level = "O2";
    
    // 编译插件
    if (plugin->Compile(config) != StatusCode::SUCCESS) {
        std::cerr << "插件编译失败" << std::endl;
        return;
    }
    
    // 准备数据
    std::vector<Tensor<float>> inputs, outputs;
    
    // 三角矩阵A
    Tensor<float> A({64, 64});
    A.fill(1.0f);
    inputs.push_back(A);
    
    // 矩阵B
    Tensor<float> B({64, 32});
    B.fill(2.0f);
    inputs.push_back(B);
    
    // 输出矩阵B (in-place)
    outputs.push_back(B);
    
    // 执行运算
    auto start = std::chrono::high_resolution_clock::now();
    StatusCode status = plugin->Execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == StatusCode::SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "TRMM执行成功！耗时: " << duration.count() << " μs" << std::endl;
    } else {
        std::cerr << "TRMM执行失败" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief GER使用示例
 */
void GerExample() {
    std::cout << "=== GER 示例 ===" << std::endl;
    
    // 创建GER插件
    auto plugin = CreateBlasJITPlugin("ger");
    if (!plugin) {
        std::cerr << "无法创建GER插件" << std::endl;
        return;
    }
    
    // 初始化插件
    if (plugin->Initialize() != StatusCode::SUCCESS) {
        std::cerr << "插件初始化失败" << std::endl;
        return;
    }
    
    // 配置插件
    auto* ger_plugin = dynamic_cast<GerJITPlugin*>(plugin.get());
    if (ger_plugin) {
        ger_plugin->SetGerParams(1.0f);  // alpha = 1.0
        ger_plugin->SetMatrixDimensions(64, 32);
        ger_plugin->SetVectorIncrements(1, 1);
    }
    
    // 配置JIT参数
    JITConfig config;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.use_shared_memory = true;
    config.optimization_level = "O2";
    
    // 编译插件
    if (plugin->Compile(config) != StatusCode::SUCCESS) {
        std::cerr << "插件编译失败" << std::endl;
        return;
    }
    
    // 准备数据
    std::vector<Tensor<float>> inputs, outputs;
    
    // 向量x
    Tensor<float> x({64});
    x.fill(1.0f);
    inputs.push_back(x);
    
    // 向量y
    Tensor<float> y({32});
    y.fill(2.0f);
    inputs.push_back(y);
    
    // 矩阵A
    Tensor<float> A({64, 32});
    A.fill(0.5f);
    inputs.push_back(A);
    
    // 输出矩阵A (in-place)
    outputs.push_back(A);
    
    // 执行运算
    auto start = std::chrono::high_resolution_clock::now();
    StatusCode status = plugin->Execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == StatusCode::SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "GER执行成功！耗时: " << duration.count() << " μs" << std::endl;
    } else {
        std::cerr << "GER执行失败" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief 运行所有示例
 */
void RunAllExamples() {
    std::cout << "🚀 开始运行BLAS JIT插件示例..." << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // 注册所有BLAS插件
    RegisterBlasJITPlugins();
    
    // 显示支持的算子
    auto supported_ops = GetSupportedBlasOperators();
    std::cout << "支持的BLAS算子数量: " << supported_ops.size() << std::endl;
    std::cout << "支持的算子: ";
    for (const auto& op : supported_ops) {
        std::cout << op << " ";
    }
    std::cout << std::endl << std::endl;
    
    // 运行各种示例
    BatchedGemmExample();
    SymmetricOpsExample();
    VectorOpsExample();
    TrmmExample();
    GerExample();
    
    std::cout << "✅ 所有示例运行完成！" << std::endl;
}

} // namespace examples
} // namespace cu_op_mem
