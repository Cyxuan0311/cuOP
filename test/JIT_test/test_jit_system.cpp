#include "jit/jit_wrapper.hpp"
#include "jit/jit_config.hpp"
#include "jit/Blas/blas_jit_plugins.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace cu_op_mem;

// 性能测试函数
template<typename T>
double MeasureExecutionTime(T&& func, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0; // 返回毫秒
}

// 打印性能结果
void PrintPerformanceResult(const std::string& name, double time_ms, int iterations) {
    std::cout << std::setw(20) << name << ": "
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << time_ms / iterations << " ms/iter)" << std::endl;
}

int main() {
    // 初始化日志
    google::InitGoogleLogging("test_jit_system");
    google::SetStderrLogging(google::INFO);
    
    std::cout << "=== cuOP JIT System Test ===" << std::endl;
    
    try {
        // 初始化全局JIT管理器
        auto& global_manager = GlobalJITManager::Instance();
        auto status = global_manager.Initialize();
        if (status != StatusCode::SUCCESS) {
            std::cerr << "Failed to initialize Global JIT Manager" << std::endl;
            return -1;
        }
        
        // 设置全局配置
        GlobalJITConfig global_config;
        global_config.enable_jit = true;
        global_config.enable_auto_tuning = true;
        global_config.enable_caching = true;
        global_config.cache_dir = "./jit_cache";
        global_config.enable_debug = true;
        global_manager.SetGlobalConfig(global_config);
        
        // 注册BLAS JIT插件
        RegisterBlasJITPlugins();
        
        // 设置GEMM算子配置
        JITConfig gemm_config;
        gemm_config.enable_jit = true;
        gemm_config.kernel_type = "tiled";
        gemm_config.tile_size = 32;
        gemm_config.block_size = 256;
        gemm_config.optimization_level = "O2";
        gemm_config.enable_tensor_core = true;
        gemm_config.enable_tma = false;
        global_manager.SetOperatorConfig("gemm", gemm_config);
        
        std::cout << "\n1. 创建测试数据..." << std::endl;
        
        // 创建测试数据
        int M = 1024, N = 1024, K = 1024;
        Tensor<float> A({static_cast<size_t>(M), static_cast<size_t>(K)});
        Tensor<float> B({static_cast<size_t>(K), static_cast<size_t>(N)});
        Tensor<float> C({static_cast<size_t>(M), static_cast<size_t>(N)});
        
        // 初始化数据
        A.fill(1.0f);
        B.fill(1.0f);
        C.zero();
        
        std::cout << "   矩阵大小: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;
        
        std::cout << "\n2. 创建原始GEMM算子..." << std::endl;
        
        // 创建原始GEMM算子
        Gemm<float> original_gemm;
        original_gemm.SetWeight(B);
        
        std::cout << "\n3. 创建JIT包装器..." << std::endl;
        
        // 创建JIT包装器
        JITWrapper<Gemm<float>> jit_gemm(std::move(original_gemm));
        
        // 配置JIT
        jit_gemm.SetJITConfig(gemm_config);
        jit_gemm.EnableJIT(true);
        jit_gemm.EnableAutoTuning(true);
        
        std::cout << "   JIT状态: " << (jit_gemm.IsJITEnabled() ? "启用" : "禁用") << std::endl;
        std::cout << "   自动调优: " << (jit_gemm.IsAutoTuningEnabled() ? "启用" : "禁用") << std::endl;
        
        std::cout << "\n4. 编译JIT内核..." << std::endl;
        
        // 编译JIT内核
        status = jit_gemm.CompileJIT();
        if (status == StatusCode::SUCCESS) {
            std::cout << "   JIT编译成功!" << std::endl;
            std::cout << "   JIT已编译: " << (jit_gemm.IsJITCompiled() ? "是" : "否") << std::endl;
        } else {
            std::cout << "   JIT编译失败，将使用原始算子" << std::endl;
        }
        
        std::cout << "\n5. 性能测试..." << std::endl;
        
        const int warmup_iterations = 10;
        const int test_iterations = 100;
        
        // 预热
        std::cout << "   预热 (" << warmup_iterations << " 次)..." << std::endl;
        for (int i = 0; i < warmup_iterations; ++i) {
            original_gemm.Forward(A, C);
        }
        
        // 测试原始算子性能
        std::cout << "   测试原始算子 (" << test_iterations << " 次)..." << std::endl;
        double original_time = MeasureExecutionTime([&]() {
            original_gemm.Forward(A, C);
        }, test_iterations);
        
        // 测试JIT算子性能
        std::cout << "   测试JIT算子 (" << test_iterations << " 次)..." << std::endl;
        double jit_time = MeasureExecutionTime([&]() {
            jit_gemm(A, C);
        }, test_iterations);
        
        std::cout << "\n6. 性能结果:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        PrintPerformanceResult("原始算子", original_time, test_iterations);
        PrintPerformanceResult("JIT算子", jit_time, test_iterations);
        
        if (jit_time > 0 && original_time > 0) {
            double speedup = original_time / jit_time;
            std::cout << std::string(50, '-') << std::endl;
            std::cout << "加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            
            if (speedup > 1.0) {
                std::cout << "JIT优化有效!" << std::endl;
            } else {
                std::cout << "JIT优化效果不明显，可能需要进一步调优" << std::endl;
            }
        }
        
        std::cout << "\n7. 获取性能分析信息..." << std::endl;
        
        // 获取性能分析信息
        auto profile = jit_gemm.GetPerformanceProfile();
        std::cout << "   执行时间: " << std::fixed << std::setprecision(3) << profile.execution_time << " s" << std::endl;
        std::cout << "   吞吐量: " << std::fixed << std::setprecision(2) << profile.throughput << " GFLOPS" << std::endl;
        std::cout << "   Kernel类型: " << profile.kernel_type << std::endl;
        std::cout << "   矩阵大小: " << profile.matrix_size[0] << "x" << profile.matrix_size[1] << "x" << profile.matrix_size[2] << std::endl;
        
        std::cout << "\n8. 获取统计信息..." << std::endl;
        
        // 获取全局统计信息
        auto global_stats = global_manager.GetStatistics();
        std::cout << "   总编译次数: " << global_stats.total_compilations << std::endl;
        std::cout << "   缓存命中次数: " << global_stats.cache_hits << std::endl;
        std::cout << "   缓存未命中次数: " << global_stats.cache_misses << std::endl;
        std::cout << "   缓存命中率: " << std::fixed << std::setprecision(2) 
                  << (global_stats.GetCacheHitRate() * 100) << "%" << std::endl;
        std::cout << "   平均编译时间: " << std::fixed << std::setprecision(3) 
                  << global_stats.GetAverageCompilationTime() << " s" << std::endl;
        std::cout << "   总编译时间: " << std::fixed << std::setprecision(3) 
                  << global_stats.total_compilation_time << " s" << std::endl;
        std::cout << "   总执行时间: " << std::fixed << std::setprecision(3) 
                  << global_stats.total_execution_time << " s" << std::endl;
        std::cout << "   当前缓存大小: " << global_stats.cache_size << " bytes" << std::endl;
        std::cout << "   活跃kernel数量: " << global_stats.active_kernels << std::endl;
        
        std::cout << "\n9. 清理资源..." << std::endl;
        
        // 清理资源
        jit_gemm.Cleanup();
        global_manager.Cleanup();
        
        std::cout << "测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 