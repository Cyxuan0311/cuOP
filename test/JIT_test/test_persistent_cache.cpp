#include "jit/jit_wrapper.hpp"
#include "jit/jit_config.hpp"
#include "jit/jit_persistent_cache.hpp"
#include "jit/Blas/blas_jit_plugins.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"
#include <glog/logging.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace cu_op_mem;

// 性能测试函数
template<typename T>
double MeasureExecutionTime(T& func, int iterations = 100) {
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
    std::cout << std::setw(30) << name << ": "
              << std::fixed << std::setprecision(3) << time_ms << " ms "
              << "(" << time_ms / iterations << " ms/iter)" << std::endl;
}

// 测试持久化缓存功能
void TestPersistentCache() {
    std::cout << "\n=== 测试持久化缓存功能 ===" << std::endl;
    
    // 初始化全局JIT管理器
    auto& global_manager = GlobalJITManager::Instance();
    auto status = global_manager.Initialize();
    if (status != StatusCode::SUCCESS) {
        std::cerr << "Failed to initialize Global JIT Manager" << std::endl;
        return;
    }
    
    // 设置全局配置
    GlobalJITConfig global_config;
    global_config.enable_jit = true;
    global_config.enable_auto_tuning = true;
    global_config.enable_caching = true;
    global_config.cache_dir = "./jit_persistent_cache";
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
    global_manager.SetOperatorConfig("gemm", gemm_config);
    
    // 创建测试数据
    int M = 1024, N = 1024, K = 1024;
    Tensor<float> A({M, K});
    Tensor<float> B({K, N});
    Tensor<float> C({M, N});
    
    // 初始化数据
    A.Fill(1.0f);
    B.Fill(1.0f);
    C.Zero();
    
    std::cout << "矩阵大小: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;
    
    // 测试1: 首次编译（无缓存）
    std::cout << "\n1. 首次编译（无缓存）..." << std::endl;
    
    Gemm<float> gemm1;
    gemm1.SetWeight(B);
    
    JITWrapper<Gemm<float>> jit_gemm1(gemm1);
    jit_gemm1.EnableJIT(true);
    
    // 预热
    jit_gemm1.Forward(A, C);
    cudaDeviceSynchronize();
    
    // 测量首次编译时间
    auto first_compile_start = std::chrono::high_resolution_clock::now();
    jit_gemm1.Forward(A, C);
    cudaDeviceSynchronize();
    auto first_compile_end = std::chrono::high_resolution_clock::now();
    
    double first_compile_time = std::chrono::duration<double>(first_compile_end - first_compile_start).count() * 1000.0;
    std::cout << "首次编译时间: " << std::fixed << std::setprecision(3) << first_compile_time << " ms" << std::endl;
    
    // 测试2: 使用持久化缓存
    std::cout << "\n2. 启用持久化缓存..." << std::endl;
    
    // 启用持久化缓存
    jit_gemm1.EnablePersistentCache(true);
    jit_gemm1.SetPersistentCacheDirectory("./jit_persistent_cache");
    
    // 再次执行（应该会保存到持久化缓存）
    jit_gemm1.Forward(A, C);
    cudaDeviceSynchronize();
    
    // 测试3: 创建新的JIT包装器（模拟重启应用）
    std::cout << "\n3. 模拟应用重启（从持久化缓存加载）..." << std::endl;
    
    Gemm<float> gemm2;
    gemm2.SetWeight(B);
    
    JITWrapper<Gemm<float>> jit_gemm2(gemm2);
    jit_gemm2.EnableJIT(true);
    jit_gemm2.EnablePersistentCache(true);
    jit_gemm2.SetPersistentCacheDirectory("./jit_persistent_cache");
    
    // 测量从持久化缓存加载的时间
    auto cache_load_start = std::chrono::high_resolution_clock::now();
    jit_gemm2.Forward(A, C);
    cudaDeviceSynchronize();
    auto cache_load_end = std::chrono::high_resolution_clock::now();
    
    double cache_load_time = std::chrono::duration<double>(cache_load_end - cache_load_start).count() * 1000.0;
    std::cout << "从持久化缓存加载时间: " << std::fixed << std::setprecision(3) << cache_load_time << " ms" << std::endl;
    
    // 计算性能提升
    if (first_compile_time > 0 && cache_load_time > 0) {
        double speedup = first_compile_time / cache_load_time;
        double time_saved = first_compile_time - cache_load_time;
        double time_saved_percent = (time_saved / first_compile_time) * 100.0;
        
        std::cout << "\n=== 性能提升分析 ===" << std::endl;
        std::cout << "首次编译时间: " << std::fixed << std::setprecision(3) << first_compile_time << " ms" << std::endl;
        std::cout << "缓存加载时间: " << std::fixed << std::setprecision(3) << cache_load_time << " ms" << std::endl;
        std::cout << "时间节省: " << std::fixed << std::setprecision(3) << time_saved << " ms (" << time_saved_percent << "%)" << std::endl;
        std::cout << "加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // 测试4: 缓存统计信息
    std::cout << "\n4. 缓存统计信息..." << std::endl;
    
    auto cache_stats = GlobalPersistentCacheManager::Instance().GetStats();
    std::cout << "总缓存内核数: " << cache_stats.total_cached_kernels.load() << std::endl;
    std::cout << "磁盘缓存命中: " << cache_stats.disk_cache_hits.load() << std::endl;
    std::cout << "磁盘缓存未命中: " << cache_stats.disk_cache_misses.load() << std::endl;
    std::cout << "总磁盘缓存大小: " << cache_stats.total_disk_cache_size.load() << " bytes" << std::endl;
    std::cout << "节省的总编译时间: " << cache_stats.total_saved_compilation_time.load() << " ms" << std::endl;
    
    // 测试5: 缓存管理功能
    std::cout << "\n5. 测试缓存管理功能..." << std::endl;
    
    // 清理过期缓存
    GlobalPersistentCacheManager::Instance().CleanupExpiredCache();
    std::cout << "已清理过期缓存" << std::endl;
    
    // 验证缓存完整性
    GlobalPersistentCacheManager::Instance().ValidateCacheIntegrity();
    std::cout << "已验证缓存完整性" << std::endl;
    
    // 测试6: 批量测试不同大小的矩阵
    std::cout << "\n6. 批量测试不同大小的矩阵..." << std::endl;
    
    std::vector<std::pair<int, int>> test_cases = {
        {256, 256},   // 小矩阵
        {512, 512},   // 中等矩阵
        {1024, 1024}, // 大矩阵
        {2048, 2048}  // 超大矩阵
    };
    
    for (const auto& test_case : test_cases) {
        int m = test_case.first;
        int k = test_case.second;
        int n = k;
        
        std::cout << "\n测试矩阵: " << m << "x" << k << " * " << k << "x" << n << std::endl;
        
        Tensor<float> test_A({m, k});
        Tensor<float> test_B({k, n});
        Tensor<float> test_C({m, n});
        
        test_A.Fill(1.0f);
        test_B.Fill(1.0f);
        test_C.Zero();
        
        Gemm<float> test_gemm;
        test_gemm.SetWeight(test_B);
        
        JITWrapper<Gemm<float>> test_jit_gemm(test_gemm);
        test_jit_gemm.EnableJIT(true);
        test_jit_gemm.EnablePersistentCache(true);
        test_jit_gemm.SetPersistentCacheDirectory("./jit_persistent_cache");
        
        // 首次执行（编译）
        auto start1 = std::chrono::high_resolution_clock::now();
        test_jit_gemm.Forward(test_A, test_C);
        cudaDeviceSynchronize();
        auto end1 = std::chrono::high_resolution_clock::now();
        
        double compile_time = std::chrono::duration<double>(end1 - start1).count() * 1000.0;
        
        // 第二次执行（从缓存加载）
        auto start2 = std::chrono::high_resolution_clock::now();
        test_jit_gemm.Forward(test_A, test_C);
        cudaDeviceSynchronize();
        auto end2 = std::chrono::high_resolution_clock::now();
        
        double cache_time = std::chrono::duration<double>(end2 - start2).count() * 1000.0;
        
        std::cout << "  编译时间: " << std::fixed << std::setprecision(3) << compile_time << " ms" << std::endl;
        std::cout << "  缓存时间: " << std::fixed << std::setprecision(3) << cache_time << " ms" << std::endl;
        
        if (compile_time > 0 && cache_time > 0) {
            double speedup = compile_time / cache_time;
            std::cout << "  加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }
    
    std::cout << "\n=== 持久化缓存测试完成 ===" << std::endl;
}

int main() {
    // 初始化日志
    google::InitGoogleLogging("test_persistent_cache");
    google::SetStderrLogging(google::INFO);
    
    std::cout << "=== cuOP JIT 持久化缓存系统测试 ===" << std::endl;
    
    try {
        TestPersistentCache();
        
    } catch (const std::exception& e) {
        std::cerr << "测试异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 