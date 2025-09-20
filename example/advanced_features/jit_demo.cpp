#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "cuda_op/detail/cuBlas/dot.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP JIT 优化演示 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 1. GEMM JIT 演示
        std::cout << "\n=== GEMM JIT 优化演示 ===" << std::endl;
        
        const int M = 1024, N = 1024, K = 1024;
        Tensor<float> A({M, K});
        Tensor<float> B({K, N});
        Tensor<float> C({M, N});
        
        // 初始化数据
        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_B(K * N, 2.0f);
        std::vector<float> h_C(M * N, 0.0f);
        
        cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(B.data(), h_B.data(), B.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(C.data(), h_C.data(), C.bytes(), cudaMemcpyHostToDevice);
        
        // 创建GEMM算子
        Gemm<float> gemm;
        gemm.SetAlpha(1.0f);
        gemm.SetBeta(0.0f);
        
        // 测试无JIT性能
        std::cout << "测试无JIT性能..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            gemm.Forward(A, B, C);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto no_jit_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 测试JIT性能
        std::cout << "测试JIT性能..." << std::endl;
        JITWrapper<Gemm<float>> jit_gemm(gemm);
        jit_gemm.EnableJIT(true);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            jit_gemm.Forward(A, B, C);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto jit_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "无JIT时间: " << no_jit_time.count() << " ms" << std::endl;
        std::cout << "JIT时间: " << jit_time.count() << " ms" << std::endl;
        std::cout << "加速比: " << (double)no_jit_time.count() / jit_time.count() << "x" << std::endl;
        
        // 获取JIT性能信息
        auto profile = jit_gemm.GetPerformanceProfile();
        std::cout << "JIT性能信息:" << std::endl;
        std::cout << "  GFLOPS: " << profile.gflops << std::endl;
        std::cout << "  带宽: " << profile.bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  缓存命中率: " << profile.cache_hit_rate << std::endl;
        
        // 2. GEMV JIT 演示
        std::cout << "\n=== GEMV JIT 优化演示 ===" << std::endl;
        
        const int M2 = 2048, N2 = 1024;
        Tensor<float> A2({M2, N2});
        Tensor<float> x2({N2});
        Tensor<float> y2({M2});
        
        // 初始化数据
        std::vector<float> h_A2(M2 * N2, 1.0f);
        std::vector<float> h_x2(N2, 2.0f);
        std::vector<float> h_y2(M2, 0.0f);
        
        cudaMemcpy(A2.data(), h_A2.data(), A2.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(x2.data(), h_x2.data(), x2.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(y2.data(), h_y2.data(), y2.bytes(), cudaMemcpyHostToDevice);
        
        // 创建GEMV算子
        Gemv<float> gemv;
        gemv.SetAlpha(1.0f);
        gemv.SetBeta(0.0f);
        
        // 测试无JIT性能
        std::cout << "测试无JIT性能..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            gemv.Forward(A2, x2, y2);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto no_jit_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 测试JIT性能
        std::cout << "测试JIT性能..." << std::endl;
        JITWrapper<Gemv<float>> jit_gemv(gemv);
        jit_gemv.EnableJIT(true);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            jit_gemv.Forward(A2, x2, y2);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto jit_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "无JIT时间: " << no_jit_time2.count() << " ms" << std::endl;
        std::cout << "JIT时间: " << jit_time2.count() << " ms" << std::endl;
        std::cout << "加速比: " << (double)no_jit_time2.count() / jit_time2.count() << "x" << std::endl;
        
        // 3. DOT JIT 演示
        std::cout << "\n=== DOT JIT 优化演示 ===" << std::endl;
        
        const int N3 = 1000000;
        Tensor<float> x3({N3});
        Tensor<float> y3({N3});
        
        // 初始化数据
        std::vector<float> h_x3(N3, 1.0f);
        std::vector<float> h_y3(N3, 2.0f);
        
        cudaMemcpy(x3.data(), h_x3.data(), x3.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(y3.data(), h_y3.data(), y3.bytes(), cudaMemcpyHostToDevice);
        
        // 创建DOT算子
        Dot<float> dot;
        
        // 测试无JIT性能
        std::cout << "测试无JIT性能..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            dot.Forward(x3, y3);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto no_jit_time3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 测试JIT性能
        std::cout << "测试JIT性能..." << std::endl;
        JITWrapper<Dot<float>> jit_dot(dot);
        jit_dot.EnableJIT(true);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            jit_dot.Forward(x3, y3);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto jit_time3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "无JIT时间: " << no_jit_time3.count() << " ms" << std::endl;
        std::cout << "JIT时间: " << jit_time3.count() << " ms" << std::endl;
        std::cout << "加速比: " << (double)no_jit_time3.count() / jit_time3.count() << "x" << std::endl;
        
        // 4. JIT 缓存演示
        std::cout << "\n=== JIT 缓存演示 ===" << std::endl;
        
        // 第一次运行（编译）
        std::cout << "第一次运行（编译）..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        jit_gemm.Forward(A, B, C);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto compile_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 第二次运行（缓存命中）
        std::cout << "第二次运行（缓存命中）..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        jit_gemm.Forward(A, B, C);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "编译时间: " << compile_time.count() << " μs" << std::endl;
        std::cout << "缓存时间: " << cache_time.count() << " μs" << std::endl;
        std::cout << "缓存加速比: " << (double)compile_time.count() / cache_time.count() << "x" << std::endl;
        
        // 5. JIT 统计信息
        std::cout << "\n=== JIT 统计信息 ===" << std::endl;
        auto stats = jit_gemm.GetJITStatistics();
        std::cout << "编译次数: " << stats.compile_count << std::endl;
        std::cout << "缓存命中次数: " << stats.cache_hit_count << std::endl;
        std::cout << "总执行次数: " << stats.total_executions << std::endl;
        std::cout << "平均编译时间: " << stats.avg_compile_time << " μs" << std::endl;
        std::cout << "平均执行时间: " << stats.avg_execution_time << " μs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "JIT演示失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

