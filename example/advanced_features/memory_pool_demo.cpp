#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "data/tensor.hpp"
#include "memory/memory_pool.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP 内存池演示 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 1. 内存池基础演示
        std::cout << "\n=== 内存池基础演示 ===" << std::endl;
        
        // 创建内存池
        MemoryPool pool;
        pool.Initialize(1024 * 1024 * 1024); // 1GB
        
        // 分配内存
        void* ptr1 = pool.Allocate(1024 * 1024); // 1MB
        void* ptr2 = pool.Allocate(2048 * 1024); // 2MB
        void* ptr3 = pool.Allocate(512 * 1024);  // 512KB
        
        std::cout << "分配内存块:" << std::endl;
        std::cout << "  块1: " << ptr1 << " (1MB)" << std::endl;
        std::cout << "  块2: " << ptr2 << " (2MB)" << std::endl;
        std::cout << "  块3: " << ptr3 << " (512KB)" << std::endl;
        
        // 获取内存池统计信息
        auto stats = pool.GetStatistics();
        std::cout << "内存池统计:" << std::endl;
        std::cout << "  总大小: " << stats.total_size << " bytes" << std::endl;
        std::cout << "  已分配: " << stats.allocated_size << " bytes" << std::endl;
        std::cout << "  可用: " << stats.available_size << " bytes" << std::endl;
        std::cout << "  分配次数: " << stats.allocation_count << std::endl;
        std::cout << "  释放次数: " << stats.deallocation_count << std::endl;
        
        // 释放内存
        pool.Deallocate(ptr1);
        pool.Deallocate(ptr2);
        pool.Deallocate(ptr3);
        
        std::cout << "内存已释放" << std::endl;
        
        // 2. 张量内存池演示
        std::cout << "\n=== 张量内存池演示 ===" << std::endl;
        
        // 创建张量（使用内存池）
        Tensor<float> A({1024, 1024});
        Tensor<float> B({1024, 1024});
        Tensor<float> C({1024, 1024});
        
        // 初始化数据
        std::vector<float> h_A(1024 * 1024, 1.0f);
        std::vector<float> h_B(1024 * 1024, 2.0f);
        std::vector<float> h_C(1024 * 1024, 0.0f);
        
        cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(B.data(), h_B.data(), B.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(C.data(), h_C.data(), C.bytes(), cudaMemcpyHostToDevice);
        
        // 创建GEMM算子
        Gemm<float> gemm;
        gemm.SetAlpha(1.0f);
        gemm.SetBeta(0.0f);
        
        // 测试性能
        std::cout << "测试GEMM性能..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            gemm.Forward(A, B, C);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "GEMM执行时间: " << duration.count() << " ms" << std::endl;
        
        // 3. 内存池性能对比
        std::cout << "\n=== 内存池性能对比 ===" << std::endl;
        
        const int num_iterations = 1000;
        const int tensor_size = 1000;
        
        // 测试使用内存池的性能
        std::cout << "测试使用内存池的性能..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            Tensor<float> temp({tensor_size, tensor_size});
            // 模拟一些操作
            cudaMemset(temp.data(), 0, temp.bytes());
        }
        
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 测试不使用内存池的性能
        std::cout << "测试不使用内存池的性能..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            float* temp_ptr;
            cudaMalloc(&temp_ptr, tensor_size * tensor_size * sizeof(float));
            cudaMemset(temp_ptr, 0, tensor_size * tensor_size * sizeof(float));
            cudaFree(temp_ptr);
        }
        
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto no_pool_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "内存池时间: " << pool_time.count() << " μs" << std::endl;
        std::cout << "无内存池时间: " << no_pool_time.count() << " μs" << std::endl;
        std::cout << "加速比: " << (double)no_pool_time.count() / pool_time.count() << "x" << std::endl;
        
        // 4. 内存池碎片化演示
        std::cout << "\n=== 内存池碎片化演示 ===" << std::endl;
        
        // 分配不同大小的内存块
        std::vector<void*> ptrs;
        for (int i = 0; i < 10; ++i) {
            size_t size = (i + 1) * 1024 * 1024; // 1MB, 2MB, 3MB, ...
            void* ptr = pool.Allocate(size);
            ptrs.push_back(ptr);
            std::cout << "分配 " << size / (1024 * 1024) << "MB 内存块" << std::endl;
        }
        
        // 释放部分内存块
        for (int i = 0; i < 5; ++i) {
            pool.Deallocate(ptrs[i * 2]); // 释放偶数索引的内存块
            std::cout << "释放 " << (i * 2 + 1) << "MB 内存块" << std::endl;
        }
        
        // 获取碎片化信息
        auto frag_stats = pool.GetFragmentationStats();
        std::cout << "碎片化统计:" << std::endl;
        std::cout << "  总块数: " << frag_stats.total_blocks << std::endl;
        std::cout << "  已分配块数: " << frag_stats.allocated_blocks << std::endl;
        std::cout << "  空闲块数: " << frag_stats.free_blocks << std::endl;
        std::cout << "  最大连续空间: " << frag_stats.max_contiguous_size << " bytes" << std::endl;
        std::cout << "  碎片化率: " << frag_stats.fragmentation_rate << "%" << std::endl;
        
        // 清理剩余内存
        for (int i = 0; i < 10; ++i) {
            if (ptrs[i] != nullptr) {
                pool.Deallocate(ptrs[i]);
            }
        }
        
        // 5. 内存池优化建议
        std::cout << "\n=== 内存池优化建议 ===" << std::endl;
        
        auto opt_stats = pool.GetOptimizationStats();
        std::cout << "优化统计:" << std::endl;
        std::cout << "  平均分配时间: " << opt_stats.avg_allocation_time << " μs" << std::endl;
        std::cout << "  平均释放时间: " << opt_stats.avg_deallocation_time << " μs" << std::endl;
        std::cout << "  缓存命中率: " << opt_stats.cache_hit_rate << "%" << std::endl;
        std::cout << "  内存利用率: " << opt_stats.memory_utilization << "%" << std::endl;
        
        if (opt_stats.memory_utilization < 80.0) {
            std::cout << "建议: 内存利用率较低，考虑减小内存池大小" << std::endl;
        }
        
        if (opt_stats.cache_hit_rate < 90.0) {
            std::cout << "建议: 缓存命中率较低，考虑增加缓存大小" << std::endl;
        }
        
        if (frag_stats.fragmentation_rate > 20.0) {
            std::cout << "建议: 碎片化率较高，考虑使用内存整理功能" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "内存池演示失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

