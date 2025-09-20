#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP GEMM 基础示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建矩阵
        const int M = 1024, N = 1024, K = 1024;
        Tensor<float> A({M, K});
        Tensor<float> B({K, N});
        Tensor<float> C({M, N});
        
        // 初始化数据
        std::vector<float> h_A(M * K, 1.0f);
        std::vector<float> h_B(K * N, 2.0f);
        std::vector<float> h_C(M * N, 0.0f);
        
        // 复制到GPU
        cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(B.data(), h_B.data(), B.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(C.data(), h_C.data(), C.bytes(), cudaMemcpyHostToDevice);
        
        // 创建GEMM算子
        Gemm<float> gemm(false, false, 1.0f, 0.0f);
        gemm.SetWeight(B);
        
        // 执行GEMM
        auto start = std::chrono::high_resolution_clock::now();
        StatusCode status = gemm.Forward(A, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "GEMM执行成功！" << std::endl;
            std::cout << "矩阵大小: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 计算性能
            double gflops = (2.0 * M * N * K) / (duration.count() * 1e6);
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            
            // 验证结果
            std::vector<float> h_result(M * N);
            cudaMemcpy(h_result.data(), C.data(), C.bytes(), cudaMemcpyDeviceToHost);
            
            // 检查第一个元素
            std::cout << "结果验证: C[0] = " << h_result[0] << " (期望: " << (2.0f * K) << ")" << std::endl;
            
        } else {
            std::cout << "GEMM执行失败！" << std::endl;
            return 1;
        }
        
        // 测试JIT优化
        std::cout << "\n=== 测试JIT优化 ===" << std::endl;
        JITWrapper<Gemm<float>> jit_gemm(std::move(gemm));
        jit_gemm.EnableJIT(true);
        
        start = std::chrono::high_resolution_clock::now();
        status = jit_gemm(A, C);
        end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "JIT GEMM执行成功！" << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 获取性能信息
            auto profile = jit_gemm.GetPerformanceProfile();
            std::cout << "性能信息: GFLOPS=" << profile.gflops 
                     << ", 带宽=" << profile.bandwidth_gb_s << " GB/s" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
