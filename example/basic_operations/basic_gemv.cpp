#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP GEMV 基础示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建矩阵和向量
        const int M = 2048, N = 1024;
        Tensor<float> A({M, N});
        Tensor<float> x({N});
        Tensor<float> y({M});
        
        // 初始化数据
        std::vector<float> h_A(M * N, 1.0f);
        std::vector<float> h_x(N, 2.0f);
        std::vector<float> h_y(M, 0.0f);
        
        // 复制到GPU
        cudaMemcpy(A.data(), h_A.data(), A.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(x.data(), h_x.data(), x.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(y.data(), h_y.data(), y.bytes(), cudaMemcpyHostToDevice);
        
        // 创建GEMV算子
        Gemv<float> gemv(false, 1.0f, 0.0f);
        gemv.SetWeight(x);
        
        // 执行GEMV
        auto start = std::chrono::high_resolution_clock::now();
        StatusCode status = gemv.Forward(A, y);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "GEMV执行成功！" << std::endl;
            std::cout << "矩阵大小: " << M << "x" << N << " * " << N << " = " << M << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 计算性能
            double gflops = (2.0 * M * N) / (duration.count() * 1e6);
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            
            // 验证结果
            std::vector<float> h_result(M);
            cudaMemcpy(h_result.data(), y.data(), y.bytes(), cudaMemcpyDeviceToHost);
            
            // 检查第一个元素
            std::cout << "结果验证: y[0] = " << h_result[0] << " (期望: " << (2.0f * N) << ")" << std::endl;
            
        } else {
            std::cout << "GEMV执行失败！" << std::endl;
            return 1;
        }
        
        // 测试重复执行性能
        std::cout << "\n=== 测试重复执行性能 ===" << std::endl;
        // 重新初始化y
        cudaMemcpy(y.data(), h_y.data(), y.bytes(), cudaMemcpyHostToDevice);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            status = gemv.Forward(A, y);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "重复执行成功！" << std::endl;
            std::cout << "10次执行时间: " << duration.count() << " ms" << std::endl;
            std::cout << "平均执行时间: " << duration.count() / 10.0 << " ms" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
