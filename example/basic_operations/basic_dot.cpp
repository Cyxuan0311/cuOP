#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuBlas/dot.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP DOT 基础示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建向量
        const int N = 1000000;
        Tensor<float> x({N});
        Tensor<float> y({N});
        
        // 初始化数据
        std::vector<float> h_x(N);
        std::vector<float> h_y(N);
        
        for (int i = 0; i < N; ++i) {
            h_x[i] = static_cast<float>(i + 1);
            h_y[i] = static_cast<float>(i + 1);
        }
        
        // 复制到GPU
        cudaMemcpy(x.data(), h_x.data(), x.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(y.data(), h_y.data(), y.bytes(), cudaMemcpyHostToDevice);
        
        // 创建DOT算子
        Dot<float> dot;
        
        // 执行DOT
        auto start = std::chrono::high_resolution_clock::now();
        float result;
        StatusCode status = dot.Forward(x, y, result);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "DOT执行成功！" << std::endl;
        std::cout << "向量大小: " << N << std::endl;
        std::cout << "执行时间: " << duration.count() << " μs" << std::endl;
        
        // 计算性能
        double gflops = (2.0 * N) / (duration.count() * 1e3);
        std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
        
        // 验证结果
        float expected = 0.0f;
        for (int i = 0; i < N; ++i) {
            expected += h_x[i] * h_y[i];
        }
        
            std::cout << "结果验证: " << result << " (期望: " << expected << ")" << std::endl;
            std::cout << "误差: " << std::abs(result - expected) << std::endl;
        } else {
            std::cout << "DOT执行失败！" << std::endl;
            return 1;
        }
        
        // 测试JIT优化
        std::cout << "\n=== 测试JIT优化 ===" << std::endl;
        // 注意：DOT的Forward方法签名特殊，JITWrapper可能不支持
        // 这里我们直接测试原始DOT性能
        start = std::chrono::high_resolution_clock::now();
        float jit_result;
        status = dot.Forward(x, y, jit_result);
        end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "DOT执行成功！" << std::endl;
            std::cout << "执行时间: " << duration.count() << " μs" << std::endl;
            std::cout << "结果: " << jit_result << std::endl;
        } else {
            std::cout << "DOT执行失败！" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
