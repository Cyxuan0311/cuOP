#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP ReLU 深度学习示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建张量 (模拟批量数据)
        const int batch_size = 32;
        const int channels = 64;
        const int height = 224;
        const int width = 224;
        
        Tensor<float> input({batch_size, channels, height, width});
        Tensor<float> output({batch_size, channels, height, width});
        
        // 初始化数据 (包含负值以测试ReLU)
        std::vector<float> h_input(batch_size * channels * height * width);
        for (int i = 0; i < h_input.size(); ++i) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        }
        
        // 复制到GPU
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        
        // 创建ReLU算子
        Relu<float> relu;
        
        // 执行ReLU
        auto start = std::chrono::high_resolution_clock::now();
        StatusCode status = relu.Forward(input, output);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "ReLU执行成功！" << std::endl;
            std::cout << "输入形状: [" << batch_size << ", " << channels << ", " << height << ", " << width << "]" << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 计算性能
            double gflops = (input.numel()) / (duration.count() * 1e6);
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            
            // 验证结果
            std::vector<float> h_output(batch_size * channels * height * width);
            cudaMemcpy(h_output.data(), output.data(), output.bytes(), cudaMemcpyDeviceToHost);
            
            // 检查前几个元素
            std::cout << "结果验证:" << std::endl;
            for (int i = 0; i < 10; ++i) {
                float expected = std::max(0.0f, h_input[i]);
                std::cout << "  input[" << i << "] = " << h_input[i] 
                         << " -> output[" << i << "] = " << h_output[i] 
                         << " (期望: " << expected << ")" << std::endl;
            }
            
            // 统计激活率
            int activated = 0;
            for (int i = 0; i < h_output.size(); ++i) {
                if (h_output[i] > 0) activated++;
            }
            double activation_rate = static_cast<double>(activated) / h_output.size();
            std::cout << "激活率: " << (activation_rate * 100) << "%" << std::endl;
            
        } else {
            std::cout << "ReLU执行失败！" << std::endl;
            return 1;
        }
        
        // 测试重复执行性能
        std::cout << "\n=== 测试重复执行性能 ===" << std::endl;
        // 重新初始化输入数据
        for (int i = 0; i < h_input.size(); ++i) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        }
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            status = relu.Forward(input, output);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "重复执行成功！" << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
