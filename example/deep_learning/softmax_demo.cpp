#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP Softmax 深度学习示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建张量 (模拟分类任务)
        const int batch_size = 32;
        const int num_classes = 1000;
        
        Tensor<float> input({batch_size, num_classes});
        Tensor<float> output({batch_size, num_classes});
        
        // 初始化数据 (模拟logits)
        std::vector<float> h_input(batch_size * num_classes);
        for (int i = 0; i < h_input.size(); ++i) {
            h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        }
        
        // 复制到GPU
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        
        // 创建Softmax算子
        Softmax<float> softmax;
        
        // 执行Softmax
        auto start = std::chrono::high_resolution_clock::now();
        StatusCode status = softmax.Forward(input, output, 1); // 在类别维度上应用softmax
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Softmax执行成功！" << std::endl;
            std::cout << "输入形状: [" << batch_size << ", " << num_classes << "]" << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 计算性能
            double gflops = (input.numel() * 3) / (duration.count() * 1e6); // 3个操作: exp, sum, div
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            
            // 验证结果
            std::vector<float> h_output(batch_size * num_classes);
            cudaMemcpy(h_output.data(), output.data(), output.bytes(), cudaMemcpyDeviceToHost);
            
            // 验证softmax属性
            std::cout << "结果验证:" << std::endl;
            for (int b = 0; b < std::min(3, batch_size); ++b) {
                float sum = 0.0f;
                float max_val = -std::numeric_limits<float>::infinity();
                int max_idx = 0;
                
                for (int c = 0; c < num_classes; ++c) {
                    int idx = b * num_classes + c;
                    sum += h_output[idx];
                    if (h_output[idx] > max_val) {
                        max_val = h_output[idx];
                        max_idx = c;
                    }
                }
                
                std::cout << "  批次 " << b << ": 和=" << sum << " (期望≈1.0), 最大概率=" << max_val 
                         << " 在类别 " << max_idx << std::endl;
            }
            
            // 计算平均概率分布
            float avg_prob = 1.0f / num_classes;
            float total_variance = 0.0f;
            for (int i = 0; i < h_output.size(); ++i) {
                float diff = h_output[i] - avg_prob;
                total_variance += diff * diff;
            }
            float avg_variance = total_variance / h_output.size();
            std::cout << "平均方差: " << avg_variance << " (越小越均匀)" << std::endl;
            
        } else {
            std::cout << "Softmax执行失败！" << std::endl;
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
            status = softmax.Forward(input, output, 1);
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
