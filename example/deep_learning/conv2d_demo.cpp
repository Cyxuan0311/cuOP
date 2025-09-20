#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "util/status_code.hpp"
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "data/tensor.hpp"
#include "jit/jit_wrapper.hpp"

using namespace cu_op_mem;

int main() {
    std::cout << "=== cuOP Conv2D 深度学习示例 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    try {
        // 创建卷积层参数
        const int batch_size = 16;
        const int in_channels = 64;
        const int out_channels = 128;
        const int input_height = 224;
        const int input_width = 224;
        const int kernel_size = 3;
        const int stride = 1;
        const int padding = 1;
        
        // 计算输出尺寸
        const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        
        // 创建张量
        Tensor<float> input({batch_size, in_channels, input_height, input_width});
        Tensor<float> weight({out_channels, in_channels, kernel_size, kernel_size});
        Tensor<float> bias({out_channels});
        Tensor<float> output({batch_size, out_channels, output_height, output_width});
        
        // 初始化数据
        std::vector<float> h_input(batch_size * in_channels * input_height * input_width, 1.0f);
        std::vector<float> h_weight(out_channels * in_channels * kernel_size * kernel_size, 0.1f);
        std::vector<float> h_bias(out_channels, 0.0f);
        
        // 复制到GPU
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(weight.data(), h_weight.data(), weight.bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(bias.data(), h_bias.data(), bias.bytes(), cudaMemcpyHostToDevice);
        
        // 创建Conv2D算子
        Convolution2D<float> conv2d(in_channels, out_channels, kernel_size, kernel_size, stride, stride, padding, padding);
        conv2d.SetWeight(weight);
        conv2d.SetBias(bias);
        
        // 执行Conv2D
        auto start = std::chrono::high_resolution_clock::now();
        StatusCode status = conv2d.Forward(input, output);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == StatusCode::SUCCESS) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Conv2D执行成功！" << std::endl;
            std::cout << "输入形状: [" << batch_size << ", " << in_channels << ", " 
                     << input_height << ", " << input_width << "]" << std::endl;
            std::cout << "输出形状: [" << batch_size << ", " << out_channels << ", " 
                     << output_height << ", " << output_width << "]" << std::endl;
            std::cout << "卷积核大小: " << kernel_size << "x" << kernel_size << std::endl;
            std::cout << "步长: " << stride << ", 填充: " << padding << std::endl;
            std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
            
            // 计算性能
            long long flops = (long long)batch_size * out_channels * output_height * output_width * 
                             in_channels * kernel_size * kernel_size * 2; // 乘加操作
            double gflops = flops / (duration.count() * 1e6);
            std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
            
            // 验证结果
            std::vector<float> h_output(batch_size * out_channels * output_height * output_width);
            cudaMemcpy(h_output.data(), output.data(), output.bytes(), cudaMemcpyDeviceToHost);
            
            // 检查输出统计
            float sum = 0.0f;
            float max_val = -std::numeric_limits<float>::infinity();
            float min_val = std::numeric_limits<float>::infinity();
            
            for (int i = 0; i < h_output.size(); ++i) {
                sum += h_output[i];
                max_val = std::max(max_val, h_output[i]);
                min_val = std::min(min_val, h_output[i]);
            }
            
            std::cout << "输出统计:" << std::endl;
            std::cout << "  和: " << sum << std::endl;
            std::cout << "  最大值: " << max_val << std::endl;
            std::cout << "  最小值: " << min_val << std::endl;
            std::cout << "  平均值: " << (sum / h_output.size()) << std::endl;
            
        } else {
            std::cout << "Conv2D执行失败！" << std::endl;
            return 1;
        }
        
        // 测试重复执行性能
        std::cout << "\n=== 测试重复执行性能 ===" << std::endl;
        // 重新初始化输入数据
        for (int i = 0; i < h_input.size(); ++i) {
            h_input[i] = 1.0f;
        }
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 5; ++i) {
            status = conv2d.Forward(input, output);
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
