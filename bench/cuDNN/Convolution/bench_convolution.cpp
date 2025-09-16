#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int batch_size = 32, in_channels = 64, height = 32, width = 32; // 默认输入参数
    int out_channels = 128, kernel_h = 3, kernel_w = 3; // 默认卷积参数
    int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1; // 默认步长和填充
    
    if(argc >= 5){
        batch_size = std::atoi(argv[1]);
        in_channels = std::atoi(argv[2]);
        height = std::atoi(argv[3]);
        width = std::atoi(argv[4]);
    }
    if(argc >= 8){
        out_channels = std::atoi(argv[5]);
        kernel_h = std::atoi(argv[6]);
        kernel_w = std::atoi(argv[7]);
    }
    if(argc >= 11){
        stride_h = std::atoi(argv[8]);
        stride_w = std::atoi(argv[9]);
        pad_h = std::atoi(argv[10]);
        pad_w = std::atoi(argv[11]);
    }

    std::cout << "Convolution2D input shape: (" << batch_size << ", " << in_channels << ", " << height << ", " << width << ")" << std::endl;
    std::cout << "Convolution2D params: out_channels=" << out_channels << ", kernel=(" << kernel_h << ", " << kernel_w << "), stride=(" << stride_h << ", " << stride_w << "), pad=(" << pad_h << ", " << pad_w << ")" << std::endl;

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(in_channels), static_cast<size_t>(height), static_cast<size_t>(width)});
    
    // 计算输出尺寸
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(out_channels), static_cast<size_t>(out_height), static_cast<size_t>(out_width)});
    
    // 权重和偏置
    Tensor<float> weight({static_cast<size_t>(out_channels), static_cast<size_t>(in_channels), static_cast<size_t>(kernel_h), static_cast<size_t>(kernel_w)});
    Tensor<float> bias({static_cast<size_t>(out_channels)});

    std::vector<float> h_input(batch_size * in_channels * height * width);
    std::vector<float> h_weight(out_channels * in_channels * kernel_h * kernel_w);
    std::vector<float> h_bias(out_channels, 0.0f);
    
    for(auto& v : h_input) v = rand_float() * 2 - 1; // [-1, 1] 区间
    for(auto& v : h_weight) v = rand_float() * 0.1f - 0.05f; // 小权重初始化

    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);

    LOG(INFO) << "Start first Forward call";
    conv.Forward(input, output);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i){
        conv.Forward(input, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average Convolution2D time: " << ms / 10 << " ms" << std::endl;
    std::cout << "Throughput: " << (batch_size * in_channels * height * width * sizeof(float) + batch_size * out_channels * out_height * out_width * sizeof(float)) * 10 / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
