#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/averagepool.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int batch_size = 32, channels = 64, height = 32, width = 32; // 默认输入参数
    int pool_h = 2, pool_w = 2, stride_h = 2, stride_w = 2; // 默认池化参数
    
    if(argc >= 5){
        batch_size = std::atoi(argv[1]);
        channels = std::atoi(argv[2]);
        height = std::atoi(argv[3]);
        width = std::atoi(argv[4]);
    }
    if(argc >= 9){
        pool_h = std::atoi(argv[5]);
        pool_w = std::atoi(argv[6]);
        stride_h = std::atoi(argv[7]);
        stride_w = std::atoi(argv[8]);
    }

    std::cout << "AveragePool2D input shape: (" << batch_size << ", " << channels << ", " << height << ", " << width << ")" << std::endl;
    std::cout << "AveragePool2D params: pool=(" << pool_h << ", " << pool_w << "), stride=(" << stride_h << ", " << stride_w << ")" << std::endl;

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)});
    
    // 计算输出尺寸
    int out_height = (height - pool_h) / stride_h + 1;
    int out_width = (width - pool_w) / stride_w + 1;
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(channels), static_cast<size_t>(out_height), static_cast<size_t>(out_width)});

    std::vector<float> h_input(batch_size * channels * height * width);
    
    for(auto& v : h_input) v = rand_float() * 2 - 1; // [-1, 1] 区间

    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    AveragePool2D<float> avgpool(pool_h, pool_w, stride_h, stride_w);

    LOG(INFO) << "Start first Forward call";
    avgpool.Forward(input, output, 2, 3);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 100; ++i){
        avgpool.Forward(input, output, 2, 3);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average AveragePool2D time: " << ms / 100 << " ms" << std::endl;
    std::cout << "Throughput: " << (batch_size * channels * height * width * sizeof(float) + batch_size * channels * out_height * out_width * sizeof(float)) * 100 / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
