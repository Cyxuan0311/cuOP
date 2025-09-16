#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int batch_size = 32, channels = 64, height = 32, width = 32; // 默认参数
    float eps = 1e-5f;
    
    if(argc >= 5){
        batch_size = std::atoi(argv[1]);
        channels = std::atoi(argv[2]);
        height = std::atoi(argv[3]);
        width = std::atoi(argv[4]);
    }
    if(argc >= 6){
        eps = std::atof(argv[5]);
    }

    std::cout << "BatchNorm input shape: (" << batch_size << ", " << channels << ", " << height << ", " << width << ") eps = " << eps << std::endl;

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)});
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)});
    
    // BatchNorm参数
    Tensor<float> gamma({static_cast<size_t>(channels)});
    Tensor<float> beta({static_cast<size_t>(channels)});
    Tensor<float> running_mean({static_cast<size_t>(channels)});
    Tensor<float> running_var({static_cast<size_t>(channels)});

    std::vector<float> h_input(batch_size * channels * height * width);
    std::vector<float> h_gamma(channels, 1.0f);
    std::vector<float> h_beta(channels, 0.0f);
    std::vector<float> h_running_mean(channels, 0.0f);
    std::vector<float> h_running_var(channels, 1.0f);
    
    for(auto& v : h_input) v = rand_float() * 2 - 1; // [-1, 1] 区间

    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean.data(), h_running_mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var.data(), h_running_var.size() * sizeof(float), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;

    LOG(INFO) << "Start first Forward call";
    batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 100; ++i){
        batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average BatchNorm time: " << ms / 100 << " ms" << std::endl;
    std::cout << "Throughput: " << (batch_size * channels * height * width * sizeof(float) * 2 * 100) / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
