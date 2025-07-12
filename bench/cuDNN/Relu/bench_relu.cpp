#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    int m = 1024, n = 1024;
    if(argc == 3){
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
    }

    std::cout << "Relu input shape: (" << m << ", " << n << ")" << std::endl;

    Tensor<float> input({static_cast<size_t>(m), static_cast<size_t>(n)});
    Tensor<float> output({static_cast<size_t>(m), static_cast<size_t>(n)});

    std::vector<float> h_input(m * n);
    for(auto& v : h_input) v = rand_float() * 2 - 1; // [-1, 1] 区间

    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    Relu<float> relu;
    relu.Forward(input, output);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i){
        relu.Forward(input, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average Relu time: " << ms / 10 << " ms " << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} 