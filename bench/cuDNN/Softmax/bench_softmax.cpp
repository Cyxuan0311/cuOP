#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int batch_size = 32, seq_len = 128, hidden_size = 512; // 默认参数
    int dim = -1; // 默认在最后一个维度上计算softmax
    
    if(argc >= 4){
        batch_size = std::atoi(argv[1]);
        seq_len = std::atoi(argv[2]);
        hidden_size = std::atoi(argv[3]);
    }
    if(argc >= 5){
        dim = std::atoi(argv[4]);
    }

    std::cout << "Softmax input shape: (" << batch_size << ", " << seq_len << ", " << hidden_size << ") dim = " << dim << std::endl;

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});

    std::vector<float> h_input(batch_size * seq_len * hidden_size);
    for(auto& v : h_input) v = rand_float() * 4 - 2; // [-2, 2] 区间，避免数值溢出

    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    Softmax<float> softmax;

    LOG(INFO) << "Start first Forward call";
    softmax.Forward(input, output, dim);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 100; ++i){
        softmax.Forward(input, output, dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average Softmax time: " << ms / 100 << " ms" << std::endl;
    std::cout << "Throughput: " << (batch_size * seq_len * hidden_size * sizeof(float) * 2 * 100) / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
