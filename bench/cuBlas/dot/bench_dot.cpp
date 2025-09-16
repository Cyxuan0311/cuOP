#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuBlas/dot.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int n = 1024 * 1024; // 默认向量大小
    
    if(argc >= 2){
        n = std::atoi(argv[1]);
    }

    std::cout << "DOT shape: (" << n << ")" << std::endl;

    Tensor<float> x({static_cast<size_t>(n)});
    Tensor<float> y({static_cast<size_t>(n)});

    std::vector<float> h_x(n), h_y(n);
    for(auto& v : h_x) v = rand_float();
    for(auto& v : h_y) v = rand_float();

    cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    Dot<float> dot;

    LOG(INFO) << "Start first Forward call";
    float result;
    StatusCode status = dot.Forward(x, y, result);
    cudaDeviceSynchronize();

    std::cout << "First DOT result: " << result << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 100; ++i){
        float temp_result;
        dot.Forward(x, y, temp_result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average DOT time: " << ms / 100 << " ms" << std::endl;
    std::cout << "Throughput: " << (n * sizeof(float) * 2 * 100) / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
