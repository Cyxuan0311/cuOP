#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuBlas/scal.hpp"
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
    float alpha = 2.5f;   // 默认标量值
    
    if(argc >= 2){
        n = std::atoi(argv[1]);
    }
    if(argc >= 3){
        alpha = std::atof(argv[2]);
    }

    std::cout << "SCAL shape: (" << n << ") with alpha = " << alpha << std::endl;

    Tensor<float> x({static_cast<size_t>(n)});

    std::vector<float> h_x(n);
    for(auto& v : h_x) v = rand_float();

    cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    Scal<float> scal(alpha);

    LOG(INFO) << "Start first Forward call";
    scal.Forward(x);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 100; ++i){
        scal.Forward(x);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average SCAL time: " << ms / 100 << " ms" << std::endl;
    std::cout << "Throughput: " << (n * sizeof(float) * 100) / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
