#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    int m = 1024,k = 1024,n = 1024;
    if(argc == 4){
        m = std::atoi(argv[1]);
        k = std::atoi(argv[2]);
        n = std::atoi(argv[3]);
    }

    std::cout<<"GEMM shape: ("<<m<<" , "<<k<<") x ("<<" , "<<n<<")"<<std::endl;

    Tensor<float> input({static_cast<size_t>(m),static_cast<size_t>(k)});
    Tensor<float> weight({static_cast<size_t>(k),static_cast<size_t>(n)});
    Tensor<float> output({static_cast<size_t>(m),static_cast<size_t>(n)});

    std::vector<float> h_input(m*k),h_weight(k*n);
    for(auto& v : h_input) v = rand_float();
    for(auto& v : h_weight) v = rand_float();

    cudaMemcpy(input.data(),h_input.data(),h_input.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(),h_weight.data(),h_weight.size()*sizeof(float),cudaMemcpyHostToDevice);

    Gemm<float> gemm;
    gemm.SetWeight(weight);

    gemm.Forward(input,output);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0;i<10;++i){
        gemm.Forward(input,output);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout<<"Average GEMM time: "<<ms / 10<<" ms "<<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}