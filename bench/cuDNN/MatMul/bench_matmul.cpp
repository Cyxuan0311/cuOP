#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuDNN/matmul.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int m = 1024, k = 1024, n = 1024; // 默认矩阵大小
    bool transA = false, transB = false; // 默认不转置
    
    if(argc >= 4){
        m = std::atoi(argv[1]);
        k = std::atoi(argv[2]);
        n = std::atoi(argv[3]);
    }
    if(argc >= 6){
        transA = std::atoi(argv[4]) != 0;
        transB = std::atoi(argv[5]) != 0;
    }

    std::cout << "MatMul shape: (" << m << ", " << k << ") x (" << k << ", " << n << ") transA=" << transA << " transB=" << transB << std::endl;

    // 根据转置情况调整矩阵维度
    int A_rows = transA ? k : m;
    int A_cols = transA ? m : k;
    int B_rows = transB ? n : k;
    int B_cols = transB ? k : n;
    
    Tensor<float> A({static_cast<size_t>(A_rows), static_cast<size_t>(A_cols)});
    Tensor<float> B({static_cast<size_t>(B_rows), static_cast<size_t>(B_cols)});
    Tensor<float> C({static_cast<size_t>(m), static_cast<size_t>(n)});

    std::vector<float> h_A(A_rows * A_cols), h_B(B_rows * B_cols);
    for(auto& v : h_A) v = rand_float() * 0.1f - 0.05f; // 小权重初始化
    for(auto& v : h_B) v = rand_float() * 0.1f - 0.05f;

    cudaMemcpy(A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    MatMul<float> matmul(transA, transB);

    LOG(INFO) << "Start first Forward call";
    matmul.Forward(A, B, C, -1);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i){
        matmul.Forward(A, B, C, -1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average MatMul time: " << ms / 10 << " ms" << std::endl;
    std::cout << "Throughput: " << (m * k * n * sizeof(float) * 2 * 10) / (ms / 1000.0f) / 1e9 << " GFLOPS" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
