#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuBlas/symm.hpp"
#include "data/tensor.hpp"

using namespace cu_op_mem;

float rand_float(){
    return static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char** argv){
    cudaSetDevice(0);
    cudaFree(0); // 强制初始化 CUDA 上下文
    google::InitGoogleLogging(argv[0]);
    
    int m = 1024, n = 1024; // 默认矩阵大小
    float alpha = 1.0f, beta = 0.0f; // 默认标量值
    
    if(argc >= 3){
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
    }
    if(argc >= 4){
        alpha = std::atof(argv[3]);
    }
    if(argc >= 5){
        beta = std::atof(argv[4]);
    }

    std::cout << "SYMM shape: (" << m << ", " << m << ") x (" << m << ", " << n << ") with alpha = " << alpha << ", beta = " << beta << std::endl;

    // A是m×m对称矩阵，B是m×n矩阵，C是m×n矩阵
    Tensor<float> A({static_cast<size_t>(m), static_cast<size_t>(m)});
    Tensor<float> B({static_cast<size_t>(m), static_cast<size_t>(n)});
    Tensor<float> C({static_cast<size_t>(m), static_cast<size_t>(n)});

    std::vector<float> h_A(m * m), h_B(m * n), h_C(m * n);
    
    // 生成对称矩阵A
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < m; ++j){
            if(i <= j){
                h_A[i * m + j] = rand_float();
            } else {
                h_A[i * m + j] = h_A[j * m + i]; // 对称性
            }
        }
    }
    
    for(auto& v : h_B) v = rand_float();
    for(auto& v : h_C) v = rand_float();

    cudaMemcpy(A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);

    LOG(INFO) << "Start first Forward call";
    symm.Forward(B, C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i){
        symm.Forward(B, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average SYMM time: " << ms / 10 << " ms" << std::endl;
    std::cout << "Throughput: " << (m * m * sizeof(float) + m * n * sizeof(float) * 2) * 10 / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
