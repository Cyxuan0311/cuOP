#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "cuda_op/detail/cuBlas/trsm.hpp"
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
    float alpha = 1.0f; // 默认标量值
    
    if(argc >= 3){
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
    }
    if(argc >= 4){
        alpha = std::atof(argv[3]);
    }

    std::cout << "TRSM shape: (" << m << ", " << m << ") x (" << m << ", " << n << ") with alpha = " << alpha << std::endl;

    // A是m×m上三角矩阵，B是m×n矩阵
    Tensor<float> A({static_cast<size_t>(m), static_cast<size_t>(m)});
    Tensor<float> B({static_cast<size_t>(m), static_cast<size_t>(n)});

    std::vector<float> h_A(m * m), h_B(m * n);
    
    // 生成上三角矩阵A
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < m; ++j){
            if(i <= j){
                h_A[i * m + j] = rand_float() + 1.0f; // 确保对角元素不为0
            } else {
                h_A[i * m + j] = 0.0f; // 下三角为0
            }
        }
    }
    
    for(auto& v : h_B) v = rand_float();

    cudaMemcpy(A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    Trsm<float> trsm(0, 1, 0, 0, alpha);
    trsm.SetMatrixA(A);

    LOG(INFO) << "Start first Forward call";
    trsm.Forward(B, B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Start timing loop";
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i){
        trsm.Forward(B, B);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Average TRSM time: " << ms / 10 << " ms" << std::endl;
    std::cout << "Throughput: " << (m * m * sizeof(float) + m * n * sizeof(float)) * 10 / (ms / 1000.0f) / 1e9 << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
