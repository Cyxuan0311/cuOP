#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/matmul.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace cu_op_mem;

class MatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        google::InitGoogleLogging("MatMulTest");
    }
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(MatMulTest, Forward2D) {
    // 2D 输入测试
    float h_A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    float h_B[12] = {1,0,0,0,0,1,0,0,0,0,1,0}; // 3x4
    float h_C[8] = {0};
    Tensor<float> A({2, 3});
    Tensor<float> B({3, 4});
    Tensor<float> C({2, 4});
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    MatMul<float> matmul;
    matmul.Forward(A, B, C);
    float h_result[8] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    float expected[8] = {1,2,3,0,4,5,6,0};
    for (int i=0;i<8;i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(MatMulTest, ForwardBatch3D) {
    // 3D batch matmul: [2, 2, 3] x [2, 3, 4] -> [2, 2, 4]
    float h_A[12] = {
        1,2,3,4,5,6, // batch 0, 2x3
        7,8,9,10,11,12 // batch 1, 2x3
    };
    float h_B[24] = {
        1,0,0,0,0,1,0,0,0,0,1,0, // batch 0, 3x4
        2,0,0,0,0,2,0,0,0,0,2,0 // batch 1, 3x4
    };
    Tensor<float> A({2,2,3});
    Tensor<float> B({2,3,4});
    Tensor<float> C;
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    MatMul<float> matmul;
    matmul.Forward(A, B, C);
    float h_result[16] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // batch 0: [1,2,3;4,5,6] x [1,0,0,0;0,1,0,0;0,0,1,0]
    float expected0[8] = {1,2,3,0,4,5,6,0};
    // batch 1: [7,8,9;10,11,12] x [2,0,0,0;0,2,0,0;0,0,2,0]
    float expected1[8] = {14,16,18,0,20,22,24,0};
    for (int i=0;i<8;i++) {
        EXPECT_NEAR(h_result[i], expected0[i], 1e-5);
        EXPECT_NEAR(h_result[8+i], expected1[i], 1e-5);
    }
} 