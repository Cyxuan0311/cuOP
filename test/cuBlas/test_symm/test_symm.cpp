#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/symm.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class SymmTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("SymmTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(SymmTest, ForwardFloatBasic) {
    // 创建2x2对称矩阵A
    float h_A[4] = {1.0f, 2.0f, 2.0f, 3.0f}; // 对称矩阵
    float h_B[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2矩阵
    float h_C[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 输出矩阵
    float alpha = 1.0f, beta = 0.0f;
    
    // 期望结果: C = alpha * A * B + beta * C
    // A = [1 2; 2 3], B = [1 2; 3 4]
    // A*B = [7 10; 11 16]
    float expected[4] = {7.0f, 10.0f, 11.0f, 16.0f};

    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});
    Tensor<float> C({2, 2});
    
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);
    symm.Forward(B, C);

    float h_result[4] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SymmTest, ForwardFloatWithBeta) {
    // 创建2x2对称矩阵A
    float h_A[4] = {1.0f, 2.0f, 2.0f, 3.0f}; // 对称矩阵
    float h_B[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2矩阵
    float h_C[4] = {0.5f, 1.0f, 1.5f, 2.0f}; // 初始C矩阵
    float alpha = 1.0f, beta = 2.0f;
    
    // 期望结果: C = alpha * A * B + beta * C
    // A*B = [7 10; 11 16], beta*C = [1 2; 3 4]
    // 结果 = [8 12; 14 20]
    float expected[4] = {8.0f, 12.0f, 14.0f, 20.0f};

    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});
    Tensor<float> C({2, 2});
    
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);
    symm.Forward(B, C);

    float h_result[4] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SymmTest, ForwardFloatZeroAlpha) {
    // 创建2x2对称矩阵A
    float h_A[4] = {1.0f, 2.0f, 2.0f, 3.0f}; // 对称矩阵
    float h_B[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2矩阵
    float h_C[4] = {0.5f, 1.0f, 1.5f, 2.0f}; // 初始C矩阵
    float alpha = 0.0f, beta = 1.0f;
    
    // 期望结果: C = alpha * A * B + beta * C = beta * C
    float expected[4] = {0.5f, 1.0f, 1.5f, 2.0f};

    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});
    Tensor<float> C({2, 2});
    
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);
    symm.Forward(B, C);

    float h_result[4] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SymmTest, ForwardFloatIdentity) {
    // 创建2x2单位矩阵A（对称）
    float h_A[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // 单位矩阵
    float h_B[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2矩阵
    float h_C[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 输出矩阵
    float alpha = 1.0f, beta = 0.0f;
    
    // 期望结果: C = alpha * I * B + beta * C = B
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});
    Tensor<float> C({2, 2});
    
    cudaMemcpy(A.data(), h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C, sizeof(h_C), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);
    symm.Forward(B, C);

    float h_result[4] = {0};
    cudaMemcpy(h_result, C.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SymmTest, ForwardFloatLargeMatrix) {
    const int n = 8;
    std::vector<float> h_A(n * n);
    std::vector<float> h_B(n * n);
    std::vector<float> h_C(n * n, 0.0f);
    std::vector<float> expected(n * n);
    float alpha = 1.0f, beta = 0.0f;
    
    // 创建对称矩阵A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i <= j) {
                h_A[i * n + j] = static_cast<float>(i + j + 1);
            } else {
                h_A[i * n + j] = h_A[j * n + i]; // 对称性
            }
        }
    }
    
    // 创建矩阵B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i * n + j] = static_cast<float>(i * n + j + 1);
        }
    }
    
    // 计算期望结果
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            expected[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                expected[i * n + j] += h_A[i * n + k] * h_B[k * n + j];
            }
        }
    }

    Tensor<float> A({static_cast<size_t>(n), static_cast<size_t>(n)});
    Tensor<float> B({static_cast<size_t>(n), static_cast<size_t>(n)});
    Tensor<float> C({static_cast<size_t>(n), static_cast<size_t>(n)});
    
    cudaMemcpy(A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C.data(), h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice);

    Symm<float> symm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, alpha, beta);
    symm.SetWeight(A);
    symm.Forward(B, C);

    std::vector<float> h_result(n * n);
    cudaMemcpy(h_result.data(), C.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * n; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-3); // 使用稍大的容差
    }
}
