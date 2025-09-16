#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/axpy.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class AxpyTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("AxpyTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(AxpyTest, ForwardFloatBasic) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float alpha = 2.0f;
    float expected[4] = {2.5f, 5.0f, 7.5f, 10.0f}; // alpha*x + y

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Axpy<float> axpy(alpha);
    axpy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(AxpyTest, ForwardFloatZeroAlpha) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float alpha = 0.0f;
    float expected[4] = {0.5f, 1.0f, 1.5f, 2.0f}; // y unchanged

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Axpy<float> axpy(alpha);
    axpy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(AxpyTest, ForwardFloatNegativeAlpha) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float alpha = -1.0f;
    float expected[4] = {-0.5f, -1.0f, -1.5f, -2.0f}; // -x + y

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Axpy<float> axpy(alpha);
    axpy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(AxpyTest, ForwardFloatLargeVector) {
    const int n = 1024;
    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> expected(n);
    float alpha = 2.5f;
    
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i * 0.5f);
        expected[i] = alpha * h_x[i] + h_y[i];
    }

    Tensor<float> x({static_cast<size_t>(n)});
    Tensor<float> y({static_cast<size_t>(n)});
    
    cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    Axpy<float> axpy(alpha);
    axpy.Forward(x, y);

    std::vector<float> h_result(n);
    cudaMemcpy(h_result.data(), y.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}
