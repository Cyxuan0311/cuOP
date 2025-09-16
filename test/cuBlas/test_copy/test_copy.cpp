#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/copy.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class CopyTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("CopyTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(CopyTest, ForwardFloatBasic) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Copy<float> copy;
    copy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(CopyTest, ForwardFloatOverwrite) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Copy<float> copy;
    copy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(CopyTest, ForwardFloatNegative) {
    float h_x[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[4] = {-1.0f, -2.0f, -3.0f, -4.0f};

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Copy<float> copy;
    copy.Forward(x, y);

    float h_result[4] = {0};
    cudaMemcpy(h_result, y.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(CopyTest, ForwardFloatLargeVector) {
    const int n = 1024;
    std::vector<float> h_x(n);
    std::vector<float> h_y(n, 0.0f);
    std::vector<float> expected(n);
    
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i * 0.1f);
        expected[i] = h_x[i];
    }

    Tensor<float> x({static_cast<size_t>(n)});
    Tensor<float> y({static_cast<size_t>(n)});
    
    cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    Copy<float> copy;
    copy.Forward(x, y);

    std::vector<float> h_result(n);
    cudaMemcpy(h_result.data(), y.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(CopyTest, ForwardFloatSameTensor) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> x({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    Copy<float> copy;
    copy.Forward(x, x); // 复制到自己

    float h_result[4] = {0};
    cudaMemcpy(h_result, x.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}
