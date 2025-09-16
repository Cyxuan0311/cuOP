#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/dot.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class DotTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("DotTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(DotTest, ForwardFloatBasic) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float expected = 1.0f*2.0f + 2.0f*3.0f + 3.0f*4.0f + 4.0f*5.0f; // 40.0f

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, y, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST_F(DotTest, ForwardFloatZero) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected = 0.0f;

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, y, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST_F(DotTest, ForwardFloatNegative) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float expected = -30.0f; // 1*(-1) + 2*(-2) + 3*(-3) + 4*(-4)

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, y, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST_F(DotTest, ForwardFloatSameVector) {
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected = 30.0f; // 1*1 + 2*2 + 3*3 + 4*4

    Tensor<float> x({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, x, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    EXPECT_NEAR(result, expected, 1e-5);
}

TEST_F(DotTest, ForwardFloatLargeVector) {
    const int n = 1024;
    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    float expected = 0.0f;
    
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i * 0.5f);
        expected += h_x[i] * h_y[i];
    }

    Tensor<float> x({static_cast<size_t>(n)});
    Tensor<float> y({static_cast<size_t>(n)});
    
    cudaMemcpy(x.data(), h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, y, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    // 使用相对误差测试，因为大向量计算可能有累积误差
    float relative_error = std::abs(result - expected) / std::abs(expected);
    EXPECT_LT(relative_error, 1e-3); // 相对误差小于0.1%
}

TEST_F(DotTest, ForwardFloatOrthogonal) {
    float h_x[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float h_y[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float expected = 0.0f;

    Tensor<float> x({4});
    Tensor<float> y({4});
    
    cudaMemcpy(x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(y.data(), h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    Dot<float> dot;
    float result;
    StatusCode status = dot.Forward(x, y, result);
    
    EXPECT_EQ(status, StatusCode::SUCCESS);
    EXPECT_NEAR(result, expected, 1e-5);
}
