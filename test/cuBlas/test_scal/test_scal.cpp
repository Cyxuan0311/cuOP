#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/scal.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class ScalTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("ScalTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(ScalTest, ForwardFloatBasic) {
    float h_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float alpha = 2.5f;
    float expected[4] = {2.5f, 5.0f, 7.5f, 10.0f};

    Tensor<float> input({4});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);

    Scal<float> scal(alpha);
    scal.Forward(input);

    float h_result[4] = {0};
    cudaMemcpy(h_result, input.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(ScalTest, ForwardFloatZero) {
    float h_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float alpha = 0.0f;
    float expected[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> input({4});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);

    Scal<float> scal(alpha);
    scal.Forward(input);

    float h_result[4] = {0};
    cudaMemcpy(h_result, input.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(ScalTest, ForwardFloatNegative) {
    float h_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float alpha = -1.0f;
    float expected[4] = {-1.0f, -2.0f, -3.0f, -4.0f};

    Tensor<float> input({4});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);

    Scal<float> scal(alpha);
    scal.Forward(input);

    float h_result[4] = {0};
    cudaMemcpy(h_result, input.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(ScalTest, ForwardFloatLargeVector) {
    const int n = 1024;
    std::vector<float> h_input(n);
    std::vector<float> expected(n);
    float alpha = 3.14f;
    
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i);
        expected[i] = static_cast<float>(i) * alpha;
    }

    Tensor<float> input({static_cast<size_t>(n)});
    
    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    Scal<float> scal(alpha);
    scal.Forward(input);

    std::vector<float> h_result(n);
    cudaMemcpy(h_result.data(), input.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}
