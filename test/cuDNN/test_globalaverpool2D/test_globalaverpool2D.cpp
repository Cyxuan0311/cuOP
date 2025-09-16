#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/globalaverpool.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace cu_op_mem;

class GlobalAveragePool2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        google::InitGoogleLogging("GlobalAveragePool2DTest");
    }
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(GlobalAveragePool2DTest, Forward2D) {
    // 2D 输入测试
    float h_input[4] = {1, 2, 3, 4}; // 2x2
    float h_output[1] = {0};
    Tensor<float> input({2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    GlobalAveragePool2D<float> pool;
    pool.Forward(input, output, 2, 3);
    float h_result[1] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    float expected = (1+2+3+4)/4.0f;
    EXPECT_NEAR(h_result[0], expected, 1e-5);
}

TEST_F(GlobalAveragePool2DTest, Forward4D) {
    // 4D 输入测试 [N, C, H, W] = [1, 1, 2, 2]
    float h_input[4] = {1, 2, 3, 4};
    float h_output[1] = {0};
    Tensor<float> input({1, 1, 2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    GlobalAveragePool2D<float> pool;
    pool.Forward(input, output, 2, 3);
    float h_result[1] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    float expected = (1+2+3+4)/4.0f;
    EXPECT_NEAR(h_result[0], expected, 1e-5);
}

TEST_F(GlobalAveragePool2DTest, Forward4DBatchChannel) {
    // 4D 输入测试 [N, C, H, W] = [2, 2, 2, 2]
    float h_input[16] = {
        1, 2, 3, 4, 5, 6, 7, 8, // N=0, C=0,1
        9, 10, 11, 12, 13, 14, 15, 16 // N=1, C=0,1
    };
    Tensor<float> input({2, 2, 2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    GlobalAveragePool2D<float> pool;
    pool.Forward(input, output, 2, 3);
    float h_result[4] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // 每个 [N, C] 独立池化
    for (int i = 0; i < 4; ++i) {
        float sum = 0;
        for (int j = 0; j < 4; ++j) sum += h_input[i*4 + j];
        float expected = sum / 4.0f;
        EXPECT_NEAR(h_result[i], expected, 1e-5);
    }
} 