#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/averagepool.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace cu_op_mem;

class AveragePool2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        google::InitGoogleLogging("AveragePool2DTest");
    }
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(AveragePool2DTest, Forward2D) {
    // 2D 输入测试 - 使用1x1池化窗口，步长1x1（应该保持原值）
    float h_input[4] = {1, 2, 3, 4}; // 2x2
    float h_output[4] = {0};
    Tensor<float> input({2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    AveragePool2D<float> pool(1, 1, 1, 1); // 1x1池化，步长1x1
    pool.Forward(input, output, 2, 3);
    float h_result[4] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // 对于1x1池化，输出应该与输入相同
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(AveragePool2DTest, Forward4D) {
    // 4D 输入测试 [N, C, H, W] = [1, 1, 2, 2] - 使用1x1池化
    float h_input[4] = {1, 2, 3, 4};
    float h_output[4] = {0};
    Tensor<float> input({1, 1, 2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    AveragePool2D<float> pool(1, 1, 1, 1); // 1x1池化，步长1x1
    pool.Forward(input, output, 2, 3);
    float h_result[4] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // 对于1x1池化，输出应该与输入相同
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(AveragePool2DTest, Forward4DBatchChannel) {
    // 4D 输入测试 [N, C, H, W] = [2, 2, 2, 2] - 使用1x1池化
    float h_input[16] = {
        1, 2, 3, 4, 5, 6, 7, 8, // N=0, C=0,1
        9, 10, 11, 12, 13, 14, 15, 16 // N=1, C=0,1
    };
    Tensor<float> input({2, 2, 2, 2});
    Tensor<float> output;
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    AveragePool2D<float> pool(1, 1, 1, 1); // 1x1池化，步长1x1
    pool.Forward(input, output, 2, 3);
    float h_result[16] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // 对于1x1池化，输出应该与输入相同
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
} 