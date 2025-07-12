#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/relu.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class ReluTest : public ::testing::Test {
protected:
    void SetUp() override {
        google::InitGoogleLogging("ReluTest");
    }
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(ReluTest, Forward_int_number){
    float h_input[6] = {1, -2, 3, -4, 0, 5}; // 2x3
    float h_output[6] = {0};                 // 2x3

    Tensor<float> input({2,3});
    Tensor<float> output({2,3});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Relu<float> relu;
    relu.Forward(input, output);

    float h_result[6] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 期望结果: [1,0,3,0,0,5]
    float expected[6] = {1,0,3,0,0,5};
    for (int i=0; i<6; i++){
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(ReluTest, Forward_float_number){
    float h_input[6] = {1.7, -2, 3.877, -4.79606, 0, 5}; // 2x3
    float h_output[6] = {0};                 // 2x3

    Tensor<float> input({2,3});
    Tensor<float> output({2,3});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Relu<float> relu;
    relu.Forward(input, output);

    float h_result[6] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 期望结果: [1,0,3,0,0,5]
    float expected[6] = {1.7,0,3.877,0,0,5};
    for (int i=0; i<6; i++){
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}