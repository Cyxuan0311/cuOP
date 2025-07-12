#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/gemv.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class GemvTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("GemvTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(GemvTest, ForwardFloatNoTranspose){
    float h_input[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    float h_weight[3] = {1, 2, 3};         // 3
    float h_output[2] = {0};               // 2

    Tensor<float> input({2,3});
    Tensor<float> weight({3});
    Tensor<float> output({2});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Gemv<float> gemv(false, 1.0f, 0.0f);
    gemv.SetWeight(weight);
    gemv.Forward(input, output);

    float h_result[2] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 期望结果: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    float expected[2] = {14, 32};
    for (int i=0; i<2; i++){
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}
