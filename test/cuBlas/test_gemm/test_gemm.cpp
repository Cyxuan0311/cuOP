#include <gtest/gtest.h>
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <glog/logging.h>
#include "data/tensor"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class GemmTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("GemmTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(GemmTest,ForwardFloatNoTranspose){
    float h_input[6] = {1, 2, 3, 4, 5, 6};
    float h_weight[12] = {1,0,0,0,0,1,0,0,0,0,1,0};
    float h_output[8] = {0};

    Tensor<float> input({2,3});
    Tensor<float> weight({3,4});
    Tensor<float> output({2,4});

    cudaMemcpy(input.data(),h_input,sizeof(h_input),cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(),h_weight,sizeof(h_weight),cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(),h_output,sizeof(h_output),cudaMemcpyHostToDevice);

    Gemm<float> gemm(false,false,1.0f,0.0f);
    gemm.SetWeight(weight);
    gemm.Forward(input, output);

    float h_result[8] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    float expected[8] = {1,2,3,0,4,5,6,0};
    for (int i=0;i<8;i++){
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}
