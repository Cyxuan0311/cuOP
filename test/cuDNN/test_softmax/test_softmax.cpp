#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <cmath>

using namespace cu_op_mem;

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        google::InitGoogleLogging("SoftmaxTest");
    }
    void TearDown() override {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(SoftmaxTest, ForwardSingleRow) {
    // 测试单行数据
    float h_input[3] = {1.0f, 2.0f, 3.0f};  // 1x3
    float h_output[3] = {0};                 // 1x3

    Tensor<float> input({1, 3});
    Tensor<float> output({1, 3});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Softmax<float> softmax;
    softmax.Forward(input, output);

    float h_result[3] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 计算期望值
    float sum = expf(1.0f) + expf(2.0f) + expf(3.0f);
    float expected[3] = {
        expf(1.0f)/sum,
        expf(2.0f)/sum,
        expf(3.0f)/sum
    };

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SoftmaxTest, ForwardMultiRow) {
    // 测试多行数据(2行3列)
    float h_input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};  // 2x3
    float h_output[6] = {0};                                    // 2x3

    Tensor<float> input({2, 3});
    Tensor<float> output({2, 3});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Softmax<float> softmax;
    softmax.Forward(input, output);

    float h_result[6] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 计算期望值(每行独立计算)
    // 第一行
    float sum1 = expf(1.0f) + expf(2.0f) + expf(3.0f);
    float expected1[3] = {
        expf(1.0f)/sum1,
        expf(2.0f)/sum1,
        expf(3.0f)/sum1
    };
    
    // 第二行
    float sum2 = expf(-1.0f) + expf(0.0f) + expf(1.0f);
    float expected2[3] = {
        expf(-1.0f)/sum2,
        expf(0.0f)/sum2,
        expf(1.0f)/sum2
    };

    // 验证第一行
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(h_result[i], expected1[i], 1e-5);
    }
    
    // 验证第二行
    for (int i = 3; i < 6; i++) {
        EXPECT_NEAR(h_result[i], expected2[i-3], 1e-5);
    }
}

TEST_F(SoftmaxTest, ForwardNegativeValues) {
    // 测试负值输入
    float h_input[4] = {-2.0f, -1.0f, 0.0f, 1.0f};  // 1x4
    float h_output[4] = {0};                        // 1x4

    Tensor<float> input({1, 4});
    Tensor<float> output({1, 4});

    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);

    Softmax<float> softmax;
    softmax.Forward(input, output);

    float h_result[4] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 计算期望值
    float sum = expf(-2.0f) + expf(-1.0f) + expf(0.0f) + expf(1.0f);
    float expected[4] = {
        expf(-2.0f)/sum,
        expf(-1.0f)/sum,
        expf(0.0f)/sum,
        expf(1.0f)/sum
    };

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], expected[i], 1e-5);
    }
}

TEST_F(SoftmaxTest, Forward4DTensorDim3) {
    // 测试4维张量，在最后一维做softmax
    // shape: [2, 2, 2, 2]
    float h_input[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        -1, 0, 1, 2, -2, -1, 0, 1
    };
    float h_output[16] = {0};
    Tensor<float> input({2, 2, 2, 2});
    Tensor<float> output({2, 2, 2, 2});
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    Softmax<float> softmax;
    softmax.Forward(input, output, 3); // 在最后一维做softmax
    float h_result[16] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    // 期望结果：每个最后一维做softmax
    // 例如 [a, b] -> [exp(a)/(exp(a)+exp(b)), exp(b)/(exp(a)+exp(b))]
    for (int i = 0; i < 8; ++i) {
        float a = h_input[2*i];
        float b = h_input[2*i+1];
        float ea = expf(a);
        float eb = expf(b);
        float sum = ea + eb;
        float expected0 = ea / sum;
        float expected1 = eb / sum;
        EXPECT_NEAR(h_result[2*i], expected0, 1e-5);
        EXPECT_NEAR(h_result[2*i+1], expected1, 1e-5);
    }
}