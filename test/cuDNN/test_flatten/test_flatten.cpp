#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/flatten.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class FlattenTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("FlattenTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(FlattenTest, Forward2D) {
    // 测试2D张量展平
    const int height = 3;
    const int width = 4;
    
    // 输入数据
    float h_input[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    
    float h_output[12] = {0};
    
    Tensor<float> input({height, width});
    Tensor<float> output({height * width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[12] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, Forward3D) {
    // 测试3D张量展平
    const int depth = 2;
    const int height = 3;
    const int width = 4;
    
    // 输入数据
    float h_input[24] = {
        // 第一个深度
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        // 第二个深度
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    };
    
    float h_output[24] = {0};
    
    Tensor<float> input({depth, height, width});
    Tensor<float> output({depth * height * width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[24] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < 24; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, Forward4D) {
    // 测试4D张量展平
    const int batch_size = 2;
    const int channels = 2;
    const int height = 2;
    const int width = 3;
    
    // 输入数据
    float h_input[24] = {
        // 第一个样本
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  // 第一个通道
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, // 第二个通道
        // 第二个样本
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, // 第一个通道
        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f  // 第二个通道
    };
    
    float h_output[24] = {0};
    
    Tensor<float> input({batch_size, channels, height, width});
    Tensor<float> output({batch_size * channels * height * width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[24] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < 24; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, Forward1D) {
    // 测试1D张量展平（应该保持不变）
    const int size = 5;
    
    // 输入数据
    float h_input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    float h_output[5] = {0};
    
    Tensor<float> input({size});
    Tensor<float> output({size});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[5] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, ForwardLargeTensor) {
    // 测试大张量展平
    const int batch_size = 4;
    const int channels = 8;
    const int height = 16;
    const int width = 16;
    
    const int total_size = batch_size * channels * height * width;
    std::vector<float> h_input(total_size);
    std::vector<float> h_output(total_size, 0.0f);
    
    // 初始化输入数据
    for (int i = 0; i < total_size; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(channels), 
                        static_cast<size_t>(height), static_cast<size_t>(width)});
    Tensor<float> output({static_cast<size_t>(total_size)});
    
    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    std::vector<float> h_result(total_size);
    cudaMemcpy(h_result.data(), output.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < total_size; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, ForwardNegativeValues) {
    // 测试负值张量展平
    const int height = 2;
    const int width = 3;
    
    // 输入数据（包含负值）
    float h_input[6] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
    
    float h_output[6] = {0};
    
    Tensor<float> input({height, width});
    Tensor<float> output({height * width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[6] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同
    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-5);
    }
}

TEST_F(FlattenTest, ForwardFloatPrecision) {
    // 测试浮点精度
    const int height = 2;
    const int width = 2;
    
    // 输入数据（包含小数）
    float h_input[4] = {1.234567f, 2.345678f, 3.456789f, 4.567890f};
    
    float h_output[4] = {0};
    
    Tensor<float> input({height, width});
    Tensor<float> output({height * width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    
    Flatten<float> flatten;
    flatten.Forward(input, output, 0);
    
    float h_result[4] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果与输入相同（使用适当的精度）
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(h_result[i], h_input[i], 1e-6);
    }
}
