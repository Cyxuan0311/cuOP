#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <cmath>

using namespace cu_op_mem;

class BatchNormTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("BatchNormTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(BatchNormTest, Forward2DBasic) {
    // 测试2D输入: [batch_size, channels]
    const int batch_size = 4;
    const int channels = 2;
    const float eps = 1e-5f;
    
    // 输入数据
    float h_input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; // 4x2
    float h_output[8] = {0};
    
    // BatchNorm参数
    float h_gamma[2] = {1.0f, 1.0f};     // 缩放参数
    float h_beta[2] = {0.0f, 0.0f};      // 偏移参数
    float h_running_mean[2] = {0.0f, 0.0f};  // 运行均值
    float h_running_var[2] = {1.0f, 1.0f};   // 运行方差

    Tensor<float> input({batch_size, channels});
    Tensor<float> output({batch_size, channels});
    Tensor<float> gamma({channels});
    Tensor<float> beta({channels});
    Tensor<float> running_mean({channels});
    Tensor<float> running_var({channels});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean, sizeof(h_running_mean), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var, sizeof(h_running_var), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;
    batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);

    float h_result[8] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零（基本功能测试）
    bool all_zero = true;
    for (int i = 0; i < 8; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "BatchNorm result should not be all zeros";
}

TEST_F(BatchNormTest, Forward4DBasic) {
    // 测试4D输入: [batch_size, channels, height, width]
    const int batch_size = 2;
    const int channels = 2;
    const int height = 2;
    const int width = 2;
    const float eps = 1e-5f;
    
    // 输入数据
    float h_input[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,  // 第一个样本，第一个通道
        5.0f, 6.0f, 7.0f, 8.0f,  // 第一个样本，第二个通道
        9.0f, 10.0f, 11.0f, 12.0f, // 第二个样本，第一个通道
        13.0f, 14.0f, 15.0f, 16.0f // 第二个样本，第二个通道
    };
    float h_output[16] = {0};
    
    // BatchNorm参数
    float h_gamma[2] = {1.0f, 1.0f};
    float h_beta[2] = {0.0f, 0.0f};
    float h_running_mean[2] = {0.0f, 0.0f};
    float h_running_var[2] = {1.0f, 1.0f};

    Tensor<float> input({batch_size, channels, height, width});
    Tensor<float> output({batch_size, channels, height, width});
    Tensor<float> gamma({channels});
    Tensor<float> beta({channels});
    Tensor<float> running_mean({channels});
    Tensor<float> running_var({channels});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean, sizeof(h_running_mean), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var, sizeof(h_running_var), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;
    batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);

    float h_result[16] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 16; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "BatchNorm result should not be all zeros";
}

TEST_F(BatchNormTest, ForwardWithGammaBeta) {
    // 测试带缩放和偏移参数的BatchNorm
    const int batch_size = 3;
    const int channels = 1;
    const float eps = 1e-5f;
    
    float h_input[3] = {1.0f, 2.0f, 3.0f};
    float h_output[3] = {0};
    
    // 设置非平凡的gamma和beta
    float h_gamma[1] = {2.0f};     // 缩放因子
    float h_beta[1] = {1.0f};      // 偏移
    float h_running_mean[1] = {0.0f};
    float h_running_var[1] = {1.0f};

    Tensor<float> input({batch_size, channels});
    Tensor<float> output({batch_size, channels});
    Tensor<float> gamma({channels});
    Tensor<float> beta({channels});
    Tensor<float> running_mean({channels});
    Tensor<float> running_var({channels});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean, sizeof(h_running_mean), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var, sizeof(h_running_var), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;
    batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);

    float h_result[3] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 3; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "BatchNorm result should not be all zeros";
}

TEST_F(BatchNormTest, ForwardLargeTensor) {
    // 测试大张量
    const int batch_size = 8;
    const int channels = 4;
    const int height = 4;
    const int width = 4;
    const float eps = 1e-5f;
    
    const int total_size = batch_size * channels * height * width;
    std::vector<float> h_input(total_size);
    std::vector<float> h_output(total_size, 0.0f);
    
    // 初始化输入数据
    for (int i = 0; i < total_size; i++) {
        h_input[i] = static_cast<float>(i % 10) - 5.0f; // 范围 [-5, 4]
    }
    
    // BatchNorm参数
    std::vector<float> h_gamma(channels, 1.0f);
    std::vector<float> h_beta(channels, 0.0f);
    std::vector<float> h_running_mean(channels, 0.0f);
    std::vector<float> h_running_var(channels, 1.0f);

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(channels), 
                        static_cast<size_t>(height), static_cast<size_t>(width)});
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(channels), 
                         static_cast<size_t>(height), static_cast<size_t>(width)});
    Tensor<float> gamma({static_cast<size_t>(channels)});
    Tensor<float> beta({static_cast<size_t>(channels)});
    Tensor<float> running_mean({static_cast<size_t>(channels)});
    Tensor<float> running_var({static_cast<size_t>(channels)});
    
    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean.data(), h_running_mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var.data(), h_running_var.size() * sizeof(float), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;
    batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps);

    std::vector<float> h_result(total_size);
    cudaMemcpy(h_result.data(), output.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < total_size; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "BatchNorm result should not be all zeros";
}

TEST_F(BatchNormTest, ForwardZeroEpsilon) {
    // 测试零epsilon值（应该使用默认值）
    const int batch_size = 2;
    const int channels = 1;
    const float eps = 0.0f; // 零epsilon
    
    float h_input[2] = {1.0f, 2.0f};
    float h_output[2] = {0};
    
    float h_gamma[1] = {1.0f};
    float h_beta[1] = {0.0f};
    float h_running_mean[1] = {0.0f};
    float h_running_var[1] = {1.0f};

    Tensor<float> input({batch_size, channels});
    Tensor<float> output({batch_size, channels});
    Tensor<float> gamma({channels});
    Tensor<float> beta({channels});
    Tensor<float> running_mean({channels});
    Tensor<float> running_var({channels});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);
    cudaMemcpy(running_mean.data(), h_running_mean, sizeof(h_running_mean), cudaMemcpyHostToDevice);
    cudaMemcpy(running_var.data(), h_running_var, sizeof(h_running_var), cudaMemcpyHostToDevice);

    BatchNorm<float> batchnorm;
    // 应该能正常处理零epsilon
    EXPECT_NO_THROW(batchnorm.Forward(input, output, gamma, beta, running_mean, running_var, eps));
}
