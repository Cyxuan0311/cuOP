#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/layernorm.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>
#include <cmath>

using namespace cu_op_mem;

class LayerNormTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("LayerNormTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(LayerNormTest, Forward2DBasic) {
    // 测试2D输入: [batch_size, features]
    const int batch_size = 2;
    const int features = 4;
    const int dim = -1; // 在最后一个维度上归一化
    const float eps = 1e-5f;
    
    // 输入数据
    float h_input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; // 2x4
    float h_output[8] = {0};
    
    // LayerNorm参数
    float h_gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};     // 缩放参数
    float h_beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};      // 偏移参数

    Tensor<float> input({batch_size, features});
    Tensor<float> output({batch_size, features});
    Tensor<float> gamma({features});
    Tensor<float> beta({features});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    layernorm.Forward(input, output, gamma, beta, dim, eps);

    float h_result[8] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 8; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "LayerNorm result should not be all zeros";
}

TEST_F(LayerNormTest, Forward3DBasic) {
    // 测试3D输入: [batch_size, seq_len, hidden_size]
    const int batch_size = 2;
    const int seq_len = 3;
    const int hidden_size = 4;
    const int dim = -1; // 在最后一个维度上归一化
    const float eps = 1e-5f;
    
    // 输入数据
    float h_input[24] = {
        1.0f, 2.0f, 3.0f, 4.0f,  // 第一个样本，第一个序列
        5.0f, 6.0f, 7.0f, 8.0f,  // 第一个样本，第二个序列
        9.0f, 10.0f, 11.0f, 12.0f, // 第一个样本，第三个序列
        13.0f, 14.0f, 15.0f, 16.0f, // 第二个样本，第一个序列
        17.0f, 18.0f, 19.0f, 20.0f, // 第二个样本，第二个序列
        21.0f, 22.0f, 23.0f, 24.0f  // 第二个样本，第三个序列
    };
    float h_output[24] = {0};
    
    // LayerNorm参数
    float h_gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> input({batch_size, seq_len, hidden_size});
    Tensor<float> output({batch_size, seq_len, hidden_size});
    Tensor<float> gamma({hidden_size});
    Tensor<float> beta({hidden_size});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    layernorm.Forward(input, output, gamma, beta, dim, eps);

    float h_result[24] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 24; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "LayerNorm result should not be all zeros";
}

TEST_F(LayerNormTest, ForwardWithGammaBeta) {
    // 测试带缩放和偏移参数的LayerNorm
    const int batch_size = 2;
    const int features = 3;
    const int dim = -1;
    const float eps = 1e-5f;
    
    float h_input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_output[6] = {0};
    
    // 设置非平凡的gamma和beta
    float h_gamma[3] = {2.0f, 1.5f, 0.5f};     // 缩放因子
    float h_beta[3] = {1.0f, 0.5f, -0.5f};     // 偏移

    Tensor<float> input({batch_size, features});
    Tensor<float> output({batch_size, features});
    Tensor<float> gamma({features});
    Tensor<float> beta({features});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    layernorm.Forward(input, output, gamma, beta, dim, eps);

    float h_result[6] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 6; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "LayerNorm result should not be all zeros";
}

TEST_F(LayerNormTest, ForwardDifferentDim) {
    // 测试在不同维度上归一化
    const int batch_size = 2;
    const int seq_len = 3;
    const int features = 2;
    const int dim = 1; // 在第二个维度上归一化
    const float eps = 1e-5f;
    
    float h_input[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float h_output[12] = {0};
    
    // LayerNorm参数（在dim=1上归一化，所以参数数量应该是seq_len）
    float h_gamma[3] = {1.0f, 1.0f, 1.0f};
    float h_beta[3] = {0.0f, 0.0f, 0.0f};

    Tensor<float> input({batch_size, seq_len, features});
    Tensor<float> output({batch_size, seq_len, features});
    Tensor<float> gamma({seq_len});
    Tensor<float> beta({seq_len});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    layernorm.Forward(input, output, gamma, beta, dim, eps);

    float h_result[12] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);

    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < 12; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "LayerNorm result should not be all zeros";
}

TEST_F(LayerNormTest, ForwardLargeTensor) {
    // 测试大张量
    const int batch_size = 4;
    const int seq_len = 8;
    const int hidden_size = 16;
    const int dim = -1;
    const float eps = 1e-5f;
    
    const int total_size = batch_size * seq_len * hidden_size;
    std::vector<float> h_input(total_size);
    std::vector<float> h_output(total_size, 0.0f);
    
    // 初始化输入数据
    for (int i = 0; i < total_size; i++) {
        h_input[i] = static_cast<float>(i % 20) - 10.0f; // 范围 [-10, 9]
    }
    
    // LayerNorm参数
    std::vector<float> h_gamma(hidden_size, 1.0f);
    std::vector<float> h_beta(hidden_size, 0.0f);

    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), 
                        static_cast<size_t>(hidden_size)});
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), 
                         static_cast<size_t>(hidden_size)});
    Tensor<float> gamma({static_cast<size_t>(hidden_size)});
    Tensor<float> beta({static_cast<size_t>(hidden_size)});
    
    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    layernorm.Forward(input, output, gamma, beta, dim, eps);

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
    EXPECT_FALSE(all_zero) << "LayerNorm result should not be all zeros";
}

TEST_F(LayerNormTest, ForwardZeroEpsilon) {
    // 测试零epsilon值
    const int batch_size = 2;
    const int features = 3;
    const int dim = -1;
    const float eps = 0.0f; // 零epsilon
    
    float h_input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_output[6] = {0};
    
    float h_gamma[3] = {1.0f, 1.0f, 1.0f};
    float h_beta[3] = {0.0f, 0.0f, 0.0f};

    Tensor<float> input({batch_size, features});
    Tensor<float> output({batch_size, features});
    Tensor<float> gamma({features});
    Tensor<float> beta({features});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(output.data(), h_output, sizeof(h_output), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma.data(), h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);
    cudaMemcpy(beta.data(), h_beta, sizeof(h_beta), cudaMemcpyHostToDevice);

    LayerNorm<float> layernorm;
    // 应该能正常处理零epsilon
    EXPECT_NO_THROW(layernorm.Forward(input, output, gamma, beta, dim, eps));
}
