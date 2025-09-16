#include <gtest/gtest.h>
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include <glog/logging.h>
#include "data/tensor.hpp"
#include <cuda_runtime.h>

using namespace cu_op_mem;

class ConvolutionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            google::InitGoogleLogging("ConvolutionTest");
        }
        void TearDown() override {
            google::ShutdownGoogleLogging();
        }
};

TEST_F(ConvolutionTest, ForwardBasic) {
    // 测试基本卷积操作
    const int batch_size = 1;
    const int in_channels = 1;
    const int height = 4;
    const int width = 4;
    const int out_channels = 1;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    
    // 输入数据 (1x1x4x4)
    float h_input[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    // 权重数据 (1x1x3x3) - 简单的边缘检测核
    float h_weight[9] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 0.0f, -2.0f,
        1.0f, 0.0f, -1.0f
    };
    
    // 偏置数据 (1)
    float h_bias[1] = {0.0f};
    
    // 计算输出尺寸
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    Tensor<float> input({batch_size, in_channels, height, width});
    Tensor<float> weight({out_channels, in_channels, kernel_h, kernel_w});
    Tensor<float> bias({out_channels});
    Tensor<float> output({batch_size, out_channels, out_height, out_width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias, sizeof(h_bias), cudaMemcpyHostToDevice);
    
    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, 
                             stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);
    conv.Forward(input, output);
    
    float h_result[out_height * out_width] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < out_height * out_width; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Convolution result should not be all zeros";
}

TEST_F(ConvolutionTest, ForwardMultiChannel) {
    // 测试多通道卷积
    const int batch_size = 1;
    const int in_channels = 2;
    const int height = 3;
    const int width = 3;
    const int out_channels = 1;
    const int kernel_h = 2;
    const int kernel_w = 2;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 0;
    const int pad_w = 0;
    
    // 输入数据 (1x2x3x3)
    float h_input[18] = {
        // 第一个通道
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        // 第二个通道
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f
    };
    
    // 权重数据 (1x2x2x2)
    float h_weight[8] = {
        1.0f, 1.0f,  // 第一个通道的核
        1.0f, 1.0f,
        0.5f, 0.5f,  // 第二个通道的核
        0.5f, 0.5f
    };
    
    // 偏置数据
    float h_bias[1] = {0.0f};
    
    // 计算输出尺寸
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    Tensor<float> input({batch_size, in_channels, height, width});
    Tensor<float> weight({out_channels, in_channels, kernel_h, kernel_w});
    Tensor<float> bias({out_channels});
    Tensor<float> output({batch_size, out_channels, out_height, out_width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias, sizeof(h_bias), cudaMemcpyHostToDevice);
    
    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, 
                             stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);
    conv.Forward(input, output);
    
    float h_result[out_height * out_width] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < out_height * out_width; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Multi-channel convolution result should not be all zeros";
}

TEST_F(ConvolutionTest, ForwardWithStride) {
    // 测试带步长的卷积
    const int batch_size = 1;
    const int in_channels = 1;
    const int height = 4;
    const int width = 4;
    const int out_channels = 1;
    const int kernel_h = 2;
    const int kernel_w = 2;
    const int stride_h = 2;
    const int stride_w = 2;
    const int pad_h = 0;
    const int pad_w = 0;
    
    // 输入数据
    float h_input[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    // 权重数据 (简单的平均池化核)
    float h_weight[4] = {
        0.25f, 0.25f,
        0.25f, 0.25f
    };
    
    // 偏置数据
    float h_bias[1] = {0.0f};
    
    // 计算输出尺寸
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    Tensor<float> input({batch_size, in_channels, height, width});
    Tensor<float> weight({out_channels, in_channels, kernel_h, kernel_w});
    Tensor<float> bias({out_channels});
    Tensor<float> output({batch_size, out_channels, out_height, out_width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias, sizeof(h_bias), cudaMemcpyHostToDevice);
    
    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, 
                             stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);
    conv.Forward(input, output);
    
    float h_result[out_height * out_width] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < out_height * out_width; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Strided convolution result should not be all zeros";
}

TEST_F(ConvolutionTest, ForwardWithPadding) {
    // 测试带填充的卷积
    const int batch_size = 1;
    const int in_channels = 1;
    const int height = 2;
    const int width = 2;
    const int out_channels = 1;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    
    // 输入数据
    float h_input[4] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    
    // 权重数据 (3x3核)
    float h_weight[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    
    // 偏置数据
    float h_bias[1] = {0.0f};
    
    // 计算输出尺寸
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    Tensor<float> input({batch_size, in_channels, height, width});
    Tensor<float> weight({out_channels, in_channels, kernel_h, kernel_w});
    Tensor<float> bias({out_channels});
    Tensor<float> output({batch_size, out_channels, out_height, out_width});
    
    cudaMemcpy(input.data(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight, sizeof(h_weight), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias, sizeof(h_bias), cudaMemcpyHostToDevice);
    
    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, 
                             stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);
    conv.Forward(input, output);
    
    float h_result[out_height * out_width] = {0};
    cudaMemcpy(h_result, output.data(), sizeof(h_result), cudaMemcpyDeviceToHost);
    
    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < out_height * out_width; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Padded convolution result should not be all zeros";
}

TEST_F(ConvolutionTest, ForwardLargeTensor) {
    // 测试大张量
    const int batch_size = 2;
    const int in_channels = 3;
    const int height = 8;
    const int width = 8;
    const int out_channels = 4;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    
    const int input_size = batch_size * in_channels * height * width;
    const int weight_size = out_channels * in_channels * kernel_h * kernel_w;
    const int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int output_size = batch_size * out_channels * out_height * out_width;
    
    std::vector<float> h_input(input_size);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(out_channels, 0.0f);
    std::vector<float> h_output(output_size, 0.0f);
    
    // 初始化输入数据
    for (int i = 0; i < input_size; i++) {
        h_input[i] = static_cast<float>(i % 10) / 10.0f; // 范围 [0, 0.9]
    }
    
    // 初始化权重数据
    for (int i = 0; i < weight_size; i++) {
        h_weight[i] = static_cast<float>(i % 5) / 5.0f - 0.4f; // 范围 [-0.4, 0.4]
    }
    
    Tensor<float> input({static_cast<size_t>(batch_size), static_cast<size_t>(in_channels), 
                        static_cast<size_t>(height), static_cast<size_t>(width)});
    Tensor<float> weight({static_cast<size_t>(out_channels), static_cast<size_t>(in_channels), 
                         static_cast<size_t>(kernel_h), static_cast<size_t>(kernel_w)});
    Tensor<float> bias({static_cast<size_t>(out_channels)});
    Tensor<float> output({static_cast<size_t>(batch_size), static_cast<size_t>(out_channels), 
                         static_cast<size_t>(out_height), static_cast<size_t>(out_width)});
    
    cudaMemcpy(input.data(), h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.data(), h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias.data(), h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    Convolution2D<float> conv(in_channels, out_channels, kernel_h, kernel_w, 
                             stride_h, stride_w, pad_h, pad_w);
    conv.SetWeight(weight);
    conv.SetBias(bias);
    conv.Forward(input, output);
    
    std::vector<float> h_result(output_size);
    cudaMemcpy(h_result.data(), output.data(), h_result.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果不为零
    bool all_zero = true;
    for (int i = 0; i < output_size; i++) {
        if (std::abs(h_result[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Large tensor convolution result should not be all zeros";
}
