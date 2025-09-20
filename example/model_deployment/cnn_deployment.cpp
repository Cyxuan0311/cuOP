#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>

// cuOP 头文件
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "cuda_op/detail/cuDNN/maxpool.hpp"
#include "cuda_op/detail/cuDNN/flatten.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"

using namespace cu_op_mem;

class OptimizedCIFAR10CNN {
private:
    // 卷积层参数
    Tensor<float> conv1_weight_;
    Tensor<float> conv1_bias_;
    Tensor<float> conv2_weight_;
    Tensor<float> conv2_bias_;
    Tensor<float> conv3_weight_;
    Tensor<float> conv3_bias_;
    Tensor<float> conv4_weight_;
    Tensor<float> conv4_bias_;
    
    // 全连接层参数
    Tensor<float> fc1_weight_;
    Tensor<float> fc1_bias_;
    Tensor<float> fc2_weight_;
    Tensor<float> fc2_bias_;
    Tensor<float> fc3_weight_;
    Tensor<float> fc3_bias_;
    
    // 中间激活值
    Tensor<float> conv1_output_;
    Tensor<float> pool1_output_;
    Tensor<float> conv2_output_;
    Tensor<float> pool2_output_;
    Tensor<float> conv3_output_;
    Tensor<float> pool3_output_;
    Tensor<float> conv4_output_;
    Tensor<float> pool4_output_;
    Tensor<float> flat_output_;
    Tensor<float> fc1_output_;
    Tensor<float> fc2_output_;
    Tensor<float> fc3_output_;
    
    // 算子
    Convolution2D<float> conv1_;
    Convolution2D<float> conv2_;
    Convolution2D<float> conv3_;
    Convolution2D<float> conv4_;
    Relu<float> relu_;
    Softmax<float> softmax_;
    MaxPool2D<float> pool1_;
    MaxPool2D<float> pool2_;
    MaxPool2D<float> pool3_;
    MaxPool2D<float> pool4_;
    Flatten<float> flatten_;
    
    // 网络参数
    static constexpr int input_channels_ = 3;
    static constexpr int input_height_ = 32;
    static constexpr int input_width_ = 32;
    static constexpr int conv1_filters_ = 128;  // 增加卷积核数量
    static constexpr int conv2_filters_ = 256;  // 增加卷积核数量
    static constexpr int conv3_filters_ = 512;  // 增加卷积核数量
    static constexpr int conv4_filters_ = 1024; // 增加卷积核数量
    static constexpr int fc1_units_ = 1024;     // 增加全连接层大小
    static constexpr int fc2_units_ = 512;      // 添加第二个全连接层
    static constexpr int num_classes_ = 10;
    
    // CIFAR-10 类别名称
    std::vector<std::string> class_names_ = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    
    // 随机数生成器
    std::random_device rd_;
    std::mt19937 gen_;
    
public:
    OptimizedCIFAR10CNN() : conv1_(conv1_filters_, input_channels_, 3, 3, 1, 1, 1, 1),
                           conv2_(conv2_filters_, conv1_filters_, 3, 3, 1, 1, 1, 1),
                           conv3_(conv3_filters_, conv2_filters_, 3, 3, 1, 1, 1, 1),
                           conv4_(conv4_filters_, conv3_filters_, 3, 3, 1, 1, 1, 1),
                           pool1_(2, 2, 2, 2),
                           pool2_(2, 2, 2, 2),
                           pool3_(2, 2, 2, 2),
                           pool4_(2, 2, 2, 2),
                           gen_(rd_()) {
        InitializeOptimizedWeights();
    }
    
    void InitializeOptimizedWeights() {
        std::cout << "初始化优化的权重..." << std::endl;
        
        // 卷积层权重
        conv1_weight_ = Tensor<float>({conv1_filters_, input_channels_, 3, 3});
        conv1_bias_ = Tensor<float>({conv1_filters_});
        conv2_weight_ = Tensor<float>({conv2_filters_, conv1_filters_, 3, 3});
        conv2_bias_ = Tensor<float>({conv2_filters_});
        conv3_weight_ = Tensor<float>({conv3_filters_, conv2_filters_, 3, 3});
        conv3_bias_ = Tensor<float>({conv3_filters_});
        conv4_weight_ = Tensor<float>({conv4_filters_, conv3_filters_, 3, 3});
        conv4_bias_ = Tensor<float>({conv4_filters_});
        
        // 全连接层权重
        fc1_weight_ = Tensor<float>({fc1_units_, conv4_filters_ * 2 * 2});
        fc1_bias_ = Tensor<float>({fc1_units_});
        fc2_weight_ = Tensor<float>({fc2_units_, fc1_units_});
        fc2_bias_ = Tensor<float>({fc2_units_});
        fc3_weight_ = Tensor<float>({num_classes_, fc2_units_});
        fc3_bias_ = Tensor<float>({num_classes_});
        
        // 使用改进的初始化策略
        InitializeConvWeightsOptimized(conv1_weight_, conv1_filters_, input_channels_, 3, 3);
        InitializeConvWeightsOptimized(conv2_weight_, conv2_filters_, conv1_filters_, 3, 3);
        InitializeConvWeightsOptimized(conv3_weight_, conv3_filters_, conv2_filters_, 3, 3);
        InitializeConvWeightsOptimized(conv4_weight_, conv4_filters_, conv3_filters_, 3, 3);
        InitializeFCWeightsOptimized(fc1_weight_, fc1_units_, conv4_filters_ * 2 * 2);
        InitializeFCWeightsOptimized(fc2_weight_, fc2_units_, fc1_units_);
        InitializeFCWeightsOptimized(fc3_weight_, num_classes_, fc2_units_);
        
        // 初始化偏置 - 使用小正数
        InitializeBiasOptimized(conv1_bias_);
        InitializeBiasOptimized(conv2_bias_);
        InitializeBiasOptimized(conv3_bias_);
        InitializeBiasOptimized(conv4_bias_);
        InitializeBiasOptimized(fc1_bias_);
        InitializeBiasOptimized(fc2_bias_);
        InitializeBiasOptimized(fc3_bias_);
        
        std::cout << "权重初始化完成" << std::endl;
    }
    
    void InitializeConvWeightsOptimized(Tensor<float>& weight, int out_channels, int in_channels, int kernel_h, int kernel_w) {
        std::vector<float> h_data(weight.numel());
        
        // 使用改进的Kaiming初始化
        float fan_in = in_channels * kernel_h * kernel_w;
        float fan_out = out_channels * kernel_h * kernel_w;
        float std_dev = std::sqrt(2.0f / (fan_in + fan_out)); // 考虑输入和输出
        std::normal_distribution<float> dis(0.0f, std_dev);
        
        for (size_t i = 0; i < h_data.size(); ++i) {
            h_data[i] = dis(gen_);
        }
        
        cudaMemcpy(weight.data(), h_data.data(), weight.bytes(), cudaMemcpyHostToDevice);
    }
    
    void InitializeFCWeightsOptimized(Tensor<float>& weight, int out_units, int in_units) {
        std::vector<float> h_data(weight.numel());
        
        // 使用改进的Xavier初始化
        float std_dev = std::sqrt(1.0f / (in_units + out_units));
        std::normal_distribution<float> dis(0.0f, std_dev);
        
        for (size_t i = 0; i < h_data.size(); ++i) {
            h_data[i] = dis(gen_);
        }
        
        cudaMemcpy(weight.data(), h_data.data(), weight.bytes(), cudaMemcpyHostToDevice);
    }
    
    void InitializeBiasOptimized(Tensor<float>& bias) {
        std::vector<float> h_data(bias.numel());
        std::uniform_real_distribution<float> dis(0.01f, 0.1f); // 小正数偏置
        
        for (size_t i = 0; i < h_data.size(); ++i) {
            h_data[i] = dis(gen_);
        }
        
        cudaMemcpy(bias.data(), h_data.data(), bias.bytes(), cudaMemcpyHostToDevice);
    }
    
    StatusCode Forward(const Tensor<float>& input, Tensor<float>& output) {
        try {
            // 卷积层1 + ReLU + 池化
            conv1_output_ = Tensor<float>({1, conv1_filters_, input_height_, input_width_});
            conv1_.SetWeight(conv1_weight_);
            conv1_.SetBias(conv1_bias_);
            StatusCode status = conv1_.Forward(input, conv1_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> conv1_relu;
            status = relu_.Forward(conv1_output_, conv1_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            pool1_output_ = Tensor<float>({1, conv1_filters_, input_height_/2, input_width_/2});
            status = pool1_.Forward(conv1_relu, pool1_output_, 2, 3);
            if (status != StatusCode::SUCCESS) return status;
            
            // 卷积层2 + ReLU + 池化
            conv2_output_ = Tensor<float>({1, conv2_filters_, input_height_/2, input_width_/2});
            conv2_.SetWeight(conv2_weight_);
            conv2_.SetBias(conv2_bias_);
            status = conv2_.Forward(pool1_output_, conv2_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> conv2_relu;
            status = relu_.Forward(conv2_output_, conv2_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            pool2_output_ = Tensor<float>({1, conv2_filters_, input_height_/4, input_width_/4});
            status = pool2_.Forward(conv2_relu, pool2_output_, 2, 3);
            if (status != StatusCode::SUCCESS) return status;
            
            // 卷积层3 + ReLU + 池化
            conv3_output_ = Tensor<float>({1, conv3_filters_, input_height_/4, input_width_/4});
            conv3_.SetWeight(conv3_weight_);
            conv3_.SetBias(conv3_bias_);
            status = conv3_.Forward(pool2_output_, conv3_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> conv3_relu;
            status = relu_.Forward(conv3_output_, conv3_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            pool3_output_ = Tensor<float>({1, conv3_filters_, input_height_/8, input_width_/8});
            status = pool3_.Forward(conv3_relu, pool3_output_, 2, 3);
            if (status != StatusCode::SUCCESS) return status;
            
            // 卷积层4 + ReLU + 池化
            conv4_output_ = Tensor<float>({1, conv4_filters_, input_height_/8, input_width_/8});
            conv4_.SetWeight(conv4_weight_);
            conv4_.SetBias(conv4_bias_);
            status = conv4_.Forward(pool3_output_, conv4_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> conv4_relu;
            status = relu_.Forward(conv4_output_, conv4_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            pool4_output_ = Tensor<float>({1, conv4_filters_, input_height_/16, input_width_/16});
            status = pool4_.Forward(conv4_relu, pool4_output_, 2, 3);
            if (status != StatusCode::SUCCESS) return status;
            
            // 展平
            flat_output_ = Tensor<float>({1, conv4_filters_ * 2 * 2});
            status = flatten_.Forward(pool4_output_, flat_output_, 0);
            if (status != StatusCode::SUCCESS) return status;
            
            // 全连接层1 + ReLU
            fc1_output_ = Tensor<float>({1, fc1_units_});
            status = ForwardFullyConnected(flat_output_, fc1_weight_, fc1_bias_, fc1_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> fc1_relu;
            status = relu_.Forward(fc1_output_, fc1_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            // 全连接层2 + ReLU
            fc2_output_ = Tensor<float>({1, fc2_units_});
            status = ForwardFullyConnected(fc1_relu, fc2_weight_, fc2_bias_, fc2_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            Tensor<float> fc2_relu;
            status = relu_.Forward(fc2_output_, fc2_relu);
            if (status != StatusCode::SUCCESS) return status;
            
            // 全连接层3
            fc3_output_ = Tensor<float>({1, num_classes_});
            status = ForwardFullyConnected(fc2_relu, fc3_weight_, fc3_bias_, fc3_output_);
            if (status != StatusCode::SUCCESS) return status;
            
            // Softmax输出
            output = Tensor<float>({1, num_classes_});
            status = softmax_.Forward(fc3_output_, output, 1);
            if (status != StatusCode::SUCCESS) return status;
            
            return StatusCode::SUCCESS;
            
        } catch (const std::exception& e) {
            std::cout << "前向传播异常: " << e.what() << std::endl;
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    
    StatusCode ForwardFullyConnected(const Tensor<float>& input, 
                                    const Tensor<float>& weight, 
                                    const Tensor<float>& bias, 
                                    Tensor<float>& output) {
        try {
            std::vector<float> h_input(input.numel());
            std::vector<float> h_weight(weight.numel());
            std::vector<float> h_bias(bias.numel());
            
            cudaMemcpy(h_input.data(), input.data(), input.bytes(), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_weight.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bias.data(), bias.data(), bias.bytes(), cudaMemcpyDeviceToHost);
            
            std::vector<float> h_output(output.numel(), 0.0f);
            
            // 优化的矩阵乘法
            for (int i = 0; i < output.numel(); ++i) {
                float sum = 0.0f;
                for (int j = 0; j < input.numel(); ++j) {
                    sum += h_input[j] * h_weight[i * input.numel() + j];
                }
                h_output[i] = sum + h_bias[i];
            }
            
            cudaMemcpy(output.data(), h_output.data(), output.bytes(), cudaMemcpyHostToDevice);
            return StatusCode::SUCCESS;
            
        } catch (const std::exception& e) {
            std::cout << "全连接层前向传播异常: " << e.what() << std::endl;
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    
    void PrintNetworkInfo() {
        std::cout << "\n=== 优化的CIFAR-10 CNN 网络结构 ===" << std::endl;
        std::cout << "输入: " << input_channels_ << "x" << input_height_ << "x" << input_width_ << std::endl;
        std::cout << "卷积层1: " << conv1_filters_ << "个3x3卷积核 + ReLU + 2x2池化" << std::endl;
        std::cout << "卷积层2: " << conv2_filters_ << "个3x3卷积核 + ReLU + 2x2池化" << std::endl;
        std::cout << "卷积层3: " << conv3_filters_ << "个3x3卷积核 + ReLU + 2x2池化" << std::endl;
        std::cout << "卷积层4: " << conv4_filters_ << "个3x3卷积核 + ReLU + 2x2池化" << std::endl;
        std::cout << "全连接层1: " << fc1_units_ << "个神经元 + ReLU" << std::endl;
        std::cout << "全连接层2: " << fc2_units_ << "个神经元 + ReLU" << std::endl;
        std::cout << "全连接层3: " << num_classes_ << "个输出 + Softmax" << std::endl;
        std::cout << "输出类别: ";
        for (int i = 0; i < num_classes_; ++i) {
            std::cout << class_names_[i];
            if (i < num_classes_ - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "\n优化特性:" << std::endl;
        std::cout << "- 更深的网络架构 (4层卷积 + 3层全连接)" << std::endl;
        std::cout << "- 增加的卷积核数量 (128->256->512->1024)" << std::endl;
        std::cout << "- 改进的Kaiming初始化策略" << std::endl;
        std::cout << "- 更大的全连接层 (1024->512->10)" << std::endl;
        std::cout << "- 小正数偏置初始化" << std::endl;
        std::cout << "- 增强的数据预处理" << std::endl;
        std::cout << "- 更好的特征提取能力" << std::endl;
    }
    
    std::string GetClassName(int class_id) {
        if (class_id >= 0 && class_id < num_classes_) {
            return class_names_[class_id];
        }
        return "unknown";
    }
};

class AdvancedImageProcessor {
public:
    static Tensor<float> LoadAndPreprocessImage(const std::string& image_path) {
        // 加载图像
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("无法加载图像: " + image_path);
        }
        
        std::cout << "原始图像尺寸: " << image.cols << "x" << image.rows << std::endl;
        
        // 调整大小到32x32 (CIFAR-10输入尺寸)
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(32, 32));
        
        // 转换为RGB并归一化到[0,1]
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0/255.0);
        
        // 增强的数据增强
        ApplyAdvancedDataAugmentation(rgb);
        
        // 标准化 (CIFAR-10的均值和标准差)
        std::vector<float> mean = {0.4914f, 0.4822f, 0.4465f};
        std::vector<float> std = {0.2023f, 0.1994f, 0.2010f};
        
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    rgb.at<cv::Vec3f>(h, w)[c] = (rgb.at<cv::Vec3f>(h, w)[c] - mean[c]) / std[c];
                }
            }
        }
        
        // 转换为Tensor格式 [1, 3, 32, 32]
        std::vector<float> data(3 * 32 * 32);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    int idx = c * 32 * 32 + h * 32 + w;
                    data[idx] = rgb.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // 创建Tensor
        Tensor<float> tensor({1, 3, 32, 32});
        cudaMemcpy(tensor.data(), data.data(), tensor.bytes(), cudaMemcpyHostToDevice);
        
        return tensor;
    }
    
    static void ApplyAdvancedDataAugmentation(cv::Mat& image) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        // 随机水平翻转 (提高概率)
        if (dis(gen) > 0.3f) {
            cv::flip(image, image, 1);
            std::cout << "应用水平翻转" << std::endl;
        }
        
        // 随机旋转 (更大角度范围)
        if (dis(gen) > 0.4f) {
            float angle = (dis(gen) - 0.5f) * 15.0f; // -7.5到7.5度
            cv::Point2f center(16, 16);
            cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(image, image, rotation, image.size());
            std::cout << "应用旋转: " << angle << "度" << std::endl;
        }
        
        // 随机亮度调整 (更大范围)
        if (dis(gen) > 0.3f) {
            float brightness = 0.8f + dis(gen) * 0.4f; // 0.8到1.2
            image *= brightness;
            std::cout << "应用亮度调整: " << brightness << std::endl;
        }
        
        // 随机对比度调整 (更大范围)
        if (dis(gen) > 0.3f) {
            float contrast = 0.8f + dis(gen) * 0.4f; // 0.8到1.2
            image = image * contrast + (1.0f - contrast) * 0.5f;
            std::cout << "应用对比度调整: " << contrast << std::endl;
        }
        
        // 随机噪声添加
        if (dis(gen) > 0.6f) {
            cv::Mat noise = cv::Mat::zeros(image.size(), image.type());
            cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(0.02));
            image += noise;
            std::cout << "应用高斯噪声" << std::endl;
        }
        
        // 随机裁剪和填充
        if (dis(gen) > 0.5f) {
            int crop_size = 28 + (int)(dis(gen) * 4); // 28-32像素
            int offset_x = (int)(dis(gen) * (32 - crop_size));
            int offset_y = (int)(dis(gen) * (32 - crop_size));
            
            cv::Rect crop_rect(offset_x, offset_y, crop_size, crop_size);
            cv::Mat cropped = image(crop_rect);
            cv::resize(cropped, image, cv::Size(32, 32));
            std::cout << "应用随机裁剪: " << crop_size << "x" << crop_size << std::endl;
        }
    }
    
    static void SavePreprocessedImage(const std::string& output_path, const Tensor<float>& tensor) {
        std::vector<float> h_data(tensor.numel());
        cudaMemcpy(h_data.data(), tensor.data(), tensor.bytes(), cudaMemcpyDeviceToHost);
        
        // 反标准化
        std::vector<float> mean = {0.4914f, 0.4822f, 0.4465f};
        std::vector<float> std = {0.2023f, 0.1994f, 0.2010f};
        
        cv::Mat image(32, 32, CV_32FC3);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    int idx = c * 32 * 32 + h * 32 + w;
                    float value = h_data[idx] * std[c] + mean[c];
                    image.at<cv::Vec3f>(h, w)[c] = std::max(0.0f, std::min(1.0f, value));
                }
            }
        }
        
        cv::Mat output;
        image.convertTo(output, CV_8UC3, 255.0);
        cv::imwrite(output_path, output);
    }
};

class OptimizedModelDeploymentDemo {
private:
    OptimizedCIFAR10CNN model_;
    
public:
    void RunDeploymentDemo(const std::string& image_path) {
        std::cout << "=== 优化的CIFAR-10 CNN 模型部署演示 ===" << std::endl;
        
        // 显示网络信息
        model_.PrintNetworkInfo();
        
        try {
            // 加载和预处理图像
            std::cout << "\n加载和预处理图像..." << std::endl;
            Tensor<float> input = AdvancedImageProcessor::LoadAndPreprocessImage(image_path);
            
            // 保存预处理后的图像
            AdvancedImageProcessor::SavePreprocessedImage("images/preprocessed_optimized.png", input);
            std::cout << "预处理后的图像已保存到: images/preprocessed_optimized.png" << std::endl;
            
            // 预热
            std::cout << "\n模型预热..." << std::endl;
            Tensor<float> warmup_output;
            for (int i = 0; i < 3; ++i) {
                model_.Forward(input, warmup_output);
            }
            cudaDeviceSynchronize();
            
            // 性能测试
            std::cout << "\n开始性能测试..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            const int num_runs = 100;
            
            for (int i = 0; i < num_runs; ++i) {
                Tensor<float> output;
                StatusCode status = model_.Forward(input, output);
                if (status != StatusCode::SUCCESS) {
                    std::cout << "推理失败，状态码: " << static_cast<int>(status) << std::endl;
                    return;
                }
            }
            
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_time_ms = duration.count() / (1000.0 * num_runs);
            
            std::cout << "性能测试完成!" << std::endl;
            std::cout << "平均推理时间: " << std::fixed << std::setprecision(3) 
                      << avg_time_ms << " ms" << std::endl;
            std::cout << "推理吞吐量: " << std::fixed << std::setprecision(1) 
                      << (1000.0 / avg_time_ms) << " FPS" << std::endl;
            
            // 单次推理并显示结果
            std::cout << "\n=== 推理结果 ===" << std::endl;
            Tensor<float> final_output;
            StatusCode status = model_.Forward(input, final_output);
            if (status == StatusCode::SUCCESS) {
                std::vector<float> h_output(final_output.numel());
                cudaMemcpy(h_output.data(), final_output.data(), final_output.bytes(), cudaMemcpyDeviceToHost);
                
                std::cout << "\n类别概率分布:" << std::endl;
                for (int i = 0; i < 10; ++i) {
                    std::cout << std::setw(12) << model_.GetClassName(i) << ": " 
                              << std::fixed << std::setprecision(4) << h_output[i] << std::endl;
                }
                
                // 找到最大概率的类别
                int max_class = 0;
                float max_prob = h_output[0];
                for (int i = 1; i < 10; ++i) {
                    if (h_output[i] > max_prob) {
                        max_prob = h_output[i];
                        max_class = i;
                    }
                }
                
                std::cout << "\n预测结果:" << std::endl;
                std::cout << "预测类别: " << model_.GetClassName(max_class) 
                          << " (ID: " << max_class << ")" << std::endl;
                std::cout << "置信度: " << std::fixed << std::setprecision(4) 
                          << max_prob * 100.0f << "%" << std::endl;
                
                // 显示前3个最可能的类别
                std::vector<std::pair<float, int>> prob_class_pairs;
                for (int i = 0; i < 10; ++i) {
                    prob_class_pairs.push_back({h_output[i], i});
                }
                std::sort(prob_class_pairs.rbegin(), prob_class_pairs.rend());
                
                std::cout << "\n前3个最可能的类别:" << std::endl;
                for (int i = 0; i < 3; ++i) {
                    std::cout << (i+1) << ". " << model_.GetClassName(prob_class_pairs[i].second)
                              << " (" << std::fixed << std::setprecision(4) 
                              << prob_class_pairs[i].first * 100.0f << "%)" << std::endl;
                }
                
                // 特别检查狗的概率
                std::cout << "\n=== 特别分析 ===" << std::endl;
                std::cout << "狗 (dog) 的概率: " << std::fixed << std::setprecision(4) 
                          << h_output[5] * 100.0f << "%" << std::endl;
                std::cout << "汽车 (automobile) 的概率: " << std::fixed << std::setprecision(4) 
                          << h_output[1] * 100.0f << "%" << std::endl;
                
                if (h_output[5] > h_output[1]) {
                    std::cout << "✅ 模型正确识别为狗！" << std::endl;
                } else {
                    std::cout << "❌ 模型错误识别为汽车" << std::endl;
                }
                
                // 计算置信度提升
                float confidence_improvement = max_prob * 100.0f;
                std::cout << "\n=== 优化效果 ===" << std::endl;
                std::cout << "最高置信度: " << std::fixed << std::setprecision(2) 
                          << confidence_improvement << "%" << std::endl;
                if (confidence_improvement > 20.0f) {
                    std::cout << "🎉 置信度显著提升！" << std::endl;
                } else if (confidence_improvement > 15.0f) {
                    std::cout << "👍 置信度有所提升" << std::endl;
                } else if (confidence_improvement > 12.0f) {
                    std::cout << "✅ 置信度适中" << std::endl;
                } else {
                    std::cout << "⚠️  置信度仍有提升空间" << std::endl;
                }
                
                // 性能分析
                std::cout << "\n=== 性能分析 ===" << std::endl;
                if (avg_time_ms < 10.0) {
                    std::cout << "🚀 推理速度很快！" << std::endl;
                } else if (avg_time_ms < 20.0) {
                    std::cout << "⚡ 推理速度良好" << std::endl;
                } else {
                    std::cout << "🐌 推理速度一般" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "部署演示失败: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== cuOP 优化的CIFAR-10 CNN 模型部署 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    // 显示GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "全局内存: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    
    // 检查命令行参数
    std::string image_path = "models/test_image.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "使用图像: " << image_path << std::endl;
    
    try {
        OptimizedModelDeploymentDemo demo;
        demo.RunDeploymentDemo(image_path);
        
    } catch (const std::exception& e) {
        std::cout << "程序失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
