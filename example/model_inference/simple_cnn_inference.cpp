#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

// cuOP 头文件
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "cuda_op/detail/cuDNN/softmax.hpp"
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "cuda_op/detail/cuDNN/maxpool.hpp"
#include "cuda_op/detail/cuDNN/flatten.hpp"
#include "data/tensor.hpp"
#include "util/status_code.hpp"

using namespace cu_op_mem;

class SimpleCNN {
private:
    // 卷积层参数
    Tensor<float> conv1_weight_;
    Tensor<float> conv1_bias_;
    Tensor<float> conv2_weight_;
    Tensor<float> conv2_bias_;
    
    // 中间激活值
    Tensor<float> conv1_output_;
    Tensor<float> pool1_output_;
    Tensor<float> conv2_output_;
    Tensor<float> pool2_output_;
    Tensor<float> flat_output_;
    Tensor<float> final_output_;
    
    // 算子
    Convolution2D<float> conv1_;
    Convolution2D<float> conv2_;
    Relu<float> relu_;
    Softmax<float> softmax_;
    MaxPool2D<float> pool1_;
    MaxPool2D<float> pool2_;
    Flatten<float> flatten_;
    
    // 网络参数
    static constexpr int input_channels_ = 3;
    static constexpr int input_height_ = 32;
    static constexpr int input_width_ = 32;
    static constexpr int conv1_filters_ = 32;
    static constexpr int conv2_filters_ = 64;
    static constexpr int num_classes_ = 10;
    
public:
    SimpleCNN() : conv1_(conv1_filters_, input_channels_, 3, 3, 1, 1, 1, 1),
                   conv2_(conv2_filters_, conv1_filters_, 3, 3, 1, 1, 1, 1),
                   pool1_(2, 2, 2, 2),
                   pool2_(2, 2, 2, 2) {
        InitializeWeights();
    }
    
    void InitializeWeights() {
        std::cout << "初始化网络权重..." << std::endl;
        
        // 卷积层1: 3x3卷积，3输入通道，32输出通道
        conv1_weight_ = Tensor<float>({conv1_filters_, input_channels_, 3, 3});
        conv1_bias_ = Tensor<float>({conv1_filters_});
        
        // 卷积层2: 3x3卷积，32输入通道，64输出通道
        conv2_weight_ = Tensor<float>({conv2_filters_, conv1_filters_, 3, 3});
        conv2_bias_ = Tensor<float>({conv2_filters_});
        
        // 随机初始化权重
        InitializeTensorRandom(conv1_weight_);
        InitializeTensorRandom(conv1_bias_);
        InitializeTensorRandom(conv2_weight_);
        InitializeTensorRandom(conv2_bias_);
        
        std::cout << "权重初始化完成" << std::endl;
    }
    
    
    void InitializeTensorRandom(Tensor<float>& tensor) {
        std::vector<float> h_data(tensor.numel());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 0.1f);
        
        for (size_t i = 0; i < h_data.size(); ++i) {
            h_data[i] = dis(gen);
        }
        
        cudaMemcpy(tensor.data(), h_data.data(), tensor.bytes(), cudaMemcpyHostToDevice);
    }
    
    StatusCode Forward(const Tensor<float>& input, Tensor<float>& output) {
        try {
            // 卷积层1 + ReLU
            conv1_output_ = Tensor<float>({1, conv1_filters_, input_height_, input_width_});
            conv1_.SetWeight(conv1_weight_);
            conv1_.SetBias(conv1_bias_);
            StatusCode status = conv1_.Forward(input, conv1_output_);
            if (status != StatusCode::SUCCESS) {
                std::cout << "卷积层1前向传播失败" << std::endl;
                return status;
            }
            
            // ReLU激活
            Tensor<float> conv1_relu;
            status = relu_.Forward(conv1_output_, conv1_relu);
            if (status != StatusCode::SUCCESS) {
                std::cout << "ReLU激活失败" << std::endl;
                return status;
            }
            
            // 池化层1
            pool1_output_ = Tensor<float>({1, conv1_filters_, input_height_/2, input_width_/2});
            status = pool1_.Forward(conv1_relu, pool1_output_, 2, 3);
            if (status != StatusCode::SUCCESS) {
                std::cout << "池化层1前向传播失败" << std::endl;
                return status;
            }
            
            // 卷积层2 + ReLU
            conv2_output_ = Tensor<float>({1, conv2_filters_, input_height_/2, input_width_/2});
            conv2_.SetWeight(conv2_weight_);
            conv2_.SetBias(conv2_bias_);
            status = conv2_.Forward(pool1_output_, conv2_output_);
            if (status != StatusCode::SUCCESS) {
                std::cout << "卷积层2前向传播失败" << std::endl;
                return status;
            }
            
            // ReLU激活
            Tensor<float> conv2_relu;
            status = relu_.Forward(conv2_output_, conv2_relu);
            if (status != StatusCode::SUCCESS) {
                std::cout << "ReLU激活失败" << std::endl;
                return status;
            }
            
            // 池化层2
            pool2_output_ = Tensor<float>({1, conv2_filters_, input_height_/4, input_width_/4});
            status = pool2_.Forward(conv2_relu, pool2_output_, 2, 3);
            if (status != StatusCode::SUCCESS) {
                std::cout << "池化层2前向传播失败" << std::endl;
                return status;
            }
            
            // 展平
            flat_output_ = Tensor<float>({1, conv2_filters_ * 8 * 8});
            status = flatten_.Forward(pool2_output_, flat_output_, 0);
            if (status != StatusCode::SUCCESS) {
                std::cout << "展平层前向传播失败" << std::endl;
                return status;
            }
            
            // 简化的分类层：使用全局平均池化 + 线性变换
            // 这里我们直接使用展平后的特征进行简单的分类
            final_output_ = Tensor<float>({1, num_classes_});
            
            // 简单的线性分类：将特征映射到10个类别
            // 这里我们使用一个简化的方法：将特征分成10组，每组取平均值
            std::vector<float> h_features(flat_output_.numel());
            cudaMemcpy(h_features.data(), flat_output_.data(), flat_output_.bytes(), cudaMemcpyDeviceToHost);
            
            std::vector<float> h_output(num_classes_, 0.0f);
            int features_per_class = flat_output_.numel() / num_classes_;
            
            for (int i = 0; i < num_classes_; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < features_per_class; ++j) {
                    int idx = i * features_per_class + j;
                    if (idx < flat_output_.numel()) {
                        sum += h_features[idx];
                    }
                }
                h_output[i] = sum / features_per_class;
            }
            
            cudaMemcpy(final_output_.data(), h_output.data(), final_output_.bytes(), cudaMemcpyHostToDevice);
            
            // Softmax输出
            output = Tensor<float>({1, num_classes_});
            status = softmax_.Forward(final_output_, output, 1);
            if (status != StatusCode::SUCCESS) {
                std::cout << "Softmax激活失败" << std::endl;
                return status;
            }
            
            return StatusCode::SUCCESS;
            
        } catch (const std::exception& e) {
            std::cout << "前向传播异常: " << e.what() << std::endl;
            return StatusCode::UNKNOWN_ERROR;
        }
    }
    
    void PrintNetworkInfo() {
        std::cout << "\n=== 网络结构信息 ===" << std::endl;
        std::cout << "输入: " << input_channels_ << "x" << input_height_ << "x" << input_width_ << std::endl;
        std::cout << "卷积层1: " << conv1_filters_ << "个3x3卷积核 + ReLU" << std::endl;
        std::cout << "池化层1: 2x2最大池化" << std::endl;
        std::cout << "卷积层2: " << conv2_filters_ << "个3x3卷积核 + ReLU" << std::endl;
        std::cout << "池化层2: 2x2最大池化" << std::endl;
        std::cout << "展平层: 将特征展平" << std::endl;
        std::cout << "分类层: 简化的线性分类" << std::endl;
        std::cout << "输出: " << num_classes_ << "个类别概率" << std::endl;
    }
};

class ModelInferenceDemo {
private:
    SimpleCNN model_;
    
public:
    void RunInferenceDemo() {
        std::cout << "=== 简单CNN模型推理演示 ===" << std::endl;
        
        // 显示网络信息
        model_.PrintNetworkInfo();
        
        // 创建测试输入
        Tensor<float> input({1, 3, 32, 32});
        std::vector<float> h_input(3 * 32 * 32);
        
        // 生成随机测试数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t i = 0; i < h_input.size(); ++i) {
            h_input[i] = dis(gen);
        }
        
        cudaMemcpy(input.data(), h_input.data(), input.bytes(), cudaMemcpyHostToDevice);
        
        std::cout << "\n开始模型推理..." << std::endl;
        
        // 预热
        Tensor<float> warmup_output;
        for (int i = 0; i < 3; ++i) {
            model_.Forward(input, warmup_output);
        }
        cudaDeviceSynchronize();
        
        // 性能测试
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
        
        std::cout << "推理完成!" << std::endl;
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(3) 
                  << avg_time_ms << " ms" << std::endl;
        std::cout << "推理吞吐量: " << std::fixed << std::setprecision(1) 
                  << (1000.0 / avg_time_ms) << " FPS" << std::endl;
        
        // 显示最后一次推理的结果
        Tensor<float> final_output;
        StatusCode status = model_.Forward(input, final_output);
        if (status == StatusCode::SUCCESS) {
            std::vector<float> h_output(final_output.numel());
            cudaMemcpy(h_output.data(), final_output.data(), final_output.bytes(), cudaMemcpyDeviceToHost);
            
            std::cout << "\n输出概率分布:" << std::endl;
            for (int i = 0; i < 10; ++i) {
                std::cout << "类别 " << i << ": " << std::fixed << std::setprecision(4) 
                          << h_output[i] << std::endl;
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
            
            std::cout << "\n预测类别: " << max_class << " (概率: " 
                      << std::fixed << std::setprecision(4) << max_prob << ")" << std::endl;
        }
    }
};

int main() {
    std::cout << "=== cuOP 模型推理演示 ===" << std::endl;
    
    // 初始化CUDA
    cudaSetDevice(0);
    cudaFree(0);
    
    // 显示GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "全局内存: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    
    try {
        ModelInferenceDemo demo;
        demo.RunInferenceDemo();
        
    } catch (const std::exception& e) {
        std::cout << "演示失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}