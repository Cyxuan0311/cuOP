#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "data/tensor.hpp"
#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

// 算子融合类型枚举
enum class FusionType {
    CONV_RELU,           // 卷积 + ReLU
    CONV_BN_RELU,        // 卷积 + BatchNorm + ReLU
    CONV_BN,             // 卷积 + BatchNorm
    MATMUL_RELU,         // 矩阵乘法 + ReLU
    LAYERNORM_RELU,      // LayerNorm + ReLU
    SOFTMAX_DROPOUT,     // Softmax + Dropout
    BATCHNORM_RELU,      // BatchNorm + ReLU
    MAXPOOL_RELU         // MaxPool + ReLU
};

// 融合算子基类
template<typename T>
class FusedOperator {
public:
    virtual ~FusedOperator() = default;
    virtual StatusCode Forward(const std::vector<Tensor<T>*>& inputs, 
                              std::vector<Tensor<T>*>& outputs) = 0;
    virtual StatusCode Backward(const std::vector<Tensor<T>*>& inputs,
                               const std::vector<Tensor<T>*>& outputs,
                               std::vector<Tensor<T>*>& grad_inputs) = 0;
};

// 卷积 + ReLU 融合算子
template<typename T>
class ConvReluFused : public FusedOperator<T> {
public:
    ConvReluFused(int in_channels, int out_channels, int kernel_h, int kernel_w,
                  int stride_h, int stride_w, int pad_h, int pad_w);
    
    StatusCode Forward(const std::vector<Tensor<T>*>& inputs, 
                      std::vector<Tensor<T>*>& outputs) override;
    StatusCode Backward(const std::vector<Tensor<T>*>& inputs,
                       const std::vector<Tensor<T>*>& outputs,
                       std::vector<Tensor<T>*>& grad_inputs) override;
    
    void SetWeight(const Tensor<T>& weight);
    void SetBias(const Tensor<T>& bias);

private:
    int in_channels_, out_channels_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    Tensor<T> weight_;
    Tensor<T> bias_;
};

// 卷积 + BatchNorm + ReLU 融合算子
template<typename T>
class ConvBnReluFused : public FusedOperator<T> {
public:
    ConvBnReluFused(int in_channels, int out_channels, int kernel_h, int kernel_w,
                    int stride_h, int stride_w, int pad_h, int pad_w);
    
    StatusCode Forward(const std::vector<Tensor<T>*>& inputs, 
                      std::vector<Tensor<T>*>& outputs) override;
    StatusCode Backward(const std::vector<Tensor<T>*>& inputs,
                       const std::vector<Tensor<T>*>& outputs,
                       std::vector<Tensor<T>*>& grad_inputs) override;
    
    void SetWeight(const Tensor<T>& weight);
    void SetBias(const Tensor<T>& bias);
    void SetGamma(const Tensor<T>& gamma);
    void SetBeta(const Tensor<T>& beta);

private:
    int in_channels_, out_channels_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    Tensor<T> weight_;
    Tensor<T> bias_;
    Tensor<T> gamma_;
    Tensor<T> beta_;
    Tensor<T> running_mean_;
    Tensor<T> running_var_;
    T eps_;
};

// 矩阵乘法 + ReLU 融合算子
template<typename T>
class MatMulReluFused : public FusedOperator<T> {
public:
    MatMulReluFused(bool transA = false, bool transB = false);
    
    StatusCode Forward(const std::vector<Tensor<T>*>& inputs, 
                      std::vector<Tensor<T>*>& outputs) override;
    StatusCode Backward(const std::vector<Tensor<T>*>& inputs,
                       const std::vector<Tensor<T>*>& outputs,
                       std::vector<Tensor<T>*>& grad_inputs) override;
    
    void SetWeight(const Tensor<T>& weight);
    void SetBias(const Tensor<T>& bias);

private:
    bool transA_, transB_;
    Tensor<T> weight_;
    Tensor<T> bias_;
};

// 融合算子工厂
template<typename T>
class FusedOperatorFactory {
public:
    static std::unique_ptr<FusedOperator<T>> Create(FusionType type, 
                                                    const std::vector<int>& params);
};

} // namespace cu_op_mem
