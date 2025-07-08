#pragma once

#include "cuda_op/abstract/operator.hpp"
#include <cublas_v2.h>

namespace cu_op_mem {

template <typename T>
class Gemv : public Operator<T> {
public:
    // 构造函数（默认不转置，alpha=1, beta=0）
    explicit Gemv(bool transA = false, T alpha = 1, T beta = 0);
    
    // 析构函数（自动释放CUBLAS句柄）
    ~Gemv();

    // 前向计算：output = alpha * (transA ? A^T : A) * weight + beta * output
    void Forward(const Tensor<T>& input, Tensor<T>& output) override;

    // 设置权重向量（需与input的列维度匹配）
    void SetWeight(const Tensor<T>& weight);

private:
    bool transA_;          // 是否转置输入矩阵
    T alpha_, beta_;       // 缩放系数
    Tensor<T> weight_;     // 权重向量
    cublasHandle_t handle_; // CUBLAS上下文句柄
};

} // namespace cu_op_mem
