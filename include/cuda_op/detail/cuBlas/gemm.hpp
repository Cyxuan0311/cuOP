#pragma once

#include "cuda_op/abstract/operator.hpp"
#include <cublas_v2.h>

namespace cu_op_mem {

template <typename T>
class Gemm : public Operator<T> {
    public:
        Gemm(bool transA = false,bool transB = false,T alpha = 1,T beta = 0);

        ~Gemm();
        
        // 移动构造函数
        Gemm(Gemm&& other) noexcept 
            : transA_(other.transA_), transB_(other.transB_), 
              alpha_(other.alpha_), beta_(other.beta_),
              weight_(std::move(other.weight_)), handle_(other.handle_) {
            other.handle_ = nullptr;
        }
        
        // 移动赋值操作符
        Gemm& operator=(Gemm&& other) noexcept {
            if (this != &other) {
                if (handle_) {
                    cublasDestroy(handle_);
                }
                transA_ = other.transA_;
                transB_ = other.transB_;
                alpha_ = other.alpha_;
                beta_ = other.beta_;
                weight_ = std::move(other.weight_);
                handle_ = other.handle_;
                other.handle_ = nullptr;
            }
            return *this;
        }
        
        // 删除复制构造函数和复制赋值操作符
        Gemm(const Gemm&) = delete;
        Gemm& operator=(const Gemm&) = delete;

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override;

        void SetWeight(const Tensor<T>& weight);
        
        // 函数调用操作符，用于JITWrapper
        StatusCode operator()(const Tensor<T>& input, Tensor<T>& output) {
            return Forward(input, output);
        }

    private:
        bool transA_,transB_;
        T alpha_,beta_;
        Tensor<T> weight_;
        cublasHandle_t handle_;

};

}