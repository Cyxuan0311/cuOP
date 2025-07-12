#pragma once

#include "cuda_op/abstract/operator.hpp"
#include <cublas_v2.h>

namespace cu_op_mem {

template <typename T>
class Gemm : public Operator<T> {
    public:
        Gemm(bool transA = false,bool transB = false,T alpha = 1,T beta = 0);

        ~Gemm();

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override;

        void SetWeight(const Tensor<T>& weight);

    private:
        bool transA_,transB_;
        T alpha_,beta_;
        Tensor<T> weight_;
        cublasHandle_t handle_;

};

}