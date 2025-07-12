#pragma once

#include "cuda_op/abstract/operator.hpp"
#include <cublas_v2.h>

namespace cu_op_mem{
    
template <typename T>
class Axpy : public Operator<T> {
    public:
        Axpy(T alpha = 1);

        ~Axpy();

        StatusCode Forward(const Tensor<T>& x,Tensor<T>& y) override;

        void SetAlpha(T alpha);

    private:
        T alpha_;
        cublasHandle_t handle_;
};


}