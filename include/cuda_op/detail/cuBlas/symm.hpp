#pragma once 

#include "cuda_op/abstract/operator.hpp"
#include <cublas_v2.h>

namespace cu_op_mem {

template <typename T>
class Symm : public Operator<T> {
    public:
        Symm(cublasSideMode_t side = CUBLAS_SIDE_LEFT,
             cublasSideMode_t uplo = CUBLAS_FILL_MODE_UPPER,
             T alpha = 1, T beta = 0);

        ~Symm();

        void Forward(const Tensor<T>& input,Tensor<T>& output) override;

        void SetWeight(const Tensor<T>& weight);
        
    private:
        cublasSideMode_t side_;
        cublasFillMode_t uplo_;
        T alpha_,beta_;
        Tensor<T> weight_;
        cublasHandle_t handle_;
};

}
