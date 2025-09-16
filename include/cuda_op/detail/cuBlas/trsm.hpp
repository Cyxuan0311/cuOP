#pragma once 

#include "cuda_op/abstract/operator.hpp"
#include "util/status_code.hpp"
#include "data/tensor.hpp"

namespace cu_op_mem {

template <typename T>
class Trsm : public Operator<T> {
    public:
        Trsm(int side = 0,int uplo = 0,int trans = 0,int diag = 0,T alpha =1);

        ~Trsm();

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override;

        void SetAlpha(T alpha);
        void SetMatrixA(const Tensor<T>& A);
    private:
        int side_,uplo_,trans_,diag_;
        T alpha_;
        Tensor<T> matrix_A_;
};

}