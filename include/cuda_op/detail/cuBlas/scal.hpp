#pragma once 

#include "cuda_op/abstract/operator.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem {

template <typename T>
class Scal : public Operator<T> {
    public:
        Scal(T alpha = 1);
        ~Scal();

        StatusCode Forward(Tensor<T>& x);

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
            return StatusCode::UNSUPPORTED_TYPE;
        }

        void SetAlpha(T alpha);

    private:
        T alpha_;
};

}