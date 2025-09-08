#pragma once 

#include "cuda_op/abstract/operator.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem{

template <typename T>
class Dot : public Operator<T> {
    public:
        Dot();
        ~Dot();

        StatusCode Forward(const Tensor<T>& x,const Tensor<T>& y,T& result);

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
            return StatusCode::UNSUPPORTED_TYPE;
        }
};

}