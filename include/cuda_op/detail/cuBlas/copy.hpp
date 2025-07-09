#pragma once 

#include "cuda_op/abstract/operator.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem {

template <typename T>
class Copy : public Operator<T> {

    public:
        Copy();

        ~Copy();

        StatusCode Forward(const Tensor<T>& x,Tensor<T>& y);

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
            return StatusCode::UNSUPPORTED_TYPE;
        }

};

}