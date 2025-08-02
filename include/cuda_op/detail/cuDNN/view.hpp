#ifndef CUDA_OP_VIEW_OP_H_
#define CUDA_OP_VIEW_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class View : public Operator<T> {
    public:
        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output,const std::vector<std::size_t>& offset,const std::vector<std::size_t>& new_shape);

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
            return StatusCode::UNSUPPORTED_TYPE;
        }
};

}

#endif