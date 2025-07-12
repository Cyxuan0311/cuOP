#ifndef CUDA_OP_SOFTMAX_OP_H_
#define CUDA_OP_SOFTMAX_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class Softmax : public Operator<T {
    StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override;
};

}


#endif
