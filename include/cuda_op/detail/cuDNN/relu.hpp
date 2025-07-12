#ifndef CUDA_OP_RELU_OP_H_
#define CUDA_OP_RELU_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class Relu : public Operator<T> {
   public:
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output) override;
};

} // namespace cu_op_mem

#endif