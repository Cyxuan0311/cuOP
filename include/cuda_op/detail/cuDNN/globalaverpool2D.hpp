#ifndef CUDA_OP_GLOBALAVERPOOL2D_OP_H_
#define CUDA_OP_GLOBALAVERPOOL2D_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class GlobalAveragePool2D : public Operator<T> {
   public:
    GlobalAveragePool2D() = default;
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h = 2, int dim_w = 3);
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output) override {
        return Forward(input, output, 2, 3);
    }
};

} // namespace cu_op_mem

#endif
