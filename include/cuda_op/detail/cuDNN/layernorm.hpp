#ifndef CUDA_OP_LAYERNORM_OP_H_
#define CUDA_OP_LAYERNORM_OP_H_

#include "cuda_op/abstract/operator.hpp"
#include <vector>

namespace cu_op_mem {

template <typename T>
class LayerNorm : public Operator<T> {
   public:
    // Forward: input, output, gamma, beta, eps, normalized_dim
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output,
                       const Tensor<T>& gamma, const Tensor<T>& beta,
                       int normalized_dim = -1, T eps = static_cast<T>(1e-5));
    // 不支持无参数的 override
    StatusCode Forward(const Tensor<T>&, Tensor<T>&) override {
        return StatusCode::UNSUPPORTED_TYPE;
    }
};

} // namespace cu_op_mem

#endif
