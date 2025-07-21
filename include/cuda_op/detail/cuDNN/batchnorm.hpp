#ifndef CUDA_OP_BATCHNORM_OP_H_
#define CUDA_OP_BATCHNORM_OP_H_

#include "cuda_op/abstract/operator.hpp"
#include <vector>

namespace cu_op_mem {

template <typename T>
class BatchNorm : public Operator<T> {
   public:
    // Forward: input, output, gamma, beta, running_mean, running_var, eps
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output,
                       const Tensor<T>& gamma, const Tensor<T>& beta,
                       Tensor<T>& running_mean, Tensor<T>& running_var,
                       T eps = static_cast<T>(1e-5));
    // 不支持无参数的 override
    StatusCode Forward(const Tensor<T>&, Tensor<T>&) override {
        return StatusCode::UNSUPPORTED_TYPE;
    }
};

} // namespace cu_op_mem

#endif
