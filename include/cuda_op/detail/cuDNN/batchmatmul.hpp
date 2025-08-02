#ifndef CUDA_OP_BATCHMATMUL_OP_H_
#define CUDA_OP_BATCHMATMUL_OP_H_

#include "cuda_op/abstract/operator.hpp"
#include <vector>

namespace cu_op_mem {

template <typename T>
class BatchMatMul : public Operator<T> {
   public:
    // 支持 batch 维度，默认 batch_dim=0 表示第0维为 batch
    StatusCode Forward(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, int batch_dim = 0);
    // 不支持无 batch 的 override
    StatusCode Forward(const Tensor<T>&, Tensor<T>&) override {
        return StatusCode::UNSUPPORTED_TYPE;
    }
};

} // namespace cu_op_mem

#endif