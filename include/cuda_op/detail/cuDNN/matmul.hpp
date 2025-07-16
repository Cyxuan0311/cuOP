#ifndef CUDA_OP_MATMUL_OP_H_
#define CUDA_OP_MATMUL_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class MatMul : public Operator<T> {
   public:
    MatMul(bool transA = false, bool transB = false)
        : transA_(transA), transB_(transB) {}
    // 支持 batch 维度，默认 batch_dim=-1 表示无 batch
    StatusCode Forward(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, int batch_dim = -1);
    StatusCode Forward(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
        return Forward(A, B, C, -1);
    }
   private:
    bool transA_;
    bool transB_;
};

} // namespace cu_op_mem

#endif
