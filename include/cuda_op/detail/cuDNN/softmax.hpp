#ifndef CUDA_OP_SOFTMAX_OP_H_
#define CUDA_OP_SOFTMAX_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class Softmax : public Operator<T> {
    public:
        // 新增 dim 参数，默认-1（最后一维），向后兼容
        StatusCode Forward(const Tensor<T>& input, Tensor<T>& output, int dim = -1);
        // 保留原接口，调用新接口
        StatusCode Forward(const Tensor<T>& input, Tensor<T>& output) override {
            return Forward(input, output, -1);
        }
};

}


#endif
