#ifndef CUDA_OP_FLATTEN_OP_H_
#define CUDA_OP_FLATTEN_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

    template <typename T>
    class Flatten : public Operator<T> {
        public:
            StatusCode Forward(const Tensor<T>& input,Tensor<T>& output,int batch_dim = 0);

            StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override {
                return Forward(input,output,0);
            }
    };

}


#endif