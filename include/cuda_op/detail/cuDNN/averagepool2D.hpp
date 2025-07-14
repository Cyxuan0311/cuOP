#ifndef CUDA_OP_AVERAGEPOOL2D_OP_H_
#define CUDA_OP_AVERAGEPOOL2D_OP_H_

#include "cuda_op/abstract/operator.hpp"

namespace cu_op_mem {

template <typename T>
class AveragePool2D : public Operator<T> {
   public:
    AveragePool2D(int pool_height, int pool_width, int stride_height = 1, int stride_width = 1)
        : pool_height_(pool_height),
          pool_width_(pool_width),
          stride_height_(stride_height),
          stride_width_(stride_width) {}

    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output) override;

   private:
    int pool_height_;
    int pool_width_;
    int stride_height_;
    int stride_width_;
};

} // namespace cu_op_mem

#endif
