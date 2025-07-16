#ifndef GLOBALMAXPOOL2D_CUH
#define GLOBALMAXPOOL2D_CUH

#include "data/tensor.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem {

template <typename T>
class GlobalMaxPool2D {
    public:
        GlobalMaxPool2D() = default;

        StatusCode Forward(const Tensor<T>& input, Tensor<T>& output, int dim_h = 2, int dim_w = 3);
        StatusCode Forward(const Tensor<T>& input, Tensor<T>& output) override {
            return Forward(input, output, 2, 3);
        }
};

}

#endif