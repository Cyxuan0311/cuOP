#ifndef GLOBALMAXPOOL2D_CUH
#define GLOBALMAXPOOL2D_CUH

#include "data/tensor.hpp"
#include "util/status_code.hpp"

namespace cu_op_mem {

template <typename T>
class GlobalMaxPool2D {
    public:
        GlobalMaxPool2D() = default;

        StatusCode Forward(const Tensor<T>& input,Tensor<T>& output) override;
};

}

#endif