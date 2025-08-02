#pragma once
#include "util/status_code.hpp"
#include "tensor/tensor.hpp"
#include <vector>

namespace cu_op_mem {

template <typename T>
class Convolution2D {
public:
    Convolution2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
                  int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0);

    void SetWeight(const Tensor<T>& weight); // weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    void SetBias(const Tensor<T>& bias);     // bias shape: [out_channels]
    StatusCode Forward(const Tensor<T>& input, Tensor<T>& output);

private:
    int in_channels_, out_channels_, kernel_h_, kernel_w_;
    int stride_h_, stride_w_, pad_h_, pad_w_;
    Tensor<T> weight_;
    Tensor<T> bias_;
};

} // namespace cu_op_mem