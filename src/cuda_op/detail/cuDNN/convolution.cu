#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <vector>

namespace cu_op_mem {

// Im2Col kernel: NCHW -> col
template <typename T>
__global__ void im2col_kernel(const T* data_im, int channels, int height, int width,
                              int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int out_h, int out_w, T* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int c_per_col = channels * kernel_h * kernel_w;
    int total_cols = out_h * out_w;
    int total = c_per_col * total_cols;
    if (index < total) {
        int col = index % total_cols;
        int c = index / total_cols;
        int w_out = col % out_w;
        int h_out = col / out_w;
        int k_w = c % kernel_w;
        int k_h = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);

        int im_row = h_out * stride_h - pad_h + k_h;
        int im_col = w_out * stride_w - pad_w + k_w;
        int im_idx = (c_im * height + im_row) * width + im_col;

        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width)
            data_col[index] = data_im[im_idx];
        else
            data_col[index] = 0;
    }
}

template <typename T>
Convolution2D<T>::Convolution2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
                                int stride_h, int stride_w, int pad_h, int pad_w)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {}

template <typename T>
void Convolution2D<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
void Convolution2D<T>::SetBias(const Tensor<T>& bias) {
    bias_ = Tensor<T>(bias.shape());
    cudaMemcpy(bias_.data(), bias.data(), bias.bytes(), cudaMemcpyDeviceToDevice);
}

template <typename T>
StatusCode Convolution2D<T>::Forward(const Tensor<T>& input, Tensor<T>& output) {
    // input: [N, C, H, W]
    const auto& in_shape = input.shape();
    if (in_shape.size() != 4) {
        LOG(ERROR) << "Convolution2D only supports 4D input";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H = static_cast<int>(in_shape[2]);
    int W = static_cast<int>(in_shape[3]);
    int out_h = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int out_w = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    std::vector<std::size_t> out_shape = {static_cast<std::size_t>(N), static_cast<std::size_t>(out_channels_), static_cast<std::size_t>(out_h), static_cast<std::size_t>(out_w)};
    output.resize(out_shape);

    // im2col buffer: [C * kernel_h * kernel_w, out_h * out_w]
    Tensor<T> col({static_cast<std::size_t>(C * kernel_h_ * kernel_w_), static_cast<std::size_t>(out_h * out_w)});

    // weight: [out_channels, in_channels, kernel_h, kernel_w] -> [out_channels, C * kernel_h * kernel_w]
    // bias: [out_channels]
    Gemm<T> gemm(false, false, 1.0, 0.0);

    for (int n = 0; n < N; ++n) {
        const T* input_ptr = input.data() + n * C * H * W;
        T* col_ptr = col.data();

        int threads = 256;
        int total = C * kernel_h_ * kernel_w_ * out_h * out_w;
        int blocks = (total + threads - 1) / threads;
        im2col_kernel<T><<<blocks, threads>>>(input_ptr, C, H, W, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, out_h, out_w, col_ptr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "im2col kernel failed: " << cudaGetErrorString(err);
            return StatusCode::CUDA_ERROR;
        }

        // output[n]: [out_channels, out_h * out_w]
        Tensor<T> out_mat({static_cast<std::size_t>(out_channels_), static_cast<std::size_t>(out_h * out_w)});
        gemm.SetWeight(weight_.reshape({static_cast<std::size_t>(out_channels_), static_cast<std::size_t>(C * kernel_h_ * kernel_w_)}));
        StatusCode s = gemm.Forward(col, out_mat);
        if (s != StatusCode::SUCCESS) return s;

        // 加bias
        if (bias_.size() == static_cast<std::size_t>(out_channels_)) {
            int total_out = out_channels_ * out_h * out_w;
            int threads2 = 256;
            int blocks2 = (total_out + threads2 - 1) / threads2;
            const T* bias_ptr = bias_.data();
            T* out_ptr = out_mat.data();
            // bias add kernel
            auto bias_add_kernel = [] __global__ (T* out, const T* bias, int out_c, int out_hw, int total) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total) {
                    int c = idx / out_hw;
                    out[idx] += bias[c];
                }
            };
            bias_add_kernel<<<blocks2, threads2>>>(out_ptr, bias_ptr, out_channels_, out_h * out_w, total_out);
            cudaError_t err2 = cudaGetLastError();
            if (err2 != cudaSuccess) {
                LOG(ERROR) << "bias add kernel failed: " << cudaGetErrorString(err2);
                return StatusCode::CUDA_ERROR;
            }
        }

        // 拷贝到output
        cudaMemcpy(output.data() + n * out_channels_ * out_h * out_w, out_mat.data(), out_mat.bytes(), cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Convolution2D<float>;
template class Convolution2D<double>;

} //