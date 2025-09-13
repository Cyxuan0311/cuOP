#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "cuda_op/detail/cuBlas/gemm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <vector>

namespace cu_op_mem {

// 优化的Im2Col kernel: NCHW -> col，使用共享内存和向量化访问
template <typename T>
__global__ void im2col_optimized_kernel(const T* data_im, int channels, int height, int width,
                                        int kernel_h, int kernel_w, int pad_h, int pad_w,
                                        int stride_h, int stride_w,
                                        int out_h, int out_w, T* data_col) {
    // 使用共享内存缓存输入数据
    extern __shared__ T shared_data[];
    const int TILE_SIZE = 32;
    const int CHANNELS_PER_BLOCK = min(channels, 4); // 每个block处理的通道数
    
    int c_per_col = channels * kernel_h * kernel_w;
    int total_cols = out_h * out_w;
    
    // 协作加载输入数据到共享内存
    for (int c = 0; c < CHANNELS_PER_BLOCK; ++c) {
        int global_c = blockIdx.z * CHANNELS_PER_BLOCK + c;
        if (global_c >= channels) break;
        
        for (int h = threadIdx.y; h < height; h += blockDim.y) {
            for (int w = threadIdx.x; w < width; w += blockDim.x) {
                int shared_idx = c * height * width + h * width + w;
                int global_idx = global_c * height * width + h * width + w;
                if (shared_idx < blockDim.x * blockDim.y * CHANNELS_PER_BLOCK) {
                    shared_data[shared_idx] = data_im[global_idx];
                }
            }
        }
    }
    __syncthreads();
    
    // 计算输出
    int index = blockIdx.x * blockDim.x + threadIdx.x;
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
        
        // 检查边界
        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
            // 优先从共享内存读取
            int block_c = c_im % CHANNELS_PER_BLOCK;
            if (block_c < CHANNELS_PER_BLOCK) {
                int shared_idx = block_c * height * width + im_row * width + im_col;
                if (shared_idx < blockDim.x * blockDim.y * CHANNELS_PER_BLOCK) {
                    data_col[index] = shared_data[shared_idx];
                } else {
                    int im_idx = (c_im * height + im_row) * width + im_col;
                    data_col[index] = data_im[im_idx];
                }
            } else {
                int im_idx = (c_im * height + im_row) * width + im_col;
                data_col[index] = data_im[im_idx];
            }
        } else {
            data_col[index] = 0;
        }
    }
}

// 融合的卷积kernel，直接进行卷积计算而不使用im2col
template <typename T>
__global__ void convolution_fused_kernel(const T* input, const T* weight, const T* bias, T* output,
                                        int N, int C, int H, int W, int out_channels,
                                        int kernel_h, int kernel_w, int stride_h, int stride_w,
                                        int pad_h, int pad_w, int out_h, int out_w) {
    int n = blockIdx.z;
    int out_c = blockIdx.y;
    int out_h_idx = blockIdx.x / out_w;
    int out_w_idx = blockIdx.x % out_w;
    
    if (n >= N || out_c >= out_channels || out_h_idx >= out_h || out_w_idx >= out_w) return;
    
    T sum = 0;
    
    // 计算输入区域的起始位置
    int in_h_start = out_h_idx * stride_h - pad_h;
    int in_w_start = out_w_idx * stride_w - pad_w;
    
    // 遍历所有输入通道和卷积核
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;
                
                // 边界检查
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int input_idx = ((n * C + c) * H + in_h) * W + in_w;
                    int weight_idx = ((out_c * C + c) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    // 写入输出
    int output_idx = ((n * out_channels + out_c) * out_h + out_h_idx) * out_w + out_w_idx;
    output[output_idx] = sum;
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

    // 使用融合的卷积kernel
    dim3 block_size(16, 16);
    dim3 grid_size(out_h * out_w, out_channels_, N);
    
    const T* bias_ptr = (bias_.numel() == static_cast<std::size_t>(out_channels_)) ? bias_.data() : nullptr;
    
    convolution_fused_kernel<T><<<grid_size, block_size>>>(
        input.data(), weight_.data(), bias_ptr, output.data(),
        N, C, H, W, out_channels_,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, out_h, out_w);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "Convolution fused kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

// 显式实例化
template class Convolution2D<float>;
template class Convolution2D<double>;

} //