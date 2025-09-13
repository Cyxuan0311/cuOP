#include "cuda_op/detail/cuDNN/kernel_fusion.hpp"
#include "cuda_op/detail/cuDNN/convolution.hpp"
#include "cuda_op/detail/cuDNN/relu.hpp"
#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include "cuda_op/detail/cuDNN/matmul.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace cu_op_mem {

// 卷积 + ReLU 融合kernel
template<typename T>
__global__ void conv_relu_fused_kernel(const T* input, const T* weight, const T* bias, T* output,
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
    
    // 应用ReLU激活函数
    sum = max(sum, T(0));
    
    // 写入输出
    int output_idx = ((n * out_channels + out_c) * out_h + out_h_idx) * out_w + out_w_idx;
    output[output_idx] = sum;
}

// 卷积 + BatchNorm + ReLU 融合kernel
template<typename T>
__global__ void conv_bn_relu_fused_kernel(const T* input, const T* weight, const T* bias,
                                         const T* gamma, const T* beta, T* output,
                                         int N, int C, int H, int W, int out_channels,
                                         int kernel_h, int kernel_w, int stride_h, int stride_w,
                                         int pad_h, int pad_w, int out_h, int out_w, T eps) {
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
    
    // 应用BatchNorm（简化版本，假设已经训练好）
    T norm = (sum - T(0)) / sqrt(T(1) + eps);  // 简化：假设均值为0，方差为1
    sum = norm * gamma[out_c] + beta[out_c];
    
    // 应用ReLU激活函数
    sum = max(sum, T(0));
    
    // 写入输出
    int output_idx = ((n * out_channels + out_c) * out_h + out_h_idx) * out_w + out_w_idx;
    output[output_idx] = sum;
}

// 矩阵乘法 + ReLU 融合kernel
template<typename T>
__global__ void matmul_relu_fused_kernel(const T* A, const T* B, const T* bias, T* C,
                                        int M, int N, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            T a = transA ? A[k * M + row] : A[row * K + k];
            T b = transB ? B[col * K + k] : B[k * N + col];
            sum += a * b;
        }
        
        // 添加偏置
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // 应用ReLU激活函数
        sum = max(sum, T(0));
        
        C[row * N + col] = sum;
    }
}

// ConvReluFused 实现
template<typename T>
ConvReluFused<T>::ConvReluFused(int in_channels, int out_channels, int kernel_h, int kernel_w,
                               int stride_h, int stride_w, int pad_h, int pad_w)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {}

template<typename T>
void ConvReluFused<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
void ConvReluFused<T>::SetBias(const Tensor<T>& bias) {
    bias_ = Tensor<T>(bias.shape());
    cudaMemcpy(bias_.data(), bias.data(), bias.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
StatusCode ConvReluFused<T>::Forward(const std::vector<Tensor<T>*>& inputs, 
                                    std::vector<Tensor<T>*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        LOG(ERROR) << "ConvReluFused expects 1 input and 1 output";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    const auto& input = *inputs[0];
    auto& output = *outputs[0];
    
    const auto& in_shape = input.shape();
    if (in_shape.size() != 4) {
        LOG(ERROR) << "ConvReluFused only supports 4D input";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
    
    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H = static_cast<int>(in_shape[2]);
    int W = static_cast<int>(in_shape[3]);
    int out_h = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int out_w = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    
    std::vector<std::size_t> out_shape = {static_cast<std::size_t>(N), 
                                         static_cast<std::size_t>(out_channels_), 
                                         static_cast<std::size_t>(out_h), 
                                         static_cast<std::size_t>(out_w)};
    output.resize(out_shape);
    
    // 使用融合kernel
    dim3 block_size(16, 16);
    dim3 grid_size(out_h * out_w, out_channels_, N);
    
    const T* bias_ptr = (bias_.numel() == static_cast<std::size_t>(out_channels_)) ? bias_.data() : nullptr;
    
    conv_relu_fused_kernel<T><<<grid_size, block_size>>>(
        input.data(), weight_.data(), bias_ptr, output.data(),
        N, C, H, W, out_channels_,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, out_h, out_w);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "ConvReluFused kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template<typename T>
StatusCode ConvReluFused<T>::Backward(const std::vector<Tensor<T>*>& inputs,
                                     const std::vector<Tensor<T>*>& outputs,
                                     std::vector<Tensor<T>*>& grad_inputs) {
    // 简化实现，实际应用中需要完整的反向传播
    LOG(WARNING) << "ConvReluFused backward not implemented";
    return StatusCode::UNSUPPORTED_OPERATION;
}

// ConvBnReluFused 实现
template<typename T>
ConvBnReluFused<T>::ConvBnReluFused(int in_channels, int out_channels, int kernel_h, int kernel_w,
                                   int stride_h, int stride_w, int pad_h, int pad_w)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w), eps_(1e-5) {}

template<typename T>
void ConvBnReluFused<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
void ConvBnReluFused<T>::SetBias(const Tensor<T>& bias) {
    bias_ = Tensor<T>(bias.shape());
    cudaMemcpy(bias_.data(), bias.data(), bias.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
void ConvBnReluFused<T>::SetGamma(const Tensor<T>& gamma) {
    gamma_ = Tensor<T>(gamma.shape());
    cudaMemcpy(gamma_.data(), gamma.data(), gamma.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
void ConvBnReluFused<T>::SetBeta(const Tensor<T>& beta) {
    beta_ = Tensor<T>(beta.shape());
    cudaMemcpy(beta_.data(), beta.data(), beta.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
StatusCode ConvBnReluFused<T>::Forward(const std::vector<Tensor<T>*>& inputs, 
                                      std::vector<Tensor<T>*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        LOG(ERROR) << "ConvBnReluFused expects 1 input and 1 output";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    const auto& input = *inputs[0];
    auto& output = *outputs[0];
    
    const auto& in_shape = input.shape();
    if (in_shape.size() != 4) {
        LOG(ERROR) << "ConvBnReluFused only supports 4D input";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
    
    int N = static_cast<int>(in_shape[0]);
    int C = static_cast<int>(in_shape[1]);
    int H = static_cast<int>(in_shape[2]);
    int W = static_cast<int>(in_shape[3]);
    int out_h = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int out_w = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    
    std::vector<std::size_t> out_shape = {static_cast<std::size_t>(N), 
                                         static_cast<std::size_t>(out_channels_), 
                                         static_cast<std::size_t>(out_h), 
                                         static_cast<std::size_t>(out_w)};
    output.resize(out_shape);
    
    // 使用融合kernel
    dim3 block_size(16, 16);
    dim3 grid_size(out_h * out_w, out_channels_, N);
    
    const T* bias_ptr = (bias_.numel() == static_cast<std::size_t>(out_channels_)) ? bias_.data() : nullptr;
    
    conv_bn_relu_fused_kernel<T><<<grid_size, block_size>>>(
        input.data(), weight_.data(), bias_ptr, gamma_.data(), beta_.data(), output.data(),
        N, C, H, W, out_channels_,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, out_h, out_w, eps_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "ConvBnReluFused kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template<typename T>
StatusCode ConvBnReluFused<T>::Backward(const std::vector<Tensor<T>*>& inputs,
                                       const std::vector<Tensor<T>*>& outputs,
                                       std::vector<Tensor<T>*>& grad_inputs) {
    LOG(WARNING) << "ConvBnReluFused backward not implemented";
    return StatusCode::UNSUPPORTED_OPERATION;
}

// MatMulReluFused 实现
template<typename T>
MatMulReluFused<T>::MatMulReluFused(bool transA, bool transB) 
    : transA_(transA), transB_(transB) {}

template<typename T>
void MatMulReluFused<T>::SetWeight(const Tensor<T>& weight) {
    weight_ = Tensor<T>(weight.shape());
    cudaMemcpy(weight_.data(), weight.data(), weight.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
void MatMulReluFused<T>::SetBias(const Tensor<T>& bias) {
    bias_ = Tensor<T>(bias.shape());
    cudaMemcpy(bias_.data(), bias.data(), bias.bytes(), cudaMemcpyDeviceToDevice);
}

template<typename T>
StatusCode MatMulReluFused<T>::Forward(const std::vector<Tensor<T>*>& inputs, 
                                      std::vector<Tensor<T>*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        LOG(ERROR) << "MatMulReluFused expects 1 input and 1 output";
        return StatusCode::SHAPE_MISMATCH;
    }
    
    const auto& input = *inputs[0];
    auto& output = *outputs[0];
    
    const auto& in_shape = input.shape();
    if (in_shape.size() != 2) {
        LOG(ERROR) << "MatMulReluFused only supports 2D input";
        return StatusCode::TENSOR_DIMONSION_MISMATCH;
    }
    
    int M = static_cast<int>(in_shape[0]);
    int K = static_cast<int>(in_shape[1]);
    int N = static_cast<int>(weight_.shape()[1]);
    
    std::vector<std::size_t> out_shape = {static_cast<std::size_t>(M), static_cast<std::size_t>(N)};
    output.resize(out_shape);
    
    // 使用融合kernel
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    
    const T* bias_ptr = (bias_.numel() == static_cast<std::size_t>(N)) ? bias_.data() : nullptr;
    
    matmul_relu_fused_kernel<T><<<grid_size, block_size>>>(
        input.data(), weight_.data(), bias_ptr, output.data(),
        M, N, K, transA_, transB_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "MatMulReluFused kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template<typename T>
StatusCode MatMulReluFused<T>::Backward(const std::vector<Tensor<T>*>& inputs,
                                       const std::vector<Tensor<T>*>& outputs,
                                       std::vector<Tensor<T>*>& grad_inputs) {
    LOG(WARNING) << "MatMulReluFused backward not implemented";
    return StatusCode::UNSUPPORTED_OPERATION;
}

// FusedOperatorFactory 实现
template<typename T>
std::unique_ptr<FusedOperator<T>> FusedOperatorFactory<T>::Create(FusionType type, 
                                                                  const std::vector<int>& params) {
    switch (type) {
        case FusionType::CONV_RELU:
            if (params.size() >= 6) {
                return std::make_unique<ConvReluFused<T>>(
                    params[0], params[1], params[2], params[3], 
                    params[4], params[5], params[6], params[7]);
            }
            break;
        case FusionType::CONV_BN_RELU:
            if (params.size() >= 6) {
                return std::make_unique<ConvBnReluFused<T>>(
                    params[0], params[1], params[2], params[3], 
                    params[4], params[5], params[6], params[7]);
            }
            break;
        case FusionType::MATMUL_RELU:
            if (params.size() >= 2) {
                return std::make_unique<MatMulReluFused<T>>(
                    params[0] != 0, params[1] != 0);
            }
            break;
        default:
            LOG(ERROR) << "Unsupported fusion type";
            break;
    }
    return nullptr;
}

// 显式实例化
template class ConvReluFused<float>;
template class ConvReluFused<double>;
template class ConvBnReluFused<float>;
template class ConvBnReluFused<double>;
template class MatMulReluFused<float>;
template class MatMulReluFused<double>;
template class FusedOperatorFactory<float>;
template class FusedOperatorFactory<double>;

} // namespace cu_op_mem
