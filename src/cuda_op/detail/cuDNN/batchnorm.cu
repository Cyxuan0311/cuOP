#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

// 为float类型创建独立的kernel函数
__global__ void batchnorm_forward_shared_kernel_float(const float* input, float* output,
                                                      const float* gamma, const float* beta,
                                                      float* running_mean, float* running_var,
                                                      int N, int C, int HxW, float eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int num = N * HxW;
    extern __shared__ float batchnorm_shared_mem_float[];

    float sum = 0, sqsum = 0;
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        float v = input[(n * C + c) * HxW + hw];
        sum += v;
        sqsum += v * v;
    }
    batchnorm_shared_mem_float[tid] = sum;
    batchnorm_shared_mem_float[blockDim.x + tid] = sqsum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            batchnorm_shared_mem_float[tid] += batchnorm_shared_mem_float[tid + s];
            batchnorm_shared_mem_float[blockDim.x + tid] += batchnorm_shared_mem_float[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    float mean = batchnorm_shared_mem_float[0] / num;
    float var = batchnorm_shared_mem_float[blockDim.x] / num - mean * mean;
    if (tid == 0) {
        running_mean[c] = mean;
        running_var[c] = var;
    }
    __syncthreads();

    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        int idx = (n * C + c) * HxW + hw;
        float v = input[idx];
        float norm = (v - mean) / sqrt(var + eps);
        output[idx] = norm * gamma[c] + beta[c];
    }
}

// 为double类型创建独立的kernel函数
__global__ void batchnorm_forward_shared_kernel_double(const double* input, double* output,
                                                       const double* gamma, const double* beta,
                                                       double* running_mean, double* running_var,
                                                       int N, int C, int HxW, double eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int num = N * HxW;
    extern __shared__ double batchnorm_shared_mem_double[];

    double sum = 0, sqsum = 0;
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        double v = input[(n * C + c) * HxW + hw];
        sum += v;
        sqsum += v * v;
    }
    batchnorm_shared_mem_double[tid] = sum;
    batchnorm_shared_mem_double[blockDim.x + tid] = sqsum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            batchnorm_shared_mem_double[tid] += batchnorm_shared_mem_double[tid + s];
            batchnorm_shared_mem_double[blockDim.x + tid] += batchnorm_shared_mem_double[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    double mean = batchnorm_shared_mem_double[0] / num;
    double var = batchnorm_shared_mem_double[blockDim.x] / num - mean * mean;
    if (tid == 0) {
        running_mean[c] = mean;
        running_var[c] = var;
    }
    __syncthreads();

    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        int idx = (n * C + c) * HxW + hw;
        double v = input[idx];
        double norm = (v - mean) / sqrt(var + eps);
        output[idx] = norm * gamma[c] + beta[c];
    }
}

template <typename T>
StatusCode BatchNorm<T>::Forward(const Tensor<T>& input, Tensor<T>& output,
                                 const Tensor<T>& gamma, const Tensor<T>& beta,
                                 Tensor<T>& running_mean, Tensor<T>& running_var,
                                 T eps) {
    const auto& shape = input.shape();
    int N, C, HxW;
    if (shape.size() == 4) {
        N = shape[0];
        C = shape[1];
        HxW = shape[2] * shape[3];
    } else if (shape.size() == 2) {
        N = shape[0];
        C = shape[1];
        HxW = 1;
    } else {
        LOG(ERROR) << "BatchNorm only supports 2D [N,C] or 4D [N,C,H,W] input.";
        return StatusCode::SHAPE_MISMATCH;
    }
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    if (running_mean.data() == nullptr || running_mean.shape() != std::vector<std::size_t>{(size_t)C}) {
        running_mean = Tensor<T>({(size_t)C});
    }
    if (running_var.data() == nullptr || running_var.shape() != std::vector<std::size_t>{(size_t)C}) {
        running_var = Tensor<T>({(size_t)C});
    }
    int threads = 256;
    int blocks = C;
    size_t shared_mem = threads * 2 * sizeof(T);
    
    // 根据类型调用相应的kernel函数
    if constexpr (std::is_same_v<T, float>) {
        batchnorm_forward_shared_kernel_float<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), running_mean.data(), running_var.data(),
            N, C, HxW, eps);
    } else if constexpr (std::is_same_v<T, double>) {
        batchnorm_forward_shared_kernel_double<<<blocks, threads, shared_mem>>>(
            input.data(), output.data(),
            gamma.data(), beta.data(), running_mean.data(), running_var.data(),
            N, C, HxW, eps);
    } else {
        LOG(ERROR) << "BatchNorm only supports float and double types.";
        return StatusCode::UNSUPPORTED_OPERATION;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "BatchNorm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template class BatchNorm<float>;
template class BatchNorm<double>;

} // namespace cu_op_mem
