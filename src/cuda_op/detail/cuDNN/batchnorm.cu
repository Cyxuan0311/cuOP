#include "cuda_op/detail/cuDNN/batchnorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

template <typename T>
__global__ void batchnorm_forward_shared_kernel(const T* input, T* output,
                                                const T* gamma, const T* beta,
                                                T* running_mean, T* running_var,
                                                int N, int C, int HxW, T eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int num = N * HxW;
    extern __shared__ T sdata[];

    T sum = 0, sqsum = 0;
    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        T v = input[(n * C + c) * HxW + hw];
        sum += v;
        sqsum += v * v;
    }
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sqsum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    T mean = sdata[0] / num;
    T var = sdata[blockDim.x] / num - mean * mean;
    if (tid == 0) {
        running_mean[c] = mean;
        running_var[c] = var;
    }
    __syncthreads();

    for (int i = tid; i < num; i += blockDim.x) {
        int n = i / HxW;
        int hw = i % HxW;
        int idx = (n * C + c) * HxW + hw;
        T v = input[idx];
        T norm = (v - mean) / sqrt(var + eps);
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
    batchnorm_forward_shared_kernel<T><<<blocks, threads, shared_mem>>>(
        input.data(), output.data(),
        gamma.data(), beta.data(), running_mean.data(), running_var.data(),
        N, C, HxW, eps);
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
