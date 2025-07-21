#include "cuda_op/detail/cuDNN/layernorm.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>

namespace cu_op_mem {

template <typename T>
__global__ void layernorm_forward_parallel_kernel(const T* input, T* output,
                                                 const T* gamma, const T* beta,
                                                 int batch, int norm_size, int norm_stride, T eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;
    const T* in_ptr = input + row * norm_stride;
    T* out_ptr = output + row * norm_stride;
    extern __shared__ T sdata[]; // sdata[0:blockDim.x] for sum, [blockDim.x:2*blockDim.x] for sqsum

    // 1. 每个线程处理部分元素
    T sum = 0, sqsum = 0;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        T v = in_ptr[i];
        sum += v;
        sqsum += v * v;
    }
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sqsum;
    __syncthreads();

    // 2. block内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    T mean = sdata[0] / norm_size;
    T var = sdata[blockDim.x] / norm_size - mean * mean;
    __syncthreads();

    // 3. 并行归一化写回
    for (int i = tid; i < norm_size; i += blockDim.x) {
        T v = in_ptr[i];
        T norm = (v - mean) / sqrt(var + eps);
        out_ptr[i] = norm * gamma[i] + beta[i];
    }
}

template <typename T>
StatusCode LayerNorm<T>::Forward(const Tensor<T>& input, Tensor<T>& output,
                                 const Tensor<T>& gamma, const Tensor<T>& beta,
                                 int normalized_dim, T eps) {
    const auto& shape = input.shape();
    int ndim = shape.size();
    if (normalized_dim < 0) normalized_dim += ndim;
    if (normalized_dim < 0 || normalized_dim >= ndim) {
        LOG(ERROR) << "LayerNorm: normalized_dim out of range.";
        return StatusCode::SHAPE_MISMATCH;
    }
    // 计算batch和norm_size
    int batch = 1, norm_size = 1;
    for (int i = 0; i < normalized_dim; ++i) batch *= shape[i];
    for (int i = normalized_dim; i < ndim; ++i) norm_size *= shape[i];
    int norm_stride = norm_size;
    if (output.data() == nullptr || output.shape() != shape) {
        output = Tensor<T>(shape);
    }
    int threads = 256;
    int blocks = batch;
    size_t shared_mem = threads * 2 * sizeof(T);
    layernorm_forward_parallel_kernel<T><<<blocks, threads, shared_mem>>>(
        input.data(), output.data(),
        gamma.data(), beta.data(), batch, norm_size, norm_stride, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "LayerNorm kernel failed: " << cudaGetErrorString(err);
        return StatusCode::CUDA_ERROR;
    }
    cudaDeviceSynchronize();
    return StatusCode::SUCCESS;
}

template class LayerNorm<float>;
template class LayerNorm<double>;

} // namespace cu_op_mem
